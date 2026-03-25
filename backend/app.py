"""
Leap Backend — Flask API
========================
Three engineering capabilities:
  1. Resume parsing       — OpenAI GPT-4o extracts profile from uploaded PDF/DOCX
  2. Job matching         — MongoDB Job_data matched against extracted skills
  3. Intelligent chat     — GPT-4o with full context: profile + jobs + UK 70-pt visa rules

Run:
    pip install -r requirements.txt
    python app.py
"""

import os, json, re, uuid
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from pymongo import MongoClient, TEXT
from bson import ObjectId
from openai import OpenAI
import PyPDF2, docx

# ── config ────────────────────────────────────────────────────────────────────
MONGO_URI   = os.getenv("MONGO_URI","your-mongo-uri-here")
OPENAI_KEY  = os.getenv("OPENAI_KEY", "your-openai-key-here")
DB_NAME     = "UK_job"
COLLECTION  = "Job_data"
PORT        = int(os.getenv("PORT", 5000))

app   = Flask(__name__)
CORS(app, origins="*")
oai   = OpenAI(api_key=OPENAI_KEY)

# In-memory session store (replace with Redis/MongoDB for production)
SESSIONS: dict[str, dict] = {}

# ── MongoDB ───────────────────────────────────────────────────────────────────
def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    return client[DB_NAME]

# ── helpers ───────────────────────────────────────────────────────────────────
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId): return str(o)
        if isinstance(o, datetime): return o.isoformat()
        return super().default(o)

def jresp(data, status=200):
    return Response(json.dumps(data, cls=JSONEncoder),
                    status=status, mimetype="application/json")

def extract_text_from_file(file_storage) -> str:
    """Extract raw text from uploaded PDF or DOCX."""
    filename  = file_storage.filename.lower()
    file_bytes = file_storage.read()

    if filename.endswith(".pdf"):
        import io
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    elif filename.endswith((".doc", ".docx")):
        import io
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    return file_bytes.decode("utf-8", errors="ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  UK 70-POINT VISA RULES ENGINE
# ══════════════════════════════════════════════════════════════════════════════

VISA_RULES = """
UK SKILLED WORKER VISA — 70-POINT SYSTEM (2024)

MANDATORY POINTS (50 pts — all three required):
  A. Job offer from UK-licensed sponsor                        → +20 pts
  B. Role at or above RQF Level 3 (A-level equivalent)        → +20 pts
  C. English language (B1 CEFR / IELTS 4.0+)                 → +10 pts

TRADEABLE POINTS (need 20 more after mandatory 50):
  D. Salary ≥ £38,700/yr (general threshold 2024)             → +20 pts
     OR Salary ≥ £30,960 (new entrant / shortage occ.)        → +20 pts
  E. Job is on the Immigration Salary List (shortage)         → +20 pts  (also unlocks lower salary)
  F. Relevant PhD qualification                               → +10 pts
  G. Relevant STEM PhD                                         → +20 pts

NOTES:
- Total needed: 70 pts minimum
- Mandatory 50 + at least one tradeable route = 70
- Shortage occupation roles: agriculture, nursing, engineering, IT, teaching, social work
- Points D, E, F/G are tradeable — you only need enough to reach 70
- English requirement: passport from majority-English-speaking country OR IELTS/equivalent OR degree taught in English
- CoS (Certificate of Sponsorship) issued by employer after visa application
"""

def calculate_visa_score(profile: dict) -> dict:
    """
    Calculate UK 70-point visa score from user profile.
    Returns score + breakdown + gap analysis.
    """
    score       = 0
    breakdown   = {}
    suggestions = []

    # A: Sponsor job offer — we infer this from whether they have UK job offer
    has_offer = profile.get("has_job_offer", False)
    breakdown["sponsor_offer"] = {"pts": 20 if has_offer else 0, "label": "Sponsor job offer", "status": "earned" if has_offer else "pending"}
    if has_offer:
        score += 20
    else:
        suggestions.append({"action": "Apply to sponsored roles on Leap", "pts": 20, "priority": "high"})

    # B: Role level — assume RQF3+ for professional roles
    role = (profile.get("role") or "").lower()
    professional_keywords = ["engineer","developer","analyst","manager","designer","architect",
                              "consultant","scientist","doctor","nurse","teacher","accountant",
                              "lawyer","pharmacist","researcher","director","specialist","coordinator"]
    is_professional = any(k in role for k in professional_keywords) or profile.get("experience_years", 0) >= 2
    breakdown["role_level"] = {"pts": 20 if is_professional else 0, "label": "Role at RQF Level 3+", "status": "earned" if is_professional else "pending"}
    if is_professional:
        score += 20
    else:
        suggestions.append({"action": "Ensure target role is RQF Level 3+", "pts": 20, "priority": "high"})

    # C: English language — assume YES for Indian professionals (native English education)
    english_ok = profile.get("english_ok", True)
    breakdown["english"] = {"pts": 10 if english_ok else 0, "label": "English B1+", "status": "earned" if english_ok else "pending"}
    if english_ok:
        score += 10
    else:
        suggestions.append({"action": "Take IELTS (target 4.0+)", "pts": 10, "priority": "high"})

    # D: Salary threshold
    salary_gbp = profile.get("expected_salary_gbp", 0)
    if salary_gbp >= 38700:
        score += 20
        breakdown["salary"] = {"pts": 20, "label": f"Salary ≥ £38,700 (yours: £{salary_gbp:,})", "status": "earned"}
    elif salary_gbp >= 30960:
        score += 20
        breakdown["salary"] = {"pts": 20, "label": f"Salary ≥ £30,960 new-entrant (yours: £{salary_gbp:,})", "status": "earned"}
    else:
        breakdown["salary"] = {"pts": 0, "label": "Salary meets threshold (£38,700+)", "status": "pending"}
        suggestions.append({"action": "Target roles paying £38,700+ — or £30,960 on shortage list", "pts": 20, "priority": "high"})

    # E: Shortage occupation
    is_shortage = profile.get("is_shortage_role", False)
    breakdown["shortage"] = {"pts": 20 if is_shortage else 0, "label": "Shortage occupation role", "status": "earned" if is_shortage else "optional"}
    if is_shortage:
        score += 20
    else:
        suggestions.append({"action": "Check if your role is on the Immigration Salary List", "pts": 20, "priority": "medium"})

    # F/G: PhD
    edu = (profile.get("education") or "").lower()
    has_phd  = "phd" in edu or "doctorate" in edu
    is_stem  = any(k in role for k in ["engineer","scientist","developer","data","research","biotech","chem","phys","math"])
    if has_phd and is_stem:
        score += 20
        breakdown["phd"] = {"pts": 20, "label": "STEM PhD", "status": "earned"}
    elif has_phd:
        score += 10
        breakdown["phd"] = {"pts": 10, "label": "Relevant PhD", "status": "earned"}
    else:
        breakdown["phd"] = {"pts": 0, "label": "PhD (optional boost)", "status": "optional"}
        suggestions.append({"action": "A relevant PhD adds +10 or +20 pts", "pts": 10, "priority": "low"})

    return {
        "total":       score,
        "target":      70,
        "gap":         max(70 - score, 0),
        "qualifies":   score >= 70,
        "breakdown":   breakdown,
        "suggestions": sorted(suggestions, key=lambda x: {"high":0,"medium":1,"low":2}[x["priority"]]),
        "percentage":  round((score / 70) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  JOB MATCHING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def match_jobs_to_profile(profile: dict, filters: dict = None, limit: int = 20) -> list:
    """
    Query MongoDB Job_data and score jobs against user profile.
    filters: optional dict from chat intent (role_filter, city_filter, salary_min)
    """
    try:
        db  = get_db()
        col = db[COLLECTION]

        # Build MongoDB query
        query = {}
        skills   = [s.lower() for s in (profile.get("skills") or [])]
        role_kw  = (profile.get("role") or "").lower()

        # Apply chat filters if present
        if filters:
            if filters.get("role_filter"):
                query["$or"] = [
                    {"title":       {"$regex": filters["role_filter"], "$options": "i"}},
                    {"description": {"$regex": filters["role_filter"], "$options": "i"}},
                ]
            if filters.get("city_filter"):
                query["location"] = {"$regex": filters["city_filter"], "$options": "i"}
            if filters.get("salary_min"):
                query["salary_min"] = {"$gte": filters["salary_min"]}
            if filters.get("sponsored_only"):
                query["visa_sponsored"] = True

        # Fetch candidates
        raw_jobs  = list(col.find(query).limit(limit * 3))

        # Score each job against profile
        scored = []
        for job in raw_jobs:
            job_text = " ".join([
                str(job.get("title", "")),
                str(job.get("description", "")),
                str(job.get("requirements", "")),
            ]).lower()

            score = 0

            # Role title match
            if role_kw and role_kw in job_text:
                score += 30

            # Skills match
            matched_skills = []
            for skill in skills:
                if skill in job_text:
                    score += 10
                    matched_skills.append(skill)

            # Experience match
            exp_years = profile.get("experience_years", 0)
            job_exp   = job.get("experience_required", 0) or 0
            if isinstance(job_exp, str):
                nums = re.findall(r'\d+', str(job_exp))
                job_exp = int(nums[0]) if nums else 0
            if exp_years >= job_exp:
                score += 15

            # Salary match
            salary_min = job.get("salary_min") or job.get("salary") or 0
            if isinstance(salary_min, str):
                nums = re.findall(r'\d+', str(salary_min).replace(",",""))
                salary_min = int(nums[0]) if nums else 0
            if salary_min > 0:
                score += 5  # has salary info

            scored.append({
                "_id":            str(job.get("_id", "")),
                "title":          job.get("title") or job.get("job_title") or "Role",
                "company":        job.get("company") or job.get("employer") or "Company",
                "location":       job.get("location") or job.get("city") or "UK",
                "salary":         _format_salary(job),
                "salary_gbp":     salary_min,
                "visa_sponsored": bool(job.get("visa_sponsored") or job.get("sponsored")),
                "description":    (job.get("description") or "")[:300],
                "requirements":   job.get("requirements") or [],
                "matched_skills": matched_skills,
                "match_score":    min(score, 100),
                "posted_date":    str(job.get("posted_date") or job.get("date_posted") or ""),
                "apply_url":      job.get("apply_url") or job.get("url") or "",
            })

        # Sort by match score, return top N
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        return scored[:limit]

    except Exception as e:
        print(f"Job matching error: {e}")
        return _fallback_jobs(profile)


def _format_salary(job: dict) -> str:
    """Format salary field regardless of schema."""
    for field in ["salary", "salary_range", "salary_min", "pay"]:
        val = job.get(field)
        if val:
            s = str(val).strip()
            if s and s != "0":
                if "£" not in s and s.replace(",","").isdigit():
                    return f"£{int(s.replace(',','')):,}/yr"
                return s
    return "Competitive"


def _fallback_jobs(profile: dict) -> list:
    """Fallback sample jobs if DB is unavailable."""
    role = profile.get("role", "Professional")
    return [
        {"_id": "f1", "title": f"Senior {role}", "company": "HSBC Technology",
         "location": "London", "salary": "£72,000/yr", "salary_gbp": 72000,
         "visa_sponsored": True, "matched_skills": [], "match_score": 85,
         "description": "Exciting opportunity for an experienced professional.", "requirements": [], "apply_url": ""},
        {"_id": "f2", "title": role, "company": "Barclays Digital",
         "location": "Manchester", "salary": "£58,000/yr", "salary_gbp": 58000,
         "visa_sponsored": True, "matched_skills": [], "match_score": 78,
         "description": "Join our growing UK team.", "requirements": [], "apply_url": ""},
        {"_id": "f3", "title": f"Lead {role}", "company": "Amazon UK",
         "location": "Edinburgh", "salary": "£65,000/yr", "salary_gbp": 65000,
         "visa_sponsored": True, "matched_skills": [], "match_score": 72,
         "description": "Shape the future of technology.", "requirements": [], "apply_url": ""},
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME PARSING
# ══════════════════════════════════════════════════════════════════════════════

RESUME_PARSE_PROMPT = """
You are a professional resume parser. Extract the following from the resume text and return ONLY valid JSON.

Return this exact structure:
{
  "name": "full name",
  "email": "email if found",
  "phone": "phone if found",
  "role": "current or most recent job title",
  "experience_years": <integer years total experience>,
  "current_company": "current employer name",
  "education": "highest qualification (e.g. B.Tech Computer Science, IIT Delhi)",
  "skills": ["skill1", "skill2", ...],  // top 8-10 technical/professional skills
  "languages": ["English", ...],
  "certifications": ["cert1", ...],
  "summary": "2-3 sentence professional summary based on the resume",
  "english_ok": true
}

Be precise. Extract only what is explicitly in the resume. Return ONLY the JSON, no other text.

RESUME TEXT:
{resume_text}
"""

def parse_resume_with_openai(text: str) -> dict:
    resp = oai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",  "content": "You are a precise resume parser. Return only valid JSON."},
            {"role": "user",    "content": RESUME_PARSE_PROMPT.format(resume_text=text[:6000])},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """
You are Leap's UK Career Advisor — an expert on UK Skilled Worker Visa, sponsored jobs, and career strategy for Indian professionals moving to the UK.

USER PROFILE:
- Name: {name}
- Role: {role}
- Experience: {experience_years} years
- Skills: {skills}
- Education: {education}
- Current company: {current_company}

VISA SCORE: {visa_score}/70 pts (gap: {visa_gap} pts)
VISA BREAKDOWN:
{visa_breakdown}

TOP JOB MATCHES ({job_count} total):
{top_jobs}

UK 70-POINT VISA RULES:
{visa_rules}

YOUR CAPABILITIES:
1. Filter jobs: Extract job requirements from user message → call filter_jobs tool
2. Improve visa score: Explain exactly what the user needs to reach 70 pts
3. Answer visa questions: Use the rules above to give precise answers
4. Suggest next steps: Actionable, specific advice based on their actual profile
5. Show sponsored roles: Identify which jobs are visa-sponsored

GUARDRAILS — NEVER:
- Answer questions unrelated to UK jobs, visa, career, or skills
- Make up job listings or visa rules not in your context
- Give legal advice (recommend consulting UKVI for legal questions)
- Respond to jokes, trivia, or off-topic requests

If asked something off-topic, say:
"I'm focused on helping you with your UK career journey. I can help with job matching, visa score improvement, sponsored roles, and career strategy. What would you like to know?"

RESPONSE STYLE:
- Concise but complete
- After every response, suggest 2-3 relevant follow-up questions the user might want to ask
- Format follow-ups as: **You might also want to ask:**\n- "question 1"\n- "question 2"
- When filtering jobs, describe what filter you applied and how many results remain
- Use specific numbers from their profile (their actual score, their actual skills)
"""

INTENT_CLASSIFIER = """
Classify this user message. Return JSON only:
{
  "intent": "filter_jobs" | "visa_question" | "improve_score" | "show_sponsored" | "general_career" | "off_topic",
  "filters": {
    "role_filter": "string or null",
    "city_filter": "string or null",
    "salary_min": number_or_null,
    "sponsored_only": boolean
  }
}

Message: "{message}"
"""

def classify_intent(message: str) -> dict:
    """Lightweight intent classification using gpt-4o-mini."""
    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": INTENT_CLASSIFIER.format(message=message)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=150,
        )
        return json.loads(resp.choices[0].message.content)
    except:
        return {"intent": "general_career", "filters": {}}


def build_system_prompt(session: dict) -> str:
    profile     = session.get("profile", {})
    visa_data   = session.get("visa_score", {})
    jobs        = session.get("jobs", [])

    # Format top 5 jobs for context
    top_jobs_text = ""
    for i, j in enumerate(jobs[:5], 1):
        top_jobs_text += (
            f"{i}. {j['title']} @ {j['company']} | {j['location']} | "
            f"{j['salary']} | Match: {j['match_score']}% | "
            f"Visa sponsored: {'Yes' if j['visa_sponsored'] else 'Unknown'}\n"
        )

    # Format visa breakdown
    breakdown_text = ""
    for k, v in (visa_data.get("breakdown") or {}).items():
        status_icon = "✓" if v["status"] == "earned" else ("~" if v["status"] == "optional" else "✗")
        breakdown_text += f"  {status_icon} {v['label']}: {v['pts']} pts ({v['status']})\n"

    return SYSTEM_PROMPT_TEMPLATE.format(
        name             = profile.get("name", "User"),
        role             = profile.get("role", "Professional"),
        experience_years = profile.get("experience_years", 0),
        skills           = ", ".join(profile.get("skills") or []),
        education        = profile.get("education", "Not specified"),
        current_company  = profile.get("current_company", "Not specified"),
        visa_score       = visa_data.get("total", 0),
        visa_gap         = visa_data.get("gap", 70),
        visa_breakdown   = breakdown_text or "  Not yet calculated",
        job_count        = len(jobs),
        top_jobs         = top_jobs_text or "  No jobs loaded yet",
        visa_rules       = VISA_RULES,
    )


def chat_stream(session: dict, message: str, history: list):
    """Stream GPT-4o response with full context."""
    system_prompt = build_system_prompt(session)

    messages = [{"role": "system", "content": system_prompt}]
    # Include last 10 turns of history
    for turn in history[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    stream = oai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
        temperature=0.4,
        max_tokens=800,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# ══════════════════════════════════════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── Session management ────────────────────────────────────────────────────────

@app.route("/api/session/create", methods=["POST"])
def create_session():
    """Create session from mad-lib answers."""
    body = request.json or {}
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "id":           session_id,
        "madlib":       body,
        "profile":      {},
        "visa_score":   {},
        "jobs":         [],
        "chat_history": [],
        "created_at":   datetime.now(timezone.utc).isoformat(),
    }
    return jresp({"session_id": session_id})


@app.route("/api/session/<sid>", methods=["GET"])
def get_session(sid):
    s = SESSIONS.get(sid)
    if not s:
        return jresp({"error": "Session not found"}, 404)
    return jresp(s)


# ── Resume parsing ────────────────────────────────────────────────────────────

@app.route("/api/resume/parse", methods=["POST"])
def parse_resume():
    """
    Accept PDF/DOCX upload.
    Returns: extracted profile + visa score + matched jobs.
    """
    sid  = request.form.get("session_id", "")
    file = request.files.get("resume")

    if not file:
        return jresp({"error": "No file uploaded"}, 400)

    # Extract text
    try:
        raw_text = extract_text_from_file(file)
    except Exception as e:
        return jresp({"error": f"Could not read file: {e}"}, 400)

    if len(raw_text.strip()) < 50:
        return jresp({"error": "File appears to be empty or unreadable"}, 400)

    # Parse with OpenAI
    try:
        profile = parse_resume_with_openai(raw_text)
    except Exception as e:
        return jresp({"error": f"Resume parsing failed: {e}"}, 500)

    # Merge mad-lib answers
    if sid and sid in SESSIONS:
        madlib = SESSIONS[sid].get("madlib", {})
        if madlib.get("role") and not profile.get("role"):
            profile["role"] = madlib["role"]
        if madlib.get("goal"):
            profile["goal"] = madlib["goal"]
        profile["salary_range_inr"] = madlib.get("salary", "")
        profile["experience_years"] = profile.get("experience_years") or _exp_from_string(madlib.get("exp",""))

    # Estimate expected UK salary
    profile["expected_salary_gbp"] = _estimate_uk_salary(profile)

    # Calculate visa score
    visa_data = calculate_visa_score(profile)

    # Match jobs
    jobs = match_jobs_to_profile(profile)

    # Store in session
    if sid and sid in SESSIONS:
        SESSIONS[sid]["profile"]    = profile
        SESSIONS[sid]["visa_score"] = visa_data
        SESSIONS[sid]["jobs"]       = jobs

    return jresp({
        "profile":    profile,
        "visa_score": visa_data,
        "jobs":       jobs,
    })


def _exp_from_string(s: str) -> int:
    """'3-5 years' → 4"""
    nums = re.findall(r'\d+', str(s))
    if len(nums) >= 2:
        return (int(nums[0]) + int(nums[1])) // 2
    if len(nums) == 1:
        return int(nums[0])
    return 3


def _estimate_uk_salary(profile: dict) -> int:
    """Rough UK salary estimate based on role and experience."""
    role = (profile.get("role") or "").lower()
    exp  = profile.get("experience_years", 3) or 3

    base_map = {
        "software engineer": 60000, "developer": 55000, "data analyst": 50000,
        "data scientist": 65000, "product manager": 70000, "designer": 50000,
        "finance": 55000, "accountant": 45000, "nurse": 38000, "doctor": 65000,
        "teacher": 38000, "marketing": 45000, "operations": 45000,
        "consultant": 60000, "analyst": 50000, "engineer": 55000,
    }
    base = 45000
    for k, v in base_map.items():
        if k in role:
            base = v
            break

    # Scale for experience
    multiplier = 1 + (min(exp, 12) * 0.025)
    return int(base * multiplier)


# ── Jobs ──────────────────────────────────────────────────────────────────────

@app.route("/api/jobs", methods=["GET"])
def get_jobs():
    """Get jobs for a session (with optional filters)."""
    sid     = request.args.get("session_id", "")
    s       = SESSIONS.get(sid, {})
    profile = s.get("profile", {})

    filters = {
        "role_filter":   request.args.get("role"),
        "city_filter":   request.args.get("city"),
        "salary_min":    int(request.args.get("salary_min", 0)) or None,
        "sponsored_only":request.args.get("sponsored") == "true",
    }
    filters = {k: v for k, v in filters.items() if v}

    jobs = match_jobs_to_profile(profile, filters=filters)

    if sid and sid in SESSIONS:
        SESSIONS[sid]["jobs"] = jobs

    return jresp({"jobs": jobs, "total": len(jobs)})


# ── Visa score ────────────────────────────────────────────────────────────────

@app.route("/api/visa-score", methods=["GET", "POST"])
def visa_score():
    sid = request.args.get("session_id") or (request.json or {}).get("session_id", "")
    s   = SESSIONS.get(sid, {})

    if request.method == "POST":
        # Allow updating profile fields (e.g. from chat suggestions)
        updates = request.json or {}
        updates.pop("session_id", None)
        if "profile" in s:
            s["profile"].update(updates)
            s["visa_score"] = calculate_visa_score(s["profile"])
            SESSIONS[sid] = s

    return jresp(s.get("visa_score") or calculate_visa_score(s.get("profile", {})))


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Non-streaming chat endpoint.
    Classifies intent → optionally re-fetches jobs → streams GPT-4o response.
    """
    body    = request.json or {}
    sid     = body.get("session_id", "")
    message = (body.get("message") or "").strip()

    if not message:
        return jresp({"error": "No message provided"}, 400)

    s = SESSIONS.get(sid, {"profile": {}, "visa_score": {}, "jobs": [], "chat_history": []})

    # Classify intent
    intent_data = classify_intent(message)
    intent      = intent_data.get("intent", "general_career")
    filters     = intent_data.get("filters", {}) or {}

    # If job filter intent — refresh jobs with new filters
    jobs_updated = False
    if intent == "filter_jobs" and any(filters.values()):
        new_jobs = match_jobs_to_profile(s.get("profile", {}), filters=filters)
        if sid and sid in SESSIONS:
            SESSIONS[sid]["jobs"] = new_jobs
            s["jobs"] = new_jobs
        jobs_updated = True

    # If visa improvement intent — update score based on message
    if intent == "improve_score":
        # Re-calculate with any new info mentioned in message
        new_profile = s.get("profile", {}).copy()
        if "phd" in message.lower():
            new_profile["education"] = (new_profile.get("education") or "") + " PhD"
        if "ielts" in message.lower():
            new_profile["english_ok"] = True
        new_visa = calculate_visa_score(new_profile)
        if sid and sid in SESSIONS:
            SESSIONS[sid]["visa_score"] = new_visa
            s["visa_score"] = new_visa

    # Generate response
    history = s.get("chat_history", [])
    full_response = ""
    for token in chat_stream(s, message, history):
        full_response += token

    # Save to history
    if sid and sid in SESSIONS:
        SESSIONS[sid]["chat_history"].append({"role": "user",      "content": message})
        SESSIONS[sid]["chat_history"].append({"role": "assistant", "content": full_response})

    return jresp({
        "response":     full_response,
        "intent":       intent,
        "jobs_updated": jobs_updated,
        "jobs":         s.get("jobs", []) if jobs_updated else None,
        "visa_score":   s.get("visa_score") if intent == "improve_score" else None,
    })


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream_route():
    """Streaming chat endpoint — sends tokens as SSE."""
    body    = request.json or {}
    sid     = body.get("session_id", "")
    message = (body.get("message") or "").strip()

    if not message:
        return jresp({"error": "No message"}, 400)

    s = SESSIONS.get(sid, {"profile": {}, "visa_score": {}, "jobs": [], "chat_history": []})

    # Intent + filter (same as non-streaming)
    intent_data  = classify_intent(message)
    intent       = intent_data.get("intent", "general_career")
    filters      = intent_data.get("filters", {}) or {}

    if intent == "filter_jobs" and any(filters.values()):
        new_jobs = match_jobs_to_profile(s.get("profile", {}), filters=filters)
        if sid and sid in SESSIONS:
            SESSIONS[sid]["jobs"] = new_jobs
            s["jobs"] = new_jobs

    history   = s.get("chat_history", [])
    collected = []

    def generate():
        # Send intent metadata first
        yield f"data: {json.dumps({'type':'meta','intent':intent,'filters':filters})}\n\n"

        for token in chat_stream(s, message, history):
            collected.append(token)
            yield f"data: {json.dumps({'type':'token','content':token})}\n\n"

        full = "".join(collected)
        if sid and sid in SESSIONS:
            SESSIONS[sid]["chat_history"].append({"role": "user",      "content": message})
            SESSIONS[sid]["chat_history"].append({"role": "assistant", "content": full})

        # Send updated jobs if filter was applied
        updated_jobs = SESSIONS.get(sid, {}).get("jobs", []) if intent == "filter_jobs" else None
        yield f"data: {json.dumps({'type':'done','jobs':updated_jobs})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Booking (dummy) ───────────────────────────────────────────────────────────

@app.route("/api/booking", methods=["POST"])
def create_booking():
    """Dummy booking — stores in memory (connect to MongoDB for production)."""
    body = request.json or {}
    booking = {
        "id":          str(uuid.uuid4()),
        "name":        body.get("name"),
        "email":       body.get("email"),
        "whatsapp":    body.get("whatsapp"),
        "date":        body.get("date"),
        "time":        body.get("time"),
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "status":      "confirmed",
    }
    return jresp({"booking": booking, "message": "Booking confirmed! You'll receive a WhatsApp reminder."})


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    try:
        db = get_db()
        db.command("ping")
        db_status = "connected"
        job_count = db[COLLECTION].count_documents({})
    except Exception as e:
        db_status = f"error: {e}"
        job_count = 0

    return jresp({
        "status":    "ok",
        "db":        db_status,
        "job_count": job_count,
        "time":      datetime.now(timezone.utc).isoformat(),
    })


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nLeap Backend starting on port {PORT}")
    print(f"MongoDB: {MONGO_URI[:50]}...")
    print(f"OpenAI:  {OPENAI_KEY[:20]}...\n")
    app.run(host="0.0.0.0", port=PORT, debug=True)
