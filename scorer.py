"""
scorer.py — Core candidate scoring engine for Redrob Hackathon
No external API calls. Pure Python + pandas. CPU-only. <5 min for 100K candidates.

Scoring breakdown:
  35% — Title & career fit
  25% — Skill depth (AI/ML relevance)
  20% — Experience quality
  15% — Behavioral availability (redrob_signals)
   5% — Soft fit signals
"""

import json
import re
from datetime import datetime, date
from typing import Optional

# ─── JD-derived constants ────────────────────────────────────────────────────

# Must-have AI/ML skills from JD
CORE_AI_SKILLS = {
    # Embeddings & retrieval (JD says "absolutely need")
    "embeddings", "sentence-transformers", "vector search", "semantic search",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "bge", "e5", "retrieval", "dense retrieval", "hybrid search",
    "vector database", "vector db", "ann", "approximate nearest neighbor",

    # Ranking & evaluation (JD: "absolutely need")
    "ranking", "learning to rank", "ltr", "ndcg", "mrr", "map", "a/b testing",
    "offline evaluation", "reranking", "re-ranking", "bm25", "information retrieval",

    # LLM / NLP (JD: "would like")
    "llm", "large language model", "fine-tuning", "fine tuning", "lora", "qlora",
    "peft", "rag", "retrieval augmented generation", "nlp", "transformers",
    "bert", "gpt", "hugging face", "huggingface", "langchain", "llamaindex",

    # ML production
    "mlops", "ml pipeline", "model serving", "inference", "tensorflow", "pytorch",
    "scikit-learn", "sklearn", "xgboost", "recommendation system", "recommender",
    "feature store", "embedding drift", "index refresh",
}

# Skills from JD "explicitly do NOT want" — penalize if these dominate
UNRELATED_SKILLS = {
    "computer vision", "image classification", "object detection", "cnn",
    "speech recognition", "tts", "text to speech", "asr",
    "photoshop", "illustrator", "figma", "canva",
    "tailwind", "react", "vue", "angular", "css", "html",
    "seo", "google ads", "digital marketing", "social media",
    "accounting", "tally", "quickbooks", "erp",
    "autocad", "solidworks", "mechanical design",
}

# Good company types (product companies)
PRODUCT_COMPANY_SIGNALS = {
    "startup", "saas", "product", "platform", "fintech", "edtech", "healthtech",
    "ai", "ml", "series", "backed", "funded", "ventures",
}

# Bad companies per JD
SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "l&t infotech",
    "ltimindtree", "mindtree", "persistent", "mastech", "niit technologies",
}

# Good titles for this role
GOOD_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "applied scientist", "applied ml", "applied ai",
    "nlp engineer", "search engineer", "ranking engineer",
    "senior engineer", "staff engineer", "principal engineer",
    "data scientist", "research engineer", "founding engineer",
}

# Red flag titles (keywords from sample_submission.csv showing the trap)
BAD_TITLES = {
    "hr manager", "human resources", "recruiter", "talent acquisition",
    "marketing manager", "content writer", "graphic designer",
    "accountant", "finance manager", "sales executive", "sales manager",
    "operations manager", "project manager", "business analyst",
    "civil engineer", "mechanical engineer", "customer support",
    "product manager",  # borderline, but not for this role
}

# Preferred Indian cities from JD
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurgaon", "gurugram", "hyderabad",
    "mumbai", "bangalore", "bengaluru", "chennai", "india",
}

# Education tiers (from schema)
EDU_TIER_SCORE = {
    "tier_1": 1.0,   # IITs, IISc, NITs top, IIMs
    "tier_2": 0.75,
    "tier_3": 0.5,
    "tier_4": 0.25,
    "tier_5": 0.1,
    None: 0.35,
}

TODAY = date.today()


# ─── Helper utilities ─────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.lower().strip() if text else ""


def days_since(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (TODAY - d).days
    except Exception:
        return None


def title_is_bad(title: str) -> bool:
    t = normalize(title)
    return any(bad in t for bad in BAD_TITLES)


def title_is_good(title: str) -> bool:
    t = normalize(title)
    return any(good in t for good in GOOD_TITLES)


def company_is_services(company: str) -> bool:
    c = normalize(company)
    return any(s in c for s in SERVICES_COMPANIES)


def is_product_company(company: str, industry: str, size: str) -> bool:
    c = normalize(company)
    ind = normalize(industry or "")
    # Clear services = False
    if any(s in c for s in SERVICES_COMPANIES):
        return False
    # Clear product signals
    if any(p in ind for p in ["saas", "ai", "fintech", "edtech", "startup", "software product"]):
        return True
    if any(p in c for p in PRODUCT_COMPANY_SIGNALS):
        return True
    # Small/mid size usually product
    if size in ("1-10", "11-50", "51-200", "201-500", "501-1000"):
        return True
    return False


def skill_name_matches_ai(skill_name: str) -> bool:
    s = normalize(skill_name)
    return any(core in s for core in CORE_AI_SKILLS)


def skill_is_unrelated(skill_name: str) -> bool:
    s = normalize(skill_name)
    return any(u in s for u in UNRELATED_SKILLS)


# ─── Honeypot detection ───────────────────────────────────────────────────────

def detect_honeypot(candidate: dict) -> float:
    """
    Returns a honeypot penalty multiplier: 1.0 = clean, 0.0 = certain honeypot.
    Checks:
      - Title-skill mismatch (HR manager with 9 AI skills)
      - Impossible tenure (8 yrs at company founded 3 yrs ago)
      - Expert in 10+ skills with 0 total endorsements
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    current_title = profile.get("current_title", "")
    penalty = 1.0

    # 1. Title-skill mismatch: non-AI title but claims many AI skills
    if title_is_bad(current_title):
        ai_skill_count = sum(1 for s in skills if skill_name_matches_ai(s["name"]))
        if ai_skill_count >= 6:
            penalty *= 0.05  # strong honeypot signal
        elif ai_skill_count >= 3:
            penalty *= 0.3

    # 2. Impossible tenure: years at company > company age inferred from start
    for job in career:
        if job.get("is_current") and job.get("start_date"):
            start = days_since(job["start_date"])
            yoe = profile.get("years_of_experience", 0) or 0
            if start and yoe > 0:
                # If claimed YOE >> months at current company and all career is 1 company
                if len(career) == 1 and yoe > (start / 365) + 1.5:
                    penalty *= 0.4

    # 3. Expert proficiency with 0 endorsements across many skills
    expert_zero_endorse = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("endorsements", 0) == 0
    )
    if expert_zero_endorse >= 5:
        penalty *= 0.3
    elif expert_zero_endorse >= 3:
        penalty *= 0.6

    return min(penalty, 1.0)


# ─── Component 1: Title & Career Fit (35%) ───────────────────────────────────

def score_title_career(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    current_title = profile.get("current_title", "")
    score = 0.0

    # Current title match
    if title_is_good(current_title):
        score += 0.45
    elif title_is_bad(current_title):
        score += 0.0  # floor
    else:
        # Partial — engineer/developer/scientist not in bad list
        t = normalize(current_title)
        if any(x in t for x in ["engineer", "developer", "scientist", "researcher", "architect"]):
            score += 0.25
        else:
            score += 0.1

    # Career progression analysis
    product_company_months = 0
    services_only = True
    ai_ml_roles = 0
    total_months = 0

    for job in career:
        months = job.get("duration_months", 0) or 0
        total_months += months
        company = job.get("company", "")
        industry = job.get("industry", "")
        size = job.get("current_company_size") or job.get("company_size", "")
        title = job.get("title", "")
        desc = job.get("description", "") or ""

        if is_product_company(company, industry, size):
            product_company_months += months
            services_only = False

        if title_is_good(title):
            ai_ml_roles += 1

        # Check for production deployment in descriptions
        if any(kw in normalize(desc) for kw in [
            "production", "deployed", "real users", "scale", "serving",
            "shipped", "launched", "live", "million", "thousand users",
        ]):
            score += 0.03  # small bonus per production mention, capped later

    # Services-only career = penalty (JD explicit)
    if services_only and career:
        score *= 0.4

    # Product company experience ratio
    if total_months > 0:
        product_ratio = min(product_company_months / total_months, 1.0)
        score += 0.3 * product_ratio

    # AI/ML role history
    score += min(ai_ml_roles * 0.08, 0.2)

    return min(score, 1.0)


# ─── Component 2: Skill Depth (25%) ──────────────────────────────────────────

def score_skills(candidate: dict) -> float:
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    ai_skill_score = 0.0
    unrelated_penalty = 0.0
    total_skills = len(skills)

    for skill in skills:
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "beginner")
        endorsements = skill.get("endorsements", 0) or 0
        duration = skill.get("duration_months", 0) or 0

        if skill_name_matches_ai(name):
            # Base score by proficiency
            prof_score = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}.get(
                proficiency, 0.3
            )
            # Endorsement credibility
            endorse_bonus = min(endorsements / 50, 0.3)
            # Duration (real use vs listed)
            duration_bonus = min(duration / 36, 0.2)  # max bonus at 3yr
            ai_skill_score += prof_score + endorse_bonus + duration_bonus

        elif skill_is_unrelated(name):
            unrelated_penalty += 0.05

    # Normalize: good candidates have 5-10 solid AI skills
    normalized = min(ai_skill_score / 8.0, 1.0)

    # Penalize if unrelated skills dominate
    unrelated_ratio = unrelated_penalty / max(total_skills * 0.05, 0.01)
    normalized *= max(1.0 - unrelated_ratio * 0.3, 0.5)

    return min(normalized, 1.0)


# ─── Component 3: Experience Quality (20%) ───────────────────────────────────

def score_experience(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])

    yoe = profile.get("years_of_experience", 0) or 0
    score = 0.0

    # JD says 5-9 years is the sweet spot
    if 5 <= yoe <= 9:
        score += 0.4
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        score += 0.3
    elif 3 <= yoe < 4 or 11 < yoe <= 13:
        score += 0.2
    elif yoe >= 13:
        score += 0.15  # too senior / potential mismatch
    else:
        score += 0.05  # too junior

    # Education tier
    best_tier = None
    for edu in education:
        tier = edu.get("tier")
        if tier and (best_tier is None or EDU_TIER_SCORE.get(tier, 0) > EDU_TIER_SCORE.get(best_tier, 0)):
            best_tier = tier
    score += 0.2 * EDU_TIER_SCORE.get(best_tier, 0.35)

    # Production deployment signals in career descriptions
    prod_signals = 0
    for job in career:
        desc = normalize(job.get("description", "") or "")
        if any(kw in desc for kw in ["production", "deployed to", "real users", "at scale", "million requests"]):
            prod_signals += 1

    score += min(prod_signals * 0.08, 0.25)

    # Research-only penalty (JD explicitly says no)
    all_research = all(
        any(kw in normalize(j.get("industry", "") + j.get("title", ""))
            for kw in ["research", "university", "lab", "academia", "phd"])
        for j in career
    ) if career else False

    if all_research:
        score *= 0.3

    # Pure LangChain/framework-only signals (JD: "framework enthusiasts")
    summary = normalize(profile.get("summary", "") or "")
    if summary.count("langchain") >= 2 and "production" not in summary:
        score *= 0.7

    return min(score, 1.0)


# ─── Component 4: Behavioral Availability (15%) ───────────────────────────────

def score_behavioral(candidate: dict) -> float:
    signals = candidate.get("redrob_signals", {})
    if not signals:
        return 0.3  # neutral if no signals

    score = 0.5  # start neutral

    # Open to work (big signal)
    if signals.get("open_to_work_flag"):
        score += 0.15

    # Last active recency
    days_inactive = days_since(signals.get("last_active_date"))
    if days_inactive is not None:
        if days_inactive <= 14:
            score += 0.15
        elif days_inactive <= 30:
            score += 0.1
        elif days_inactive <= 90:
            score += 0.05
        elif days_inactive > 180:
            score -= 0.15
        elif days_inactive > 120:
            score -= 0.1

    # Response rate (very important for actual hireability)
    response_rate = signals.get("recruiter_response_rate", 0) or 0
    if response_rate >= 0.6:
        score += 0.15
    elif response_rate >= 0.3:
        score += 0.08
    elif response_rate < 0.1:
        score -= 0.15

    # Notice period (JD: "love sub-30-day notice")
    notice = signals.get("notice_period_days", 60) or 60
    if notice <= 15:
        score += 0.1
    elif notice <= 30:
        score += 0.07
    elif notice > 90:
        score -= 0.1

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 50) or 50
    if completeness >= 80:
        score += 0.05
    elif completeness < 50:
        score -= 0.05

    # Interview completion rate
    icr = signals.get("interview_completion_rate", 0.5) or 0.5
    score += (icr - 0.5) * 0.1

    # Saved by recruiters recently (market validation)
    saved = signals.get("saved_by_recruiters_30d", 0) or 0
    if saved >= 5:
        score += 0.05

    return max(0.0, min(score, 1.0))


# ─── Component 5: Soft Fit Signals (5%) ──────────────────────────────────────

def score_soft_fit(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])

    score = 0.0

    # Location preference
    location = normalize(profile.get("location", "") + " " + profile.get("country", ""))
    if any(loc in location for loc in PREFERRED_LOCATIONS):
        score += 0.3
    if signals.get("willing_to_relocate"):
        score += 0.15

    # GitHub activity (indicates active builder)
    github_score = signals.get("github_activity_score", 0) or 0
    score += min(github_score / 100, 0.25)

    # Summary quality (async writing = proxy for "writes a lot" culture fit)
    summary = profile.get("summary", "") or ""
    word_count = len(summary.split())
    if word_count >= 100:
        score += 0.15
    elif word_count >= 50:
        score += 0.08

    # Startup experience
    for job in career:
        size = job.get("company_size") or job.get("current_company_size", "")
        if size in ("1-10", "11-50", "51-200"):
            score += 0.1
            break

    return min(score, 1.0)


# ─── Final weighted score ─────────────────────────────────────────────────────

WEIGHTS = {
    "title_career": 0.35,
    "skills": 0.25,
    "experience": 0.20,
    "behavioral": 0.15,
    "soft_fit": 0.05,
}


def score_candidate(candidate: dict) -> dict:
    """Score a single candidate. Returns dict with component scores + final score."""

    # Detect honeypot first
    honeypot_mult = detect_honeypot(candidate)

    components = {
        "title_career": score_title_career(candidate),
        "skills": score_skills(candidate),
        "experience": score_experience(candidate),
        "behavioral": score_behavioral(candidate),
        "soft_fit": score_soft_fit(candidate),
    }

    weighted = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    final = weighted * honeypot_mult

    return {
        "candidate_id": candidate["candidate_id"],
        "final_score": round(final, 6),
        "honeypot_mult": round(honeypot_mult, 3),
        **{f"score_{k}": round(v, 4) for k, v in components.items()},
    }


def generate_reasoning(candidate: dict, scores: dict) -> str:
    """Generate a concise 1-2 sentence reasoning string for the submission."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0) or 0
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    country = profile.get("country", "")

    # Top AI skills
    ai_skills = [
        s["name"] for s in candidate.get("skills", [])
        if skill_name_matches_ai(s["name"]) and s.get("endorsements", 0) >= 5
    ][:3]

    notice = signals.get("notice_period_days", "?")
    response_rate = signals.get("recruiter_response_rate", 0)
    open_to_work = signals.get("open_to_work_flag", False)

    parts = []

    # Part 1: Role + skills
    skill_str = ", ".join(ai_skills) if ai_skills else "ML/AI background"
    parts.append(
        f"{title} with {yoe:.1f} yrs; {skill_str}"
        + (f"; based in {location}" if location else "")
    )

    # Part 2: Availability + caveats
    avail_parts = []
    if open_to_work:
        avail_parts.append("actively looking")
    if isinstance(notice, (int, float)):
        avail_parts.append(f"{int(notice)}d notice")
    if response_rate:
        avail_parts.append(f"response rate {response_rate:.0%}")

    if scores.get("honeypot_mult", 1.0) < 0.5:
        parts.append("flagged: title-skill mismatch")
    elif avail_parts:
        parts.append("; ".join(avail_parts))

    return ". ".join(parts) + "."
