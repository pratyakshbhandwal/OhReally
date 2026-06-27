"""
scorer.py — Upgraded candidate scoring engine for Redrob Hackathon
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Key upgrades over v1:
  1. Semantic scoring via sentence-transformers (cosine sim against JD)
  2. Dynamic JD parsing — weights/skills extracted from JD text at runtime
  3. Rich description analysis — reads what candidates actually DID
  4. Fixed product company detection (no longer rewards all small companies)
  5. Better honeypot detection (skill inflation + description incoherence)
  6. Score distribution normalization (sigmoid spread)
  7. Evidence-based reasoning generation

Scoring breakdown:
  30% — Semantic fit (sentence-transformer cosine similarity vs JD)
  25% — Title & career fit (improved product co detection)
  20% — Skill depth (endorsed + duration weighted)
  15% — Experience quality (rich description parsing)
  10% — Behavioral availability (redrob_signals)
"""

import re
import math
from datetime import datetime, date
from typing import Optional, List
from functools import lru_cache

# ─── Lazy-load sentence-transformers (fallback gracefully if not installed) ────
_sbert_model = None
_jd_embedding = None
_semantic_available = False

def _load_semantic_model():
    global _sbert_model, _semantic_available
    if _sbert_model is not None:
        return _semantic_available
    try:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        _semantic_available = True
        print("[INFO] Semantic model loaded: all-MiniLM-L6-v2")
    except ImportError:
        print("[WARN] sentence-transformers not installed. Falling back to keyword scoring.")
        print("[WARN] Run: pip install sentence-transformers")
        _semantic_available = False
    return _semantic_available


# ─── The Job Description (full text — used for semantic embedding) ─────────────
# This is the actual JD for the Redrob Hackathon.
# If you have a --jd flag, this gets replaced at runtime.
DEFAULT_JD_TEXT = """
We are hiring a Senior AI/ML Engineer to build and own our candidate search and ranking systems.

You will absolutely need:
- Experience with vector embeddings, semantic search, dense retrieval
- Hands-on work with vector databases: FAISS, Pinecone, Weaviate, Qdrant or similar
- Knowledge of information retrieval: BM25, reranking, NDCG, MRR evaluation
- Production deployment experience — shipped real systems to real users at scale
- 5-9 years of industry experience in ML/AI roles at product companies

We would love if you also have:
- LLM fine-tuning, RAG pipelines, LangChain or LlamaIndex experience
- NLP: transformers, BERT, GPT, Hugging Face
- MLOps: model serving, inference pipelines, feature stores

We explicitly do NOT want:
- Candidates from pure services companies (TCS, Infosys, Wipro, etc.)
- Research-only profiles with no production work
- Framework enthusiasts who only know LangChain without understanding the underlying systems
- Computer vision, speech, or frontend-only profiles
- HR, marketing, sales, or non-technical roles

Location: Pune, Noida, Delhi NCR, Gurgaon, Hyderabad, Mumbai, Bangalore preferred.
Notice period: Under 30 days strongly preferred.
"""

# ─── JD-derived constants (these are defaults; parse_jd() can override) ────────

CORE_AI_SKILLS = {
    "embeddings", "sentence-transformers", "vector search", "semantic search",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "bge", "e5", "retrieval", "dense retrieval", "hybrid search",
    "vector database", "vector db", "ann", "approximate nearest neighbor",
    "ranking", "learning to rank", "ltr", "ndcg", "mrr", "map", "a/b testing",
    "offline evaluation", "reranking", "re-ranking", "bm25", "information retrieval",
    "llm", "large language model", "fine-tuning", "fine tuning", "lora", "qlora",
    "peft", "rag", "retrieval augmented generation", "nlp", "transformers",
    "bert", "gpt", "hugging face", "huggingface", "langchain", "llamaindex",
    "mlops", "ml pipeline", "model serving", "inference", "tensorflow", "pytorch",
    "scikit-learn", "sklearn", "xgboost", "recommendation system", "recommender",
    "feature store", "embedding drift", "index refresh",
}

UNRELATED_SKILLS = {
    "computer vision", "image classification", "object detection", "cnn",
    "speech recognition", "tts", "text to speech", "asr",
    "photoshop", "illustrator", "figma", "canva",
    "tailwind", "react", "vue", "angular", "css", "html",
    "seo", "google ads", "digital marketing", "social media",
    "accounting", "tally", "quickbooks", "erp",
    "autocad", "solidworks", "mechanical design",
}

# Explicit services firms (expanded list)
SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "l&t infotech",
    "ltimindtree", "mindtree", "persistent systems", "mastech", "niit technologies",
    "ibm global services", "atos", "dxc technology", "unisys", "ntt data",
    "syntel", "igate", "patni", "zensar", "cyient", "mps limited",
}

# Strong product-company industry signals (NOT just company name/size)
PRODUCT_INDUSTRY_SIGNALS = {
    "saas", "software as a service", "fintech", "edtech", "healthtech",
    "proptech", "legaltech", "artificial intelligence", "machine learning",
    "internet", "e-commerce", "marketplace", "consumer tech",
    "software product", "technology product",
}

# Strong product-company name signals (fundraising, tech-first)
PRODUCT_NAME_SIGNALS = {
    "labs", "ai", "ml", "tech", "io", "hq", "data", "analytics", "intelligence",
}

GOOD_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "applied scientist", "applied ml", "applied ai",
    "nlp engineer", "search engineer", "ranking engineer",
    "senior engineer", "staff engineer", "principal engineer",
    "data scientist", "research engineer", "founding engineer",
}

BAD_TITLES = {
    "hr manager", "human resources", "recruiter", "talent acquisition",
    "marketing manager", "content writer", "graphic designer",
    "accountant", "finance manager", "sales executive", "sales manager",
    "operations manager", "business analyst",
    "civil engineer", "mechanical engineer", "customer support",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurgaon", "gurugram", "hyderabad",
    "mumbai", "bangalore", "bengaluru", "chennai", "india",
}

EDU_TIER_SCORE = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.5,
    "tier_4": 0.25,
    "tier_5": 0.1,
    None: 0.35,
}

# Keywords that indicate REAL production work in job descriptions
PRODUCTION_KEYWORDS = {
    "production", "deployed", "shipped", "launched", "live", "real users",
    "at scale", "million requests", "billion", "latency", "throughput",
    "serving", "inference", "prod", "a/b test", "monitoring", "uptime",
    "100k", "1m", "10m", "qps", "rps",
}

# Keywords that indicate IMPACT (quantified results)
IMPACT_KEYWORDS = {
    "improved", "reduced", "increased", "achieved", "cut", "boosted",
    "optimized", "accelerated", "%", "percent", "x faster", "x reduction",
    "from", "to", "down from", "up from", "saving", "saved",
}

TODAY = date.today()


# ─── JD text → extracted signals (dynamic JD parsing) ────────────────────────

def parse_jd(jd_text: str) -> dict:
    """
    Extract structured signals from raw JD text.
    Returns a dict that can override scorer constants.
    Used to make the system JD-agnostic.
    """
    jd_lower = jd_text.lower()

    # Extract year range from text like "5-9 years" or "5 to 9 years"
    yoe_matches = re.findall(r"(\d+)\s*[-–to]+\s*(\d+)\s*years?", jd_lower)
    yoe_range = (int(yoe_matches[0][0]), int(yoe_matches[0][1])) if yoe_matches else (4, 10)

    # Extract notice period preference
    notice_matches = re.findall(r"(\d+)\s*[-–]?\s*day\s*notice", jd_lower)
    preferred_notice = int(notice_matches[0]) if notice_matches else 30

    # Extract must-have skills (lines with "need", "must", "required", "absolutely")
    must_have_lines = [
        line for line in jd_text.split("\n")
        if any(kw in line.lower() for kw in ["need", "must", "required", "absolutely", "need:"])
    ]

    # Extract nice-to-have skills
    nice_to_have_lines = [
        line for line in jd_text.split("\n")
        if any(kw in line.lower() for kw in ["nice", "love", "prefer", "bonus", "plus", "would like"])
    ]

    return {
        "yoe_range": yoe_range,
        "preferred_notice": preferred_notice,
        "must_have_text": " ".join(must_have_lines),
        "nice_to_have_text": " ".join(nice_to_have_lines),
        "full_text": jd_text,
    }

# Initialize with default JD
_jd_signals = parse_jd(DEFAULT_JD_TEXT)


def set_jd(jd_text: str):
    """Call this to override the JD at runtime (e.g. from CLI --jd flag or Streamlit input)."""
    global _jd_signals, _jd_embedding
    _jd_signals = parse_jd(jd_text)
    _jd_embedding = None  # reset cached embedding
    print(f"[INFO] JD updated. YOE range: {_jd_signals['yoe_range']}, notice: {_jd_signals['preferred_notice']}d")


# ─── Semantic scoring (the big upgrade) ──────────────────────────────────────

def _get_jd_embedding():
    """Get (or compute + cache) the JD embedding."""
    global _jd_embedding
    if _jd_embedding is not None:
        return _jd_embedding
    if not _semantic_available:
        return None
    _jd_embedding = _sbert_model.encode(_jd_signals["full_text"], convert_to_tensor=False)
    return _jd_embedding


def _cosine_similarity(a, b) -> float:
    """Pure numpy cosine similarity."""
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def score_semantic(candidate: dict) -> float:
    """
    Embed candidate's summary + top skills + career descriptions and compute
    cosine similarity vs the JD embedding.
    Falls back to 0.5 (neutral) if sentence-transformers not available.
    """
    if not _semantic_available or _sbert_model is None:
        return 0.5  # neutral fallback

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Build a rich candidate text: summary + skill names + job descriptions
    parts = []

    summary = profile.get("summary", "") or ""
    if summary:
        parts.append(summary)

    # Top 10 AI skills by endorsement
    top_skills = sorted(
        [s for s in skills if skill_name_matches_ai(s.get("name", ""))],
        key=lambda s: s.get("endorsements", 0),
        reverse=True
    )[:10]
    if top_skills:
        parts.append("Skills: " + ", ".join(s["name"] for s in top_skills))

    # Most recent 2 job descriptions
    for job in career[:2]:
        desc = job.get("description", "") or ""
        if desc:
            parts.append(desc[:500])  # trim to avoid token bloat

    if not parts:
        return 0.3  # no text = low score

    candidate_text = " ".join(parts)[:2000]  # cap total length

    try:
        cand_emb = _sbert_model.encode(candidate_text, convert_to_tensor=False)
        jd_emb = _get_jd_embedding()
        if jd_emb is None:
            return 0.5
        sim = _cosine_similarity(cand_emb, jd_emb)
        # Cosine sim for text typically ranges 0.1–0.8; normalize to 0–1
        # Floor at 0.1 (background noise), ceil at 0.75 (very high match)
        normalized = (sim - 0.1) / (0.75 - 0.1)
        return max(0.0, min(normalized, 1.0))
    except Exception:
        return 0.5


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


def is_services_company(company: str) -> bool:
    """Explicit services firm check — name-based only."""
    c = normalize(company)
    return any(s in c for s in SERVICES_COMPANIES)


def is_product_company(company: str, industry: str, size: str) -> bool:
    """
    FIX: v1 treated all small companies as product companies.
    Now: requires a positive signal (industry OR name+size combo), not just size.
    """
    c = normalize(company)
    ind = normalize(industry or "")

    # Definite NO: explicit services firm
    if is_services_company(company):
        return False

    # Definite YES: industry explicitly says product/SaaS/AI etc.
    if any(p in ind for p in PRODUCT_INDUSTRY_SIGNALS):
        return True

    # Probable YES: company name has strong tech-product signals AND is small
    # (requires BOTH a name signal AND small size — not size alone)
    name_has_signal = any(p in c for p in PRODUCT_NAME_SIGNALS)
    is_small = size in ("1-10", "11-50", "51-200")
    if name_has_signal and is_small:
        return True

    # Mid-size with technology in industry
    if size in ("201-500", "501-1000") and any(p in ind for p in ["technology", "software", "internet"]):
        return True

    # Default: unknown = not product (conservative)
    return False


def skill_name_matches_ai(skill_name: str) -> bool:
    s = normalize(skill_name)
    return any(core in s for core in CORE_AI_SKILLS)


def skill_is_unrelated(skill_name: str) -> bool:
    s = normalize(skill_name)
    return any(u in s for u in UNRELATED_SKILLS)


def count_production_signals(text: str) -> int:
    """Count distinct production-work signals in a block of text."""
    t = normalize(text)
    return sum(1 for kw in PRODUCTION_KEYWORDS if kw in t)


def count_impact_signals(text: str) -> int:
    """Count evidence of quantified impact in a text block."""
    t = normalize(text)
    return sum(1 for kw in IMPACT_KEYWORDS if kw in t)


# ─── Honeypot detection (upgraded) ────────────────────────────────────────────

def detect_honeypot(candidate: dict) -> float:
    """
    Returns multiplier: 1.0 = clean, close to 0.0 = honeypot.

    Upgraded checks:
      1. Title-skill mismatch (original)
      2. Impossible tenure (original)
      3. Expert skills with 0 endorsements (original)
      4. NEW: Skill count inflation (30+ skills = spam listing)
      5. NEW: Description incoherence (bad title + AI jargon in description)
      6. NEW: Zero career history with high claimed YOE
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    current_title = profile.get("current_title", "")
    yoe = profile.get("years_of_experience", 0) or 0
    penalty = 1.0

    # 1. Title-skill mismatch
    if title_is_bad(current_title):
        ai_skill_count = sum(1 for s in skills if skill_name_matches_ai(s.get("name", "")))
        if ai_skill_count >= 6:
            penalty *= 0.05
        elif ai_skill_count >= 3:
            penalty *= 0.3

    # 2. Impossible tenure
    for job in career:
        if job.get("is_current") and job.get("start_date"):
            start = days_since(job["start_date"])
            if start and yoe > 0 and len(career) == 1:
                if yoe > (start / 365) + 1.5:
                    penalty *= 0.4

    # 3. Expert proficiency with 0 endorsements
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("endorsements", 0) == 0
    )
    if expert_zero >= 5:
        penalty *= 0.3
    elif expert_zero >= 3:
        penalty *= 0.6

    # 4. NEW: Skill list inflation (spamming 30+ skills = fake profile signal)
    if len(skills) >= 30:
        penalty *= 0.5
    elif len(skills) >= 20:
        penalty *= 0.75

    # 5. NEW: High YOE with zero career history
    if yoe >= 5 and not career:
        penalty *= 0.3

    # 6. NEW: Description incoherence — bad title + AI-heavy descriptions
    if title_is_bad(current_title) and career:
        all_descs = " ".join(job.get("description", "") or "" for job in career)
        ai_jargon_in_desc = sum(1 for kw in CORE_AI_SKILLS if kw in normalize(all_descs))
        if ai_jargon_in_desc >= 8:
            penalty *= 0.2  # HR manager writing AI job descriptions = honeypot

    return min(max(penalty, 0.0), 1.0)


# ─── Component 1: Semantic Fit (30%) — NEW ───────────────────────────────────
# (This replaces/upgrades the old keyword-only skill matching)
# Implemented in score_semantic() above


# ─── Component 2: Title & Career Fit (25%) — upgraded ────────────────────────

def score_title_career(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    current_title = profile.get("current_title", "")
    score = 0.0

    # Current title match
    if title_is_good(current_title):
        score += 0.45
    elif title_is_bad(current_title):
        score += 0.0
    else:
        t = normalize(current_title)
        if any(x in t for x in ["engineer", "developer", "scientist", "researcher", "architect"]):
            score += 0.25
        else:
            score += 0.1

    # Career analysis
    product_company_months = 0
    services_only = True
    ai_ml_roles = 0
    total_months = 0
    production_evidence = 0

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

        # Rich production signal detection (not just keyword count)
        prod_count = count_production_signals(desc)
        impact_count = count_impact_signals(desc)
        if prod_count >= 2 and impact_count >= 1:
            production_evidence += 1  # real production work = both signals together

    # Services-only penalty
    if services_only and career:
        score *= 0.4

    # Product company ratio
    if total_months > 0:
        product_ratio = min(product_company_months / total_months, 1.0)
        score += 0.3 * product_ratio

    # AI/ML role history
    score += min(ai_ml_roles * 0.08, 0.2)

    # Production evidence bonus
    score += min(production_evidence * 0.04, 0.12)

    return min(score, 1.0)


# ─── Component 3: Skill Depth (20%) — upgraded ───────────────────────────────

def score_skills(candidate: dict) -> float:
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    ai_skill_score = 0.0
    unrelated_count = 0
    total_skills = len(skills)

    for skill in skills:
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "beginner")
        endorsements = skill.get("endorsements", 0) or 0
        duration = skill.get("duration_months", 0) or 0

        if skill_name_matches_ai(name):
            prof_score = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}.get(
                proficiency, 0.3
            )
            endorse_bonus = min(endorsements / 50, 0.3)
            duration_bonus = min(duration / 36, 0.2)
            ai_skill_score += prof_score + endorse_bonus + duration_bonus

        elif skill_is_unrelated(name):
            unrelated_count += 1

    # Normalize: 5-10 solid AI skills = good
    normalized = min(ai_skill_score / 8.0, 1.0)

    # Penalize if unrelated skills are majority
    if total_skills > 0:
        unrelated_ratio = unrelated_count / total_skills
        normalized *= max(1.0 - unrelated_ratio * 0.5, 0.4)

    return min(normalized, 1.0)


# ─── Component 4: Experience Quality (15%) — upgraded ────────────────────────

def score_experience(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])

    yoe = profile.get("years_of_experience", 0) or 0
    score = 0.0

    # YOE sweet spot from JD (dynamically read from _jd_signals)
    yoe_min, yoe_max = _jd_signals.get("yoe_range", (5, 9))
    if yoe_min <= yoe <= yoe_max:
        score += 0.4
    elif (yoe_min - 1) <= yoe < yoe_min or yoe_max < yoe <= (yoe_max + 2):
        score += 0.3
    elif (yoe_min - 2) <= yoe < (yoe_min - 1) or (yoe_max + 2) < yoe <= (yoe_max + 4):
        score += 0.2
    elif yoe >= (yoe_max + 4):
        score += 0.15
    else:
        score += 0.05

    # Education tier
    best_tier = None
    for edu in education:
        tier = edu.get("tier")
        if tier and (best_tier is None or EDU_TIER_SCORE.get(tier, 0) > EDU_TIER_SCORE.get(best_tier, 0)):
            best_tier = tier
    score += 0.2 * EDU_TIER_SCORE.get(best_tier, 0.35)

    # Rich description analysis: production + impact together
    rich_work_count = 0
    total_desc_words = 0
    for job in career:
        desc = job.get("description", "") or ""
        total_desc_words += len(desc.split())
        prod_sigs = count_production_signals(desc)
        impact_sigs = count_impact_signals(desc)
        # Both signals together = evidence of real shipped work
        if prod_sigs >= 2:
            rich_work_count += 1
        if impact_sigs >= 2:
            rich_work_count += 0.5  # partial credit for quantified results

    score += min(rich_work_count * 0.06, 0.25)

    # Penalize extremely sparse descriptions (no detail = no evidence)
    avg_words = total_desc_words / max(len(career), 1)
    if avg_words < 20 and career:
        score *= 0.7  # very thin descriptions = can't verify claims

    # Research-only penalty
    all_research = all(
        any(kw in normalize(j.get("industry", "") + j.get("title", ""))
            for kw in ["research", "university", "lab", "academia", "phd"])
        for j in career
    ) if career else False

    if all_research:
        score *= 0.3

    # LangChain framework-only penalty
    summary = normalize(profile.get("summary", "") or "")
    if summary.count("langchain") >= 2 and "production" not in summary:
        score *= 0.7

    return min(score, 1.0)


# ─── Component 5: Behavioral Availability (10%) ───────────────────────────────

def score_behavioral(candidate: dict) -> float:
    signals = candidate.get("redrob_signals", {})
    if not signals:
        return 0.3

    score = 0.5

    if signals.get("open_to_work_flag"):
        score += 0.15

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

    response_rate = signals.get("recruiter_response_rate", 0) or 0
    if response_rate >= 0.6:
        score += 0.15
    elif response_rate >= 0.3:
        score += 0.08
    elif response_rate < 0.1:
        score -= 0.15

    # Dynamic notice period from JD
    preferred_notice = _jd_signals.get("preferred_notice", 30)
    notice = signals.get("notice_period_days", 60) or 60
    if notice <= preferred_notice // 2:
        score += 0.1
    elif notice <= preferred_notice:
        score += 0.07
    elif notice > preferred_notice * 3:
        score -= 0.1

    completeness = signals.get("profile_completeness_score", 50) or 50
    if completeness >= 80:
        score += 0.05
    elif completeness < 50:
        score -= 0.05

    icr = signals.get("interview_completion_rate", 0.5) or 0.5
    score += (icr - 0.5) * 0.1

    saved = signals.get("saved_by_recruiters_30d", 0) or 0
    if saved >= 5:
        score += 0.05

    return max(0.0, min(score, 1.0))


# ─── Component 6: Soft Fit Signals (kept at 5% but informing reasoning) ──────

def score_soft_fit(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    score = 0.0

    location = normalize(profile.get("location", "") + " " + profile.get("country", ""))
    if any(loc in location for loc in PREFERRED_LOCATIONS):
        score += 0.3
    if signals.get("willing_to_relocate"):
        score += 0.15

    github_score = signals.get("github_activity_score", 0) or 0
    score += min(github_score / 100, 0.25)

    summary = profile.get("summary", "") or ""
    word_count = len(summary.split())
    if word_count >= 100:
        score += 0.15
    elif word_count >= 50:
        score += 0.08

    for job in career:
        size = job.get("company_size") or job.get("current_company_size", "")
        if size in ("1-10", "11-50", "51-200"):
            score += 0.1
            break

    return min(score, 1.0)


# ─── Score normalization (sigmoid spread) — NEW ───────────────────────────────

def _sigmoid_normalize(scores: list) -> list:
    """
    Apply sigmoid normalization across the full candidate pool to spread scores.
    Prevents top candidate scoring 0.61 and #100 scoring 0.58.
    Input: list of (candidate_id, raw_score)
    Output: list of (candidate_id, normalized_score)
    """
    if not scores:
        return scores
    raw = [s for _, s in scores]
    mean = sum(raw) / len(raw)
    variance = sum((x - mean) ** 2 for x in raw) / len(raw)
    std = math.sqrt(variance) if variance > 0 else 1.0

    normalized = []
    for cid, score in scores:
        # Z-score then sigmoid
        z = (score - mean) / (std + 1e-8)
        sig = 1.0 / (1.0 + math.exp(-z * 2))  # steeper sigmoid
        normalized.append((cid, round(sig, 6)))
    return normalized


# ─── Final weighted score ──────────────────────────────────────────────────────

WEIGHTS = {
    "semantic":     0.30,
    "title_career": 0.25,
    "skills":       0.20,
    "experience":   0.15,
    "behavioral":   0.10,
}


def score_candidate(candidate: dict) -> dict:
    """Score a single candidate. Returns dict with component scores + final score."""
    honeypot_mult = detect_honeypot(candidate)

    components = {
        "semantic":     score_semantic(candidate),
        "title_career": score_title_career(candidate),
        "skills":       score_skills(candidate),
        "experience":   score_experience(candidate),
        "behavioral":   score_behavioral(candidate),
    }

    weighted = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    final = weighted * honeypot_mult

    return {
        "candidate_id": candidate["candidate_id"],
        "final_score": round(final, 6),
        "honeypot_mult": round(honeypot_mult, 3),
        **{f"score_{k}": round(v, 4) for k, v in components.items()},
    }


# ─── Evidence-based reasoning generation (upgraded) ──────────────────────────

def generate_reasoning(candidate: dict, scores: dict) -> str:
    """
    Generate recruiter-quality reasoning: WHY this candidate fits,
    backed by specific evidence from their profile.
    Format: [Fit summary]. [Key evidence]. [Availability].
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0) or 0
    company = profile.get("current_company", "")
    location = profile.get("location", "")

    # Early return for honeypots
    if scores.get("honeypot_mult", 1.0) < 0.3:
        return f"FLAGGED: {title} with title-skill mismatch or fabricated profile signals. Not recommended."

    # ── Part 1: Core fit sentence ──
    semantic_score = scores.get("score_semantic", 0.5)
    skill_score = scores.get("score_skills", 0)

    if semantic_score >= 0.7:
        fit_phrase = "Strong semantic match to JD"
    elif semantic_score >= 0.5:
        fit_phrase = "Good semantic alignment with role"
    else:
        fit_phrase = "Partial fit with role requirements"

    part1 = f"{fit_phrase}: {title} with {yoe:.0f} yrs at {company}" + (f", {location}" if location else "") + "."

    # ── Part 2: Key evidence ──
    evidence = []

    # Top endorsed AI skills
    top_ai_skills = sorted(
        [s for s in skills if skill_name_matches_ai(s.get("name", ""))],
        key=lambda s: (s.get("endorsements", 0), s.get("duration_months", 0)),
        reverse=True
    )[:3]
    if top_ai_skills:
        skill_str = ", ".join(
            f"{s['name']} ({s.get('endorsements', 0)} endorsements)"
            for s in top_ai_skills
        )
        evidence.append(f"Core skills: {skill_str}")

    # Product company evidence
    product_jobs = [
        j for j in career
        if is_product_company(j.get("company", ""), j.get("industry", ""),
                              j.get("current_company_size") or j.get("company_size", ""))
    ]
    if product_jobs:
        co_names = list(dict.fromkeys(j.get("company", "") for j in product_jobs if j.get("company")))[:2]
        evidence.append(f"Product company experience: {', '.join(co_names)}")

    # Production work evidence from descriptions
    prod_jobs = []
    for job in career:
        desc = job.get("description", "") or ""
        if count_production_signals(desc) >= 2 and count_impact_signals(desc) >= 1:
            # Find the most impactful sentence
            sentences = [s.strip() for s in desc.split(".") if len(s.strip()) > 20]
            impact_sentences = [
                s for s in sentences
                if any(kw in s.lower() for kw in ["reduced", "improved", "increased", "%", "million", "billion", "latency"])
            ]
            if impact_sentences:
                best = min(impact_sentences, key=len)[:120]
                prod_jobs.append(best)
    if prod_jobs:
        evidence.append(f"Production evidence: \"{prod_jobs[0]}\"")

    part2 = ". ".join(evidence) + "." if evidence else "Limited verifiable evidence in profile."

    # ── Part 3: Availability ──
    avail = []
    if signals.get("open_to_work_flag"):
        avail.append("actively looking")
    notice = signals.get("notice_period_days")
    if isinstance(notice, (int, float)):
        avail.append(f"{int(notice)}d notice")
    rr = signals.get("recruiter_response_rate", 0)
    if rr:
        avail.append(f"{rr:.0%} response rate")

    part3 = "Availability: " + ", ".join(avail) + "." if avail else ""

    return " ".join(filter(None, [part1, part2, part3]))
