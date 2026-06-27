# 🎯 OhReally — AI Candidate Ranker v2

> Ranks candidates the way a great recruiter would — semantic understanding, not keyword counting.
> No GPU. No paid APIs. Fully offline, CPU-only. Runs in under 5 minutes.

---

## What's New in v2

| Upgrade | v1 | v2 |
|---|---|---|
| **Core scoring** | Keyword matching | Semantic similarity (sentence-transformers) |
| **JD handling** | Hardcoded constants | Dynamic — paste any JD, scores adapt |
| **Description analysis** | 5 hardcoded keywords | Production + impact signal detection |
| **Product co detection** | All small co's = product | Requires industry/name signal |
| **Score distribution** | Raw weighted sum | Sigmoid normalized (decisive ranking) |
| **Reasoning** | Template-filled stats | Evidence: skills + company + impact quotes |
| **Honeypot detection** | 3 checks | 6 checks including skill inflation |

---

## File Structure

```
OhReally/
├── scorer.py          ← All scoring logic (semantic + keyword hybrid)
├── rank.py            ← CLI entry point with --jd flag
├── app.py             ← Streamlit demo (paste any JD, see live results)
├── api.py             ← FastAPI backend
├── requirements.txt   ← Now includes sentence-transformers
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

The semantic model (`all-MiniLM-L6-v2`, ~80MB) downloads automatically on first run.
It runs entirely on CPU — no GPU needed.

---

## Running

**CLI (generates submission CSV):**
```bash
# Basic
python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv

# With a custom JD file
python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv --jd ./job_description.txt

# Skip semantic if sentence-transformers not installed
python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv --no-semantic
```

**Streamlit demo:**
```bash
streamlit run app.py
```
- Paste **any JD** → system adapts scoring automatically
- Upload candidates → see ranked shortlist with evidence-based reasoning
- Download submission CSV

---

## How Scoring Works

```
candidates.jsonl
      │
      ▼
detect_honeypot()           ← 6 checks, multiplier 0.05–1.0
      │
      ▼
score_semantic()      ×30%  ← cosine similarity: candidate text vs JD embedding
score_title_career()  ×25%  ← title match + product company (fixed detection)
score_skills()        ×20%  ← endorsed AI skills, proficiency, duration
score_experience()    ×15%  ← YOE from JD, rich description analysis
score_behavioral()    ×10%  ← open to work, response rate, notice period
      │
      ▼
sigmoid_normalize()         ← spreads scores for decisive ranking
      │
      ▼
top 100 → team_xxx.csv
```

---

## Honeypot Detection (6 checks)

| Signal | Penalty |
|---|---|
| Non-AI title + 6+ AI skills | 0.05 |
| Non-AI title + 3–5 AI skills | 0.30 |
| 5+ expert skills, 0 endorsements | 0.30 |
| 30+ skills total (inflation) | 0.50 |
| High YOE, no career history | 0.30 |
| Bad title + AI jargon in descriptions | 0.20 |

---

## Semantic Model

Uses `all-MiniLM-L6-v2` from the `sentence-transformers` library:
- ~80MB download, cached locally
- CPU inference: ~0.5ms per candidate
- 100K candidates ≈ 50–90 seconds total

The JD is embedded once and cached. Candidate text = summary + top skills + recent job descriptions.