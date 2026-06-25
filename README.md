# Redrob Hackathon — Intelligent Candidate Ranker

Ranks 100K candidates for the Redrob AI Senior AI Engineer JD — the way a great recruiter would. No keyword matching. No API calls. Fully offline, CPU-only, under 5 minutes.

## Quick start

```bash
git clone https://github.com/your-team/redrob-ranker
cd redrob-ranker
pip install -r requirements.txt

# Run the ranker
python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv

# Validate your output
python validate_submission.py team_xxx.csv

# Run the demo UI
streamlit run app.py
```

## Architecture

```
candidates.jsonl (100K)
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │           scorer.py                          │
  │                                              │
  │  1. Honeypot detection  (mult 0.0–1.0)      │
  │  2. Title & Career Fit      ×35%            │
  │  3. Skill Depth             ×25%            │
  │  4. Experience Quality      ×20%            │
  │  5. Behavioral Availability ×15%            │
  │  6. Soft Fit Signals         ×5%            │
  │                                              │
  │  final = weighted_sum × honeypot_mult       │
  └─────────────────────────────────────────────┘
        │
        ▼
  team_xxx.csv  (top 100 ranked candidates)
```

## Scoring components

| Component | Weight | What it measures |
|---|---|---|
| Title & Career Fit | 35% | Current title AI/ML match, product company history, production deployments, no pure services |
| Skill Depth | 25% | Endorsed + long-duration AI skills (embeddings, vector DBs, ranking eval). Penalizes unrelated skills. |
| Experience Quality | 20% | 5–9 yr sweet spot, education tier, not research-only, not pure LangChain/framework |
| Behavioral Availability | 15% | Last active, response rate, open to work, notice period, profile completeness |
| Soft Fit Signals | 5% | GitHub activity, location preference, startup experience, writing quality |

## Honeypot detection

The dataset contains ~80 trap candidates (e.g. HR Managers with 9 AI skills). The ranker detects these via:
- **Title–skill mismatch**: non-AI title claiming many AI skills → multiplier drops to 0.05
- **Impossible tenure**: YOE > company age
- **Expert with 0 endorsements**: 5+ expert-level skills with zero endorsements

Honeypot multiplier ranges 0.0–1.0 and is applied to the final weighted score.

## Constraints satisfied

- ✅ No network calls during ranking
- ✅ No GPU
- ✅ No external API calls
- ✅ Runs in < 5 min on 16GB CPU
- ✅ Pure Python + pandas + numpy + tqdm

## File structure

```
redrob-ranker/
├── rank.py         # Entry point — run this to generate submission
├── scorer.py       # All scoring logic
├── app.py          # Streamlit demo UI
├── api.py          # FastAPI backend (optional)
├── requirements.txt
└── README.md
```

## Team

| Member | Role | Owns |
|---|---|---|
| Member 1 | AI/LLM Engineer | scorer.py — JD parsing, skill matching, weights |
| Member 2 | Data Engineer | rank.py — pipeline, feature engineering, submission output |
| Member 3 | ML/Evaluation | Evaluation, deck, results analysis |
| Member 4 | Full-Stack | app.py, api.py, deployment, README |

## Reproduce command

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv
```

Expected runtime: ~60–90 seconds for 100K candidates on a standard laptop CPU.
