# 🤖 Redrob AI — Intelligent Candidate Ranker

> Ranks 100K candidates the way a great recruiter would — not by keyword count, but by actually understanding who fits the role.
> No API calls. No GPU. No paid services. Fully offline, CPU-only.

---

## 📁 File Structure

```
redrob_ranker/
├── .streamlit/
│   └── config.toml          # Raises Streamlit upload limit to 1000MB
├── scorer.py                # All scoring logic — the brain
├── rank.py                  # CLI entry point — run this to generate submission
├── app.py                   # Streamlit demo UI
├── api.py                   # FastAPI backend (optional)
├── requirements.txt         # All dependencies
├── submission_metadata.yaml # Fill in your team details before submitting
├── .gitignore               # Excludes candidates.jsonl and output CSVs
└── README.md
```

> ⚠️ **Do NOT commit `candidates.jsonl` or `team_xxx.csv` to GitHub.** They are already in `.gitignore`. The data file is 465MB — keep it only on your local machine.

---

## ⚙️ Prerequisites

- Python **3.10 or higher**
- pip
- The `candidates.jsonl` file from the hackathon bundle (place it inside the project folder)
- The `validate_submission.py` file from the hackathon bundle (place it inside the project folder)

---

## 🚀 Setup — Run This Once

**Step 1 — Clone the repo**

```bash
git clone https://github.com/YOUR_USERNAME/redrob-ranker.git
cd redrob-ranker
```

**Step 2 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 3 — Add the data files**

Copy these two files from the hackathon zip into the project folder:

```
redrob_ranker/
├── candidates.jsonl          ← paste here (from hackathon bundle)
└── validate_submission.py    ← paste here (from hackathon bundle)
```

---

## 🏃 Running the Ranker (generates the submission CSV)

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv
```

- ⏱ Takes ~60–90 seconds on a standard laptop
- 📊 Shows a live progress bar
- ✅ Outputs `team_xxx.csv` with the top 100 ranked candidates

---

## ✅ Validating the Output

```bash
python validate_submission.py team_xxx.csv
```

You should see:
```
All validations PASSED!
```

If you see errors, do not submit until they are fixed.

---

## 🌐 Running the Demo UI (Streamlit)

```bash
streamlit run app.py
```

Opens at **http://localhost:8501** in your browser.

- Upload a `.jsonl` file (use a sample of ~500 candidates for the demo)
- Click **Run Ranking**
- See ranked results with score breakdowns and a CSV download button

> 💡 **Large file tip:** The `.streamlit/config.toml` already raises the upload limit to 1000MB so `candidates.jsonl` can be uploaded directly if needed.

---

## 🔌 Running the API (Optional)

If you want to use the FastAPI backend instead of calling scorer directly:

```bash
uvicorn api:app --reload --port 8000
```

API will be live at **http://localhost:8000**

- `GET  /health` → health check
- `POST /rank`   → send candidates JSON, get ranked results back
- `GET  /docs`   → auto-generated Swagger UI

---

## ☁️ Deploying the Demo to Streamlit Cloud (Free)

1. Push this repo to GitHub (without `candidates.jsonl`)
2. Go to **https://streamlit.io/cloud** and sign in with GitHub
3. Click **New app**
4. Select your repo → set main file to `app.py` → click **Deploy**
5. Copy the live URL and paste it into `submission_metadata.yaml` under `sandbox_url`

---

## 📊 How the Scoring Works

```
candidates.jsonl (100K records)
        │
        ▼
  detect_honeypot()           ← multiplier 0.05–1.0 (kills fake profiles)
        │
        ▼
  score_title_career()  ×35%  ← job title, product company, career progression
  score_skills()        ×25%  ← endorsed AI skills, proficiency, duration
  score_experience()    ×20%  ← years of experience, education tier, production work
  score_behavioral()    ×15%  ← last active, response rate, notice period
  score_soft_fit()      ×5%   ← GitHub, location, startup experience
        │
        ▼
  final_score = weighted_sum × honeypot_multiplier
        │
        ▼
  top 100 sorted → team_xxx.csv
```

| Component | Weight | Key signals |
|---|---|---|
| Title & Career Fit | 35% | AI/ML title match, product vs services company, production deployments |
| Skill Depth | 25% | Endorsed + long-duration AI skills (embeddings, FAISS, vector DBs, ranking eval) |
| Experience Quality | 20% | 5–9 yr sweet spot, education tier, shipped to real users |
| Behavioral Availability | 15% | Last active date, recruiter response rate, open to work, notice period |
| Soft Fit Signals | 5% | GitHub activity score, city preference, startup company size |

---

## 🪤 Honeypot Detection

The dataset has trap candidates — e.g. HR Managers with 9 AI skills listed. Our system catches these via:

| Trap type | Penalty multiplier |
|---|---|
| Non-AI title + 6 or more AI skills | 0.05 (near-eliminated) |
| Non-AI title + 3–5 AI skills | 0.30 |
| 5+ expert skills with zero endorsements | 0.30 |
| Impossible tenure vs claimed YOE | 0.40 |

---

## 📋 Before Submitting

1. Rename your output CSV to match your team ID:
```bash
# Windows
rename team_xxx.csv team_YOURTEAMID.csv

# Mac/Linux
mv team_xxx.csv team_YOURTEAMID.csv
```

2. Fill in `submission_metadata.yaml` with your team ID, team name, and sandbox URL

3. Run the validator one final time:
```bash
python validate_submission.py team_YOURTEAMID.csv
```

4. Submit: **GitHub repo link + team_YOURTEAMID.csv + PDF deck**

---

## 👥 Team

| Member | Role | Owns |
|---|---|---|
| Member 1 | AI/LLM Engineer | scorer.py — skill matching, JD parsing, scoring weights |
| Member 2 | Data Engineer | rank.py — pipeline, feature engineering, CSV output |
| Member 3 | ML/Evaluation | Evaluation metrics, results analysis, PDF deck |
| Member 4 | Full-Stack | app.py, api.py, Streamlit Cloud deploy, README |

---

## ✅ Constraints Satisfied

- ✅ No network calls during ranking
- ✅ No GPU required
- ✅ No paid APIs or external services
- ✅ Runs in under 5 minutes on 16GB CPU
- ✅ Single reproduce command
- ✅ Output passes validate_submission.py with 0 errors