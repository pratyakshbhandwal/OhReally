# AI Recruiter - Redrob Hackathon Submission

Welcome to our offline AI ranking system designed for the Redrob Intelligent Candidate Discovery Challenge! 

This repository fulfills the Stage 3 (Code Reproduction) and Stage 1 (Sandbox) requirements. It is built to be extremely fast, strictly offline, and highly forensic in its ranking.

## 🛠️ Step 1: Sandbox Demo (Streamlit)

Want to see our AI dynamically catch traps and rank candidates in real-time?
1. Open our Sandbox here: **[INSERT YOUR STREAMLIT CLOUD LINK HERE]**
2. On the left sidebar, click **Upload Candidates File** and upload any `.jsonl` or `.json` file containing candidate profiles.
3. Paste a Job Description in the main text box.
4. Click **Analyze & Rank Candidates**. The app will bypass traps, rank the top 100, and give you a button to download the compliant `submission.csv`!

---

## 💻 Step 2: Reproducing the Submission (CLI)

To reproduce the exact `submission.csv` on the full 100K dataset locally (Stage 3 Verification):

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Place the dataset:**
Ensure the `candidates.jsonl` file is inside the `data/` folder.

3. **Run the magic command:**
Execute this single command from the root folder:
```bash
python generate_submission.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

---

## 🧠 How Our Algorithm Works
We built a highly optimized, completely offline architecture to meet the strict 5-minute CPU constraint while outsmarting the honeypots.

1. **Semantic Search:** We embed the JD and use a local `all-MiniLM-L6-v2` Sentence Transformer via ChromaDB to pull a massive net of 10,000 top semantic matches (bypassing keyword-stuffers).
2. **Trap Forensics (`src/trap_detector.py`):** We procedurally filter out the hidden honeypots: Civil Engineers stuffed with AI skills, Title-Chasers (fast turnover), and Consulting-only profiles. 
3. **Behavioral Heuristics (`src/local_ranker.py`):** We aggressively boost responsive candidates and mathematically penalize unavailable ones using the `redrob_signals`.
4. **Factual Reasoning:** The output reasoning strings dynamically inject real data (years of experience, explicitly matched skills, etc.) directly from the candidate JSON, ensuring 0% hallucination and 100% varied justifications.
