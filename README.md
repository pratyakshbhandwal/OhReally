# AI Recruiter - Redrob Hackathon Submission

This repository contains the offline AI ranking system designed for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## 1. Reproducing the Submission (CLI)

To reproduce the exact `submission.csv` file from the candidate dataset (Stage 3 Verification), run the following single command from the root of the repository:

```bash
python generate_submission.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

### How it works
This system uses a highly optimized, completely offline architecture to meet the strict 5-minute CPU constraint.
1. **Semantic Search:** We embed the JD and use a local `all-MiniLM-L6-v2` Sentence Transformer via ChromaDB to instantly retrieve the top semantic matches.
2. **Trap Detection:** The `src/trap_detector.py` module automatically identifies and drops honeypots, keyword stuffers, and completely unrelated titles (e.g. Civil Engineers with random AI keywords).
3. **Behavioral Heuristics:** The `src/local_ranker.py` calculates a behavioral multiplier based on the `redrob_signals` to aggressively boost highly-responsive candidates and penalize unavailable ones.

---

## 2. Running the Sandbox App (Streamlit)

We also built an interactive web application that fulfills the Sandbox requirement (Stage 1). You can run it locally with the following commands:

```bash
# 1. Create and activate a Python Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 2. Install all the required AI libraries
pip install -r requirements.txt

# 3. Start the application
streamlit run app.py
```

A browser window will automatically open with your AI Recruiter Dashboard.
Note: The Streamlit app also integrates with the Gemini API for advanced qualitative filtering, but the main `submission.csv` is generated entirely offline via `generate_submission.py` to adhere strictly to the compute and network constraints.
