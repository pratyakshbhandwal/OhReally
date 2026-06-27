<div align="center">
  <h1>🚀 AI Recruiter</h1>
  <h3>Redrob Intelligent Candidate Discovery Challenge</h3>
  <p>An ultra-fast, offline-first semantic search & heuristic ranking engine designed to identify the top 100 AI Engineers out of 100,000+ candidates.</p>
</div>

---

## 🌟 Overview

This repository fulfills both the **Stage 1 (Sandbox)** and **Stage 3 (Code Reproduction)** requirements for the hackathon. It is engineered to be extremely fast (processing 100K profiles in seconds on a CPU), strictly offline (no hosted LLMs), and highly forensic in its ranking logic.

## 🎯 Key Features

- **⚡ Blazing Fast Semantic Search:** Uses a local `all-MiniLM-L6-v2` Sentence Transformer and ChromaDB to instantly retrieve candidates.
- **🛡️ Advanced Trap Detection:** Procedurally filters out honeypots, Civil Engineers keyword-stuffing AI skills, title-chasers, and framework wrappers.
- **📊 Behavioral Heuristics:** Mathematically rewards responsive candidates and penalizes unavailable ones based on `redrob_signals`.
- **🔎 Factual Reasoning:** Generates 100% varied, hallucination-free justifications by injecting real profile data (YoE, explicit skills) directly into the reasoning strings.

---

## 💻 1. Reproducing the Submission (CLI)
*For Stage 3 Verification on the full 100K dataset.*

**Step 1:** Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

**Step 2:** Place your dataset:
Drop the massive `candidates.jsonl` file directly into the `data/` folder.

**Step 3:** Run the Generation Script:
Execute this single command from the root folder:
```bash
python generate_submission.py --candidates ./data/candidates.jsonl --out ./submission.csv
```
> **✅ Success:** In just a few seconds, the engine will bypass all traps, score the candidates, and generate a perfectly formatted `submission.csv` containing the strictly ranked Top 100 matches!

---

## 🛠️ 2. Sandbox Demo (Streamlit)
*For an interactive UI that visually demonstrates the engine.*

Want to see our AI dynamically catch traps and rank candidates in real-time? 

1. **Access the Live App:** 👉 **[Launch Streamlit Sandbox Here]** *(Replace with your live Streamlit Cloud link)*
2. **Upload Data:** On the left sidebar, click **Upload Candidates File** and drag-and-drop your `.jsonl`, `.json`, or `.csv` candidate file.
3. **Input the Role:** Paste your target Job Description into the main text box.
4. **Rank:** Click **Analyze & Rank Candidates** to watch the AI engine work! 

*(Note: You can also run the sandbox locally by executing `streamlit run app.py`)*

---

<div align="center">
  <i>Built with ❤️ for the Redrob Hack2Skill Hackathon</i>
</div>
