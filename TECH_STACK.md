# 🛠️ AI Recruiter - Technology Stack & Architecture

This document outlines the core technologies, libraries, and custom architectural modules used to build this offline, privacy-first AI candidate ranking system.

---

## 1. Core Artificial Intelligence (The Brain)
Instead of relying on hosted LLMs (like OpenAI or Gemini) which violate the strict offline-compute constraints, this system utilizes a local **Embedding Pipeline**.

* **`sentence-transformers` (HuggingFace):** 
  * **Model:** `all-MiniLM-L6-v2`
  * **Purpose:** Converts raw text (Job Descriptions and Candidate Profiles) into dense 384-dimensional mathematical vectors. It allows the system to understand the *contextual meaning* of a resume rather than just looking for exact keyword matches.
* **`chromadb` (ChromaDB):**
  * **Purpose:** An open-source Vector Database. It stores the 100,000 candidate embeddings locally. During execution, it performs blazing-fast **Semantic Search** using cosine similarity to instantly retrieve the top candidates whose vectors point in the same mathematical direction as the Job Description.

## 2. Data Processing (The Engine)
* **`pandas`:** 
  * **Purpose:** The industry standard for data manipulation. Used to parse the massive `candidates.jsonl` and `submission.csv` files.
* **Python Generators (`chunksize` streaming):**
  * **Purpose:** Advanced memory management. By streaming the JSONL file in chunks using `pd.read_json(..., chunksize=1000)`, the system prevents Out of Memory (OOM) crashes on constrained 1GB cloud servers.

## 3. User Interface & Deployment (The Sandbox)
* **`streamlit`:** 
  * **Purpose:** A lightweight Python web framework used to build the interactive Stage 1 Sandbox demo. It allows users to upload custom candidate files and visualize the AI ranking process in real-time without needing complex Javascript frontends.
* **Streamlit Community Cloud:** 
  * **Purpose:** The hosting environment for the live Sandbox. It automatically pulls from GitHub and provisions a Linux container to serve the web application.

## 4. Custom Forensic Architecture (Our Secret Weapon)
These are the proprietary Python modules written completely from scratch to satisfy the complex heuristics and trap-detection requirements of the hackathon.

* **Trap Detector (`src/trap_detector.py`):**
  * A mathematical filter that acts like a forensic investigator to bypass dataset honeypots. It automatically flags and penalizes:
    * **Keyword Stuffers:** Candidates with an improbably high number of listed skills.
    * **Title Mismatches:** Candidates (e.g., "Civil Engineers") whose actual titles do not align with the JD, even if they artificially injected AI keywords into their profiles.
    * **Title Chasers:** Candidates with extremely high turnover rates (less than 1.5 years per role).
* **Behavioral Heuristics (`src/local_ranker.py`):**
  * Utilizes the `redrob_signals` JSON block to calculate a dynamic score multiplier. Highly responsive and available candidates receive mathematical boosts, while unresponsive candidates drop out of the Top 100.
* **Factual Reasoning Generator (`src/local_ranker.py`):**
  * A procedural string-generation engine that injects actual JSON datapoints (Years of Experience, matched skills, response rates) into the output reasoning. This ensures the output is 100% varied and eliminates AI hallucinations.

---
*Built for the Redrob Intelligent Candidate Discovery Challenge.*
