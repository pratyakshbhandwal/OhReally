import streamlit as st
import pandas as pd
import sys
import os

# Ensure src module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_candidates, format_candidate_for_embedding
from retriever import CandidateRetriever
from ranker import LLMRanker

st.set_page_config(page_title="AI Recruiter Dashboard", page_icon="🕵️", layout="wide")

st.title("🚀 RedrobAI Hack2Skill: Intelligent Candidate Ranking")
st.markdown("""
This system goes beyond keyword matching. It uses **Semantic Search (ChromaDB)** to find the most relevant candidates, and a **Generative LLM (Gemini)** to deeply analyze their true fit based on career history, skills, and behavioral signals.
""")

# Load API Key
api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

# Load Data
@st.cache_resource
def init_system():
    cand_path_jsonl = "data/candidates.jsonl"
    cand_path_csv = "data/candidates.csv"
    cand_path = cand_path_jsonl if os.path.exists(cand_path_jsonl) else cand_path_csv
    df = load_candidates(cand_path)
    
    retriever = CandidateRetriever()
    
    # Only index if the database is newly created (i.e., first run)
    if retriever.is_new:
        with st.spinner("⏳ First time setup: Generating embeddings for 100k candidates... This will take a while!"):
            texts = [format_candidate_for_embedding(row) for _, row in df.iterrows()]
            retriever.index_candidates(df, texts)
            
    return df, retriever

df, retriever = init_system()

st.sidebar.header("Data Overview")
st.sidebar.info(f"Loaded {len(df)} candidate profiles.")
if st.sidebar.checkbox("Show Candidate Database"):
    st.dataframe(df)

st.header("1. Job Description")
default_jd = open("data/job_description.txt").read() if os.path.exists("data/job_description.txt") else ""
job_description = st.text_area("Paste the Job Description here:", value=default_jd, height=300)

top_k = st.slider("Number of top candidates to retrieve (Stage 1):", min_value=1, max_value=len(df), value=3)

if st.button("Analyze & Rank Candidates", type="primary"):
    if not job_description.strip():
        st.error("Please enter a job description.")
    elif not os.getenv("GEMINI_API_KEY") and not api_key:
        st.error("Please provide a Gemini API Key in the sidebar to run the LLM Ranker.")
    else:
        with st.spinner("🔍 Stage 1: Running Semantic Search to retrieve Top Candidates..."):
            results = retriever.search(job_description, top_k=top_k)
            
            # Fetch candidates from the dataframe based on retrieved IDs
            retrieved_ids = [int(id_str) for id_str in results['ids'][0]]
            shortlist_df = df[df['id'].isin(retrieved_ids)]
            
            st.subheader(f"✅ Top {top_k} Semantically Matched Candidates")
            st.dataframe(shortlist_df[['name', 'current_role', 'skills', 'experience_years']])
            
            # Convert to dict for LLM
            shortlist_dicts = shortlist_df.to_dict('records')

        with st.spinner("🧠 Stage 2: LLM Analyzing Candidate Profiles deeply..."):
            try:
                ranker = LLMRanker()
                report = ranker.rank_candidates(job_description, shortlist_dicts)
                
                st.subheader("🏆 Final Recruiter Report & Ranking")
                st.markdown(report)
            except Exception as e:
                st.error(f"Error calling LLM API: {e}")
