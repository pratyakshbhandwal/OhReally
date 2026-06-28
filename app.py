import streamlit as st
import pandas as pd
import sys
import os

# Ensure src module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_candidates, format_candidate_for_embedding
from retriever import CandidateRetriever
from local_ranker import LocalRanker

st.set_page_config(page_title="AI Recruiter Dashboard", page_icon="🕵️", layout="wide")

st.title("🚀 RedrobAI Hack2Skill: Intelligent Candidate Ranking")
st.markdown("""
This is the sandbox environment for our offline AI Recruiter. It uses **Semantic Search (ChromaDB)** to find the most relevant candidates, and a **Local Heuristics Ranker** to deeply analyze their true fit based on career history, trap-detection, and behavioral signals.
""")

@st.cache_resource
def init_system(cand_path):
    df = load_candidates(cand_path)
    
    # PROTECT STREAMLIT CLOUD FROM MEMORY CRASHES
    if len(df) > 1000:
        df = df.head(1000)
        
    retriever = CandidateRetriever()
    
    # Only index if the database is newly created (i.e., first run)
    if retriever.is_new:
        with st.spinner(f"⏳ First time setup: Generating embeddings for {len(df)} candidates... This may take a minute!"):
            texts = [format_candidate_for_embedding(row) for _, row in df.iterrows()]
            retriever.index_candidates(df, texts)
            
    return df, retriever

uploaded_file = st.sidebar.file_uploader("Upload Candidates File (JSONL)", type=["jsonl", "json", "csv"])

cand_path = None
if uploaded_file is not None:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getvalue())
        cand_path = tmp.name
elif os.path.exists("data/candidates.jsonl"):
    cand_path = "data/candidates.jsonl"
elif os.path.exists("data/sample_candidates.json"):
    cand_path = "data/sample_candidates.json"

if not cand_path:
    st.info("👋 Welcome to the Sandbox! Please upload a candidate JSON/JSONL file in the sidebar to begin.")
    st.stop()

try:
    df, retriever = init_system(cand_path)
    if len(df) == 1000:
        st.warning("⚠️ **Massive File Uploaded:** Streamlit Cloud has a strict 1GB RAM limit. We have safely capped the Sandbox to process the first 1,000 candidates to prevent server crashes. Please run the CLI command to process the full 100K dataset.")
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

st.sidebar.header("Data Overview")
st.sidebar.info(f"Loaded {len(df)} candidate profiles.")
if st.sidebar.checkbox("Show Candidate Database"):
    st.dataframe(df)

st.header("1. Job Description")
default_jd = open("data/job_description.txt").read() if os.path.exists("data/job_description.txt") else ""
job_description = st.text_area("Paste the Job Description here:", value=default_jd, height=300)

if st.button("Analyze & Rank Candidates", type="primary"):
    if not job_description.strip():
        st.error("Please enter a job description.")
    else:
        with st.spinner("🔍 Stage 1: Running Semantic Search to retrieve Top Candidates..."):
            # We use 10,000 to cast a wide net and bypass keyword-stuffer honeypots
            results = retriever.search(job_description, top_k=10000)
            
            # Fetch candidates from the dataframe based on retrieved IDs
            retrieved_ids = results['ids'][0] # ChromaDB returns strings
            distances = results['distances'][0]
            
            shortlist_df = df[df['id'].astype(str).isin(retrieved_ids)]
            preferred_cols = ['name', 'current_role', 'skills', 'experience_years']
            display_cols = [col for col in preferred_cols if col in shortlist_df.columns]
            if not display_cols:
                display_cols = [col for col in shortlist_df.columns if col != 'id'][:5] # Fallback to first 5 cols
                
            st.dataframe(shortlist_df[display_cols])
            
            # Convert to dict for Local Ranker
            shortlist_dicts = shortlist_df.to_dict('records')
            dist_map = dict(zip(retrieved_ids, distances))
            cand_distances = [dist_map.get(str(c.get('id', c.get('candidate_id'))), 1.0) for c in shortlist_dicts]

        with st.spinner("🧠 Stage 2: Local AI Ranking and Trap Detection..."):
            ranker = LocalRanker()
            ranked_candidates = ranker.rank_candidates(shortlist_dicts, cand_distances, job_description)
            
            # Hackathon rule: Exactly 100 rows of data
            top_100 = ranked_candidates[:100]
            
            st.subheader("🏆 Final Top 100 Candidates")
            
            output = []
            prev_score = float('inf')
            for i, c in enumerate(top_100):
                raw_score = float(c.get('final_score', 0.0))
                score = min(raw_score, prev_score)
                prev_score = score
                
                output.append({
                    'candidate_id': c.get('candidate_id', c.get('id')),
                    'rank': i + 1,
                    'score': round(score, 4),
                    'reasoning': c.get('reasoning', 'No reasoning generated.')
                })
                
            sub_df = pd.DataFrame(output, columns=['candidate_id', 'rank', 'score', 'reasoning'])
            st.dataframe(sub_df)
            
            csv = sub_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download submission.csv",
                data=csv,
                file_name='submission.csv',
                mime='text/csv',
            )
