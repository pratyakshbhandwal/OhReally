import argparse
import pandas as pd
from src.retriever import CandidateRetriever
from src.local_ranker import LocalRanker
from src.data_loader import load_candidates, format_candidate_for_embedding

def generate_submission(candidates_path, output_path):
    print("Loading Job Description...")
    # Assuming JD is static for the hackathon
    with open("data/job_description.txt", "r") as f:
        jd = f.read()
        
    print(f"Loading Candidates from {candidates_path}...")
    df = load_candidates(candidates_path)
    
    # Check if retriever has index
    retriever = CandidateRetriever()
    if retriever.is_new:
        print("Building ChromaDB index... this will take a while.")
        texts = [format_candidate_for_embedding(row) for _, row in df.iterrows()]
        retriever.index_candidates(df, texts)
        
    print("Querying Semantic Search for top matches...")
    results = retriever.search(jd, top_k=10000)
    
    retrieved_ids = results['ids'][0]
    distances = results['distances'][0]
    
    # Fetch candidates from dataframe
    shortlist_df = df[df['id'].astype(str).isin(retrieved_ids)]
    
    # Map id to distance
    dist_map = dict(zip(retrieved_ids, distances))
    shortlist_dicts = shortlist_df.to_dict('records')
    cand_distances = [dist_map.get(str(c['id']), 1.0) for c in shortlist_dicts]
    
    print("Applying Local Ranking and filtering Traps...")
    ranker = LocalRanker()
    ranked_candidates = ranker.rank_candidates(shortlist_dicts, cand_distances, jd)
    
    # Take top 100
    top_100 = ranked_candidates[:100]
    
    print(f"Generating {output_path}...")
    output = []
    
    # Track previous score to ensure monotonic non-increasing
    prev_score = float('inf')
    
    for i, c in enumerate(top_100):
        cid = c.get('candidate_id', c.get('id'))
        reasoning = c.get('reasoning', 'No reasoning generated.')
        raw_score = float(c.get('final_score', 0.0))
        
        # Enforce monotonically non-increasing score constraint
        score = min(raw_score, prev_score)
        prev_score = score
        
        rank = i + 1
        
        output.append({
            'candidate_id': cid,
            'rank': rank,
            'score': round(score, 4),
            'reasoning': reasoning
        })
        
    # Format requires exact columns in exact order
    sub_df = pd.DataFrame(output, columns=['candidate_id', 'rank', 'score', 'reasoning'])
    sub_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Successfully wrote {len(sub_df)} candidates to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Redrob Hackathon Submission CSV")
    parser.add_argument("--candidates", type=str, default=None, help="Path to candidates file")
    parser.add_argument("--out", type=str, default="submission.csv", help="Output path for the CSV")
    args = parser.parse_args()
    
    import os
    import time
    
    cand_path = args.candidates
    if not cand_path:
        if os.path.exists("data/candidates.jsonl"):
            cand_path = "data/candidates.jsonl"
        elif os.path.exists("data/sample_candidates.json"):
            cand_path = "data/sample_candidates.json"
        else:
            print("Error: Could not find any candidates file in the data/ folder.")
            exit(1)
    start = time.time()
    generate_submission(cand_path, args.out)
    print(f"Total execution time: {time.time() - start:.2f} seconds")
