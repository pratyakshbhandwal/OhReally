import pandas as pd
import json
import os

def load_job_description(file_path: str) -> str:
    """Loads the job description from a text file."""
    with open(file_path, 'r') as file:
        return file.read()

def load_candidates(file_path: str, limit: int = None) -> pd.DataFrame:
    """Loads candidates from a CSV or JSONL file."""
    if file_path.endswith('.jsonl'):
        if limit:
            df = next(pd.read_json(file_path, lines=True, chunksize=limit))
        else:
            df = pd.read_json(file_path, lines=True)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
        if limit:
            df = df.head(limit)
    else:
        df = pd.read_csv(file_path)
        if limit:
            df = df.head(limit)
        
    # Ensure there is always an 'id' column for ChromaDB and Streamlit filtering
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)
        
    # Force IDs to be completely unique strings (ChromaDB crashes if duplicate)
    df['id'] = df['id'].astype(str)
    if df['id'].duplicated().any():
        df['id'] = df['id'] + "_" + df.index.astype(str)
        
    return df

def format_candidate_for_embedding(row: pd.Series) -> str:
    """Formats a candidate's profile into a rich text string for vector embedding dynamically."""
    def _safe_str(val):
        if isinstance(val, (list, dict, tuple)):
            return str(val)
        if pd.isna(val): 
            return ""
        return str(val)
        
    parts = []
    for col, val in row.items():
        if col == 'id': continue
        clean_val = _safe_str(val)
        if clean_val:
            parts.append(f"{str(col).replace('_', ' ').title()}: {clean_val}")
            
    return ". ".join(parts)

def format_candidate_for_llm(row: pd.Series) -> str:
    """Formats a candidate's profile into a structured string/JSON for LLM evaluation dynamically."""
    candidate_dict = {}
    for col, val in row.items():
        if pd.isna(val):
            candidate_dict[col] = None
        else:
            candidate_dict[col] = val
    return json.dumps(candidate_dict, indent=2)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jd_path = os.path.join(base_dir, "data", "job_description.txt")
    jd = load_job_description(jd_path)
    print("Loaded JD length:", len(jd))
    
    cand_path_jsonl = os.path.join(base_dir, "data", "candidates.jsonl")
    cand_path_csv = os.path.join(base_dir, "data", "candidates.csv")
    cand_path = cand_path_jsonl if os.path.exists(cand_path_jsonl) else cand_path_csv
    df = load_candidates(cand_path)
    print(f"Loaded {len(df)} candidates.")
    print("Sample embedding text:", format_candidate_for_embedding(df.iloc[0]))
