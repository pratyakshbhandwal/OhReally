import pandas as pd
import json
import os

def load_job_description(file_path: str) -> str:
    """Loads the job description from a text file."""
    with open(file_path, 'r') as file:
        return file.read()

def load_candidates(file_path: str) -> pd.DataFrame:
    """Loads candidates from a CSV or JSONL file."""
    if file_path.endswith('.jsonl'):
        df = pd.read_json(file_path, lines=True)
    else:
        df = pd.read_csv(file_path)
        
    # Ensure there is always an 'id' column for ChromaDB and Streamlit filtering
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)
        
    return df

def format_candidate_for_embedding(row: pd.Series) -> str:
    """Formats a candidate's profile into a rich text string for vector embedding."""
    text = f"Role: {row.get('current_role', '')}. "
    text += f"Experience: {row.get('experience_years', 0)} years. "
    text += f"Skills: {row.get('skills', '')}. "
    text += f"Career History: {row.get('career_history', '')}. "
    text += f"Behavioral: {row.get('behavioral_signals', '')}."
    return text

def format_candidate_for_llm(row: pd.Series) -> str:
    """Formats a candidate's profile into a structured string/JSON for LLM evaluation."""
    candidate_dict = {
        "id": row.get('id'),
        "name": row.get('name'),
        "current_role": row.get('current_role'),
        "experience_years": row.get('experience_years'),
        "skills": row.get('skills'),
        "career_history": row.get('career_history'),
        "behavioral_signals": row.get('behavioral_signals')
    }
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
