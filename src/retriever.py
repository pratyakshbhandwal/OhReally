import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from typing import List

class CandidateRetriever:
    def __init__(self, collection_name="candidates"):
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(base_dir, "chroma_db")
        # Use Persistent ChromaDB to save time on restarts
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Use a lightweight sentence-transformer model for embeddings
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Try to get existing collection
        try:
            self.collection = self.client.get_collection(name=collection_name, embedding_function=self.embedding_fn)
            self.is_new = False
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name, 
                embedding_function=self.embedding_fn
            )
            self.is_new = True

    def index_candidates(self, df: pd.DataFrame, texts: List[str]):
        """Indexes candidates into ChromaDB."""
        ids = df['id'].astype(str).tolist()
        metadatas = df.to_dict('records')
        
        # Convert any non-string values in metadata to string/int/float for Chroma
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (list, dict, tuple)):
                    clean_meta[k] = str(v)
                elif pd.isna(v):
                    continue
                elif isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)

        # ChromaDB has a maximum batch size (e.g., 5461). We chunk the insertions.
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                documents=texts[i:i + batch_size],
                metadatas=clean_metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )

    def search(self, query: str, top_k: int = 5):
        """Searches for top_k candidates matching the query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results

if __name__ == "__main__":
    from data_loader import load_candidates, format_candidate_for_embedding
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand_path_jsonl = os.path.join(base_dir, "data", "candidates.jsonl")
    cand_path_csv = os.path.join(base_dir, "data", "candidates.csv")
    cand_path = cand_path_jsonl if os.path.exists(cand_path_jsonl) else cand_path_csv
    df = load_candidates(cand_path)
    texts = [format_candidate_for_embedding(row) for _, row in df.iterrows()]
    retriever = CandidateRetriever()
    retriever.index_candidates(df, texts)
    print("Indexed candidates successfully.")
    res = retriever.search("Looking for someone with deep NLP and LLM experience.")
    print("Search results:")
    for doc in res['documents'][0]:
        print("-", doc)
