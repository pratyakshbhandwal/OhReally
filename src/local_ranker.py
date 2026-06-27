import pandas as pd
from typing import List, Dict

class LocalRanker:
    def get_jd_match_multiplier(self, row: pd.Series) -> float:
        """Parses the JSON schema for explicit JD matches."""
        multiplier = 1.0
        
        vector_dbs = ['pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch', 'elasticsearch', 'faiss']
        eval_frameworks = ['ndcg', 'mrr', 'map', 'a/b test']
        retrieval = ['sentence-transformers', 'retrieval', 'rag', 'ranking']
        
        has_vdb = False
        has_eval = False
        has_retrieval = False
        
        skills = row.get('skills', [])
        if isinstance(skills, list):
            for s in skills:
                name = s.get('name', '').lower()
                if any(v in name for v in vector_dbs): has_vdb = True
                if any(e in name for e in eval_frameworks): has_eval = True
                if any(r in name for r in retrieval): has_retrieval = True
                
        career = row.get('career_history', [])
        if isinstance(career, list):
            for job in career:
                desc = job.get('description', '').lower()
                if any(v in desc for v in vector_dbs): has_vdb = True
                if any(e in desc for e in eval_frameworks): has_eval = True
                if any(r in desc for r in retrieval): has_retrieval = True
                
        if has_vdb: multiplier *= 1.3
        if has_eval: multiplier *= 1.3
        if has_retrieval: multiplier *= 1.3
        
        try:
            yoe = float(row.get('profile', {}).get('years_of_experience', 0))
            if 5 <= yoe <= 10:
                multiplier *= 1.2
            elif yoe < 4:
                multiplier *= 0.7 
        except:
            pass
            
        return multiplier

    def rank_candidates(self, candidates: List[Dict], distances: List[float], job_description: str) -> List[Dict]:
        """
        Ranks candidates using local heuristics: Semantic distance + Behavioral Multipliers + Schema Matches.
        """
        from src.trap_detector import (
            calculate_behavioral_score, 
            is_honeypot, 
            is_keyword_stuffer,
            is_unrelated_title,
            is_pure_consulting,
            is_title_chaser,
            is_pure_research,
            is_langchain_enthusiast
        )
        
        ranked = []
        for i, cand in enumerate(candidates):
            dist = distances[i] if i < len(distances) else 1.0
            semantic_score = max(0, 100 * (1.0 - (dist / 2.0)))
            
            row = pd.Series(cand)
            
            beh_multiplier = calculate_behavioral_score(row)
            jd_multiplier = self.get_jd_match_multiplier(row)
            
            final_score = semantic_score * beh_multiplier * jd_multiplier
            
            reasons = []
            if is_keyword_stuffer(row): 
                final_score *= 0.2
                reasons.append("Flagged as keyword stuffer.")
            if is_unrelated_title(row): 
                final_score *= 0.1
                reasons.append("Current title completely unrelated to engineering.")
            if is_pure_consulting(row): 
                final_score *= 0.5
                reasons.append("Experience entirely in consulting firms.")
            if is_title_chaser(row): 
                final_score *= 0.6
                reasons.append("High job turnover (title chaser).")
            if is_pure_research(row): 
                final_score *= 0.7
                reasons.append("Experience mostly in pure research/academia.")
            if is_langchain_enthusiast(row): 
                final_score *= 0.6
                reasons.append("Flagged as framework enthusiast without core ML depth.")
                
            if is_honeypot(row):
                continue
                
            # Construct final reasoning
            if not reasons:
                reasoning = f"Strong semantic match with solid JD alignment. "
                if jd_multiplier > 1.5:
                    reasoning += "Direct experience with Vector DBs and Ranking evaluation. "
                if beh_multiplier > 1.2:
                    reasoning += "Excellent logistics (location/notice) and engagement signals."
                elif beh_multiplier < 0.8:
                    reasoning += "Good skills but poor logistics/engagement signals."
            else:
                reasoning = " ".join(reasons)
                
            cand['final_score'] = final_score
            cand['reasoning'] = reasoning
            ranked.append(cand)
            
        ranked.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        return ranked
