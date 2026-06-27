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
            
            title = row.get('profile', {}).get('current_title', 'Unknown')
            try:
                yoe = float(row.get('profile', {}).get('years_of_experience', 0.0))
            except:
                yoe = 0.0
                
            try:
                rr = float(row.get('redrob_signals', {}).get('recruiter_response_rate', 0.0))
            except:
                rr = 0.0
                
            skills = row.get('skills', [])
            num_skills = len(skills) if isinstance(skills, list) else 0
            
            reasons = []
            if is_keyword_stuffer(row): 
                final_score *= 0.2
                reasons.append(f"an improbable {num_skills} skills listed")
            if is_unrelated_title(row): 
                final_score *= 0.1
                reasons.append(f"an unrelated title ({title})")
            if is_pure_consulting(row): 
                final_score *= 0.5
                reasons.append("an entirely consulting-based career history")
            if is_title_chaser(row): 
                final_score *= 0.6
                reasons.append("a high job turnover rate")
            if is_pure_research(row): 
                final_score *= 0.7
                reasons.append("a purely academic/research background")
            if is_langchain_enthusiast(row): 
                final_score *= 0.6
                reasons.append("framework wrapper skills without core ML depth")
                
            if is_honeypot(row):
                continue
                
            # Construct final reasoning with high variance and specific facts
            if not reasons:
                skill_names = [s.get('name', '') for s in skills[:3]] if isinstance(skills, list) else []
                skills_str = ", ".join(skill_names) if skill_names else "relevant data skills"
                
                parts = []
                if jd_multiplier >= 1.3:
                    parts.append(f"Exceptional JD match: {yoe} years as a {title} with explicit Vector DB or ranking evaluation experience.")
                else:
                    parts.append(f"Solid semantic alignment for a {title} bringing {yoe} years of experience and proficiency in {skills_str}.")
                    
                if beh_multiplier > 1.1:
                    parts.append(f"Logistics are highly favorable (Response rate: {rr:.2f}).")
                elif beh_multiplier < 0.9:
                    parts.append(f"Engagement signals are slightly concerning (Response rate: {rr:.2f}).")
                reasoning = " ".join(parts)
            else:
                penalty_str = " and ".join(reasons)
                reasoning = f"Candidate is a {title} with {yoe} yrs experience, but was penalized for {penalty_str}. (Response rate: {rr:.2f})"
                
            cand['final_score'] = final_score
            cand['reasoning'] = reasoning
            ranked.append(cand)
            
        ranked.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        return ranked
