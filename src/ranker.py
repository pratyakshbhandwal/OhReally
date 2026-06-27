import google.generativeai as genai
import os
from typing import List, Dict
import json

class LLMRanker:
    def __init__(self):
        # API key will be read from environment variable GOOGLE_API_KEY automatically if set.
        # Alternatively, we configure it directly.
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        
        # Initialize Gemini 1.5 Pro or Flash
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')

    def rank_candidates(self, job_description: str, candidates: List[Dict]) -> str:
        """
        Uses an LLM to evaluate the shortlisted candidates against the job description.
        Returns a detailed recruiter report.
        """
        prompt = f"""
You are an expert technical recruiter and hiring manager.
Your task is to evaluate a shortlist of candidates against a Job Description.
Do NOT just keyword match. Look at the full picture: career history, behavioral signals, skills, and experience.

### Job Description:
{job_description}

### Candidates (JSON):
{json.dumps(candidates, indent=2)}

### Output Instructions:
For each candidate, provide:
1. **Name** and **Current Role**
2. **Fit Score (0-100)**: A realistic score on how well they fit the actual needs of the role.
3. **Recruiter Notes**: A short, bulleted paragraph explaining *why* they fit or don't fit based on deep semantic understanding (e.g., "While they lack X, their history in Y shows they can easily adapt").
4. **Recommendation**: "Strong Hire", "Hire", "Interview", "Pass".

Rank the candidates from highest Fit Score to lowest.
Format the output nicely using Markdown.
"""
        response = self.model.generate_content(prompt)
        return response.text

if __name__ == "__main__":
    # Test
    ranker = LLMRanker()
    jd = "Need a strong Python backend engineer with database experience."
    cands = [{"id": 1, "name": "Diana", "skills": "Java, SQL", "experience_years": 3, "career_history": "Backend API development", "behavioral_signals": "Fast learner"}]
    print(ranker.rank_candidates(jd, cands))
