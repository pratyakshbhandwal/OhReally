"""
api.py — FastAPI backend for Redrob Hackathon ranker

Run locally:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /rank          — rank a list of candidates
    GET  /health        — health check
"""

import time
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scorer import score_candidate, generate_reasoning, WEIGHTS

app = FastAPI(
    title="Redrob Candidate Ranker API",
    description="Ranks candidates intelligently — no keyword matching, no API calls.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────────

class RankRequest(BaseModel):
    candidates: list[dict]
    top_n: int = 100
    weights: Optional[dict] = None   # override default weights


class RankedCandidate(BaseModel):
    rank: int
    candidate_id: str
    score: float
    reasoning: str
    score_title_career: float
    score_skills: float
    score_experience: float
    score_behavioral: float
    score_soft_fit: float
    honeypot_flag: bool


class RankResponse(BaseModel):
    total_input: int
    top_n: int
    elapsed_seconds: float
    results: list[RankedCandidate]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "message": "Redrob ranker is running"}


@app.post("/rank", response_model=RankResponse)
def rank_candidates(req: RankRequest):
    if not req.candidates:
        raise HTTPException(status_code=400, detail="No candidates provided")
    if len(req.candidates) > 100_000:
        raise HTTPException(status_code=400, detail="Max 100,000 candidates per request")

    # Override weights if provided
    if req.weights:
        total = sum(req.weights.values()) or 1
        for k in WEIGHTS:
            if k in req.weights:
                WEIGHTS[k] = req.weights[k] / total

    t0 = time.time()
    results = []

    for c in req.candidates:
        try:
            scored = score_candidate(c)
            scored["_reasoning"] = generate_reasoning(c, scored)
            scored["_candidate"] = c
            results.append(scored)
        except Exception:
            pass

    # Sort: score desc, then candidate_id asc for ties
    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    top = results[: req.top_n]

    elapsed = time.time() - t0
    prev_score = 1.0
    ranked = []

    for i, r in enumerate(top):
        score = round(min(r["final_score"], prev_score), 4)
        prev_score = score
        ranked.append(
            RankedCandidate(
                rank=i + 1,
                candidate_id=r["candidate_id"],
                score=score,
                reasoning=r["_reasoning"],
                score_title_career=r.get("score_title_career", 0),
                score_skills=r.get("score_skills", 0),
                score_experience=r.get("score_experience", 0),
                score_behavioral=r.get("score_behavioral", 0),
                score_soft_fit=r.get("score_soft_fit", 0),
                honeypot_flag=r.get("honeypot_mult", 1.0) < 0.5,
            )
        )

    return RankResponse(
        total_input=len(req.candidates),
        top_n=len(ranked),
        elapsed_seconds=round(elapsed, 2),
        results=ranked,
    )
