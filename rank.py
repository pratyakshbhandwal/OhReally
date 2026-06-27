#!/usr/bin/env python3
"""
rank.py — Upgraded entry point for Redrob Hackathon

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv
    python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv --jd ./my_jd.txt
    python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv --no-semantic

Constraints met:
  - No network calls
  - No GPU (sentence-transformers runs CPU-only)
  - <5 min on 16GB CPU
  - Graceful fallback if sentence-transformers not installed
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import scorer as scorer_module
from scorer import score_candidate, generate_reasoning, set_jd, _load_semantic_model, _sigmoid_normalize


def load_candidates(path: str):
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    candidates = []
    if path.endswith(".gz"):
        import gzip
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(path, "r", encoding="utf-8")

    with opener() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return candidates


def rank_candidates(candidates: list, top_n: int = 100) -> list:
    print(f"[INFO] Scoring {len(candidates):,} candidates...")
    t0 = time.time()

    results = []
    for c in tqdm(candidates, desc="Scoring", unit="cand", ncols=80):
        try:
            scored = score_candidate(c)
            scored["_candidate"] = c
            results.append(scored)
        except Exception as e:
            results.append({
                "candidate_id": c.get("candidate_id", "UNKNOWN"),
                "final_score": 0.0,
                "honeypot_mult": 1.0,
                "_candidate": c,
            })

    elapsed = time.time() - t0
    print(f"[INFO] Scoring done in {elapsed:.1f}s")

    # Apply sigmoid normalization across the full pool
    print("[INFO] Normalizing score distribution...")
    id_score_pairs = [(r["candidate_id"], r["final_score"]) for r in results]
    normalized_pairs = _sigmoid_normalize(id_score_pairs)
    score_map = dict(normalized_pairs)
    for r in results:
        r["final_score_normalized"] = score_map.get(r["candidate_id"], r["final_score"])

    # Sort by normalized score
    results.sort(key=lambda x: (-x["final_score_normalized"], x["candidate_id"]))
    return results[:top_n]


def build_submission(top_candidates: list, out_path: str):
    rows = []
    for rank_idx, scored in enumerate(top_candidates, start=1):
        cand = scored.pop("_candidate")
        reasoning = generate_reasoning(cand, scored)

        # Use normalized score for submission
        score = round(min(max(scored.get("final_score_normalized", scored["final_score"]), 0.0), 1.0), 4)

        rows.append({
            "candidate_id": scored["candidate_id"],
            "rank": rank_idx,
            "score": score,
            "reasoning": reasoning,
        })

    # Enforce non-increasing scores (spec requirement)
    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i - 1]["score"]:
            rows[i]["score"] = rows[i - 1]["score"]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Submission written to: {out_path}")
    print(f"[INFO] Top 5 preview:")
    for r in rows[:5]:
        print(f"  #{r['rank']:>3}  {r['candidate_id']}  score={r['score']:.4f}  {r['reasoning'][:90]}...")


def main():
    parser = argparse.ArgumentParser(description="Redrob Hackathon — Upgraded Candidate Ranker")
    parser.add_argument("--candidates", default="./candidates.jsonl")
    parser.add_argument("--out", default="./team_xxx.csv")
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--jd", default=None, help="Path to a .txt file containing the job description")
    parser.add_argument("--no-semantic", action="store_true", help="Skip semantic scoring (faster, less accurate)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Redrob Hackathon — Intelligent Candidate Ranker v2")
    print("=" * 60)

    # Load custom JD if provided
    if args.jd:
        jd_path = Path(args.jd)
        if jd_path.exists():
            jd_text = jd_path.read_text(encoding="utf-8")
            set_jd(jd_text)
            print(f"[INFO] JD loaded from: {args.jd}")
        else:
            print(f"[WARN] JD file not found: {args.jd}. Using default JD.")

    # Load semantic model (unless skipped)
    if not args.no_semantic:
        print("[INFO] Loading semantic model (all-MiniLM-L6-v2)...")
        semantic_ok = _load_semantic_model()
        if semantic_ok:
            print("[INFO] Semantic scoring: ENABLED")
        else:
            print("[INFO] Semantic scoring: DISABLED (install sentence-transformers to enable)")
    else:
        print("[INFO] Semantic scoring: SKIPPED (--no-semantic flag)")

    t_start = time.time()

    print(f"[INFO] Loading candidates from: {args.candidates}")
    candidates = load_candidates(args.candidates)
    print(f"[INFO] Loaded {len(candidates):,} candidates")

    top = rank_candidates(candidates, top_n=args.top)
    build_submission(top, args.out)

    total = time.time() - t_start
    print(f"\n[DONE] Total time: {total:.1f}s")
    print(f"[DONE] Run: python validate_submission.py {args.out}")


if __name__ == "__main__":
    main()
