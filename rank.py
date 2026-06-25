#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon entry point

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv

Constraints met:
  - No network calls
  - No GPU
  - <5 min on 16GB CPU
  - Pure Python + pandas + numpy
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scorer import score_candidate, generate_reasoning


def load_candidates(path: str):
    """Stream candidates from .jsonl or .jsonl.gz"""
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
    """Score all candidates, return top_n sorted by score desc."""
    print(f"[INFO] Scoring {len(candidates):,} candidates...")
    t0 = time.time()

    results = []
    for c in tqdm(candidates, desc="Scoring", unit="cand", ncols=80):
        try:
            scored = score_candidate(c)
            scored["_candidate"] = c  # keep reference for reasoning
            results.append(scored)
        except Exception as e:
            # Never crash on a single bad record
            results.append({
                "candidate_id": c.get("candidate_id", "UNKNOWN"),
                "final_score": 0.0,
                "honeypot_mult": 1.0,
                "_candidate": c,
            })

    elapsed = time.time() - t0
    print(f"[INFO] Scoring done in {elapsed:.1f}s")

    # Sort by final_score desc, break ties by candidate_id asc (per spec)
    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    return results[:top_n]


def build_submission(top_candidates: list, out_path: str):
    """Write submission.csv in required format."""
    rows = []
    for rank_idx, scored in enumerate(top_candidates, start=1):
        cand = scored.pop("_candidate")
        reasoning = generate_reasoning(cand, scored)

        # Clamp score to [0, 1] and ensure non-increasing
        score = round(min(max(scored["final_score"], 0.0), 1.0), 4)

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

    # Write CSV
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Submission written to: {out_path}")
    print(f"[INFO] Top 5 preview:")
    for r in rows[:5]:
        print(f"  #{r['rank']:>3}  {r['candidate_id']}  score={r['score']:.4f}  {r['reasoning'][:80]}...")


def main():
    parser = argparse.ArgumentParser(description="Redrob Hackathon Candidate Ranker")
    parser.add_argument(
        "--candidates",
        default="./candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--out",
        default="./team_xxx.csv",
        help="Output CSV path (name it your team ID)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="How many candidates to output (default: 100)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Redrob Hackathon — Intelligent Candidate Ranker")
    print("=" * 60)

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
