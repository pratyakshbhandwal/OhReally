"""
app.py — Streamlit demo UI for Redrob Hackathon
Deploy free: https://streamlit.io/cloud

Run locally:
    streamlit run app.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Add parent dir so scorer is importable
sys.path.insert(0, str(Path(__file__).parent))
from scorer import score_candidate, generate_reasoning, GOOD_TITLES, BAD_TITLES

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob AI — Candidate Ranker",
    page_icon="🤖",
    layout="wide",
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.rank-badge {
    display:inline-block;
    background:#1d4ed8;
    color:white;
    border-radius:6px;
    padding:2px 10px;
    font-weight:bold;
    font-size:14px;
    min-width:36px;
    text-align:center;
}
.score-bar-wrap { background:#e5e7eb; border-radius:8px; height:8px; margin:4px 0 8px 0; }
.score-bar { background:#1d4ed8; border-radius:8px; height:8px; }
.honeypot-warn { color:#b91c1c; font-size:12px; font-weight:600; }
.good-signal { color:#15803d; font-size:12px; }
.stat-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 16px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🤖 Redrob AI — Candidate Ranker")
st.caption("Ranks candidates the way a great recruiter would — not by keyword count.")

st.markdown("---")

# ─── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    top_n = st.slider("Candidates to rank", min_value=10, max_value=100, value=20, step=10)

    st.markdown("---")
    st.markdown("**Scoring weights**")
    w_title = st.slider("Title & Career Fit", 0, 50, 35)
    w_skills = st.slider("Skill Depth", 0, 50, 25)
    w_exp = st.slider("Experience Quality", 0, 40, 20)
    w_behav = st.slider("Behavioral Availability", 0, 30, 15)
    w_soft = st.slider("Soft Fit Signals", 0, 20, 5)

    total_w = w_title + w_skills + w_exp + w_behav + w_soft
    if total_w != 100:
        st.warning(f"Weights sum to {total_w}, not 100. Results still valid (auto-normalized).")

    st.markdown("---")
    st.markdown("**Filters**")
    hide_honeypots = st.checkbox("Hide likely honeypots (< 0.3 mult)", value=True)
    min_yoe = st.number_input("Min years of experience", 0, 20, 4)
    max_yoe = st.number_input("Max years of experience", 1, 30, 15)

# ─── Upload section ───────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📂 Upload Candidates")
    uploaded = st.file_uploader(
        "Upload candidates.jsonl (or paste JSON below)",
        type=["jsonl", "json"],
        help="Upload the candidates.jsonl file from the hackathon bundle"
    )

with col2:
    st.subheader("📋 Or paste sample JSON")
    sample_json = st.text_area(
        "Paste candidate JSON array here (for quick demo)",
        height=120,
        placeholder='[{"candidate_id": "CAND_0000001", "profile": {...}, ...}]',
    )

# ─── Load candidates ──────────────────────────────────────────────────────────
candidates = []

if uploaded is not None:
    content = uploaded.read().decode("utf-8")
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                obj = json.loads(line)
                if isinstance(obj, list):
                    candidates.extend(obj)
                else:
                    candidates.append(obj)
            except json.JSONDecodeError:
                pass
    st.success(f"✅ Loaded {len(candidates):,} candidates from file")

elif sample_json.strip():
    try:
        parsed = json.loads(sample_json)
        candidates = parsed if isinstance(parsed, list) else [parsed]
        st.success(f"✅ Parsed {len(candidates)} candidate(s) from JSON input")
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")

# ─── Run ranking ──────────────────────────────────────────────────────────────
if candidates:
    # Apply weight overrides
    from scorer import WEIGHTS
    total = w_title + w_skills + w_exp + w_behav + w_soft or 100
    WEIGHTS["title_career"] = w_title / total
    WEIGHTS["skills"] = w_skills / total
    WEIGHTS["experience"] = w_exp / total
    WEIGHTS["behavioral"] = w_behav / total
    WEIGHTS["soft_fit"] = w_soft / total

    if st.button("🚀 Run Ranking", type="primary", use_container_width=True):
        with st.spinner(f"Scoring {len(candidates):,} candidates..."):
            t0 = time.time()
            results = []
            for c in candidates:
                try:
                    yoe = c.get("profile", {}).get("years_of_experience", 0) or 0
                    if yoe < min_yoe or yoe > max_yoe:
                        continue
                    scored = score_candidate(c)
                    if hide_honeypots and scored.get("honeypot_mult", 1.0) < 0.3:
                        continue
                    scored["_candidate"] = c
                    scored["_reasoning"] = generate_reasoning(c, scored)
                    results.append(scored)
                except Exception:
                    pass

            results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
            results = results[:top_n]
            elapsed = time.time() - t0

        st.success(f"✅ Ranked {len(results)} candidates in {elapsed:.2f}s")
        st.session_state["results"] = results
        st.session_state["all_count"] = len(candidates)

# ─── Display results ──────────────────────────────────────────────────────────
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    all_count = st.session_state.get("all_count", len(candidates))

    st.markdown("---")
    st.subheader(f"🏆 Top {len(results)} Candidates")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates Ranked", len(results))
    m2.metric("Total Pool", f"{all_count:,}")
    avg_score = sum(r["final_score"] for r in results) / len(results)
    m3.metric("Avg Score (top)", f"{avg_score:.3f}")
    honeypot_count = sum(1 for r in results if r.get("honeypot_mult", 1.0) < 0.5)
    m4.metric("Honeypots Flagged", honeypot_count, delta=None)

    st.markdown("---")

    # ── Tabs: Cards view | Table view | Download
    tab1, tab2, tab3 = st.tabs(["📇 Card View", "📊 Table View", "⬇️ Download CSV"])

    with tab1:
        for i, r in enumerate(results):
            c = r["_candidate"]
            profile = c.get("profile", {})
            signals = c.get("redrob_signals", {})

            title = profile.get("current_title", "Unknown")
            company = profile.get("current_company", "")
            location = profile.get("location", "")
            yoe = profile.get("years_of_experience", 0) or 0
            score = r["final_score"]
            honeypot = r.get("honeypot_mult", 1.0)

            with st.expander(
                f"#{i+1}  {c['candidate_id']}  —  {title}  ({yoe:.1f} yrs)  •  score {score:.3f}",
                expanded=(i < 3),
            ):
                left, right = st.columns([2, 1])
                with left:
                    st.markdown(f"**{title}** at {company} · {location}")
                    st.markdown(f"_{r['_reasoning']}_")

                    # Score bar
                    pct = int(score * 100)
                    bar_color = "#15803d" if score >= 0.6 else "#1d4ed8" if score >= 0.4 else "#9ca3af"
                    st.markdown(
                        f'<div class="score-bar-wrap"><div class="score-bar" style="width:{pct}%;background:{bar_color}"></div></div>',
                        unsafe_allow_html=True,
                    )

                    # Component breakdown
                    comp_cols = st.columns(5)
                    labels = ["Title/Career", "Skills", "Experience", "Behavioral", "Soft"]
                    keys = ["score_title_career", "score_skills", "score_experience", "score_behavioral", "score_soft_fit"]
                    for col, lbl, key in zip(comp_cols, labels, keys):
                        val = r.get(key, 0)
                        col.metric(lbl, f"{val:.2f}")

                    if honeypot < 0.5:
                        st.markdown(
                            f'<span class="honeypot-warn">⚠️ Honeypot flag: title-skill mismatch (mult={honeypot:.2f})</span>',
                            unsafe_allow_html=True,
                        )

                with right:
                    open_to_work = signals.get("open_to_work_flag", False)
                    notice = signals.get("notice_period_days", "?")
                    response_rate = signals.get("recruiter_response_rate", 0)
                    last_active = signals.get("last_active_date", "?")
                    github = signals.get("github_activity_score", 0)

                    st.markdown("**Availability signals**")
                    st.markdown(
                        f"{'🟢 Open to work' if open_to_work else '🔴 Not open to work'}"
                    )
                    st.markdown(f"📅 Last active: `{last_active}`")
                    st.markdown(f"⏱ Notice: `{notice}` days")
                    st.markdown(f"💬 Response rate: `{response_rate:.0%}`")
                    st.markdown(f"🐙 GitHub score: `{github}`")

    with tab2:
        rows = []
        for i, r in enumerate(results):
            c = r["_candidate"]
            p = c.get("profile", {})
            rows.append({
                "Rank": i + 1,
                "ID": c["candidate_id"],
                "Title": p.get("current_title", ""),
                "Company": p.get("current_company", ""),
                "YOE": p.get("years_of_experience", 0),
                "Location": p.get("location", ""),
                "Score": round(r["final_score"], 4),
                "Title/Career": round(r.get("score_title_career", 0), 2),
                "Skills": round(r.get("score_skills", 0), 2),
                "Experience": round(r.get("score_experience", 0), 2),
                "Behavioral": round(r.get("score_behavioral", 0), 2),
                "Honeypot Mult": round(r.get("honeypot_mult", 1.0), 2),
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
                "Title/Career": st.column_config.NumberColumn(format="%.2f"),
                "Skills": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    with tab3:
        # Build valid submission CSV
        import csv, io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        prev_score = 1.0
        for i, r in enumerate(results):
            score = round(min(r["final_score"], prev_score), 4)
            prev_score = score
            writer.writerow([r["candidate_id"], i + 1, score, r["_reasoning"]])

        csv_bytes = output.getvalue().encode("utf-8")
        st.download_button(
            label="⬇️ Download submission.csv",
            data=csv_bytes,
            file_name="team_xxx.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Rename the file to your registered team ID before submitting.")

elif not candidates:
    # Show demo instructions
    st.info(
        "👆 Upload your candidates.jsonl file or paste sample JSON to get started.\n\n"
        "The ranker will score all candidates across 5 dimensions and return the top results "
        "with reasoning — no API calls, no GPU, fully offline."
    )

    with st.expander("📖 How it works"):
        st.markdown("""
**Scoring components:**

| Component | Weight | What it measures |
|---|---|---|
| Title & Career Fit | 35% | Current title matches AI/ML? Product company experience? No pure services? |
| Skill Depth | 25% | Endorsed, long-duration AI skills — penalizes keyword stuffers |
| Experience Quality | 20% | 5–9 yr sweet spot, production deployments, not research-only |
| Behavioral Availability | 15% | Last active, response rate, open to work, notice period |
| Soft Fit Signals | 5% | GitHub activity, location, startup experience |

**Honeypot detection:**
The dataset contains ~80 trap candidates (e.g. HR Managers with 9 AI skills).
Our system detects these via title-skill mismatch and applies a multiplier down to near-zero.
        """)
