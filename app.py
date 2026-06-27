"""
app.py — Upgraded Streamlit demo UI for Redrob Hackathon
Run: streamlit run app.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from scorer import (
    score_candidate, generate_reasoning, set_jd,
    _load_semantic_model, _sigmoid_normalize,
    GOOD_TITLES, BAD_TITLES, DEFAULT_JD_TEXT,
    skill_name_matches_ai, is_product_company,
    count_production_signals, count_impact_signals,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OhReally — AI Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.ohreally-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    border: 1px solid #334155;
}
.ohreally-header h1 {
    color: #f8fafc;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.ohreally-header p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}
.badge-semantic { background:#7c3aed; color:white; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-good { background:#065f46; color:#6ee7b7; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-warn { background:#7f1d1d; color:#fca5a5; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-neutral { background:#1e3a5f; color:#93c5fd; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }

.candidate-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.candidate-card:hover { border-color: #334155; }

.rank-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #475569;
    line-height: 1;
}
.rank-num.top3 { color: #7c3aed; }

.score-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f8fafc;
}
.score-chip.high { color: #4ade80; }
.score-chip.mid  { color: #60a5fa; }
.score-chip.low  { color: #94a3b8; }

.progress-track { background:#1e293b; border-radius:4px; height:6px; margin:6px 0 12px; }
.progress-fill  { border-radius:4px; height:6px; }

.evidence-block {
    background: #1e293b;
    border-left: 3px solid #7c3aed;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 13px;
    color: #cbd5e1;
    margin-top: 8px;
    font-style: italic;
}

.metric-mini { text-align:center; }
.metric-mini .val { font-size:1.3rem; font-weight:700; color:#f8fafc; }
.metric-mini .lbl { font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; }

.jd-box {
    background:#0f172a;
    border:1px solid #334155;
    border-radius:10px;
    padding:16px;
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    color:#94a3b8;
    max-height:200px;
    overflow-y:auto;
}
.signal-pill {
    display:inline-block;
    background:#1e293b;
    border:1px solid #334155;
    border-radius:20px;
    padding:2px 10px;
    font-size:12px;
    color:#94a3b8;
    margin:2px;
}
.signal-pill.active { background:#1e3a5f; border-color:#3b82f6; color:#93c5fd; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ohreally-header">
    <h1>🎯 OhReally — AI Candidate Ranker</h1>
    <p>Semantic ranking that understands candidates the way a great recruiter would — not keyword counting.</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    top_n = st.slider("Shortlist size", 5, 100, 20, 5)

    st.markdown("---")
    st.markdown("**Scoring weights**")
    w_sem    = st.slider("Semantic Fit",           0, 50, 30)
    w_title  = st.slider("Title & Career",         0, 50, 25)
    w_skills = st.slider("Skill Depth",            0, 40, 20)
    w_exp    = st.slider("Experience Quality",     0, 30, 15)
    w_behav  = st.slider("Behavioral Availability",0, 20, 10)

    total_w = w_sem + w_title + w_skills + w_exp + w_behav
    if total_w != 100:
        st.caption(f"Weights sum to {total_w} — auto-normalized ✓")

    st.markdown("---")
    st.markdown("**Filters**")
    hide_honeypots = st.checkbox("Hide honeypot candidates", value=True)
    min_yoe = st.number_input("Min YOE", 0, 20, 3)
    max_yoe = st.number_input("Max YOE", 1, 30, 15)

    st.markdown("---")
    use_semantic = st.checkbox("Enable semantic scoring", value=True)
    if use_semantic:
        st.caption("Uses all-MiniLM-L6-v2 (~80MB, CPU). Install: `pip install sentence-transformers`")

# ─── Main layout: JD | Candidates ─────────────────────────────────────────────
col_jd, col_cands = st.columns([1, 1], gap="medium")

with col_jd:
    st.markdown("#### 📋 Job Description")
    jd_text = st.text_area(
        "Paste or edit the JD — the ranker reads this dynamically",
        value=DEFAULT_JD_TEXT.strip(),
        height=300,
        help="Change this to rank for a different role. The system extracts skills, YOE range, and notice period automatically."
    )
    if jd_text:
        set_jd(jd_text)
        st.caption("✅ JD parsed — skills, YOE range and notice period extracted automatically")

with col_cands:
    st.markdown("#### 📂 Candidates")
    uploaded = st.file_uploader(
        "Upload candidates.jsonl",
        type=["jsonl", "json"],
    )
    st.caption("Or paste JSON below for a quick demo:")
    sample_json = st.text_area(
        "Paste candidate JSON array",
        height=120,
        placeholder='[{"candidate_id": "CAND_001", "profile": {...}, ...}]',
        label_visibility="collapsed"
    )

# ─── Load candidates ──────────────────────────────────────────────────────────
candidates = []
if uploaded is not None:
    content = uploaded.read().decode("utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            candidates.extend(obj) if isinstance(obj, list) else candidates.append(obj)
        except json.JSONDecodeError:
            pass
    st.success(f"✅ Loaded {len(candidates):,} candidates")

elif sample_json.strip():
    try:
        parsed = json.loads(sample_json)
        candidates = parsed if isinstance(parsed, list) else [parsed]
        st.success(f"✅ Parsed {len(candidates)} candidate(s)")
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")

# ─── Run ranking ──────────────────────────────────────────────────────────────
if candidates:
    # Apply weight overrides to scorer module
    from scorer import WEIGHTS as scorer_weights
    total = w_sem + w_title + w_skills + w_exp + w_behav or 100
    scorer_weights["semantic"]     = w_sem    / total
    scorer_weights["title_career"] = w_title  / total
    scorer_weights["skills"]       = w_skills / total
    scorer_weights["experience"]   = w_exp    / total
    scorer_weights["behavioral"]   = w_behav  / total

    run_btn = st.button("🚀 Run Ranking", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Loading semantic model..."):
            if use_semantic:
                _load_semantic_model()

        with st.spinner(f"Scoring {len(candidates):,} candidates..."):
            t0 = time.time()
            results = []
            prog = st.progress(0)

            for i, c in enumerate(candidates):
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
                prog.progress((i + 1) / len(candidates))

            prog.empty()

            # Normalize score distribution
            id_score_pairs = [(r["candidate_id"], r["final_score"]) for r in results]
            norm_pairs = _sigmoid_normalize(id_score_pairs)
            score_map = dict(norm_pairs)
            for r in results:
                r["final_score"] = score_map.get(r["candidate_id"], r["final_score"])

            results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
            results = results[:top_n]
            elapsed = time.time() - t0

        st.session_state["results"] = results
        st.session_state["all_count"] = len(candidates)
        st.success(f"✅ Ranked {len(results)} candidates in {elapsed:.2f}s")

# ─── Results ──────────────────────────────────────────────────────────────────
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    all_count = st.session_state.get("all_count", len(candidates))

    st.markdown("---")

    # Summary metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Shortlisted", len(results))
    m2.metric("Total Pool", f"{all_count:,}")
    avg_score = sum(r["final_score"] for r in results) / len(results)
    m3.metric("Avg Score", f"{avg_score:.3f}")
    honeypot_count = sum(1 for r in results if r.get("honeypot_mult", 1.0) < 0.5)
    m4.metric("Honeypots Flagged", honeypot_count)
    sem_enabled = results[0].get("score_semantic", 0) > 0 if results else False
    m5.metric("Semantic", "✅ ON" if sem_enabled else "⚠️ OFF")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🃏 Card View", "📊 Table View", "⬇️ Download CSV"])

    # ── Card View ──
    with tab1:
        for i, r in enumerate(results):
            c = r["_candidate"]
            profile = c.get("profile", {})
            signals = c.get("redrob_signals", {})
            career = c.get("career_history", [])
            skills_list = c.get("skills", [])

            title     = profile.get("current_title", "Unknown")
            company   = profile.get("current_company", "")
            location  = profile.get("location", "")
            yoe       = profile.get("years_of_experience", 0) or 0
            score     = r["final_score"]
            honeypot  = r.get("honeypot_mult", 1.0)
            sem_score = r.get("score_semantic", 0)

            score_class = "high" if score >= 0.65 else "mid" if score >= 0.45 else "low"
            rank_class  = "top3" if i < 3 else ""

            with st.expander(
                f"#{i+1}  ·  {title}  ·  {yoe:.0f} yrs  ·  score {score:.3f}",
                expanded=(i < 3),
            ):
                left, right = st.columns([3, 1])

                with left:
                    # Title + company
                    st.markdown(f"**{title}** @ {company}" + (f"  ·  📍 {location}" if location else ""))

                    # Badges
                    badge_html = ""
                    if sem_score >= 0.65:
                        badge_html += f'<span class="badge-semantic">🧠 High Semantic Match ({sem_score:.2f})</span> '
                    elif sem_score >= 0.45:
                        badge_html += f'<span class="badge-neutral">Semantic {sem_score:.2f}</span> '
                    if honeypot < 0.3:
                        badge_html += '<span class="badge-warn">⚠️ Honeypot Flagged</span>'
                    elif signals.get("open_to_work_flag"):
                        badge_html += '<span class="badge-good">✅ Open to Work</span>'
                    if badge_html:
                        st.markdown(badge_html, unsafe_allow_html=True)

                    # Reasoning (evidence-based)
                    st.markdown(f'<div class="evidence-block">{r["_reasoning"]}</div>', unsafe_allow_html=True)

                    # Score bar
                    pct = int(score * 100)
                    bar_color = "#4ade80" if score >= 0.65 else "#60a5fa" if score >= 0.45 else "#475569"
                    st.markdown(
                        f'<div class="progress-track"><div class="progress-fill" style="width:{pct}%;background:{bar_color}"></div></div>',
                        unsafe_allow_html=True,
                    )

                    # Component breakdown
                    comp_cols = st.columns(5)
                    labels = ["Semantic", "Title/Career", "Skills", "Experience", "Behavioral"]
                    keys   = ["score_semantic", "score_title_career", "score_skills", "score_experience", "score_behavioral"]
                    for col, lbl, key in zip(comp_cols, labels, keys):
                        val = r.get(key, 0)
                        col.metric(lbl, f"{val:.2f}")

                    # Top AI skills pills
                    top_ai = sorted(
                        [s for s in skills_list if skill_name_matches_ai(s.get("name", ""))],
                        key=lambda s: s.get("endorsements", 0), reverse=True
                    )[:6]
                    if top_ai:
                        pills = "".join(
                            f'<span class="signal-pill active">{s["name"]} ({s.get("endorsements",0)}⭐)</span>'
                            for s in top_ai
                        )
                        st.markdown(f"**Top AI skills:** {pills}", unsafe_allow_html=True)

                with right:
                    st.markdown("**Availability**")
                    open_flag = signals.get("open_to_work_flag", False)
                    st.markdown("🟢 Open to work" if open_flag else "⚫ Not open")
                    notice = signals.get("notice_period_days", "?")
                    st.markdown(f"⏱ **{notice}d** notice")
                    rr = signals.get("recruiter_response_rate", 0)
                    st.markdown(f"💬 **{rr:.0%}** response rate")
                    last_active = signals.get("last_active_date", "?")
                    st.markdown(f"📅 Active: `{str(last_active)[:10]}`")
                    github = signals.get("github_activity_score", 0) or 0
                    st.markdown(f"🐙 GitHub: **{github}**")

                    st.markdown("---")
                    st.markdown(f"**Honeypot mult:** `{honeypot:.2f}`")
                    st.markdown(f"**Raw score:** `{r.get('final_score', 0):.4f}`")

    # ── Table View ──
    with tab2:
        rows = []
        for i, r in enumerate(results):
            c  = r["_candidate"]
            p  = c.get("profile", {})
            rows.append({
                "Rank":      i + 1,
                "ID":        c["candidate_id"],
                "Title":     p.get("current_title", ""),
                "Company":   p.get("current_company", ""),
                "YOE":       p.get("years_of_experience", 0),
                "Location":  p.get("location", ""),
                "Score":     round(r["final_score"], 4),
                "Semantic":  round(r.get("score_semantic", 0), 2),
                "Title/Career": round(r.get("score_title_career", 0), 2),
                "Skills":    round(r.get("score_skills", 0), 2),
                "Experience":round(r.get("score_experience", 0), 2),
                "Behavioral":round(r.get("score_behavioral", 0), 2),
                "Honeypot":  round(r.get("honeypot_mult", 1.0), 2),
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Score":    st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
                "Semantic": st.column_config.NumberColumn(format="%.2f"),
                "Skills":   st.column_config.NumberColumn(format="%.2f"),
                "Honeypot": st.column_config.NumberColumn(format="%.2f"),
            }
        )

    # ── Download ──
    with tab3:
        import csv as csv_mod, io
        output = io.StringIO()
        writer = csv_mod.writer(output)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        prev_score = 1.0
        for i, r in enumerate(results):
            score = round(min(r["final_score"], prev_score), 4)
            prev_score = score
            writer.writerow([r["candidate_id"], i + 1, score, r["_reasoning"]])

        st.download_button(
            label="⬇️ Download submission CSV",
            data=output.getvalue().encode("utf-8"),
            file_name="team_xxx.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Rename to your team ID before submitting.")

elif not candidates:
    st.info("👆 Upload `candidates.jsonl` or paste JSON to get started. Edit the JD on the left to rank for any role.")

    with st.expander("📖 What's different in v2?"):
        st.markdown("""
**Upgrades over v1:**

| What changed | Why it matters |
|---|---|
| **Semantic scoring (30%)** via `sentence-transformers` | Understands "built dense retrieval pipelines" not just keyword "retrieval" |
| **Dynamic JD parsing** | Change the JD text box → the entire scoring adapts automatically |
| **Rich description analysis** | Reads what candidates actually *did*, not just their title |
| **Fixed product company detection** | v1 treated all small companies as product; v2 requires real signals |
| **Sigmoid score normalization** | Spreads scores across the range so ranking is decisive |
| **Evidence-based reasoning** | Reasoning cites specific skills, companies, and impact sentences |
| **Upgraded honeypot detection** | Catches skill inflation, description incoherence, thin profiles |
        """)
