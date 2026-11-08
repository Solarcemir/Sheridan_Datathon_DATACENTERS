import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from explain import explain_cve, generate, has_credentials

DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="Before the Breach · Analyst drill-down", layout="wide")


@st.cache_data
def load():
    return {
        "dos": pd.read_parquet(DATA / "dossier.parquet"),
        "missed": pd.read_csv(DATA / "missed_2025.csv"),
        "msum": pd.read_csv(DATA / "missed_summary.csv"),
        "info": json.loads((DATA / "build_info.json").read_text()),
    }


D = load()
DOS = D["dos"].set_index("cve_id", drop=False)
GROUP_COLOR = {"Targeted zero-days on consumer platforms": "#6a3d9a",
               "Chained / modest-looking edge & enterprise bugs": "#1f78b4",
               "End-of-life botnet targets": "#e31a1c"}


def nice(v):
    return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).replace("_", " ")


def dossier(cve: str):
    r = DOS.loc[cve].to_dict()
    where = f"Submission rank #{int(r['submission_rank'])}"
    st.markdown(f"### {cve}")
    st.caption(f"{where} · {nice(r['granite_vendor'])} {nice(r['granite_product'])} · assigner `{r['assigner']}` · "
               f"published {r['published']} · snapshot {r['snapshot_month']}")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**What the advisory says**")
        st.write(r.get("description") or "_No description cached._")
        st.caption("NVD text fetched Nov 8 for reading only; the models never saw NVD's current record.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CVSS 3 at publish", nice(r["cvss3"]))
        c2.metric("Attack vector", nice(r["cv_attack_vector"]))
        c3.metric("Privileges", nice(r["cv_privileges_required"]))
        c4.metric("User interaction", nice(r["cv_user_interaction"]))
        if isinstance(r.get("nvd_vector"), str):
            st.caption(f"Current NVD vector: `{r['nvd_vector']}` ({r['nvd_cvss']} {nice(r['nvd_severity'])}) · "
                       f"CWE {nice(r['cwe_ids'])}")
        src = [n for n, k in [("Exploit-DB", "exploitdb"), ("Metasploit", "metasploit"), ("Nuclei", "nuclei"),
                              ("GitHub PoC", "poc_github")] if r.get(k) == 1]
        st.markdown("**Public exploit code at snapshot:** " + (" · ".join(src) if src else "none")
                    + f" &nbsp;|&nbsp; **EPSS** {r['epss']:.4f} (month percentile {r['epss_pct'] * 100:.0f})")
        with st.container(border=True):
            st.markdown("**Analyst note** · written by IBM watsonx.ai")
            note = explain_cve(r)
            if note is None and has_credentials():
                if st.button("Generate with watsonx", key=f"gen_{cve}"):
                    with st.spinner("Asking watsonx..."):
                        try:
                            note = generate(r)
                        except Exception as ex:
                            st.error(f"watsonx call failed: {ex}")
            if note:
                st.markdown(note["text"])
                st.caption(f"{note['model']} on watsonx.ai · {note['generated']} · written only from the facts on "
                           "this page; display only, never a model input.")
            elif not has_credentials():
                st.caption("No note for this CVE: the watsonx free-plan token quota ran out before "
                           "every CVE was covered.")
    with right:
        st.markdown("**Where each model ranks it** (out of 72,670)")
        votes = [("Trees", "tree_rank"), ("Linear", "linear_rank"), ("Granite 8B", "llm_rank")]
        inside = 0
        for col, (name, k) in zip(st.columns(3), votes):
            rk = int(r[k])
            inside += rk <= 500
            col.metric(name, f"#{rk:,}", "in its top 500" if rk <= 500 else "outside its top 500",
                       delta_color="normal" if rk <= 500 else "inverse")
        st.caption({3: "All three models pick it on their own.", 2: "Two of three models pick it on their own.",
                    1: "Only one model picks it; the vote carried it.",
                    0: "No single model picks it; it made it on the combined vote."}[inside]
                   if r["in_submission"] else f"{inside} of 3 models have it in their own top 500.")
        st.markdown("**Granite 4.2 30B analyst profile**")
        prof = [("Product type", "product_type"), ("Internet exposure", "internet_exposed"),
                ("Install base", "install_base"), ("Used by", "used_by"), ("Typical attacker", "typical_attacker"),
                ("Attacker appeal", "attacker_appeal"), ("Impact", "impact"), ("Needs", "precondition"),
                ("End of life", "end_of_life"), ("Fix status", "fix_status")]
        st.dataframe(pd.DataFrame({"": [a for a, _ in prof], "Granite says": [nice(r[k]) for _, k in prof]}),
                     hide_index=True, width="stretch", height=388)


st.title("Before the Breach · analyst drill-down")
st.caption("Companion to the team's Breach Radar story, which covers the method, benchmark and the 500 picks. "
           "This app covers what it doesn't: where the model is blind, and a full file for each of our 500 CVEs.")

tab_blind, tab_cve = st.tabs(["Where the model was blind", "Open a CVE file"])

with tab_blind:
    st.header("22 exploited bugs no method saw coming")
    st.markdown("Every 2025 bug that reached CISA KEV, Jan–Sep. **22 of 52** sat below rank 1,000 in every model "
                "we built (baseline, Granite zero-shot, many-shot, stacked LR).")
    M, S = D["missed"], D["msum"]
    sm = S.set_index("metric")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Had public exploit code", "0%", "found ones: 73%", delta_color="off")
    k2.metric("Median EPSS percentile", "61", "found ones: 96", delta_color="off")
    k3.metric("Median days to KEV", "37.5", "found ones: 56 (missed are faster)", delta_color="off")
    k4.metric("Median CVSS 3", "7.6", "found ones: 9.0", delta_color="off")
    st.markdown("The misses look ordinary on paper and get exploited **sooner**. The standard signals "
                "(exploit code, EPSS, severity) are silent exactly where it matters.")
    a, b = st.columns([3, 2], gap="large")
    with a:
        M2 = M.copy()
        M2["epss"] = pd.to_numeric(M2.epss_pct, errors="coerce")
        M2["cvss_n"] = pd.to_numeric(M2.cvss, errors="coerce")
        fig = px.scatter(M2, x="days_to_kev", y="best_rank_any_method", color="group", size="cvss_n",
                         hover_name="cve_id", hover_data={"vendor_product": True, "epss": True, "cvss_n": False},
                         color_discrete_map=GROUP_COLOR, log_y=True, height=430,
                         labels={"days_to_kev": "days from publish to KEV", "best_rank_any_method":
                                 "best rank any model gave it (log)", "group": ""})
        fig.update_yaxes(autorange="reversed")
        fig.add_hline(y=1000, line_dash="dot", annotation_text="rank 1,000")
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
        st.caption("Each dot is a missed 2025 KEV bug; size = CVSS. Hover for vendor and EPSS.")
    with b:
        for g, color in GROUP_COLOR.items():
            sub = M[M.group == g]
            st.markdown(f"<span style='color:{color};font-weight:600'>■ {g} · {len(sub)}</span>",
                        unsafe_allow_html=True)
            st.caption(", ".join(sorted(sub.vendor_product.str.split(" / ").str[0].unique())))
        with st.expander("All 22, with descriptions"):
            st.dataframe(M[["group", "cve_id", "vendor_product", "cvss", "epss_pct", "days_to_kev",
                            "description"]].sort_values("group"), hide_index=True, width="stretch")
    st.caption("Source: chen/reports/missed.txt (2025 forward holdout, KEV by Dec 31 2025).")

with tab_cve:
    st.header("Open a CVE file")
    st.markdown("Everything we know about one CVE in one place: the advisory text, the CVSS vector, "
                "Granite's full analyst profile and how much each model believes in it.")
    dos = D["dos"]
    f1, f2, f3 = st.columns(3)
    d = dos
    sv = f1.multiselect("Vendor", sorted(d.granite_vendor.fillna("unknown").unique()))
    sa = f2.multiselect("Assigner", sorted(d.assigner.fillna("unknown").unique()))
    sp = f3.multiselect("Product type", sorted(d.product_type.fillna("unknown").unique()), format_func=nice)
    if sv: d = d[d.granite_vendor.fillna("unknown").isin(sv)]
    if sa: d = d[d.assigner.fillna("unknown").isin(sa)]
    if sp: d = d[d.product_type.fillna("unknown").isin(sp)]
    if d.empty:
        st.info("No CVEs match.")
    else:
        lab = d.apply(lambda r: (f"#{int(r.submission_rank)} " if r.in_submission else "") +
                      f"{r.cve_id} · {nice(r.granite_vendor)} {nice(r.granite_product)}", axis=1)
        pick = st.selectbox(f"CVE ({len(d)} match)", d.cve_id.tolist(), format_func=dict(zip(d.cve_id, lab)).get)
        st.divider()
        dossier(pick)
