"""Data for the Breach Radar presentation (chen/frontend/breach_radar.html).

1. Rebuilds the final ranking exactly like luis/analysis/final_build.py, with the unified LR as the only linear voter
   (tree and LLM voters and the Granite-demotion flag come from luis/derived/scored_2026_final.csv; equal-weight rank
   average; demoted CVEs pushed down; cap 3 per assigner-month).
2. Explains each of the top 500 with Luis's reason template (team_merge.py): exploit code, EPSS within month, Granite 30B's
   impact / appeal / attacker / end-of-life, nearest past KEV entry (Granite Embedding r2), assigner KEV history.
3. Aggregates for the stage panels. Only counts, shares and our own 500 picks leave this script; no raw IBM rows.

Run from the repo root on a machine with the processed data (chen's ~/datathon/processed):
  python chen/frontend/build_radar.py [path/to/processed]
Writes chen/frontend/radar_data.json and injects it into chen/frontend/breach_radar.html (between the DATA markers).
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CHEN, LUIS = REPO / "chen", REPO / "luis"
PROC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "datathon" / "processed"
OUT = Path(__file__).resolve().parent
FLAGS = [("exploitdb", "Exploit-DB"), ("metasploit", "Metasploit"), ("nuclei", "Nuclei"), ("poc_github", "GitHub PoC")]


def rank01(x):
    return pd.Series(np.asarray(x, float)).rank(pct=True).to_numpy()


def capped(df, s, cap):
    s = pd.Series(np.asarray(s, float), index=df.index)
    n = df.assign(_s=s).sort_values("_s", ascending=False, kind="stable").groupby(["assigner", "snapshot_month"]).cumcount()
    s = s.copy()
    s[n[n >= cap].index] -= 100
    return s.to_numpy()


def main() -> None:
    old = pd.read_csv(LUIS / "derived" / "scored_2026_final.csv")
    s = old[["cve_id", "snapshot_month", "assigner", "tree", "llm", "granite_demoted", "rank_final"]].copy()
    u = pd.read_parquet(CHEN / "submissions" / "lr_unified_scoreboard_2026.parquet")
    s = s.merge(u, on="cve_id", how="left")
    s["linear"] = rank01(s.lr_score.fillna(s.lr_score.min()))
    raw = s[["tree", "linear", "llm"]].mean(axis=1).to_numpy() - 50 * s.granite_demoted.to_numpy()
    nocap = s.assign(_r=raw).nlargest(500, "_r").cve_id
    s["score"] = capped(s, raw, 3)
    s = s.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    s["rank"] = np.arange(1, len(s) + 1)

    sb = pd.read_parquet(PROC / "scoreboard_2026_view.parquet").drop(columns=["snapshot_month", "assigner"])
    sb["epss_pct"] = sb.epss.fillna(0).groupby(pd.read_parquet(PROC / "scoreboard_2026_view.parquet").snapshot_month).rank(pct=True)
    s = s.merge(sb, on="cve_id", how="left")
    s["any_code"] = s[[f for f, _ in FLAGS]].max(axis=1).fillna(0).astype(int)
    ex = pd.read_parquet(CHEN / "processed" / "llm_extract.parquet").set_index("cve_id")
    em = pd.read_parquet(CHEN / "processed" / "emb_feat_scoreboard.parquet").set_index("cve_id")[["nearest_kev", "kev_sim_max"]]
    me = pd.read_parquet(CHEN / "processed" / "meta_scoreboard.parquet").set_index("cve_id")
    s = s.join(ex, on="cve_id").join(em, on="cve_id")
    s["assigner_kev"] = s.cve_id.map(me.assigner_kev)
    top = s.head(500).copy()

    h = lambda v: "" if pd.isna(v) else str(v).replace("_", " ")  # noqa: E731

    def product(r):
        v, p = h(r.vendor), h(r["product"])
        return p if (not v or v.lower() in p.lower()) else f"{v} {p}"

    def reason(r):
        parts = []
        code = [n for f, n in FLAGS if r[f] == 1]
        if code:
            parts.append("public exploit code (" + ", ".join(code) + ")")
        if r.epss_pct >= 0.95:
            parts.append(f"EPSS in the top {max(1, round((1 - r.epss_pct) * 100))}% of its month")
        if h(r.impact) and r.impact != "other":
            parts.append(h(r.impact) + (" with no login needed" if r.precondition == "none" else ""))
        if r.attacker_appeal == "high":
            parts.append(f"high attacker appeal ({h(r.typical_attacker)})")
        if r.end_of_life == "yes":
            parts.append("end-of-life product")
        if pd.notna(r.kev_sim_max) and r.kev_sim_max >= 0.85 and pd.notna(r.nearest_kev):
            parts.append(f"resembles KEV entry {r.nearest_kev} (similarity {r.kev_sim_max:.2f})")
        if pd.notna(r.assigner_kev) and r.assigner_kev >= 3:
            parts.append(f"{int(r.assigner_kev)} past KEV entries from this assigner")
        t = "; ".join(parts) or "ranked high by the combined model"
        return t[0].upper() + t[1:] + "."

    rows = [{"r": int(r["rank"]), "id": r.cve_id, "m": r.snapshot_month, "p": product(r), "cat": h(r.product_type),
             "e": None if pd.isna(r.epss) else round(float(r.epss), 4),
             "fl": "".join(str(int(r[f] == 1)) for f, _ in FLAGS),
             "v": [round(float(r.tree), 3), round(float(r.linear), 3), round(float(r.llm), 3)],
             "why": reason(r)} for _, r in top.iterrows()]

    # ---- aggregates
    allx = ex.reindex(s.cve_id)
    def mix(col, n=7):
        a = allx[col].value_counts(normalize=True) * 100
        t = top[col].value_counts(normalize=True) * 100
        keys = t.index[:n]
        return [{"k": h(k), "top": round(t[k], 1), "all": round(a.get(k, 0), 1), "lift": round(t[k] / max(a.get(k, 0), 1e-9), 1)} for k in keys]
    vtop = {k: set(s.nlargest(500, k).cve_id) for k in ("tree", "linear", "llm")}
    agree = {"3": 0, "2": 0, "1": 0, "0": 0}
    for c in top.cve_id:
        agree[str(sum(c in vtop[k] for k in vtop))] += 1
    corr = {f"{a}-{b}": round(s[a].rank().corr(s[b].rank()), 2) for a, b in (("tree", "linear"), ("tree", "llm"), ("linear", "llm"))}

    coef = pd.read_csv(CHEN / "results" / "lr_unified_coefficients.csv")
    GROUPS = [("granite", "Granite 30B reading", r"^llm_"),
              ("text", "Description text", r"text_kev|_text$|txt_|desc_len|cwe_|vuldb"),
              ("signals", "Exploit signals", r"exploitdb|metasploit|nuclei|poc_github|any_code|epss|n_flags|sibling"),
              ("history", "History & release", r"assigner|batch|out_of_band|twins|patch_tuesday|old_id|adjacent|vendor|product|ransom|n_versions|age_days|wordpress"),
              ("cvss", "CVSS", r"cvss|av_|ac_|pr_|ui_|scope")]
    def grp(f):
        for g, _, rx in GROUPS:
            if re.search(rx, f):
                return g
        return "cvss"
    coef["g"] = coef.feature.map(grp)
    coef["a"] = coef.coef_per_sd.abs()
    tot = coef.a.sum()
    weights = []
    for g, name, _ in GROUPS:
        c = coef[coef.g == g].sort_values("a", ascending=False)
        weights.append({"g": g, "name": name, "n": int(len(c)), "share": round(100 * c.a.sum() / tot, 1),
                        "top": [{"f": r.feature, "c": round(r.coef_per_sd, 3)} for r in c.head(6).itertuples()]})
    story = json.loads((LUIS / "frontend" / "story_data.json").read_text())
    data = {
        "built": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "n_scoreboard": int(len(s)), "share_code_all": round(100 * s.any_code.mean(), 1),
        "top_code": int(top.any_code.sum()), "top_oracle": int((top.assigner == "secalert_us@oracle.com").sum()),
        "top_assigners": int(top.assigner.nunique()), "overlap_two_lr_final": int(len(set(top.cve_id) & set(old.nsmallest(500, "rank_final").cve_id))),
        "cap_pushed_out": int(len(set(nocap) - set(top.cve_id))), "demoted_share": round(100 * s.granite_demoted.mean(), 1),
        "months": {"top": top.snapshot_month.value_counts().sort_index().to_dict(),
                   "all": s.snapshot_month.value_counts().sort_index().to_dict()},
        "product_type": mix("product_type"), "impact": mix("impact", 5), "attacker": mix("typical_attacker", 4),
        "vendors": top.vendor.map(h).value_counts().head(10).to_dict(),
        "epss_buckets": {"labels": ["< 0.001", "0.001–0.01", "0.01–0.1", "≥ 0.1"],
                         "top": np.histogram(top.epss.fillna(0), [-1, 0.001, 0.01, 0.1, 2])[0].tolist(),
                         "all": (np.histogram(s.epss.fillna(0), [-1, 0.001, 0.01, 0.1, 2])[0] / len(s) * 100).round(1).tolist()},
        "agree": agree, "corr": corr, "weights": weights,
        "drivers": [{"f": r.feature, "c": round(r.coef_per_sd, 3), "g": r.g} for r in coef.sort_values("a", ascending=False).head(16).itertuples()],
        "near_kev_085": int((top.kev_sim_max >= 0.85).sum()),
        "clusters": story["pre"]["clusters"][:6], "kev_by_year": story["pre"]["kev_by_year"], "kev_total": story["pre"]["kev_total"],
        "univariate": story["pre"]["univariate"][:8],
        "top500": rows,
    }
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (OUT / "radar_data.json").write_text(js)
    html = OUT / "breach_radar.html"
    if html.exists():
        t = html.read_text()
        t = re.sub(r"(<script id=\"radar-data\" type=\"application/json\">).*?(</script>)", lambda m: m.group(1) + js + m.group(2), t, flags=re.S)
        html.write_text(t)
    print(f"top 500: {len(rows)} | exploit code {data['top_code']} | Oracle {data['top_oracle']} | assigners {data['top_assigners']} | "
          f"shared with the 2-LR final {data['overlap_two_lr_final']} | pushed out by cap {data['cap_pushed_out']} | agree {agree} | corr {corr}")
    print("weights:", [(w["name"], w["n"], w["share"]) for w in weights])


if __name__ == "__main__":
    main()
