"""IBM Granite Instruct: read the top candidates like an analyst would, then re-rank.

Stage 1  LightGBM on safe features + TF-IDF text (the v1 model), scored out-of-sample:
         2024 rows by a model trained on 2025, 2025 rows by a model trained on 2024, 2026 by a model on both.
Stage 2  For the top 5% of each year, ibm-granite/granite-3.3-2b-instruct answers yes/no questions
         about the description. We read its confidence (log-odds of "yes" vs "no") from a single forward
         pass per question, which is much faster than generating text and gives continuous scores.
         The full run asks q_internet, q_enterprise and q_niche; --test shows all five:
           q_internet    usually exposed to the internet (VPN, firewall, gateway, mail/web server, remote access)?
           q_preauth     exploitable without logging in?
           q_takeover    code execution or full control?
           q_enterprise  widely used by large companies or governments?
           q_niche       small plugin, hobby project or niche product?
         Cached in derived/granite/instruct_{year}.csv. A logistic model on [stage-1 rank + the five answers],
         trained on the other year, re-ranks the candidates. Same A/B protocol as train_baseline.py.
Stage 3  For the final 2026 top 500, Granite writes product, category and a one-sentence reason.

Outputs: reports/granite_instruct_lift.csv, reports/granite_instruct_results.csv,
         derived/scored_2026_v2.csv, submissions/top500_v2.csv, reports/top500_v2_explained.csv

Usage:  python -E -P analysis/granite_instruct.py [--test]
"""
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")   # idle LightGBM threads otherwise spin and starve GPU kernel launches

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

T0 = time.time()
TEST = "--test" in sys.argv
LLM = "ibm-granite/granite-3.3-2b-instruct"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "IBM-20261107T174321Z-1-001", "IBM", "data", "data")
DER, GR, REP, SUB = (os.path.join(ROOT, p) for p in ("derived", "derived/granite", "reports", "submissions"))
CAND_FRAC = 0.05
QUESTIONS = {
    "q_internet": "Is the vulnerable product usually exposed to the internet, such as a VPN, firewall, router, "
                  "gateway, email or web server, file transfer appliance, or remote access tool?",
    "q_preauth": "Can an attacker exploit this without logging in or having an account?",
    "q_takeover": "Could exploiting this give an attacker code execution or full control of the system?",
    "q_enterprise": "Is this product widely used by large companies or governments?",
    "q_niche": "Is this a small plugin, hobby project, or niche product with few users?",
}
# q_preauth and q_takeover overlap with the regex text flags stage 1 already has (txt_unauth, txt_rce),
# so the full run asks only the three questions about exposure and popularity (~2.5x faster).
Q = ["q_internet", "q_enterprise", "q_niche"]


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- stage 1 (same as the v1 final model)
fd = pd.read_csv(os.path.join(DER, "feature_dictionary.csv"))
FEATS = [f for f in fd[fd.risk == "safe"].feature] + ["days_to_cutoff"]
LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25, verbose=-1)


def load(year, cutoff):
    df = pd.read_csv(os.path.join(DER, f"features_{year}.csv"), parse_dates=["snapshot_end", "published"])
    df["days_to_cutoff"] = (pd.Timestamp(cutoff) - df.snapshot_end).dt.days
    return df


def stage1(train, test):
    ms = [lgb.LGBMClassifier(random_state=s, **LGB_PARAMS).fit(train[FEATS], train.y) for s in range(5)]
    return np.mean([m.predict_proba(test[FEATS])[:, 1] for m in ms], axis=0)


d24_tr, d25_tr = load(2024, "2024-12-31"), load(2025, "2025-12-31")
Y = {2024: load(2024, "2024-11-04"), 2025: load(2025, "2025-11-04"), 2026: load(2026, "2026-11-04")}
Y[2024]["s1"] = stage1(d25_tr, Y[2024])
Y[2025]["s1"] = stage1(d24_tr, Y[2025])
Y[2026]["s1"] = stage1(pd.concat([d24_tr, d25_tr], ignore_index=True), Y[2026])
for y, df in Y.items():
    df["s1_pct"] = df.s1.rank(pct=True)
    df["cand"] = df.s1_pct >= 1 - CAND_FRAC
    msg = f"{y}: {df.cand.sum():,} candidates"
    if "y_nov4" in df:
        js = df.snapshot_month <= f"{y}-09"
        msg += f"; Jan-Sep positives by Nov 4 = {df[js].y_nov4.sum()}, inside the candidate pool = {df[js & df.cand].y_nov4.sum()}"
    log(msg)

# ----------------------------------------------------------------------------- Granite
SYS = "You are a vulnerability triage analyst. Answer the question about the vulnerability."
tok = model = YES = NO = None


def load_llm():
    global tok, model, YES, NO
    if model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log(f"loading {LLM}")
        tok = AutoTokenizer.from_pretrained(LLM)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.bfloat16).to("cuda").eval()
        YES = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes", " yes")})
        NO = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No", " no")})


def chat(user):
    return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def yes_logodds(users, batch=48):
    """log P(yes) - log P(no) for the first answer token, one forward pass per prompt."""
    import torch
    load_llm()
    t0 = time.time()
    chats = [chat(u) for u in users]
    order = np.argsort([len(c) for c in chats])
    out = np.zeros(len(chats))
    for b in range(0, len(chats), batch):
        idx = order[b:b + batch]
        # pad to a multiple of 32 so batch shapes repeat and the GPU reuses its kernels
        enc = tok([chats[i] for i in idx], return_tensors="pt", padding=True, pad_to_multiple_of=32).to("cuda")
        with torch.no_grad():   # logits_to_keep=1: only the last position goes through the vocab head
            lp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[:, -1, :].float(), dim=-1)
        out[idx] = (torch.logsumexp(lp[:, YES], dim=1) - torch.logsumexp(lp[:, NO], dim=1)).cpu().numpy()
        if (b // batch) % 100 == 0:
            log(f"  scored {min(b + batch, len(chats)):,}/{len(chats):,}")
    log(f"  {len(chats):,} prompts in {time.time() - t0:.0f}s")
    return out


def generate(users, max_new=90, batch=16):
    import torch
    load_llm()
    chats = [chat(u) for u in users]
    order = np.argsort([len(c) for c in chats])
    out = [""] * len(chats)
    for b in range(0, len(chats), batch):
        idx = order[b:b + batch]
        enc = tok([chats[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, t in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[i] = t.strip()
        if (b // batch) % 10 == 0:
            log(f"  wrote {min(b + batch, len(chats)):,}/{len(chats):,}")
    return out


desc = pd.read_csv(f"{DATA}/vulnerabilities.csv", dtype=str, usecols=["cve_id", "description"]).set_index("cve_id").description.fillna("")


def ask(cves, q):
    return [f"Vulnerability description:\n{desc[c][:600]}\n\nQuestion: {QUESTIONS[q]}\nAnswer with yes or no." for c in cves]


if TEST:
    sample = Y[2025][Y[2025].cand].sample(96, random_state=1).cve_id.tolist()
    load_llm()
    lens = [len(tok(chat(u))["input_ids"]) for u in ask(sample, "q_internet")]
    log(f"prompt tokens: median {int(np.median(lens))}, max {max(lens)}, attention: {model.config._attn_implementation}")
    t = time.time()
    sc = {q: yes_logodds(ask(sample, q)) for q in QUESTIONS}
    log(f"TEST: {len(sample)} CVEs x {len(Q)} questions in {time.time() - t:.1f}s")
    for i, c in enumerate(sample[:8]):
        print("----", c, "|", desc[c][:150].replace("\n", " "))
        print("   ", "  ".join(f"{q}={sc[q][i]:+.1f}" for q in QUESTIONS))
    sys.exit(0)


def extract(year, cves):
    path = os.path.join(GR, f"instruct_{year}.csv")
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["cve_id"] + Q)
    todo = [c for c in cves if c not in set(done.cve_id)]
    if todo:
        log(f"{year}: Granite answers {len(Q)} questions for {len(todo):,} CVEs")
        new = pd.DataFrame({"cve_id": todo, **{q: yes_logodds(ask(todo, q)) for q in Q}})
        done = pd.concat([done, new], ignore_index=True)
        done.to_csv(path, index=False)
    return done


for y, df in Y.items():
    Y[y] = df.merge(extract(y, df[df.cand].cve_id.tolist()), on="cve_id", how="left")

# ----------------------------------------------------------------------------- what Granite's answers say (lift)
pool = pd.concat([Y[2024][Y[2024].cand], Y[2025][Y[2025].cand]], ignore_index=True)
base = pool.y.mean()
lift = []
for q in Q:
    hi = pool[q] > 0
    lift.append({"question": q, "text": QUESTIONS[q], "granite_says_yes": int(hi.sum()), "kev_if_yes": int(pool[hi].y.sum()),
                 "kev_rate_if_yes": round(pool[hi].y.mean(), 4), "kev_rate_if_no": round(pool[~hi].y.mean(), 4),
                 "lift_yes_vs_no": round(pool[hi].y.mean() / max(pool[~hi].y.mean(), 1e-9), 2)})
lift = pd.DataFrame(lift)
lift.to_csv(os.path.join(REP, "granite_instruct_lift.csv"), index=False)
log(f"candidate pool 2024+2025: {len(pool):,} rows, KEV rate {base:.3%}")
print(lift.drop(columns="text").to_string(index=False))


# ----------------------------------------------------------------------------- re-rank and evaluate (A/B)
def hits(df, score, k=500):
    m = (df.snapshot_month <= f"{df.snapshot_end.dt.year.iloc[0]}-09").to_numpy()
    return int(df.y_nov4.to_numpy()[m][np.argsort(-score[m], kind="stable")[:k]].sum())


def stage2(train, test, cols, mode):
    tr = train[train.cand]
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))
    lr.fit(tr[cols], tr.y)
    c = test.cand.to_numpy()
    p_c = lr.predict_proba(test.loc[c, cols])[:, 1]
    if mode == "rank-avg":
        p_c = pd.Series(p_c).rank(pct=True).to_numpy() + test.loc[c, "s1_pct"].rank(pct=True).to_numpy()
    p = test.s1_pct.to_numpy().copy()
    p[c] = 10 + p_c                      # candidates stay above everyone else, re-ordered by stage 2
    return p, lr


VARIANTS = {"stage 2 logistic [s1 + Granite]": (["s1_pct"] + Q, "logistic"),
            "stage 2 rank-avg [s1, Granite]": (Q, "rank-avg"),
            "stage 2 logistic [s1 only] (control)": (["s1_pct"], "logistic")}
CAP = 5   # at most 5 CVEs per assigner per snapshot month (sibling_boost.py sweep: caps 3-8 all help, 4-5 best, A and B)


def capped(df, score, cap=CAP):
    """Push everything past the first `cap` CVEs of each (assigner, snapshot month) below the rest."""
    s = pd.Series(np.asarray(score, dtype=float), index=df.index)
    n = df.assign(_s=s).sort_values("_s", ascending=False, kind="stable").groupby(["assigner", "snapshot_month"]).cumcount()
    out = s.copy()
    out[n[n >= cap].index] -= 100
    return out.to_numpy()


def scores_for(option, train, test):
    base = option.replace(f" + cap{CAP}", "")
    if base == "stage 1 (v1 LightGBM)":
        s, lr = test.s1.to_numpy(), None
    else:
        cols, mode = VARIANTS[base]
        s, lr = stage2(train, test, cols, mode)
    return (capped(test, s) if option.endswith(f"+ cap{CAP}") else s), lr


OPTIONS = ["stage 1 (v1 LightGBM)"] + list(VARIANTS)
OPTIONS += [o + f" + cap{CAP}" for o in OPTIONS]
res = []
for name, tr_y, te_y in [("A 2024->2025", 2024, 2025), ("B 2025->2024", 2025, 2024)]:
    te = Y[te_y]
    r = {"split": name}
    for o in OPTIONS:
        r[o] = hits(te, scores_for(o, Y[tr_y], te)[0])
    res.append(r)
res = pd.DataFrame(res)
res.loc[len(res)] = ["total"] + res.iloc[:, 1:].sum().tolist()
res.to_csv(os.path.join(REP, "granite_instruct_results.csv"), index=False)
print(res.set_index("split").T.to_string())

# ----------------------------------------------------------------------------- 2026 final ranking
# pick by total A+B, then by A (the realistic past->future test), then prefer the Granite variant on ties
best = max([o for o in OPTIONS if "control" not in o],
           key=lambda c: (res[c].iloc[-1], res[c].iloc[0], "stage 2" in c, "cap" in c))
with open(os.path.join(REP, "final_ranking_choice.txt"), "w") as fh:
    fh.write(best)
both = pd.concat([Y[2024], Y[2025]], ignore_index=True)
s26 = Y[2026]
s26["score"], lr = scores_for(best, both, s26)
if lr is not None:
    cols = VARIANTS[best.replace(f" + cap{CAP}", "")][0]
    pd.Series(lr[-1].coef_[0], index=cols).sort_values().to_csv(os.path.join(REP, "granite_instruct_coefficients.csv"), header=["coef"])
s26 = s26.sort_values("score", ascending=False).reset_index(drop=True)
s26["rank"] = np.arange(1, len(s26) + 1)
log(f"2026 ranking uses: {best}")

# ----------------------------------------------------------------------------- Granite explains the top 500
top = s26.head(500).copy()
flags = lambda r: ", ".join(f for f in ("exploitdb", "metasploit", "nuclei", "poc_github") if getattr(r, f)) or "none"  # noqa: E731
prob = lambda x: f"{100 / (1 + np.exp(-x)):.0f}%"  # noqa: E731
EXPLAIN = """Vulnerability {cve}: {desc}

Signals when it was published: EPSS {epss:.3f}; public exploit code: {flags}; reads like past exploited CVEs: {txt:.0%}.
Analyst read: internet-facing {inet}, widely used by enterprises {ent}.

Reply in exactly three lines:
product: <vendor and product, at most 6 words>
category: <one of: edge or network device, enterprise server software, operating system, browser, mobile, desktop app, web app or plugin, library or developer tool, IoT or consumer device, industrial or medical, other>
reason: <one sentence, at most 30 words, telling a security engineer why to patch this first>"""
cache = os.path.join(GR, "reasons_2026.csv")
old = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame(columns=["cve_id", "raw"])
need = top[~top.cve_id.isin(old.cve_id)]
if len(need):
    log(f"Granite writes product, category and reason for {len(need)} CVEs")
    raw = generate([EXPLAIN.format(cve=r.cve_id, desc=desc[r.cve_id][:900], epss=r.epss, flags=flags(r), txt=r.text_kev_lr,
                                   inet=prob(r.q_internet), ent=prob(r.q_enterprise))
                    for r in need.itertuples()])
    old = pd.concat([old, pd.DataFrame({"cve_id": need.cve_id.values, "raw": raw})], ignore_index=True)
    old.to_csv(cache, index=False)


def field(raw, key):
    for line in str(raw).splitlines():
        if line.lower().startswith(key + ":"):
            return line.split(":", 1)[1].strip()
    return ""


old["product"] = old.raw.map(lambda r: field(r, "product"))
old["category"] = old.raw.map(lambda r: field(r, "category"))
old["reason"] = old.raw.map(lambda r: field(r, "reason"))
top = top.merge(old[["cve_id", "product", "category", "reason"]], on="cve_id", how="left")
for q in Q:
    top[q.replace("q_", "p_")] = (1 / (1 + np.exp(-top[q]))).round(3)
cols = ["rank", "cve_id", "snapshot_month", "assigner", "product", "category"] + [q.replace("q_", "p_") for q in Q] + \
       ["epss", "exploitdb", "metasploit", "nuclei", "poc_github", "text_kev_lr", "reason"]
top[cols].to_csv(os.path.join(REP, "top500_v2_explained.csv"), index=False)
s26[["rank", "cve_id", "score", "s1", "snapshot_month", "assigner"] + Q].to_csv(os.path.join(DER, "scored_2026_v2.csv"), index=False)
s26.head(500)[["cve_id"]].to_csv(os.path.join(SUB, "top500_v2.csv"), index=False)
v1 = set(pd.read_csv(os.path.join(SUB, "top500_v1.csv")).cve_id)
log(f"top500_v2 written; overlap with v1 = {len(v1 & set(top.cve_id))}")
print(top.category.str.lower().value_counts().head(12).to_string())
