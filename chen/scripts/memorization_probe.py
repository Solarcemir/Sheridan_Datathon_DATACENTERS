"""Does Granite know CVEs by ID? Ask it to describe a CVE from the ID alone.

Scoring: embed the answer (granite-embedding-english-r2) and rank the CVE's true description
among 999 random descriptions from the same year (+ itself). "Identified" = true description
ranks first. Chance = 0.1%.

Samples: all 2024-25 KEV positives, 300 random negatives per year, 500 random 2026 scoreboard CVEs.

  CUDA_VISIBLE_DEVICES=1 python scripts/memorization_probe.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed"
LLM = "ibm-granite/granite-4.2-30b"
EMB = "ibm-granite/granite-embedding-english-r2"
N_DISTRACT = 999

QUESTION = ("What is {cve}? In one sentence, name the affected vendor and product and the type "
            "of vulnerability. If you do not know this specific CVE, answer exactly: unknown.")


def load_emb(name: str) -> tuple[pd.Series, np.ndarray]:
    ids = pd.read_parquet(PROC / f"emb_{name}_ids.parquet").cve_id
    return pd.Series(np.arange(len(ids)), index=ids), np.load(PROC / f"emb_{name}.npy").astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(0)
    hist = pd.read_parquet(PROC / "historical_2024_2025.parquet", columns=["cve_id", "published", "label"])
    sb = pd.read_parquet(PROC / "scoreboard_2026_view.parquet", columns=["cve_id", "published"])
    hist["year"] = hist.published.dt.year.astype(str)
    sb["year"] = "2026"

    parts = []
    for y in ("2024", "2025"):
        h = hist[hist.year == y]
        parts.append(h[h.label == 1].assign(set=f"{y} KEV"))
        parts.append(h[h.label == 0].sample(300, random_state=0).assign(set=f"{y} not KEV"))
    parts.append(sb.sample(500, random_state=0).assign(set="2026 scoreboard"))
    probe = pd.concat(parts, ignore_index=True)

    from vllm import LLM as VLLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(LLM)
    texts = [tok.apply_chat_template([{"role": "user", "content": QUESTION.format(cve=c)}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for c in probe.cve_id]
    llm = VLLM(model=LLM, max_model_len=2048, gpu_memory_utilization=0.92, seed=0)
    outs = llm.generate(texts, SamplingParams(max_tokens=80, temperature=0.0))
    probe["answer"] = [o.outputs[0].text.strip() for o in outs]
    del llm
    torch.cuda.empty_cache()

    etok = AutoTokenizer.from_pretrained(EMB)
    emodel = AutoModel.from_pretrained(EMB, dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        enc = etok(probe.answer.tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
        ans = torch.nn.functional.normalize(emodel(**enc).last_hidden_state[:, 0].float(), dim=1).cpu().numpy()

    row_h, emb_h = load_emb("historical")
    row_s, emb_s = load_emb("scoreboard")
    pools = {"2024": hist[hist.year == "2024"].cve_id.to_numpy(),
             "2025": hist[hist.year == "2025"].cve_id.to_numpy(),
             "2026": sb.cve_id.to_numpy()}

    def vec(c, y):
        return emb_s[row_s[c]] if y == "2026" else emb_h[row_h[c]]

    ranks = []
    for a, c, y in zip(ans, probe.cve_id, probe.year):
        cand = rng.choice(pools[y][pools[y] != c], N_DISTRACT, replace=False)
        sims = np.array([vec(x, y) for x in cand]) @ a
        ranks.append(int((sims > vec(c, y) @ a).sum()) + 1)
    probe["rank"] = ranks
    probe["unknown"] = probe.answer.str.lower().str.startswith("unknown")

    out = probe.groupby("set", sort=False).agg(
        n=("cve_id", "size"),
        identified_top1_pct=("rank", lambda r: round(100 * (r == 1).mean(), 1)),
        top10_pct=("rank", lambda r: round(100 * (r <= 10).mean(), 1)),
        said_unknown_pct=("unknown", lambda u: round(100 * u.mean(), 1)),
    )
    print(out.to_string())
    probe.to_parquet(ROOT / "results" / "memorization_probe.parquet", index=False)

    print("\nsample answers (identified KEV CVEs):")
    for r in probe[(probe.rank == 1) & probe.set.str.contains("KEV") & ~probe.set.str.contains("not")].head(6).itertuples():
        print(f"  {r.cve_id}: {r.answer[:140]}")
    print("\nsample answers (not identified):")
    for r in probe[probe["rank"] > 10].sample(min(5, (probe["rank"] > 10).sum()), random_state=0).itertuples():
        print(f"  {r.cve_id} [{r.set}]: {r.answer[:140]}")
    (ROOT / "results" / "memorization_probe.json").write_text(json.dumps(out.reset_index().to_dict("records"), indent=2))


if __name__ == "__main__":
    main()
