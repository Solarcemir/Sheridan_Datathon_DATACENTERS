"""Embed every CVE description with IBM granite-embedding-english-r2 (CLS pooling, L2-normalised).

Writes processed/emb_<name>.npy (float16, rows aligned with processed/emb_<name>_ids.parquet)
for the historical 2024-2025 view and the 2026 scoreboard view.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed"
MODEL = "ibm-granite/granite-embedding-english-r2"
MAX_LEN = 512   # descriptions are short; >99% fit
BATCH = 256


@torch.no_grad()
def embed(texts: list[str], tok, model) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])  # length-sorted batches = less padding
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float16)
    for i in range(0, len(texts), BATCH):
        idx = order[i:i + BATCH]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to("cuda")
        cls = model(**enc).last_hidden_state[:, 0]
        out[idx] = torch.nn.functional.normalize(cls.float(), dim=1).cpu().numpy()
    return out


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    for name, file in [("historical", "historical_2024_2025.parquet"),
                       ("scoreboard", "scoreboard_2026_view.parquet")]:
        df = pd.read_parquet(PROC / file, columns=["cve_id", "description"])
        texts = [" ".join(str(d).split()) for d in df.description]
        e = embed(texts, tok, model)
        np.save(PROC / f"emb_{name}.npy", e)
        df[["cve_id"]].to_parquet(PROC / f"emb_{name}_ids.parquet", index=False)
        print(f"{name}: {e.shape}", flush=True)


if __name__ == "__main__":
    main()
