"""build training data for the reranker.
for each query (train+val product), retrieve top-K candidates using trained heads,
extract features, and label each candidate as relevant (1) or not (0).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from train import ProjectionHead, EMB_DIM, HIDDEN_DIM, PROJ_DIM

ML_DIR = Path(__file__).resolve().parent.parent
CACHE = ML_DIR / "data" / "cached"
MODELS = ML_DIR / "models"

OUT = CACHE / "rerank_data.parquet"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOP_K_CANDIDATES = 100
NUM_QUERIES = 20_000   # subsample to keep this tractable on macbook


def overlap(a: str, b: str) -> float:
    """simple jaccard on tokenized lowercase strings."""
    sa = set(str(a).lower().split())
    sb = set(str(b).lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    print("loading...")
    meta = pd.read_csv(CACHE / "metadata_clean.csv")
    splits = pd.read_csv(CACHE / "splits.csv")
    captions = pd.read_csv(CACHE / "captions.csv")
    meta = meta.merge(splits, on="id").merge(captions, on="id")

    img_emb = np.load(CACHE / "clip_image_embeddings.npy")
    img_ids = np.load(CACHE / "clip_image_ids.npy")
    txt_emb = np.load(CACHE / "clip_text_embeddings.npy")
    id_to_idx = {int(v): k for k, v in enumerate(img_ids)}

    # load trained heads
    ckpt = torch.load(MODELS / "best_heads.pt", map_location=DEVICE, weights_only=True)
    img_head = ProjectionHead(EMB_DIM, HIDDEN_DIM, PROJ_DIM).to(DEVICE)
    txt_head = ProjectionHead(EMB_DIM, HIDDEN_DIM, PROJ_DIM).to(DEVICE)
    img_head.load_state_dict(ckpt["img_head"])
    txt_head.load_state_dict(ckpt["txt_head"])
    img_head.eval(); txt_head.eval()

    # project everything once
    with torch.no_grad():
        img_t = torch.from_numpy(img_emb).float().to(DEVICE)
        txt_t = torch.from_numpy(txt_emb).float().to(DEVICE)
        img_p = []
        txt_p = []
        for i in range(0, len(img_t), 4096):
            img_p.append(img_head(img_t[i:i+4096]).cpu().numpy())
            txt_p.append(txt_head(txt_t[i:i+4096]).cpu().numpy())
        img_p = np.concatenate(img_p, axis=0)
        txt_p = np.concatenate(txt_p, axis=0)

    # only use train+val products as queries (so reranker doesn't see test info)
    eligible = meta[meta["split"].isin(["train", "val"])].dropna(
        subset=["articleType", "baseColour", "gender"]
    ).reset_index(drop=True)
    eligible["group_key"] = (
        eligible["articleType"].astype(str) + "|" +
        eligible["baseColour"].astype(str) + "|" +
        eligible["gender"].astype(str)
    )
    print(f"eligible queries: {len(eligible)}")

    # subsample
    rng = np.random.RandomState(42)
    sampled = eligible.sample(n=min(NUM_QUERIES, len(eligible)), random_state=42).reset_index(drop=True)
    print(f"sampled: {len(sampled)}")

    # build a meta lookup for fast feature access
    meta_idx = meta.set_index("id")

    rows = []
    for _, q in tqdm(sampled.iterrows(), total=len(sampled), desc="retrieving"):
        qid = int(q["id"])
        qi = id_to_idx[qid]
        q_group = q["group_key"]
        q_caption = q["caption"]
        q_art = q["articleType"]
        q_color = q["baseColour"]
        q_gender = q["gender"]
        q_season = q["season"] if pd.notna(q["season"]) else ""
        q_usage = q["usage"] if pd.notna(q["usage"]) else ""

        # retrieve top-K using projected text vec
        sims_proj = img_p @ txt_p[qi]
        sims_proj[qi] = -np.inf
        top_idx = np.argpartition(-sims_proj, TOP_K_CANDIDATES)[:TOP_K_CANDIDATES]
        top_idx = top_idx[np.argsort(-sims_proj[top_idx])]

        # also compute base clip sims for the feature
        sims_base = img_emb @ txt_emb[qi]

        for rank, ci in enumerate(top_idx):
            cid = int(img_ids[ci])
            if cid not in meta_idx.index:
                continue
            c = meta_idx.loc[cid]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[0]
            c_art = c["articleType"]
            c_color = c["baseColour"]
            c_gender = c["gender"]
            c_season = c["season"] if pd.notna(c["season"]) else ""
            c_usage = c["usage"] if pd.notna(c["usage"]) else ""
            c_caption = c["caption"] if "caption" in c else ""

            # label: relevant if same (articleType, baseColour, gender)
            c_group = f"{c_art}|{c_color}|{c_gender}"
            label = int(c_group == q_group)

            rows.append({
                "qid": qid,
                "cid": cid,
                "label": label,
                "rank": rank,
                "proj_sim": float(sims_proj[ci]),
                "base_sim": float(sims_base[ci]),
                "article_match": int(c_art == q_art),
                "color_match": int(c_color == q_color),
                "gender_match": int(c_gender == q_gender),
                "season_match": int(c_season == q_season) if q_season and c_season else 0,
                "usage_match": int(c_usage == q_usage) if q_usage and c_usage else 0,
                "caption_overlap": overlap(q_caption, c_caption),
            })

    df = pd.DataFrame(rows)
    print(f"\nrows: {len(df)}")
    print(f"positive rate: {df['label'].mean():.4f}")
    print(f"avg positives per query: {df.groupby('qid')['label'].sum().mean():.2f}")

    df.to_parquet(OUT, index=False)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()