"""evaluate the trained reranker on test set. NO torch imports.
loads pre-projected embeddings from project_embeddings.py output."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

ML_DIR = Path(__file__).resolve().parent.parent
CACHE = ML_DIR / "data" / "cached"
MODELS = ML_DIR / "models"

TOP_K_CANDIDATES = 100


def overlap(a, b):
    sa = set(str(a).lower().split())
    sb = set(str(b).lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    print("loading...")
    meta = pd.read_csv(CACHE / "metadata_clean.csv")
    captions = pd.read_csv(CACHE / "captions.csv")
    meta = meta.merge(captions, on="id")

    img_emb = np.load(CACHE / "clip_image_embeddings.npy")
    img_ids = np.load(CACHE / "clip_image_ids.npy")
    txt_emb = np.load(CACHE / "clip_text_embeddings.npy")
    img_p = np.load(CACHE / "img_projected.npy")
    txt_p = np.load(CACHE / "txt_projected.npy")
    id_to_idx = {int(v): k for k, v in enumerate(img_ids)}

    eval_set = pd.read_csv(CACHE / "eval_set.csv")
    eval_set["relevant_ids"] = eval_set["relevant_ids_str"].apply(
        lambda s: set(int(x) for x in s.split("|"))
    )

    booster = lgb.Booster(model_file=str(MODELS / "reranker.txt"))
    print("loaded reranker.txt")

    meta_idx = meta.set_index("id")
    ks = (1, 5, 10, 20, 50)
    max_k = max(ks)

    recalls = {k: [] for k in ks}
    maps = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}

    def _metrics(retrieved, relevant, top_k):
        top = retrieved[:top_k]
        hits = len(set(top) & relevant)
        recall = hits / len(relevant) if relevant else 0.0
        score, h = 0.0, 0
        for i_r, item in enumerate(top):
            if item in relevant:
                h += 1
                score += h / (i_r + 1)
        m = score / min(len(relevant), top_k) if relevant else 0.0
        dcg = sum(1.0 / np.log2(i_r + 2) for i_r, item in enumerate(top) if item in relevant)
        ideal_hits = min(len(relevant), top_k)
        idcg = sum(1.0 / np.log2(i_r + 2) for i_r in range(ideal_hits))
        n = dcg / idcg if idcg > 0 else 0.0
        return recall, m, n

    for _, row in tqdm(eval_set.iterrows(), total=len(eval_set), desc="reranking"):
        qid = int(row["id"])
        relevant = row["relevant_ids"] - {qid}
        if not relevant:
            continue
        q = meta_idx.loc[qid]
        if isinstance(q, pd.DataFrame):
            q = q.iloc[0]
        qi = id_to_idx[qid]

        sims_proj = img_p @ txt_p[qi]
        sims_proj[qi] = -np.inf
        top_idx = np.argpartition(-sims_proj, TOP_K_CANDIDATES)[:TOP_K_CANDIDATES]
        top_idx = top_idx[np.argsort(-sims_proj[top_idx])]
        sims_base = img_emb @ txt_emb[qi]

        q_art = q["articleType"]
        q_color = q["baseColour"]
        q_gender = q["gender"]
        q_season = q["season"] if pd.notna(q["season"]) else ""
        q_usage = q["usage"] if pd.notna(q["usage"]) else ""
        q_caption = q["caption"]

        feats = []
        cand_ids = []
        for ci in top_idx:
            cid = int(img_ids[ci])
            if cid not in meta_idx.index:
                continue
            c = meta_idx.loc[cid]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[0]
            c_season = c["season"] if pd.notna(c["season"]) else ""
            c_usage = c["usage"] if pd.notna(c["usage"]) else ""
            feats.append([
                float(sims_proj[ci]),
                float(sims_base[ci]),
                int(c["articleType"] == q_art),
                int(c["baseColour"] == q_color),
                int(c["gender"] == q_gender),
                int(c_season == q_season) if q_season and c_season else 0,
                int(c_usage == q_usage) if q_usage and c_usage else 0,
                overlap(q_caption, c["caption"]),
            ])
            cand_ids.append(cid)

        scores = booster.predict(np.array(feats, dtype=np.float32))
        order = np.argsort(-scores)
        retrieved = [cand_ids[i] for i in order]

        for k in ks:
            r, m, n = _metrics(retrieved, relevant, k)
            recalls[k].append(r); maps[k].append(m); ndcgs[k].append(n)

    summary = {}
    for k in ks:
        summary[f"recall@{k}"] = float(np.mean(recalls[k]))
        summary[f"map@{k}"] = float(np.mean(maps[k]))
        summary[f"ndcg@{k}"] = float(np.mean(ndcgs[k]))

    baseline = json.loads((CACHE / "baseline_metrics.json").read_text())
    trained = json.loads((CACHE / "trained_metrics.json").read_text())

    print("\ntest set comparison: baseline vs trained heads vs trained + reranked")
    print("-" * 90)
    print(f"{'metric':<14} {'baseline':>12} {'trained':>12} {'+rerank':>12} {'rerank vs baseline':>22}")
    print("-" * 90)
    for k in ks:
        for mname in (f"recall@{k}", f"map@{k}", f"ndcg@{k}"):
            b, t, r = baseline[mname], trained[mname], summary[mname]
            lift = (r / b - 1) * 100 if b > 0 else float("inf")
            print(f"{mname:<14} {b:>12.4f} {t:>12.4f} {r:>12.4f} {lift:>+21.1f}%")
        print()

    (CACHE / "reranked_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"saved -> {CACHE / 'reranked_metrics.json'}")


if __name__ == "__main__":
    main()