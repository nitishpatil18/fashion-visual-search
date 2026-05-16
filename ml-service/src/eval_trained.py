"""run the same retrieval eval as eval_metrics.py but using the trained projection heads.
produces apples-to-apples comparison vs baseline_metrics.json on the TEST set."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from train import ProjectionHead, EMB_DIM, HIDDEN_DIM, PROJ_DIM

ML_DIR = Path(__file__).resolve().parent.parent
CACHE = ML_DIR / "data" / "cached"
MODELS = ML_DIR / "models"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    print("loading...")
    img_emb = np.load(CACHE / "clip_image_embeddings.npy")
    img_ids = np.load(CACHE / "clip_image_ids.npy")
    txt_emb = np.load(CACHE / "clip_text_embeddings.npy")
    eval_set = pd.read_csv(CACHE / "eval_set.csv")
    eval_set["relevant_ids"] = eval_set["relevant_ids_str"].apply(
        lambda s: set(int(x) for x in s.split("|"))
    )

    ckpt = torch.load(MODELS / "best_heads.pt", map_location=DEVICE, weights_only=True)
    img_head = ProjectionHead(EMB_DIM, HIDDEN_DIM, PROJ_DIM).to(DEVICE)
    txt_head = ProjectionHead(EMB_DIM, HIDDEN_DIM, PROJ_DIM).to(DEVICE)
    img_head.load_state_dict(ckpt["img_head"])
    txt_head.load_state_dict(ckpt["txt_head"])
    img_head.eval(); txt_head.eval()
    print("loaded best_heads.pt")

    # project all embeddings
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
    print("projected: img", img_p.shape, "txt", txt_p.shape)

    id_to_idx = {int(v): k for k, v in enumerate(img_ids)}
    ks = (1, 5, 10, 20, 50)
    max_k = max(ks)

    results = {f"recall@{k}": [] for k in ks}
    results.update({f"map@{k}": [] for k in ks})
    results.update({f"ndcg@{k}": [] for k in ks})

    for _, row in tqdm(eval_set.iterrows(), total=len(eval_set), desc="evaluating"):
        qid = int(row["id"])
        relevant = row["relevant_ids"] - {qid}
        if not relevant:
            continue
        qi = id_to_idx[qid]
        sims = img_p @ txt_p[qi]
        sims[qi] = -np.inf
        top_idx = np.argpartition(-sims, max_k)[:max_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        retrieved = [int(img_ids[i]) for i in top_idx]

        for k in ks:
            top = retrieved[:k]
            hits = len(set(top) & relevant)
            results[f"recall@{k}"].append(hits / len(relevant))

            # map@k
            score = 0.0; h = 0
            for i_r, item in enumerate(top):
                if item in relevant:
                    h += 1
                    score += h / (i_r + 1)
            results[f"map@{k}"].append(score / min(len(relevant), k))

            # ndcg@k
            dcg = sum(1.0 / np.log2(i_r + 2) for i_r, item in enumerate(top) if item in relevant)
            ideal_hits = min(len(relevant), k)
            idcg = sum(1.0 / np.log2(i_r + 2) for i_r in range(ideal_hits))
            results[f"ndcg@{k}"].append(dcg / idcg if idcg > 0 else 0.0)

    summary = {m: float(np.mean(v)) for m, v in results.items()}

    # load baseline for comparison
    baseline = json.loads((CACHE / "baseline_metrics.json").read_text())

    print("\ntest set comparison (baseline clip vs trained projection heads):")
    print("-" * 65)
    print(f"{'metric':<14} {'baseline':>12} {'trained':>12} {'lift':>10}")
    print("-" * 65)
    for k in ks:
        for metric in (f"recall@{k}", f"map@{k}", f"ndcg@{k}"):
            b = baseline[metric]; t = summary[metric]
            lift = (t / b - 1) * 100 if b > 0 else float('inf')
            print(f"{metric:<14} {b:>12.4f} {t:>12.4f} {lift:>+9.1f}%")
        print()

    out = CACHE / "trained_metrics.json"
    pd.Series(summary).to_json(out, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()


# built a multi-modal fashion search engine over 44k products using clip vit-b/32 with custom projection heads trained via infonce contrastive 
# loss on 200k mined hard-negative triplets, lifting test recall@10 by 71%, ndcg@10 by 76%, and map@10 by 79% over frozen-clip baseline. m1 mps 
# training, faiss serving, evaluation with recall/map/ndcg @ k=1,5,10,20,50.