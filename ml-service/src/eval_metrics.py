"""retrieval evaluation: recall@k, map@k, ndcg@k for text-to-image search.
uses the cached clip text + image embeddings as baseline.
the same function will later evaluate fine-tuned embeddings."""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

ML_DIR = Path(__file__).resolve().parent.parent
CACHE = ML_DIR / "data" / "cached"


def load_data():
    img_emb = np.load(CACHE / "clip_image_embeddings.npy")
    img_ids = np.load(CACHE / "clip_image_ids.npy")
    txt_emb = np.load(CACHE / "clip_text_embeddings.npy")
    txt_ids = np.load(CACHE / "clip_text_ids.npy")
    eval_set = pd.read_csv(CACHE / "eval_set.csv")

    assert np.array_equal(img_ids, txt_ids), "id mismatch"
    eval_set["relevant_ids"] = eval_set["relevant_ids_str"].apply(
        lambda s: set(int(x) for x in s.split("|"))
    )
    return img_emb, img_ids, txt_emb, eval_set


def recall_at_k(retrieved, relevant, k):
    """fraction of relevant items found in top-k retrieved."""
    top_k = retrieved[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant) if relevant else 0.0


def average_precision_at_k(retrieved, relevant, k):
    """ap@k: mean of precision values at ranks where relevant items are found."""
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def ndcg_at_k(retrieved, relevant, k):
    """binary-relevance ndcg@k."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    # ideal dcg: as many 1s as possible at the top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(img_emb, img_ids, txt_emb, eval_set, ks=(1, 5, 10, 20, 50)):
    """run full retrieval eval."""
    id_to_idx = {int(i): k for k, i in enumerate(img_ids)}

    results = {f"recall@{k}": [] for k in ks}
    results.update({f"map@{k}": [] for k in ks})
    results.update({f"ndcg@{k}": [] for k in ks})

    max_k = max(ks)

    for _, row in tqdm(eval_set.iterrows(), total=len(eval_set), desc="evaluating"):
        qid = int(row["id"])
        relevant = row["relevant_ids"] - {qid}  # exclude query itself
        if not relevant:
            continue

        q_idx = id_to_idx[qid]
        q_vec = txt_emb[q_idx]
        sims = img_emb @ q_vec
        sims[q_idx] = -np.inf  # never retrieve self
        top_idx = np.argpartition(-sims, max_k)[:max_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        retrieved = [int(img_ids[i]) for i in top_idx]

        for k in ks:
            results[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            results[f"map@{k}"].append(average_precision_at_k(retrieved, relevant, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))

    summary = {m: float(np.mean(v)) for m, v in results.items()}
    return summary


def main():
    print("loading...")
    img_emb, img_ids, txt_emb, eval_set = load_data()
    print(f"image emb: {img_emb.shape}")
    print(f"text emb:  {txt_emb.shape}")
    print(f"eval queries: {len(eval_set)}\n")

    summary = evaluate(img_emb, img_ids, txt_emb, eval_set)

    print("\nbaseline clip (frozen, no fine-tuning) text -> image retrieval:")
    print("-" * 55)
    for k in (1, 5, 10, 20, 50):
        print(f"  recall@{k:<3}  {summary[f'recall@{k}']:.4f}")
    print()
    for k in (1, 5, 10, 20, 50):
        print(f"  map@{k:<3}     {summary[f'map@{k}']:.4f}")
    print()
    for k in (1, 5, 10, 20, 50):
        print(f"  ndcg@{k:<3}    {summary[f'ndcg@{k}']:.4f}")

    out_path = CACHE / "baseline_metrics.json"
    pd.Series(summary).to_json(out_path, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()