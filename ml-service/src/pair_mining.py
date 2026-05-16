"""mine (anchor, positive, hard_negative) triplets from training set
for contrastive fine-tuning. only uses train split. cached to disk."""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

ML_DIR = Path(__file__).resolve().parent.parent
CACHE = ML_DIR / "data" / "cached"

META = CACHE / "metadata_clean.csv"
SPLITS = CACHE / "splits.csv"
IMG_EMB = CACHE / "clip_image_embeddings.npy"
IMG_IDS = CACHE / "clip_image_ids.npy"

OUT = CACHE / "train_triplets.csv"

SEED = 42
NUM_TRIPLETS = 200_000       # ~5-6 triplets per train anchor
HARD_NEG_CANDIDATES = 50     # top-k similar items considered for hard neg pool
NUM_HARDS_PER_ANCHOR = 1     # one hard negative per triplet


def main():
    rng = np.random.RandomState(SEED)

    meta = pd.read_csv(META)
    splits = pd.read_csv(SPLITS)
    meta = meta.merge(splits, on="id")
    img_emb = np.load(IMG_EMB)
    img_ids = np.load(IMG_IDS)
    id_to_idx = {int(i): k for k, i in enumerate(img_ids)}

    train = meta[meta["split"] == "train"].copy()
    train = train.dropna(subset=["articleType", "baseColour", "gender"]).reset_index(drop=True)
    print(f"train products (after nan drop): {len(train)}")

    # group_key = positive class label for contrastive learning
    train["group_key"] = (
        train["articleType"].astype(str) + "|" +
        train["baseColour"].astype(str) + "|" +
        train["gender"].astype(str)
    )

    group_to_ids = train.groupby("group_key")["id"].apply(list).to_dict()
    art_to_ids = train.groupby("articleType")["id"].apply(list).to_dict()

    # keep only anchors whose group has >=2 items (need at least one positive)
    train["group_size"] = train["group_key"].map(lambda k: len(group_to_ids[k]))
    eligible = train[train["group_size"] >= 2].reset_index(drop=True)
    print(f"eligible anchors (group_size >= 2): {len(eligible)}")

    train_ids_set = set(eligible["id"].astype(int).tolist())

    # precompute embeddings restricted to train set for hard-neg search
    train_idx_in_full = np.array([id_to_idx[int(i)] for i in eligible["id"]])
    train_emb = img_emb[train_idx_in_full]   # (N_train, 512)
    train_id_arr = eligible["id"].astype(int).to_numpy()
    train_art_arr = eligible["articleType"].astype(str).to_numpy()
    train_color_arr = eligible["baseColour"].astype(str).to_numpy()

    # map global id -> position in train_emb
    id_to_train_pos = {int(i): k for k, i in enumerate(train_id_arr)}

    triplets = []
    anchor_pool = eligible.sample(n=min(NUM_TRIPLETS, len(eligible) * 10), replace=True, random_state=SEED).reset_index(drop=True)

    for _, row in tqdm(anchor_pool.iterrows(), total=len(anchor_pool), desc="mining"):
        a_id = int(row["id"])
        a_art = row["articleType"]
        a_color = row["baseColour"]
        group = group_to_ids[row["group_key"]]

        # positive: random other product from same group
        pos_candidates = [pid for pid in group if pid != a_id]
        if not pos_candidates:
            continue
        p_id = int(rng.choice(pos_candidates))

        # hard negative: same articleType but DIFFERENT color (and not same group)
        a_pos = id_to_train_pos[a_id]
        a_vec = train_emb[a_pos]
        sims = train_emb @ a_vec
        # find top candidates by similarity, then filter
        top_idx = np.argpartition(-sims, HARD_NEG_CANDIDATES)[:HARD_NEG_CANDIDATES]
        candidates = []
        for ti in top_idx:
            nid = int(train_id_arr[ti])
            if nid == a_id or nid == p_id:
                continue
            # hard negative criterion: same article, different color (most confusable)
            if train_art_arr[ti] == a_art and train_color_arr[ti] != a_color:
                candidates.append(nid)
        if not candidates:
            # fallback: same article, anything not in positive group
            candidates = [
                pid for pid in art_to_ids.get(a_art, [])
                if pid != a_id and pid != p_id and pid not in group and pid in train_ids_set
            ]
            if not candidates:
                continue
            n_id = int(rng.choice(candidates[:50]))
        else:
            n_id = int(rng.choice(candidates))

        triplets.append((a_id, p_id, n_id))

        if len(triplets) >= NUM_TRIPLETS:
            break

    print(f"\nmined {len(triplets)} triplets")
    out_df = pd.DataFrame(triplets, columns=["anchor_id", "positive_id", "negative_id"])
    out_df.to_csv(OUT, index=False)
    print(f"saved -> {OUT}")

    # quick sanity stats
    print("\nsanity (first 5 triplets):")
    for _, r in out_df.head(5).iterrows():
        a = meta[meta['id'] == r['anchor_id']].iloc[0]
        p = meta[meta['id'] == r['positive_id']].iloc[0]
        n = meta[meta['id'] == r['negative_id']].iloc[0]
        print(f"  A: {a['articleType']:<15} {a['baseColour']:<12} {a['gender']:<6}")
        print(f"  P: {p['articleType']:<15} {p['baseColour']:<12} {p['gender']:<6}")
        print(f"  N: {n['articleType']:<15} {n['baseColour']:<12} {n['gender']:<6}")
        print()


if __name__ == "__main__":
    main()