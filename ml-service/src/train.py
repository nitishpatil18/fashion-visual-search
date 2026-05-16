"""contrastive fine-tuning of small projection heads on top of frozen clip embeddings.
trains image-side and text-side heads jointly with infonce loss + mined hard negatives."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ML_DIR = Path(__file__).resolve().parent.parent
CACHE = ML_DIR / "data" / "cached"
MODELS = ML_DIR / "models"
MODELS.mkdir(parents=True, exist_ok=True)

IMG_EMB = CACHE / "clip_image_embeddings.npy"
IMG_IDS = CACHE / "clip_image_ids.npy"
TXT_EMB = CACHE / "clip_text_embeddings.npy"
TRIPLETS = CACHE / "train_triplets.csv"
SPLITS = CACHE / "splits.csv"
EVAL_SET = CACHE / "eval_set.csv"

# hyperparameters
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EMB_DIM = 512
PROJ_DIM = 512
HIDDEN_DIM = 1024
DROPOUT = 0.1

BATCH_SIZE = 512
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 15
TEMPERATURE = 0.07           # standard clip temperature
PATIENCE = 3
SEED = 42

print(f"device: {DEVICE}")
torch.manual_seed(SEED)
np.random.seed(SEED)


class ProjectionHead(nn.Module):
    """2-layer mlp with residual connection from clip's input embedding."""
    def __init__(self, in_dim=EMB_DIM, hidden=HIDDEN_DIM, out_dim=PROJ_DIM, dropout=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.ln(h + x)  # residual; x is already 512-dim
        return F.normalize(h, dim=-1)


class TripletDataset(Dataset):
    """yields (anchor_text_emb, positive_image_emb, negative_image_emb).
    anchor uses text embedding (queries are text in production).
    positive and negative use image embeddings (we retrieve images)."""
    def __init__(self, triplets_df, img_emb, txt_emb, id_to_idx):
        self.t = triplets_df.reset_index(drop=True)
        self.img = img_emb
        self.txt = txt_emb
        self.idx = id_to_idx

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        r = self.t.iloc[i]
        a = self.idx[int(r["anchor_id"])]
        p = self.idx[int(r["positive_id"])]
        n = self.idx[int(r["negative_id"])]
        return (
            torch.from_numpy(self.txt[a]).float(),
            torch.from_numpy(self.img[p]).float(),
            torch.from_numpy(self.img[n]).float(),
        )


def infonce_with_hard_negs(anchor, positive, hard_neg, temperature):
    """infonce loss using:
       - positives:    anchor vs positive (same index in batch)
       - in-batch neg: anchor vs other positives in batch
       - hard neg:     anchor vs explicit mined hard negative
    """
    # concat positives + hard_negs as candidates; positive is at index i, hard_neg at index i+B
    candidates = torch.cat([positive, hard_neg], dim=0)  # (2B, D)
    logits = anchor @ candidates.t() / temperature        # (B, 2B)
    targets = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, targets)


@torch.no_grad()
def eval_retrieval(img_head, txt_head, img_emb, img_ids, txt_emb, eval_set, device, ks=(1, 5, 10, 20)):
    img_head.eval(); txt_head.eval()
    img_t = torch.from_numpy(img_emb).float().to(device)
    txt_t = torch.from_numpy(txt_emb).float().to(device)
    img_proj = []
    txt_proj = []
    batch = 4096
    for i in range(0, len(img_t), batch):
        img_proj.append(img_head(img_t[i:i+batch]).cpu().numpy())
        txt_proj.append(txt_head(txt_t[i:i+batch]).cpu().numpy())
    img_proj = np.concatenate(img_proj, axis=0)
    txt_proj = np.concatenate(txt_proj, axis=0)

    id_to_idx = {int(v): k for k, v in enumerate(img_ids)}
    max_k = max(ks)
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    for _, row in eval_set.iterrows():
        qid = int(row["id"])
        relevant = row["_rel_set"] - {qid}
        if not relevant:
            continue
        qi = id_to_idx[qid]
        sims = img_proj @ txt_proj[qi]
        sims[qi] = -np.inf
        top_idx = np.argpartition(-sims, max_k)[:max_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        retrieved = [int(img_ids[i]) for i in top_idx]
        for k in ks:
            top = retrieved[:k]
            hits = len(set(top) & relevant)
            recalls[k].append(hits / len(relevant))
            dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(top) if item in relevant)
            ideal_hits = min(len(relevant), k)
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)

    return {f"recall@{k}": float(np.mean(recalls[k])) for k in ks}, \
           {f"ndcg@{k}":   float(np.mean(ndcgs[k]))   for k in ks}


def build_val_eval_set():
    """build a structured val eval set the same way eval_set was built for test."""
    meta = pd.read_csv(CACHE / "metadata_clean.csv")
    splits = pd.read_csv(SPLITS)
    df = meta.merge(splits, on="id")
    val = df[df["split"] == "val"].dropna(subset=["articleType", "baseColour", "gender"]).reset_index(drop=True)
    val["group_key"] = val["articleType"].astype(str) + "|" + val["baseColour"].astype(str) + "|" + val["gender"].astype(str)
    group_to_ids = val.groupby("group_key")["id"].apply(list).to_dict()
    val["relevant_ids"] = val["group_key"].map(group_to_ids)
    val["num_relevant"] = val["relevant_ids"].apply(len)
    val = val[val["num_relevant"] >= 2].reset_index(drop=True)
    val["_rel_set"] = val["relevant_ids"].apply(lambda lst: set(int(x) for x in lst))
    print(f"val eval queries: {len(val)}")
    return val[["id", "_rel_set"]]


def main():
    print("loading data...")
    img_emb = np.load(IMG_EMB)
    img_ids = np.load(IMG_IDS)
    txt_emb = np.load(TXT_EMB)
    triplets = pd.read_csv(TRIPLETS)
    id_to_idx = {int(v): k for k, v in enumerate(img_ids)}

    val_eval = build_val_eval_set()
    print(f"triplets: {len(triplets)}")

    train_ds = TripletDataset(triplets, img_emb, txt_emb, id_to_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    img_head = ProjectionHead().to(DEVICE)
    txt_head = ProjectionHead().to(DEVICE)
    params = list(img_head.parameters()) + list(txt_head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    # baseline val numbers (before any training)
    print("\nbaseline (frozen clip, identity heads) on val:")
    # identity init: heads currently random, so manually report base clip val numbers via a no-op forward
    # we evaluate the untrained heads anyway; it's a useful pre-train number
    rec0, ndcg0 = eval_retrieval(img_head, txt_head, img_emb, img_ids, txt_emb, val_eval, DEVICE)
    print({**rec0, **ndcg0})

    best_recall10 = -1
    best_epoch = -1
    patience_left = PATIENCE
    history = []

    for epoch in range(1, EPOCHS + 1):
        img_head.train(); txt_head.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}")
        for anchor_txt, pos_img, neg_img in pbar:
            anchor_txt = anchor_txt.to(DEVICE)
            pos_img = pos_img.to(DEVICE)
            neg_img = neg_img.to(DEVICE)

            a = txt_head(anchor_txt)
            p = img_head(pos_img)
            n = img_head(neg_img)
            loss = infonce_with_hard_negs(a, p, n, TEMPERATURE)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")

        rec, ndcg = eval_retrieval(img_head, txt_head, img_emb, img_ids, txt_emb, val_eval, DEVICE)
        print(f"epoch {epoch}: loss={np.mean(losses):.4f}  {rec}  {ndcg}")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **rec, **ndcg})

        r10 = rec["recall@10"]
        if r10 > best_recall10:
            best_recall10 = r10
            best_epoch = epoch
            patience_left = PATIENCE
            torch.save({"img_head": img_head.state_dict(),
                        "txt_head": txt_head.state_dict(),
                        "config": {"emb_dim": EMB_DIM, "hidden": HIDDEN_DIM, "out": PROJ_DIM}},
                       MODELS / "best_heads.pt")
            print(f"  ↑ new best recall@10 = {r10:.4f} (saved)")
        else:
            patience_left -= 1
            print(f"  no improvement. patience left: {patience_left}")
            if patience_left == 0:
                print("early stopping.")
                break

    print(f"\nbest epoch: {best_epoch}  best recall@10: {best_recall10:.4f}")
    with open(MODELS / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"history saved -> {MODELS / 'train_history.json'}")


if __name__ == "__main__":
    main()