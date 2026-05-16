"""extract clip text embeddings for all product captions and cache to disk."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

ML_DIR = Path(__file__).resolve().parent.parent
CACHE = ML_DIR / "data" / "cached"
META_CSV = CACHE / "metadata_clean.csv"
CACHE.mkdir(parents=True, exist_ok=True)

TEXT_EMB_PATH = CACHE / "clip_text_embeddings.npy"
TEXT_IDS_PATH = CACHE / "clip_text_ids.npy"
CAPTIONS_PATH = CACHE / "captions.csv"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"using device: {DEVICE}")

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 128


def build_caption(row):
    """build a richer caption than just productDisplayName.
    combines structured fields. clip text encoder handles this format well."""
    parts = [
        str(row.get("productDisplayName", "")),
        str(row.get("baseColour", "")),
        str(row.get("articleType", "")),
        str(row.get("gender", "")),
        str(row.get("usage", "")) if pd.notna(row.get("usage")) else "",
        str(row.get("season", "")) if pd.notna(row.get("season")) else "",
    ]
    parts = [p for p in parts if p and p != "nan"]
    return " ".join(parts).strip()


class CaptionDataset(Dataset):
    def __init__(self, captions, ids):
        self.captions = captions
        self.ids = ids

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx], int(self.ids[idx])


def collate(batch):
    captions, ids = zip(*batch)
    return list(captions), torch.tensor(ids)


def main():
    print("loading metadata...")
    df = pd.read_csv(META_CSV)
    df["caption"] = df.apply(build_caption, axis=1)
    print(f"{len(df)} rows. sample captions:")
    for c in df["caption"].head(5):
        print(" -", c)

    # save captions for later use
    df[["id", "caption"]].to_csv(CAPTIONS_PATH, index=False)
    print(f"saved captions -> {CAPTIONS_PATH}")

    print(f"loading model: {MODEL_NAME}")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    dataset = CaptionDataset(df["caption"].tolist(), df["id"].tolist())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=collate)

    all_embeddings = []
    all_ids = []

    with torch.no_grad():
        for captions, ids in tqdm(loader, desc="extracting text"):
            inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            text_outputs = model.text_model(**inputs)
            pooled = text_outputs.pooler_output
            features = model.text_projection(pooled)
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())
            all_ids.append(ids.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    ids = np.concatenate(all_ids, axis=0)
    print(f"embeddings shape: {embeddings.shape}")
    print(f"ids shape: {ids.shape}")

    np.save(TEXT_EMB_PATH, embeddings)
    np.save(TEXT_IDS_PATH, ids)
    print(f"saved text embeddings -> {TEXT_EMB_PATH}")
    print(f"saved text ids -> {TEXT_IDS_PATH}")


if __name__ == "__main__":
    main()