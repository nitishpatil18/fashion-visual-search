"""extract clip image embeddings for all fashion images and cache to disk."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

# paths
ML_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ML_DIR / "data" / "raw" / "fashion-dataset"
IMG_DIR = DATA_DIR / "images"
META_CSV = ML_DIR / "data" / "cached" / "metadata_clean.csv"
OUT_DIR = ML_DIR / "data" / "cached"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = OUT_DIR / "clip_image_embeddings.npy"
IDS_PATH = OUT_DIR / "clip_image_ids.npy"

# device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"using device: {DEVICE}")

# config
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
NUM_WORKERS = 0


class FashionImageDataset(Dataset):
    def __init__(self, df, img_dir, processor):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['id']}.jpg"
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # fallback: return a black image so the batch doesn't crash
            image = Image.new("RGB", (224, 224))
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixel_values, int(row["id"])


def main():
    print("loading metadata...")
    df = pd.read_csv(META_CSV)
    print(f"{len(df)} rows")

    print(f"loading model: {MODEL_NAME}")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    dataset = FashionImageDataset(df, IMG_DIR, processor)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=False,
    )

    all_embeddings = []
    all_ids = []

    with torch.no_grad():
        for pixel_values, ids in tqdm(loader, desc="extracting"):
            pixel_values = pixel_values.to(DEVICE)
            # call vision_model directly, then apply visual_projection ourselves.
            # robust across transformers v4 and v5 (v5 changed get_image_features return type).
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            pooled = vision_outputs.pooler_output
            features = model.visual_projection(pooled)
            # l2 normalize so cosine similarity = dot product later
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())
            all_ids.append(ids.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    ids = np.concatenate(all_ids, axis=0)

    print(f"embeddings shape: {embeddings.shape}")
    print(f"ids shape: {ids.shape}")

    np.save(EMB_PATH, embeddings)
    np.save(IDS_PATH, ids)
    print(f"saved embeddings -> {EMB_PATH}")
    print(f"saved ids -> {IDS_PATH}")


if __name__ == "__main__":
    main()