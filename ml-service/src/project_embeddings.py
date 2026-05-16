"""project all clip embeddings through the trained heads. save as .npy.
NO lightgbm imports — must stay isolated."""

from pathlib import Path
import numpy as np
import torch
from train import ProjectionHead, EMB_DIM, HIDDEN_DIM, PROJ_DIM

ML_DIR = Path(__file__).resolve().parent.parent
CACHE = ML_DIR / "data" / "cached"
MODELS = ML_DIR / "models"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    print(f"device: {DEVICE}")
    img_emb = np.load(CACHE / "clip_image_embeddings.npy")
    txt_emb = np.load(CACHE / "clip_text_embeddings.npy")

    ckpt = torch.load(MODELS / "best_heads.pt", map_location=DEVICE, weights_only=True)
    img_head = ProjectionHead(EMB_DIM, HIDDEN_DIM, PROJ_DIM).to(DEVICE)
    txt_head = ProjectionHead(EMB_DIM, HIDDEN_DIM, PROJ_DIM).to(DEVICE)
    img_head.load_state_dict(ckpt["img_head"])
    txt_head.load_state_dict(ckpt["txt_head"])
    img_head.eval(); txt_head.eval()

    with torch.no_grad():
        img_t = torch.from_numpy(img_emb).float().to(DEVICE)
        txt_t = torch.from_numpy(txt_emb).float().to(DEVICE)
        img_p, txt_p = [], []
        for i in range(0, len(img_t), 4096):
            img_p.append(img_head(img_t[i:i+4096]).cpu().numpy())
            txt_p.append(txt_head(txt_t[i:i+4096]).cpu().numpy())
        img_p = np.concatenate(img_p, axis=0)
        txt_p = np.concatenate(txt_p, axis=0)

    np.save(CACHE / "img_projected.npy", img_p)
    np.save(CACHE / "txt_projected.npy", txt_p)
    print(f"saved img_projected.npy: {img_p.shape}")
    print(f"saved txt_projected.npy: {txt_p.shape}")


if __name__ == "__main__":
    main()