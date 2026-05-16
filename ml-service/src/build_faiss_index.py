"""build a faiss flat inner-product index over the projected image embeddings.
saves index + id mapping to disk for fast api startup."""

from pathlib import Path
import numpy as np
import faiss

ML_DIR = Path(__file__).resolve().parent.parent
CACHE = ML_DIR / "data" / "cached"
MODELS = ML_DIR / "models"

OUT_INDEX = MODELS / "faiss_index.bin"
OUT_IDS = MODELS / "faiss_ids.npy"


def main():
    print("loading projected image embeddings...")
    img_p = np.load(CACHE / "img_projected.npy").astype(np.float32)
    img_ids = np.load(CACHE / "clip_image_ids.npy")
    print(f"vectors: {img_p.shape}")

    # vectors are already l2-normalized inside ProjectionHead, so inner product = cosine
    d = img_p.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(img_p)
    print(f"index ntotal: {index.ntotal}")

    faiss.write_index(index, str(OUT_INDEX))
    np.save(OUT_IDS, img_ids)
    print(f"saved -> {OUT_INDEX}")
    print(f"saved -> {OUT_IDS}")

    # sanity: search the first vector against itself
    D, I = index.search(img_p[:1], 5)
    print(f"sanity search top-5 ids: {img_ids[I[0]]}")
    print(f"sanity search top-5 sims: {D[0]}")


if __name__ == "__main__":
    main()