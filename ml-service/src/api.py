"""fastapi search service. text-to-image and image-to-image retrieval.
loads everything at startup. NO lightgbm + torch in same process — we use the
saved lightgbm booster which on inference-only paths doesn't trigger libomp clash."""

import io
import time
from pathlib import Path

# lightgbm MUST be imported before torch on macos arm64 to avoid libomp segfault
import lightgbm as lgb

import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

from src.train import ProjectionHead, EMB_DIM, HIDDEN_DIM, PROJ_DIM

ML_DIR = Path(__file__).resolve().parent.parent
CACHE = ML_DIR / "data" / "cached"
MODELS = ML_DIR / "models"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

TOP_K_CANDIDATES = 100
DEFAULT_TOP_K = 20
MODEL_NAME = "openai/clip-vit-base-patch32"

FEATURE_COLS = [
    "proj_sim", "base_sim",
    "article_match", "color_match", "gender_match",
    "season_match", "usage_match", "caption_overlap",
]


def overlap(a, b):
    sa = set(str(a).lower().split())
    sb = set(str(b).lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------- load everything once at startup ----------

print(f"[startup] device: {DEVICE}")

print("[startup] loading metadata...")
meta_df = pd.read_csv(CACHE / "metadata_clean.csv")
captions_df = pd.read_csv(CACHE / "captions.csv")
meta_df = meta_df.merge(captions_df, on="id")
meta_idx = meta_df.set_index("id")

print("[startup] loading raw clip embeddings (for base_sim feature)...")
img_emb_raw = np.load(CACHE / "clip_image_embeddings.npy").astype(np.float32)
img_ids = np.load(CACHE / "clip_image_ids.npy")
id_to_idx = {int(v): k for k, v in enumerate(img_ids)}

print("[startup] loading clip model...")
clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

print("[startup] loading projection heads...")
ckpt = torch.load(MODELS / "best_heads.pt", map_location=DEVICE, weights_only=True)
img_head = ProjectionHead(EMB_DIM, HIDDEN_DIM, PROJ_DIM).to(DEVICE).eval()
txt_head = ProjectionHead(EMB_DIM, HIDDEN_DIM, PROJ_DIM).to(DEVICE).eval()
img_head.load_state_dict(ckpt["img_head"])
txt_head.load_state_dict(ckpt["txt_head"])

print("[startup] loading faiss index...")
faiss_index = faiss.read_index(str(MODELS / "faiss_index.bin"))
faiss_ids = np.load(MODELS / "faiss_ids.npy")
assert np.array_equal(faiss_ids, img_ids), "faiss id order mismatch"

print("[startup] loading reranker...")
booster = lgb.Booster(model_file=str(MODELS / "reranker.txt"))

print("[startup] ready.")


# ---------- helpers ----------

@torch.no_grad()
def encode_text(query: str) -> np.ndarray:
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True, max_length=77)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = clip_model.text_model(**inputs)
    feat = clip_model.text_projection(out.pooler_output)
    feat = feat / feat.norm(dim=-1, keepdim=True)            # raw clip text emb (for base_sim)
    projected = txt_head(feat)                                # trained projection
    return feat.cpu().numpy()[0], projected.cpu().numpy()[0]


@torch.no_grad()
def encode_image(pil_image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=pil_image.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    vout = clip_model.vision_model(pixel_values=inputs["pixel_values"])
    feat = clip_model.visual_projection(vout.pooler_output)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    projected = img_head(feat)
    return feat.cpu().numpy()[0], projected.cpu().numpy()[0]


def build_features(q_meta, raw_query_vec, proj_query_vec, candidate_ids):
    """build lightgbm feature matrix for a list of candidates, vectorized."""
    q_art = q_meta.get("articleType") or ""
    q_color = q_meta.get("baseColour") or ""
    q_gender = q_meta.get("gender") or ""
    q_season = q_meta.get("season") or ""
    q_usage = q_meta.get("usage") or ""
    q_caption = q_meta.get("caption") or ""

    # filter to valid ids and get their array indices in one pass
    valid_ids = []
    valid_idxs = []
    for cid in candidate_ids:
        if cid in meta_idx.index:
            valid_ids.append(cid)
            valid_idxs.append(id_to_idx[cid])
    if not valid_ids:
        return np.zeros((0, 8), dtype=np.float32), []

    valid_idxs_arr = np.asarray(valid_idxs)

    # batched similarities: one matmul each, not 100 dot products
    proj_vecs = np.asarray([faiss_index.reconstruct(int(i)) for i in valid_idxs_arr])
    proj_sims = (proj_vecs @ proj_query_vec).astype(np.float32)
    base_sims = (img_emb_raw[valid_idxs_arr] @ raw_query_vec).astype(np.float32)

    # batch metadata via .loc on a list
    sub = meta_idx.loc[valid_ids]
    art = sub["articleType"].astype(str).to_numpy()
    color = sub["baseColour"].astype(str).to_numpy()
    gender = sub["gender"].astype(str).to_numpy()
    season = sub["season"].fillna("").astype(str).to_numpy()
    usage = sub["usage"].fillna("").astype(str).to_numpy()
    captions = sub["caption"].fillna("").astype(str).to_numpy()

    article_match = (art == q_art).astype(np.float32)
    color_match = (color == q_color).astype(np.float32)
    gender_match = (gender == q_gender).astype(np.float32)
    season_match = ((season == q_season) & (season != "") & (q_season != "")).astype(np.float32)
    usage_match = ((usage == q_usage) & (usage != "") & (q_usage != "")).astype(np.float32)

    # caption overlap (this loop is unavoidable, but it's fast on strings)
    q_tokens = set(q_caption.lower().split())
    co = np.zeros(len(valid_ids), dtype=np.float32)
    if q_tokens:
        for i, c_cap in enumerate(captions):
            c_tokens = set(c_cap.lower().split())
            if c_tokens:
                co[i] = len(q_tokens & c_tokens) / len(q_tokens | c_tokens)

    X = np.stack([
        proj_sims, base_sims, article_match, color_match, gender_match,
        season_match, usage_match, co,
    ], axis=1)
    return X.astype(np.float32), valid_ids


def search_with_projected_vec(proj_vec, raw_vec, q_meta, top_k, rerank=True):
    # stage 1: faiss top-K candidates
    D, I = faiss_index.search(proj_vec.reshape(1, -1).astype(np.float32), TOP_K_CANDIDATES)
    cand_ids = [int(img_ids[i]) for i in I[0]]

    if not rerank:
        # just return stage-1 results
        ranked = list(zip(cand_ids, D[0].tolist()))[:top_k]
    else:
        X, valid_ids = build_features(q_meta, raw_vec, proj_vec, cand_ids)
        if len(valid_ids) == 0:
            return []
        scores = booster.predict(X)
        order = np.argsort(-scores)
        ranked = [(valid_ids[i], float(scores[i])) for i in order[:top_k]]

    return [_build_result(cid, score) for cid, score in ranked]


def _build_result(cid: int, score: float):
    c = meta_idx.loc[cid]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[0]
    return {
        "id": int(cid),
        "score": float(score),
        "articleType": c["articleType"],
        "baseColour": c["baseColour"],
        "gender": c["gender"],
        "productDisplayName": c["productDisplayName"],
        "image_path": f"images/{int(cid)}.jpg",
    }


# ---------- api ----------

app = FastAPI(title="fashion search api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchResponse(BaseModel):
    query: str
    mode: str
    rerank: bool
    top_k: int
    latency_ms: float
    results: list


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "num_products": int(faiss_index.ntotal)}


@app.get("/search/text", response_model=SearchResponse)
def search_text(q: str, top_k: int = DEFAULT_TOP_K, rerank: bool = True):
    if not q or not q.strip():
        raise HTTPException(400, "empty query")
    if top_k <= 0 or top_k > 100:
        raise HTTPException(400, "top_k must be 1..100")

    t0 = time.perf_counter()
    raw_vec, proj_vec = encode_text(q)
    # for text query we don't have structured metadata; use empty fields and rely on caption overlap
    q_meta = {"caption": q}
    results = search_with_projected_vec(proj_vec, raw_vec, q_meta, top_k, rerank=rerank)
    dt = (time.perf_counter() - t0) * 1000

    return SearchResponse(
        query=q, mode="text", rerank=rerank, top_k=top_k,
        latency_ms=round(dt, 1), results=results,
    )


@app.post("/search/image", response_model=SearchResponse)
async def search_image(image: UploadFile = File(...), top_k: int = Form(DEFAULT_TOP_K), rerank: bool = Form(True)):
    if top_k <= 0 or top_k > 100:
        raise HTTPException(400, "top_k must be 1..100")
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"cannot read image: {e}")

    t0 = time.perf_counter()
    raw_vec, proj_vec = encode_image(pil_img)
    q_meta = {"caption": ""}  # no caption from raw image
    results = search_with_projected_vec(proj_vec, raw_vec, q_meta, top_k, rerank=rerank)
    dt = (time.perf_counter() - t0) * 1000

    return SearchResponse(
        query=f"<image:{image.filename}>", mode="image", rerank=rerank, top_k=top_k,
        latency_ms=round(dt, 1), results=results,
    )