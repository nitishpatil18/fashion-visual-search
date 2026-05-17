# fashion visual search

multi-modal search over 44k fashion products. clip embeddings + trained projection heads + learned reranker. supports text-to-image and image-to-image queries with sub-200ms latency.

built end-to-end: data pipeline, contrastive training, two-stage retrieval, lightgbm learning-to-rank, fastapi service, react ui.

## results

evaluated on a held-out test set of 3,878 queries (4,411 test products, 80/10/10 split stratified by article type). ground truth: products sharing (articleType, baseColour, gender) with the query, avg 22.8 relevant items per query.

| metric | base clip | + projection heads | + lightgbm rerank | lift vs base |
|--------|----------:|-------------------:|------------------:|-------------:|
| recall@1  | 0.007 | 0.011 | 0.023 | +215% |
| recall@5  | 0.029 | 0.053 | 0.105 | +256% |
| recall@10 | 0.056 | 0.095 | 0.179 | +223% |
| recall@20 | 0.100 | 0.165 | 0.269 | +170% |
| ndcg@10   | 0.057 | 0.100 | 0.155 | +172% |
| map@10    | 0.025 | 0.045 | 0.080 | +219% |

**3.2x recall@10**, **2.7x ndcg@10**, **3.2x map@10** over frozen-clip baseline. each stage contributes measurable lift.

## architecture
text query / image
│
▼
clip vit-b/32 (frozen)
│  raw 512-dim embedding
▼
trained projection head (mlp + residual + layernorm)
│  projected 512-dim embedding (l2 normalized)
▼
faiss flat ip index
│  top-100 candidates
▼
lightgbm ranker
│  rerank by 8 features
▼
top-k results

**stage 1 retrieval (faiss).** projected clip vectors indexed with `IndexFlatIP`. inner product on l2-normalized vectors equals cosine similarity. retrieves top-100 candidates in ~1ms.

**stage 2 rerank (lightgbm).** lambdarank objective. features: projected similarity, raw clip similarity, articleType match, baseColour match, gender match, season match, usage match, caption token overlap.

## training

**projection heads.** two-layer mlp (512 → 1024 → 512) with gelu, dropout 0.1, residual connection, layernorm. trained with infonce contrastive loss + mined hard negatives.

200,000 (anchor, positive, hard_negative) triplets mined from train split:
- **positive**: another product sharing (articleType, baseColour, gender) with anchor
- **hard negative**: same articleType, different baseColour, picked from clip's top-50 nearest neighbors (visually confusable items)

15 epochs max, early stopping on val recall@10, best epoch 6. 5 seconds per epoch on m5 mps with 195k training pairs at batch 512.

**lightgbm reranker.** 2,000,000 training rows (20k queries × 100 candidates). objective lambdarank, 200 trees, learning rate 0.1, num_leaves 63. trained on cpu in ~30 seconds.

feature importance (gain):

| feature | importance |
|---------|-----------:|
| caption_overlap | 2284 |
| proj_sim | 835 |
| base_sim | 595 |
| season_match | 486 |
| usage_match | 390 |
| gender_match | 165 |
| article_match | 150 |
| color_match | 82 |

caption_overlap dominates because product captions encode color, type, and gender. future work: train a separate reranker for image queries without caption_overlap.

## stack

- **ml service**: python 3.11, pytorch (mps + cpu), transformers, faiss-cpu, lightgbm, fastapi, uvicorn
- **backend**: node 22, express, multer, axios
- **frontend**: react 19, vite 8, tailwind 4
- **data**: kaggle fashion product images dataset (44,419 products with metadata)

## dataset

paramaggarwal/fashion-product-images-dataset. 44,441 product images. 7 master categories (apparel, accessories, footwear, ...), 47 subcategories, 142 article types, 46 colors. structured metadata: gender, season, usage, productDisplayName.

**preprocessing**: dropped 5 corrupt rows. captions built by concatenating `productDisplayName + baseColour + articleType + gender + usage + season`.

**splits**: 80/10/10 stratified by articleType. evaluation queries: test products with at least 2 relevant items in the test split (3,878 queries).

## run locally

requires python 3.11+, node 22+, ~15gb disk for the dataset, ~3gb ram for the ml service.

```bash
git clone https://github.com/nitishpatil18/fashion-visual-search
cd fashion-visual-search

# ml service setup
cd ml-service
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# download dataset (requires kaggle api token in ~/.kaggle/kaggle.json)
mkdir -p data/raw
kaggle datasets download -d paramaggarwal/fashion-product-images-dataset -p data/raw
unzip data/raw/fashion-product-images-dataset.zip -d data/raw
rm data/raw/fashion-product-images-dataset.zip

# run the full pipeline
python src/embeddings.py            # ~10 min on m5 mps
python src/text_embeddings.py       # ~1 min
python src/split.py                 # instant
python src/eval_metrics.py          # baseline numbers
python src/pair_mining.py           # ~4 min
python src/train.py                 # ~2 min, 15 epochs
python src/eval_trained.py          # trained heads numbers
python src/project_embeddings.py    # cache projected vectors
python src/build_rerank_data.py     # ~2 min
python src/train_reranker.py        # ~1 min
python src/eval_reranker.py         # final comparison table
python src/build_faiss_index.py     # build serving index

# install backend + frontend
cd ../backend && npm install
cd ../frontend && npm install
cd ..
npm install

# run all 3 services
npm run dev
```

open http://localhost:5173.

## project structure
fashion-search/
├── ml-service/
│   ├── src/
│   │   ├── embeddings.py           # clip image embeddings
│   │   ├── text_embeddings.py      # clip text embeddings
│   │   ├── split.py                # train/val/test, eval set
│   │   ├── eval_metrics.py         # baseline retrieval
│   │   ├── pair_mining.py          # hard negative triplets
│   │   ├── train.py                # contrastive training
│   │   ├── eval_trained.py         # trained projection heads eval
│   │   ├── project_embeddings.py   # cache projected vectors
│   │   ├── build_rerank_data.py    # generate reranker training data
│   │   ├── train_reranker.py       # lightgbm lambdarank
│   │   ├── eval_reranker.py        # full pipeline eval
│   │   ├── build_faiss_index.py    # serving index
│   │   └── api.py                  # fastapi service
│   ├── models/                     # trained artifacts (gitignored)
│   ├── data/                       # raw + cached embeddings (gitignored)
│   └── requirements.txt
├── backend/
│   └── src/server.js               # express proxy + image serving
├── frontend/
│   └── src/App.jsx                 # react + tailwind ui
└── package.json                    # npm run dev (concurrently)

## engineering notes

a few non-obvious problems hit during development:

- **clip foundation model.** the actual clip vit-b/32 weights are frozen. only the projection heads are trained. trying to train clip from scratch on a macbook is unrealistic. this is also how production retrieval systems typically work: freeze the foundation model, train cheap heads.

- **mps + libomp conflicts.** pytorch (mps), lightgbm, and faiss each ship their own openmp runtime. loading them in the same fastapi process triggered segfaults and deadlocks. fixed by (1) importing lightgbm before torch, (2) `torch.set_num_threads(1)`, (3) using a permanent cpu clip model for image encoding (mps vision_model hangs after lightgbm is loaded).

- **caption_overlap feature.** dominates the reranker because text captions encode color and article type. effective for text queries, ineffective for image queries. real fix is to train two separate rerankers, one per query mode.

- **rerank latency.** initial implementation called `faiss.reconstruct()` 100 times per query inside a loop. vectorizing the candidate feature extraction took latency from 1133ms to 95ms (12x).

## generate requirements.txt

```bash
cd ml-service
pip freeze > requirements.txt
```