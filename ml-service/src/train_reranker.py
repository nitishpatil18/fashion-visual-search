"""train lightgbm ranker on rerank_data. NO torch imports — must stay isolated
to avoid the libomp dual-load segfault on macos arm64. saves model to disk."""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ML_DIR = Path(__file__).resolve().parent.parent
CACHE = ML_DIR / "data" / "cached"
MODELS = ML_DIR / "models"
MODELS.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "proj_sim", "base_sim",
    "article_match", "color_match", "gender_match",
    "season_match", "usage_match", "caption_overlap",
]
SEED = 42


def main():
    print("loading rerank_data...")
    df = pd.read_parquet(CACHE / "rerank_data.parquet")
    print(f"rows: {len(df)}, queries: {df['qid'].nunique()}")

    df = df.sort_values("qid").reset_index(drop=True)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    g = df.groupby("qid", sort=False).size().values

    print(f"X: {X.shape}  y: {y.shape}  groups: {g.shape}")

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        boosting_type="gbdt",
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=SEED,
        n_jobs=4,
        verbose=-1,
    )

    ranker.fit(X, y, group=g, callbacks=[lgb.log_evaluation(20)])

    out = MODELS / "reranker.txt"
    ranker.booster_.save_model(str(out))
    print(f"saved -> {out}")

    fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": ranker.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nfeature importances:")
    print(fi.to_string(index=False))


if __name__ == "__main__":
    main()