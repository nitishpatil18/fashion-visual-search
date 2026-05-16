"""create train/val/test splits stratified by article type.
also build a structured eval set for retrieval metrics."""

from pathlib import Path
import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent
CACHE = ML_DIR / "data" / "cached"

META = CACHE / "metadata_clean.csv"
SPLIT_OUT = CACHE / "splits.csv"
EVAL_OUT = CACHE / "eval_set.csv"

SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10


def main():
    df = pd.read_csv(META)
    print(f"total products: {len(df)}")

    # stratified split by articleType to ensure rare classes appear in all splits
    df["split"] = "train"
    rng = np.random.RandomState(SEED)

    for atype, group in df.groupby("articleType"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(n * TEST_FRAC))
        n_val = max(1, int(n * VAL_FRAC))
        df.loc[idx[:n_test], "split"] = "test"
        df.loc[idx[n_test:n_test + n_val], "split"] = "val"

    print("\nsplit sizes:")
    print(df["split"].value_counts())

    # save splits
    df[["id", "split"]].to_csv(SPLIT_OUT, index=False)
    print(f"\nsaved splits -> {SPLIT_OUT}")

    # build a structured eval set from the test split.
    # for each test product, the "query" is its caption and "relevant" products
    # are all test products sharing the same (articleType, baseColour, gender).
    # this gives many-to-many ground truth instead of strict 1-to-1.
    test_df = df[df["split"] == "test"].copy()
    before = len(test_df)
    test_df = test_df.dropna(subset=["articleType", "baseColour", "gender"]).reset_index(drop=True)
    print(f"\ntest rows: {before}, after dropping nan in grouping cols: {len(test_df)}")

    test_df["group_key"] = (
        test_df["articleType"].astype(str) + "|" +
        test_df["baseColour"].astype(str) + "|" +
        test_df["gender"].astype(str)
    )
    group_to_ids = test_df.groupby("group_key")["id"].apply(list).to_dict()
    test_df["relevant_ids"] = test_df["group_key"].map(group_to_ids)
    test_df["num_relevant"] = test_df["relevant_ids"].apply(len)

    # only keep queries that have at least 2 relevant items (self + one more)
    test_df = test_df[test_df["num_relevant"] >= 2].reset_index(drop=True)

    print(f"\neval queries (test products with >=2 relevant items): {len(test_df)}")
    print(f"avg relevant items per query:    {test_df['num_relevant'].mean():.1f}")
    print(f"median relevant items per query: {test_df['num_relevant'].median():.1f}")

    # serialize the relevant_ids list as a pipe-separated string for csv
    test_df["relevant_ids_str"] = test_df["relevant_ids"].apply(lambda lst: "|".join(map(str, lst)))
    test_df[["id", "group_key", "num_relevant", "relevant_ids_str"]].to_csv(EVAL_OUT, index=False)
    print(f"saved eval set -> {EVAL_OUT}")


if __name__ == "__main__":
    main()