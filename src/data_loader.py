from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/cicids2017_clean.parquet")


def _day_from_name(path: Path) -> str:
    # "DoS-Wednesday-no-metadata.parquet" -> "Wednesday"
    return path.stem.split("-")[1]


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load all parquet files, clean labels, cast to float32, drop duplicates."""
    frames = []
    for f in sorted(raw_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        df.columns = df.columns.str.strip()
        feature_cols = df.columns.drop("Label")
        df[feature_cols] = df[feature_cols].astype("float32")
        df["Label"] = df["Label"].astype(str).str.strip()
        df["day"] = _day_from_name(f)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Fix the corrupted dash in "Web Attack ? Brute Force" etc.
    df["Label"] = df["Label"].str.replace(
        r"^Web Attack\s+\S\s+", "Web Attack - ", regex=True
    )

    # Global dedup (a flow can repeat across files)
    feature_cols = [c for c in df.columns if c not in ("Label", "day")]
    before = len(df)
    df = df.drop_duplicates(subset=feature_cols + ["Label"]).reset_index(drop=True)
    print(f"Removed {before - len(df):,} duplicate rows across files")

    df["is_attack"] = (df["Label"] != "Benign").astype("int8")
    df["Label"] = df["Label"].astype("category")
    df["day"] = df["day"].astype("category")
    return df


def load_clean() -> pd.DataFrame:
    """Load the cached clean dataset, building it first if needed."""
    if PROCESSED_PATH.exists():
        return pd.read_parquet(PROCESSED_PATH)
    df = load_raw()
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH)
    return df


if __name__ == "__main__":
    df = load_clean()
    feature_cols = [c for c in df.columns if c not in ("Label", "day", "is_attack")]
    print(f"Shape: {df.shape} | Memory: {df.memory_usage(deep=True).sum() / 1e6:,.0f} MB")
    print("\nLabel counts:")
    print(df["Label"].value_counts().to_string())
    print(f"\nAttack share: {df['is_attack'].mean() * 100:.2f}%")
    constant = [c for c in feature_cols if df[c].nunique() <= 1]
    print(f"\nConstant (useless) features: {constant}")
