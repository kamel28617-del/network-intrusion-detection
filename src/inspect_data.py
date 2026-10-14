from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")


def load_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def main() -> None:
    files = sorted(list(DATA_DIR.glob("*.parquet")) + list(DATA_DIR.glob("*.csv")))
    if not files:
        print(f"No data files found in {DATA_DIR.resolve()}")
        return

    all_labels = pd.Series(dtype="float64")

    for f in files:
        df = load_file(f)
        if "Label" not in df.columns:
            print(f"\n=== {f.name} === no 'Label' column. Columns: {list(df.columns)[:10]}...")
            continue

        features = df.drop(columns=["Label"])
        numeric = features.select_dtypes(include=[np.number])

        n_nan = int(features.isna().sum().sum())
        n_inf = int(np.isinf(numeric.to_numpy()).sum())
        n_dup = int(df.duplicated().sum())

        print(f"\n=== {f.name} ===")
        print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
        print(f"NaN: {n_nan:,} | Inf: {n_inf:,} | Duplicate rows: {n_dup:,}")
        print(f"Non-numeric feature columns: {list(features.columns.difference(numeric.columns))}")

        counts = df["Label"].value_counts()
        print(counts.to_string())
        all_labels = all_labels.add(counts, fill_value=0)

    print("\n=== OVERALL CLASS DISTRIBUTION ===")
    total = all_labels.sum()
    summary = pd.DataFrame(
        {"count": all_labels.astype(int), "percent": (all_labels / total * 100).round(3)}
    ).sort_values("count", ascending=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
