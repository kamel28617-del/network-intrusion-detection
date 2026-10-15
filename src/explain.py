import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data_loader import load_clean
from src.preprocess import RANDOM_STATE, split_data

FIG_DIR = Path("reports/figures")
SAMPLE = 30_000


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    features = json.loads(Path("models/features.json").read_text())
    rf = joblib.load("models/random_forest.joblib")

    df = load_clean()
    _, test_df = split_data(df)
    sample = test_df.sample(SAMPLE, random_state=RANDOM_STATE)
    X, y = sample[features], sample["is_attack"]

    perm = permutation_importance(
        rf, X, y, scoring="average_precision", n_repeats=3, random_state=RANDOM_STATE, n_jobs=1
    )

    imp = pd.DataFrame(
        {
            "feature": features,
            "impurity_importance": rf.feature_importances_,
            "permutation_importance": perm.importances_mean,
        }
    ).sort_values("permutation_importance", ascending=False)
    imp.to_csv("reports/feature_importance.csv", index=False)
    print(imp.head(15).round(4).to_string(index=False))

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["permutation_importance"])
    ax.set_xlabel("Drop in average precision when the feature is shuffled")
    ax.set_title("Top 15 features (Random Forest, permutation importance)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    print(f"\nSaved {FIG_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
