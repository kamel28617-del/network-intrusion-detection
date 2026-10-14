import json
import time
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from src.data_loader import load_clean
from src.evaluate import binary_metrics, recall_by_family
from src.preprocess import RANDOM_STATE, get_feature_cols, split_data

MODELS_DIR = Path("models")
PREDS_PATH = Path("data/processed/test_predictions.parquet")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_clean()
    features = get_feature_cols(df)
    print(f"{len(df):,} flows | {len(features)} features")

    train_df, test_df = split_data(df)
    X_train, y_train = train_df[features], train_df["is_attack"]
    X_test, y_test = test_df[features], test_df["is_attack"]
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    preds = test_df[["Label", "day", "is_attack"]].reset_index(drop=True)
    preds["Label"] = preds["Label"].astype(str)
    preds["day"] = preds["day"].astype(str)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_s = round(time.time() - t0, 1)

        score = model.predict_proba(X_test)[:, 1]
        pred = (score >= 0.5).astype("int8")

        m = binary_metrics(y_test, pred, score)
        m["fit_seconds"] = fit_s
        results[name] = m

        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
        print("\nDetection rate per attack type:")
        print(recall_by_family(test_df["Label"], pred).to_string())

        preds[f"{name}_score"] = score
        preds[f"{name}_pred"] = pred
        joblib.dump(model, MODELS_DIR / f"{name}.joblib", compress=3)

    (MODELS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    (MODELS_DIR / "features.json").write_text(json.dumps(features, indent=2))
    preds.to_parquet(PREDS_PATH)
    print(f"\nSaved models, metrics and test predictions ({PREDS_PATH}).")


if __name__ == "__main__":
    main()
