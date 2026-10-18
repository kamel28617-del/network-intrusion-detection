import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, roc_curve

PREDS_PATH = Path("data/processed/test_predictions.parquet")
VAL_PREDS_PATH = Path("data/processed/val_predictions.parquet")
THRESHOLDS_PATH = Path("models/thresholds.json")
FIG_DIR = Path("reports/figures")
MODELS = ["random_forest", "hist_gradient_boosting"]
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]


def threshold_table(y, score) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        pred = (score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": t,
                "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                "recall": round(tp / (tp + fn), 4),
                "fpr": round(fp / (fp + tn), 5),
                "false_alarms": int(fp),
                "missed": int(fn),
            }
        )
    return pd.DataFrame(rows)


def best_recall_at_fpr(y, score, max_fpr: float = 0.001):
    fpr, tpr, thr = roc_curve(y, score)
    ok = fpr <= max_fpr
    i = tpr[ok].argmax()
    return float(thr[ok][i]), float(tpr[ok][i]), float(fpr[ok][i])


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PREDS_PATH)
    y = df["is_attack"].to_numpy()
    val = pd.read_parquet(VAL_PREDS_PATH)
    y_val = val["is_attack"].to_numpy()
    chosen = {}

    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    fig_cm, axes = plt.subplots(1, len(MODELS), figsize=(11, 4.5))

    for ax_cm, name in zip(axes, MODELS):
        score = df[f"{name}_score"].to_numpy()

        print(f"\n=== {name}: effect of the alert threshold ===")
        table = threshold_table(y, score)
        print(table.to_string(index=False))
        table.to_csv(FIG_DIR.parent / f"{name}_thresholds.csv", index=False)

        # Pick the threshold on the validation set, then report it once on the untouched test set
        thr, val_rec, val_fpr = best_recall_at_fpr(y_val, val[f"{name}_score"].to_numpy())
        tn, fp, fn, tp = confusion_matrix(y, (score >= thr).astype(int), labels=[0, 1]).ravel()
        print(f"Threshold chosen on validation (FPR <= 0.1%): {thr:.3f} (val recall {val_rec:.4f}, val FPR {val_fpr:.5f})")
        print(f"Same threshold on test: recall {tp / (tp + fn):.4f}, FPR {fp / (fp + tn):.5f}, missed {fn}, false alarms {fp}")
        chosen[name] = thr

        p, r, _ = precision_recall_curve(y, score)
        ax_pr.plot(r, p, label=name)

        ConfusionMatrixDisplay.from_predictions(
            y,
            (score >= 0.5).astype(int),
            display_labels=["Benign", "Attack"],
            values_format="d",
            cmap="Blues",
            colorbar=False,
            ax=ax_cm,
        )
        ax_cm.set_title(f"{name} (threshold 0.5)")

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall curve (binary detection)")
    ax_pr.set_ylim(0.9, 1.001)
    ax_pr.legend()
    fig_pr.tight_layout()
    fig_pr.savefig(FIG_DIR / "pr_curves.png", dpi=150)

    fig_cm.tight_layout()
    fig_cm.savefig(FIG_DIR / "confusion_matrices.png", dpi=150)
    THRESHOLDS_PATH.write_text(json.dumps(chosen, indent=2))
    print(f"\nSaved figures to {FIG_DIR.resolve()} and thresholds to {THRESHOLDS_PATH}")


if __name__ == "__main__":
    main()
