import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_pred, y_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "false_positive_rate": float(fp / (fp + tn)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def recall_by_family(labels, y_pred) -> pd.DataFrame:
    """Detection rate for each attack type (binary prediction vs. original label)."""
    df = pd.DataFrame({"Label": np.asarray(labels).astype(str), "pred": np.asarray(y_pred)})
    df = df[df["Label"] != "Benign"]
    out = df.groupby("Label")["pred"].agg(detected="sum", total="count")
    out["recall"] = (out["detected"] / out["total"]).round(4)
    return out.sort_values("total", ascending=False)
