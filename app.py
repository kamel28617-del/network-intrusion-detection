import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NIDS Dashboard", layout="wide")

PREDS_PATH = Path("data/processed/test_predictions.parquet")
MODELS = {"Random Forest": "random_forest", "Hist Gradient Boosting": "hist_gradient_boosting"}


@st.cache_data
def load_preds() -> pd.DataFrame:
    return pd.read_parquet(PREDS_PATH)


def confusion(y: np.ndarray, flagged: np.ndarray) -> dict:
    tp = int(((y == 1) & flagged).sum())
    fp = int(((y == 0) & flagged).sum())
    fn = int(((y == 1) & ~flagged).sum())
    tn = int(((y == 0) & ~flagged).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


st.title("Network Intrusion Detection System")
st.caption(
    "ML-based detection on CIC-IDS2017. This dashboard replays held-out test flows "
    "as a simulated alert feed; it is not analyzing live traffic."
)

if not PREDS_PATH.exists():
    st.error("Predictions not found. Run `python -m src.train` first.")
    st.stop()

df = load_preds()

with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Model", list(MODELS))
    threshold = st.slider("Alert threshold", 0.01, 0.99, 0.50, 0.01)
    days = st.multiselect("Capture day", sorted(df["day"].unique()), default=sorted(df["day"].unique()))
    st.caption("Lower threshold = fewer missed attacks, more false alarms.")

name = MODELS[model_label]
view = df[df["day"].isin(days)]
y = view["is_attack"].to_numpy()
score = view[f"{name}_score"].to_numpy()
flagged = score >= threshold
c = confusion(y, flagged)

precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
fpr = c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

st.subheader(f"{model_label} at threshold {threshold:.2f}")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Flows analyzed", f"{len(view):,}")
m2.metric("Alerts raised", f"{int(flagged.sum()):,}")
m3.metric("Precision", f"{precision:.2%}")
m4.metric("Recall", f"{recall:.2%}")
m5.metric("False positive rate", f"{fpr:.3%}")
st.caption(
    f"F1 {f1:.4f} | detected attacks {c['tp']:,} | missed attacks {c['fn']:,} | "
    f"false alarms {c['fp']:,}"
)

tab_overview, tab_alerts, tab_unseen, tab_features = st.tabs(
    ["Overview", "Alerts", "Unseen attacks", "Explainability"]
)

with tab_overview:
    left, right = st.columns(2)

    with left:
        st.markdown("**Detection rate per attack type**")
        att = view[view["is_attack"] == 1].copy()
        att["detected"] = att[f"{name}_score"] >= threshold
        by_type = att.groupby("Label")["detected"].agg(detected="sum", total="count")
        by_type["recall"] = (by_type["detected"] / by_type["total"]).round(4)
        st.dataframe(by_type.sort_values("total", ascending=False))
        st.caption("Types with very few flows (SQL injection, Infiltration, Heartbleed) are too small to judge.")

        st.markdown("**Confusion matrix**")
        st.dataframe(
            pd.DataFrame(
                [[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
                index=["Actual benign", "Actual attack"],
                columns=["Predicted benign", "Predicted attack"],
            )
        )

    with right:
        st.markdown("**Alerts per capture day**")
        per_day = view.assign(alert=flagged).groupby("day")["alert"].sum()
        st.bar_chart(per_day)

        st.markdown("**Score distribution (share of each class)**")
        bins = np.linspace(0, 1, 21)
        labels = [f"{b:.2f}" for b in bins[:-1]]
        benign_hist = np.histogram(score[y == 0], bins=bins)[0] / max((y == 0).sum(), 1)
        attack_hist = np.histogram(score[y == 1], bins=bins)[0] / max((y == 1).sum(), 1)
        st.bar_chart(pd.DataFrame({"benign": benign_hist, "attack": attack_hist}, index=labels))

    st.markdown("**Threshold trade-off: missed attacks vs false alarms**")
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = []
    for t in grid:
        cc = confusion(y, score >= t)
        rows.append({"threshold": t, "missed attacks": cc["fn"], "false alarms": cc["fp"]})
    st.line_chart(pd.DataFrame(rows).set_index("threshold"))

with tab_alerts:
    st.markdown("**Highest-scoring alerts**")
    alerts = view.assign(score=score, flagged=flagged)
    alerts = alerts[alerts["flagged"]].sort_values("score", ascending=False)
    alerts["verdict"] = np.where(alerts["is_attack"] == 1, "True detection", "False alarm")
    n_show = st.slider("Rows to show", 20, 500, 100, 20)
    st.dataframe(
        alerts[["score", "Label", "day", "verdict"]].head(n_show).reset_index(drop=True),
        hide_index=True,
    )
    only_fp = alerts[alerts["is_attack"] == 0]
    st.caption(f"{len(only_fp):,} false alarms among {len(alerts):,} alerts at this threshold.")

with tab_unseen:
    path = Path("reports/unseen_attacks.csv")
    st.markdown("**Held-out attack family experiment**")
    st.write(
        "Each row: models trained without that attack family, then tested on it. "
        "AUC shows whether unseen attacks are ranked above benign traffic; "
        "recall shows how many are caught at a fixed threshold."
    )
    if path.exists():
        st.dataframe(pd.read_csv(path), hide_index=True)
    else:
        st.info("Run `python -m src.unseen_attacks` to generate this table.")

with tab_features:
    img = Path("reports/figures/feature_importance.png")
    csv = Path("reports/feature_importance.csv")
    if img.exists():
        st.image(str(img))
    if csv.exists():
        st.dataframe(pd.read_csv(csv).head(20), hide_index=True)
    st.caption("Correlated features share credit, so read feature groups rather than exact ranks.")
