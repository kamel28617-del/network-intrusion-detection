from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data_loader import load_clean
from src.preprocess import RANDOM_STATE, get_feature_cols, split_data

FAMILIES = {
    "DoS": ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "Heartbleed"],
    "DDoS": ["DDoS"],
    "PortScan": ["PortScan"],
    "BruteForce (FTP/SSH)": ["FTP-Patator", "SSH-Patator"],
    "WebAttack": ["Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection"],
    "Bot": ["Bot"],
}
BENIGN_TRAIN_CAP = 300_000
BENIGN_TEST_CAP = 100_000


def main() -> None:
    df = load_clean()
    features = get_feature_cols(df)
    train_df, test_df = split_data(df)
    train_df = train_df.assign(Label=train_df["Label"].astype(str))
    test_df = test_df.assign(Label=test_df["Label"].astype(str))

    benign_train = train_df[train_df.Label == "Benign"].sample(BENIGN_TRAIN_CAP, random_state=RANDOM_STATE)
    benign_test = test_df[test_df.Label == "Benign"].sample(BENIGN_TEST_CAP, random_state=RANDOM_STATE)
    attacks_train = train_df[train_df.Label != "Benign"]

    # Anomaly detector: trained on benign traffic only, never sees any attack
    iso = IsolationForest(n_estimators=100, max_samples=256, n_jobs=-1, random_state=RANDOM_STATE)
    iso.fit(benign_train[features])
    iso_thr = np.quantile(-iso.score_samples(benign_train[features]), 0.99)
    iso_benign = -iso.score_samples(benign_test[features])

    rows = []
    for fam, labels in FAMILIES.items():
        print(f"Holding out: {fam}")
        train = pd.concat([benign_train, attacks_train[~attacks_train.Label.isin(labels)]])
        rf = RandomForestClassifier(
            n_estimators=60,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        rf.fit(train[features], train["is_attack"])

        fam_test = test_df[test_df.Label.isin(labels)]
        X_eval = pd.concat([benign_test[features], fam_test[features]])
        y_eval = np.r_[np.zeros(len(benign_test)), np.ones(len(fam_test))]

        rf_score = rf.predict_proba(X_eval)[:, 1]
        iso_score = np.r_[iso_benign, -iso.score_samples(fam_test[features])]
        is_fam = y_eval == 1

        rows.append(
            {
                "held_out_family": fam,
                "n_test": int(is_fam.sum()),
                "rf_auc": round(roc_auc_score(y_eval, rf_score), 4),
                "rf_recall@0.5": round(float((rf_score[is_fam] >= 0.5).mean()), 4),
                "rf_fpr@0.5": round(float((rf_score[~is_fam] >= 0.5).mean()), 5),
                "iso_auc": round(roc_auc_score(y_eval, iso_score), 4),
                "iso_recall": round(float((iso_score[is_fam] >= iso_thr).mean()), 4),
                "iso_fpr": round(float((iso_score[~is_fam] >= iso_thr).mean()), 5),
            }
        )

    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False))
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/unseen_attacks.csv", index=False)


if __name__ == "__main__":
    main()
