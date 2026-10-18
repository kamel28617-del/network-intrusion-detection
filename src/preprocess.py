import pandas as pd
from sklearn.model_selection import train_test_split

META_COLS = ["Label", "day", "is_attack"]
RANDOM_STATE = 42
VAL_FRACTION = 0.25  # share of the training part -> 52.5% train / 17.5% val / 30% test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """All numeric features except metadata and constant columns."""
    cols = [c for c in df.columns if c not in META_COLS]
    return [c for c in cols if df[c].nunique() > 1]


def split_data(df: pd.DataFrame, test_size: float = 0.3):
    """Stratified split on the full attack label so every class appears in both sets."""
    return train_test_split(
        df,
        test_size=test_size,
        stratify=df["Label"].astype(str),
        random_state=RANDOM_STATE,
    )


def split_train_val(train_df: pd.DataFrame, val_size: float = VAL_FRACTION):
    """Carve a validation set out of the training part. The test set is not touched."""
    return train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["Label"].astype(str),
        random_state=RANDOM_STATE,
    )
