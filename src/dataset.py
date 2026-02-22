import pandas as pd
import numpy as np
from typing import Sequence, Optional


def load_dataset(path: str) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    return pd.read_csv(path)


def create_trq_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create torque target removing the effect of torque margin (%):
    trq_target = trq_measured / (1 + trq_margin/100)
    """
    required = {"trq_measured", "trq_margin"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for trq_target: {sorted(missing)}")

    out = df.copy()
    margin = pd.to_numeric(out["trq_margin"], errors="coerce")
    measured = pd.to_numeric(out["trq_measured"], errors="coerce")

    denom = 1 + (margin / 100.0)
    out["trq_target"] = measured / denom
    return out


def create_np_ng_ratio(df: pd.DataFrame, drop_original: bool = True) -> pd.DataFrame:
    """Create np/ng ratio with basic safety checks."""
    required = {"np", "ng"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for np_ng_ratio: {sorted(missing)}")

    out = df.copy()
    np_col = pd.to_numeric(out["np"], errors="coerce")
    ng_col = pd.to_numeric(out["ng"], errors="coerce")

    # Avoid division by zero
    ng_col = ng_col.replace(0, np.nan)

    out["np_ng_ratio"] = np_col / ng_col

    if drop_original:
        out = out.drop(["np", "ng"], axis=1)
    return out


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline of feature engineering steps."""
    df = create_trq_target(df)
    out=create_np_ng_ratio(df)
    return out


def drop_features(df: pd.DataFrame, drop_features: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Drop selected columns safely."""
    if not drop_features:
        return df.copy()
    return df.drop(list(drop_features), axis=1, errors="ignore").copy()


def save_processed_dataset(df: pd.DataFrame, path: str, drop_cols: Optional[Sequence[str]] = None) -> None:
    """Save processed dataset to CSV."""
    out = drop_features(df, drop_cols)
    out.to_csv(path, index=False)
