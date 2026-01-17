import pandas as pd
import numpy as np
from sklearn.datasets import fetch_kddcup99
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_kddcup99(as_frame: bool = True) -> pd.DataFrame:
    data = fetch_kddcup99(subset=None, shuffle=True, percent10=True, return_X_y=False, as_frame=as_frame)
    df = data.frame.copy()
    # Convert bytes to str for categorical columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
        elif df[col].dtype == "bytes":
            df[col] = df[col].str.decode("utf-8")
    # The target column exists, rename if needed
    if "target" in df.columns:
        df.rename(columns={"target": "label"}, inplace=True)
    # Binary label: normal vs attack
    label_col = "label" if "label" in df.columns else df.columns[-1]
    df["binary_label"] = np.where(df[label_col].astype(str).str.contains("normal"), "normal", "attack")
    return df


def load_csv(path: str, label_column: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_column and label_column in df.columns:
        pass
    else:
        # If no label provided, create placeholder (all normal)
        df["binary_label"] = "normal"
    return df


def save_processed(df: pd.DataFrame, name: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out


def load_dataset(name: str = "kddcup99", csv: str | None = None, label_column: str | None = None) -> pd.DataFrame:
    if csv:
        df = load_csv(csv, label_column)
    elif name.lower() == "kddcup99":
        df = load_kddcup99()
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return df
