import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def split_features_labels(df: pd.DataFrame, label_col: str = "binary_label") -> Tuple[pd.DataFrame, pd.Series]:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    X = df.drop(columns=[label_col])
    y = df[label_col].astype(str)
    return X, y


def build_transformer(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == object]
    num_cols = [c for c in X.columns if c not in cat_cols]

    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    return transformer


def train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_preprocess_pipeline(X: pd.DataFrame) -> Pipeline:
    transformer = build_transformer(X)
    pipe = Pipeline(steps=[("transform", transformer)])
    return pipe
