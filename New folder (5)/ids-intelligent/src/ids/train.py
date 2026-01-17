import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from .data import load_dataset
from .preprocess import split_features_labels, train_val_test_split, make_preprocess_pipeline
from .models import (
    build_rf,
    build_svm,
    build_knn,
    build_isolation_forest,
    build_kmeans,
    AutoencoderWrapper,
)
from .evaluate import classification_metrics, print_confusion

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_supervised(model_name: str, X_train, y_train, X_val, y_val, preprocess: Pipeline):
    if model_name == "rf":
        clf = build_rf()
    elif model_name == "svm":
        clf = build_svm()
    elif model_name == "knn":
        clf = build_knn()
    else:
        raise ValueError(f"Unknown supervised model {model_name}")

    pipe = Pipeline(steps=[("pre", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    try:
        y_val_proba = pipe.predict_proba(X_val)[:, 1]
    except Exception:
        y_val_proba = None
    metrics = classification_metrics(y_val, y_val_pred, y_val_proba)
    return pipe, metrics


def train_unsupervised(model_name: str, X_train, X_val, preprocess: Pipeline):
    if model_name == "iso":
        mdl = build_isolation_forest()
        pipe = Pipeline(steps=[("pre", preprocess), ("mdl", mdl)])
        pipe.fit(X_train)
        # IsolationForest: +1 normal, -1 anomaly
        y_val_pred = np.where(pipe.named_steps["mdl"].predict(preprocess.transform(X_val)) == -1, "attack", "normal")
        metrics = {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan}
        return pipe, metrics
    elif model_name == "kmeans":
        mdl = build_kmeans()
        pipe = Pipeline(steps=[("pre", preprocess), ("mdl", mdl)])
        pipe.fit(X_train)
        # Cluster distances as anomaly proxy
        centers = pipe.named_steps["mdl"].cluster_centers_
        Xv = preprocess.transform(X_val)
        dists = np.min(((Xv[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        threshold = np.percentile(dists, 95)
        y_val_pred = np.where(dists > threshold, "attack", "normal")
        metrics = {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan}
        return pipe, metrics
    else:
        raise ValueError(f"Unknown unsupervised model {model_name}")


def train_autoencoder(X_train, X_val, preprocess: Pipeline):
    Xn = preprocess.fit_transform(X_train)
    ae = AutoencoderWrapper(input_dim=Xn.shape[1])
    ae.fit(Xn)
    return ae, {}


def main():
    parser = argparse.ArgumentParser(description="Train IDS models")
    parser.add_argument("--dataset", default="kddcup99", help="Dataset name (default: kddcup99)")
    parser.add_argument("--csv", default=None, help="Path to CSV, overrides --dataset")
    parser.add_argument("--label-column", default=None, help="Label column name in CSV")
    parser.add_argument("--models", nargs="*", default=["rf", "svm", "knn", "iso", "kmeans", "ae"], help="Models to train")
    parser.add_argument("--save", action="store_true", help="Save trained models")
    args = parser.parse_args()

    df = load_dataset(name=args.dataset, csv=args.csv, label_column=args.label_column)
    X, y = split_features_labels(df, label_col="binary_label")

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    preprocess = make_preprocess_pipeline(X_train)

    rows = []

    for m in args.models:
        print(f"Training model: {m}")
        if m in {"rf", "svm", "knn"}:
            pipe, metrics = train_supervised(m, X_train, y_train, X_val, y_val, preprocess)
            y_test_pred = pipe.predict(X_test)
            try:
                y_test_proba = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                y_test_proba = None
            test_metrics = classification_metrics(y_test, y_test_pred, y_test_proba)
            cm = print_confusion(y_test, y_test_pred)
            print("Confusion Matrix:\n", cm)
            if args.save:
                joblib.dump(pipe, MODELS_DIR / f"model_{m}.pkl")
            rows.append({"model": m, **test_metrics})
        elif m in {"iso", "kmeans"}:
            pipe, metrics = train_unsupervised(m, X_train, X_val, preprocess)
            Xtt = preprocess.transform(X_test)
            if m == "iso":
                y_test_pred = np.where(pipe.named_steps["mdl"].predict(Xtt) == -1, "attack", "normal")
            else:
                centers = pipe.named_steps["mdl"].cluster_centers_
                dists = np.min(((Xtt[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
                threshold = np.percentile(dists, 95)
                y_test_pred = np.where(dists > threshold, "attack", "normal")
            test_metrics = classification_metrics(y_test, y_test_pred, None)
            cm = print_confusion(y_test, y_test_pred)
            print("Confusion Matrix:\n", cm)
            if args.save:
                joblib.dump(pipe, MODELS_DIR / f"model_{m}.pkl")
            rows.append({"model": m, **test_metrics})
        elif m == "ae":
            ae, _ = train_autoencoder(X_train, X_val, preprocess)
            Xtt = preprocess.transform(X_test)
            y_test_pred = ae.predict(Xtt)
            test_metrics = classification_metrics(y_test, y_test_pred, None)
            cm = print_confusion(y_test, y_test_pred)
            print("Confusion Matrix:\n", cm)
            if args.save:
                torch_path = MODELS_DIR / f"model_{m}.pt"
                # Save model state dict only
                import torch
                torch.save(ae.model.state_dict(), torch_path)
                joblib.dump(preprocess, MODELS_DIR / f"preprocess_{m}.pkl")
            rows.append({"model": m, **test_metrics})
        else:
            print(f"Unknown model {m}")

    perf = pd.DataFrame(rows)
    perf_path = MODELS_DIR / "performance.csv"
    perf.to_csv(perf_path, index=False)
    print(f"Saved performance to {perf_path}")


if __name__ == "__main__":
    main()
