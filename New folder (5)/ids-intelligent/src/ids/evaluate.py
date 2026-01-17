from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="attack")),
        "recall": float(recall_score(y_true, y_pred, pos_label="attack")),
        "f1": float(f1_score(y_true, y_pred, pos_label="attack")),
    }
    if y_proba is not None:
        # Expect probability for the positive class (attack)
        try:
            auc = roc_auc_score((np.array(y_true) == "attack").astype(int), y_proba)
            metrics["roc_auc"] = float(auc)
        except Exception:
            pass
    return metrics


def print_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["normal", "attack"])
    return cm
