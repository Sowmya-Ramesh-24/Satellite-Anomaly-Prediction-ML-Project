import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y, proba, threshold=0.5) -> dict:
    """The five headline numbers. ROC AUC uses probabilities; the others use the threshold."""
    pred = (pd.Series(proba) >= threshold).astype(int)
    return {"Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "ROC AUC": roc_auc_score(y, proba)}
