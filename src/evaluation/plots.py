import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve

from src import config as C


def plot_roc(y, proba, name, ax=None):
    ax = ax or plt.subplots(figsize=(5, 4.5))[1]
    fpr, tpr, _ = roc_curve(y, proba)
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")
    return ax


def plot_confusion(y, proba, name, ax=None, threshold=C.THRESHOLD):
    ax = ax or plt.subplots(figsize=(4.5, 4))[1]
    cm = confusion_matrix(y, (np.asarray(proba) >= threshold).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=[f">{C.HORIZON_DAYS}d", f"<={C.HORIZON_DAYS}d"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix - {name}"); ax.set_xlabel("Predicted next anomaly"); ax.set_ylabel("Actual next anomaly")
    return ax
