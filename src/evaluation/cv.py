import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import config as C


def grouped_oof_proba(make_model, df, features, n_splits=C.N_SPLITS):
    """Satellite-grouped K-fold: a satellite is NEVER in both train and test.
    Returns out-of-fold P(anomaly within horizon) and the fold id of every row.
    GroupKFold is deterministic, so every model sees exactly the same folds."""
    X, y, groups = df[features].to_numpy(float), df.label.to_numpy(int), df.sat.to_numpy()
    oof, fold = np.full(len(df), np.nan), np.zeros(len(df), int)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        oof[te] = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        fold[te] = k
    return oof, fold


def save_predictions(df, proba, fold, name):
    """Store predictions so the comparison notebook can load every model's output."""
    C.RESULTS.mkdir(exist_ok=True)
    out = pd.DataFrame({"sat": df.sat.values, "start": df.start.values, "y_true": df.label.values,
                        "proba": proba, "fold": fold})
    path = C.RESULTS / f"predictions_{name}.csv"
    out.to_csv(path, index=False)
    return path
