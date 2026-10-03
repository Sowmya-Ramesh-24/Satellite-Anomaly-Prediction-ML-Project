import numpy as np
from sklearn.model_selection import KFold

from src import config as C


def grouped_oof_predictions(make_model, df, features, n_splits=C.N_SPLITS):
    """Return continuous out-of-fold predictions using shuffled K-fold CV."""
    X, y = df[features], df.tte.to_numpy(float)
    oof, fold = np.full(len(df), np.nan), np.zeros(len(df), int)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for k, (tr, te) in enumerate(kfold.split(X)):
        model = make_model()
        model.fit(X.iloc[tr], y[tr])
        oof[te] = model.predict(X.iloc[te])
        fold[te] = k
    return oof, fold
