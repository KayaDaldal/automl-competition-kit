"""Scoring functions.

Every `score_*` function follows a higher-is-better convention, so the blend
search and the model selection can maximise without knowing which metric they
are looking at. Error metrics are therefore returned negated. `report_value`
turns an internal score back into the number a human expects to read.
"""


def score_classification(y, proba, metric, n_classes):
    """metric: 'logloss' | 'accuracy' | 'f1' | 'auc'. Higher is better."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

    p = np.clip(proba, 1e-15, 1 - 1e-15)
    if metric == "auc":
        if n_classes == 2:
            return roc_auc_score(y, p[:, 1])
        return roc_auc_score(y, p, multi_class="ovr")
    if metric in ("accuracy", "f1"):
        pred = (p[:, 1] >= 0.5).astype(int) if n_classes == 2 else p.argmax(1)
        if metric == "accuracy":
            return accuracy_score(y, pred)
        return f1_score(y, pred, average="binary" if n_classes == 2 else "macro")
    return -log_loss(y, p, labels=list(range(n_classes)))


def score_regression(y, pred, metric):
    """metric: 'rmse' | 'mae' | 'rmsle' | 'r2'. Higher is better."""
    import numpy as np
    from sklearn.metrics import r2_score

    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if metric == "mae":
        return -float(np.mean(np.abs(y - pred)))
    if metric == "rmsle":
        p = np.clip(pred, 0, None)
        return -float(
            np.sqrt(np.mean((np.log1p(p) - np.log1p(np.clip(y, 0, None))) ** 2))
        )
    if metric == "r2":
        return float(r2_score(y, pred))
    return -float(np.sqrt(np.mean((y - pred) ** 2)))


def score_forecast(y, pred, metric):
    """metric: 'rmse' | 'mae' | 'rmsle' | 'smape'. Higher is better.

    Always evaluated in the original target space, never in log space.
    """
    import numpy as np

    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if metric == "mae":
        return -float(np.mean(np.abs(y - pred)))
    if metric == "rmsle":
        p = np.clip(pred, 0, None)
        return -float(
            np.sqrt(np.mean((np.log1p(p) - np.log1p(np.clip(y, 0, None))) ** 2))
        )
    if metric == "smape":
        denom = np.abs(y) + np.abs(pred)
        diff = 2.0 * np.abs(y - pred) / np.where(denom == 0, 1.0, denom)
        return -float(100.0 * np.mean(diff))
    return -float(np.sqrt(np.mean((y - pred) ** 2)))


def report_value(metric, score):
    """Turn an internal (higher-is-better) score into a human-readable value."""
    return score if metric == "r2" else -score


def best_threshold(y, p1, metric):
    """Grid-search the binary decision threshold on out-of-fold predictions.

    Only used for 'accuracy' and 'f1'; probability metrics keep 0.5.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (p1 >= t).astype(int)
        s = accuracy_score(y, pred) if metric == "accuracy" else f1_score(y, pred)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t)


def logloss(y_true, probs, n_classes):
    """Plain (positive) log loss — used by the image engine's epoch printout."""
    import numpy as np
    from sklearn.metrics import log_loss

    p = np.clip(probs, 1e-15, 1 - 1e-15)
    return log_loss(y_true, p, labels=list(range(n_classes)))


def accuracy(y_true, probs):
    import numpy as np

    return float((probs.argmax(1) == np.asarray(y_true)).mean())
