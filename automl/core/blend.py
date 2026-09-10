"""Weighted blending of model predictions, with weights fitted on out-of-fold
predictions only.

The weight search never touches the test set or any in-fold prediction, so a
blend that looks good here is not being tuned on data the models memorised.
"""


def weighted(preds, w):
    """Weighted sum of a list of prediction arrays (1-D or 2-D, all same shape)."""
    import numpy as np

    out = np.zeros_like(preds[0], dtype=float)
    for wi, p in zip(w, preds):
        out += wi * p
    return out


def blend_weights(oof_list, y, score_fn, n_iter, seed):
    """Search blend weights maximising `score_fn(y, blended)`.

    Candidates: the equal-weight average, each single model on its own, then
    `n_iter` random draws from a flat Dirichlet. Returns (weights, best_score).
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    k = len(oof_list)
    best_w = np.ones(k) / k
    best_s = score_fn(y, weighted(oof_list, best_w))
    for i in range(k):
        w = np.zeros(k)
        w[i] = 1.0
        s = score_fn(y, weighted(oof_list, w))
        if s > best_s:
            best_s, best_w = s, w
    for _ in range(n_iter):
        w = rng.dirichlet(np.ones(k))
        s = score_fn(y, weighted(oof_list, w))
        if s > best_s:
            best_s, best_w = s, w
    return best_w, best_s
