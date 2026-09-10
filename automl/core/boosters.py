"""LightGBM / XGBoost / CatBoost fitting, shared by the classification and
regression engines.

The two engines used to carry their own copies of `_fit_lgbm`, `_fit_xgb`,
`_fit_catboost` and `_gpu_caps`. The classifier and regressor variants are
merged here behind a `kind` argument rather than collapsed into one behaviour:
the parameters that genuinely differ (objective, loss, class weighting) stay
explicit.
"""


def gpu_caps():
    """Probe whether XGBoost / CatBoost can actually train on the GPU.

    Runs a three-iteration fit on random data and catches everything: a machine
    with a driver but no usable build falls back to CPU instead of crashing
    mid-run. LightGBM is always CPU here — the common pip wheel has no GPU
    support.
    """
    caps = {"xgb": False, "cat": False}
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        import shutil as _sh

        cuda = _sh.which("nvidia-smi") is not None
    if not cuda:
        print("[gpu] no CUDA -> boosters run on CPU.")
        return caps

    import numpy as np

    Xp = np.random.rand(32, 4).astype("float32")
    yp = (np.random.rand(32) > 0.5).astype(int)
    try:
        import xgboost as xgb

        xgb.XGBClassifier(
            n_estimators=3, device="cuda", tree_method="hist", verbosity=0
        ).fit(Xp, yp)
        caps["xgb"] = True
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier

        CatBoostClassifier(
            iterations=3, task_type="GPU", verbose=0, allow_writing_files=False
        ).fit(Xp, yp)
        caps["cat"] = True
    except Exception:
        pass
    active = ", ".join([k for k, v in caps.items() if v]) or "(none)"
    print("[gpu] CUDA available. GPU boosters:", active, "| LightGBM: CPU.")
    return caps


def fit_lgbm(Xtr, ytr, Xval, yval, cfg, kind, n_classes=None, spw=None, l1=False):
    """kind: 'classification' | 'regression'."""
    import lightgbm as lgb

    params = dict(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=cfg["seed"],
        n_jobs=-1,
        verbose=-1,
    )
    callbacks = [
        lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False),
        lgb.log_evaluation(0),
    ]
    if kind == "classification":
        if n_classes == 2 and spw is not None:
            params["scale_pos_weight"] = spw
        elif n_classes is not None and n_classes > 2 and spw is not None:
            params["class_weight"] = "balanced"
        model = lgb.LGBMClassifier(**params)
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], callbacks=callbacks)
        return model

    params["objective"] = "mae" if l1 else "regression"
    model = lgb.LGBMRegressor(**params)
    model.fit(
        Xtr,
        ytr,
        eval_set=[(Xval, yval)],
        eval_metric="l1" if l1 else "l2",
        callbacks=callbacks,
    )
    return model


def fit_xgb(Xtr, ytr, Xval, yval, cfg, kind, n_classes=None, spw=None, l1=False):
    """kind: 'classification' | 'regression'."""
    import xgboost as xgb

    params = dict(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=cfg["seed"],
        n_jobs=-1,
        early_stopping_rounds=cfg["early_stopping_rounds"],
    )
    if cfg.get("_gpu_caps", {}).get("xgb"):
        params["device"] = "cuda"

    if kind == "classification":
        params["eval_metric"] = "logloss" if n_classes == 2 else "mlogloss"
        if n_classes == 2 and spw is not None:
            params["scale_pos_weight"] = spw
        model = xgb.XGBClassifier(**params)
        sw = None
        if n_classes is not None and n_classes > 2 and spw is not None:
            from sklearn.utils.class_weight import compute_sample_weight

            sw = compute_sample_weight("balanced", ytr)
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], sample_weight=sw, verbose=False)
        return model

    params["objective"] = "reg:absoluteerror" if l1 else "reg:squarederror"
    params["eval_metric"] = "mae" if l1 else "rmse"
    model = xgb.XGBRegressor(**params)
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return model


def fit_catboost(
    Xtr, ytr, Xval, yval, cat_idx, cfg, kind, n_classes=None, imbalanced=False, l1=False
):
    """kind: 'classification' | 'regression'. Categorical columns stay raw."""
    from catboost import Pool

    params = dict(
        iterations=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=cfg["seed"],
        early_stopping_rounds=cfg["early_stopping_rounds"],
        verbose=0,
        allow_writing_files=False,
    )
    if cfg.get("_gpu_caps", {}).get("cat"):
        params["task_type"] = "GPU"

    if kind == "classification":
        from catboost import CatBoostClassifier

        loss = "Logloss" if n_classes == 2 else "MultiClass"
        params["loss_function"] = loss
        params["eval_metric"] = loss
        if imbalanced:
            params["auto_class_weights"] = "Balanced"
        model = CatBoostClassifier(**params)
    else:
        from catboost import CatBoostRegressor

        loss = "MAE" if l1 else "RMSE"
        params["loss_function"] = loss
        params["eval_metric"] = loss
        model = CatBoostRegressor(**params)

    model.fit(
        Pool(Xtr, ytr, cat_features=cat_idx),
        eval_set=Pool(Xval, yval, cat_features=cat_idx),
        use_best_model=True,
    )
    return model


def predict_proba(model, X, n_classes, is_cat=False, cat_idx=None):
    """Always return an [N, n_classes] probability matrix."""
    import numpy as np

    if is_cat:
        from catboost import Pool

        p = model.predict_proba(Pool(X, cat_features=cat_idx))
    else:
        p = model.predict_proba(X)
    p = np.asarray(p)
    if p.ndim == 1:
        p = np.vstack([1 - p, p]).T
    return p


def predict(model, X, is_cat=False, cat_idx=None):
    import numpy as np

    if is_cat:
        from catboost import Pool

        return np.asarray(model.predict(Pool(X, cat_features=cat_idx)), dtype="float32")
    return np.asarray(model.predict(X), dtype="float32")
