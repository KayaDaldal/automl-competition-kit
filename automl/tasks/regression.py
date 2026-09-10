# ==========================================================================
#  Tabular regression engine.
#  Interface: run_regression(cfg) -> dict
#
#  Covers numeric targets without a time axis (a dated series goes to the
#  forecast engine instead).
#  LightGBM + XGBoost + CatBoost regressors, binned-stratified K-fold,
#  OOF-weighted blend. log1p on skewed targets (and for RMSLE), reversed with
#  expm1 before scoring, so every score is in the original target space.
#  Preprocessing is the same as the classification engine, and leakage-safe.
# ==========================================================================

import gc
import time
import warnings

from ..core.blend import blend_weights, weighted
from ..core.boosters import fit_catboost, fit_lgbm, fit_xgb, gpu_caps, predict
from ..core.config import merge_defaults
from ..core.io import read_table, try_download
from ..core.metrics import report_value, score_regression
from ..core.prep import (
    detect_date_cols,
    fold_safe_target_encode,
    guess_col,
    parse_dates,
    set_seed,
)
from ..core.submission import build_submission


# --------------------------------------------------------------------------
# 1) DEFAULTS
# --------------------------------------------------------------------------
def default_regression_cfg():
    return {
        "train_path": None,        # CSV/parquet path OR DataFrame
        "test_path": None,
        "target_col": None,        # required
        "id_col": None,
        "drop_cols": None,
        "metric": "rmse",          # "rmse" | "mae" | "rmsle" | "r2"
        "n_folds": 5,
        "seed": 42,
        "high_card_threshold": 20,
        "te_smoothing": 10.0,
        "n_estimators": 10000,
        "learning_rate": 0.01,
        "early_stopping_rounds": 300,
        "time_budget_sec": 5000,
        "blend_iters": 600,
        # --- target transform / clipping ---
        "log_target": "auto",      # "auto" | True | False
        "skew_threshold": 1.0,     # |skew| above this triggers log1p in auto mode
        "non_negative": "auto",    # "auto" | True | False  (clip predictions at 0)
        "stratify_bins": True,     # bin the target for stratified K-fold
        # --- output ---
        "submission_example": None,
        "out_dir": ".",
        "download": True,
    }


# --------------------------------------------------------------------------
# 2) QUICK EDA
# --------------------------------------------------------------------------
def _auto_eda(df, target_col, feat_cols, num_cols, cat_cols, date_cols):
    y = df[target_col].astype(float)
    skew = float(y.skew())
    print("---------- QUICK EDA ----------")
    print("rows: %d | features: %d  (numeric %d, categorical %d, date %d)"
          % (len(df), len(feat_cols), len(num_cols), len(cat_cols), len(date_cols)))
    print("target: min=%.4g  median=%.4g  mean=%.4g  max=%.4g  skew=%.3f"
          % (y.min(), y.median(), y.mean(), y.max(), skew))
    miss = df[feat_cols].isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        print("missing (top 6):",
              ", ".join("%s=%d" % (k, int(v)) for k, v in miss.head(6).items()))
    hc = [c for c in cat_cols if df[c].nunique() > 20]
    if hc:
        print("high-cardinality categoricals:", hc[:8])
    print("-------------------------------")
    return skew


def _use_l1(metric):
    return metric == "mae"


# --------------------------------------------------------------------------
# 3) ENGINE
# --------------------------------------------------------------------------
def run_regression(cfg=None):
    """
    Tabular regression. Interface: run_regression(cfg) -> dict.

    Returns {submission, submission_files, cv_score, metric, time_sec,
             blend_weights, log_target, test_probs (DataFrame),
             feature_importance (DataFrame), n_folds_done}
    """
    warnings.filterwarnings("ignore")
    t0 = time.time()
    cfg = merge_defaults(cfg, default_regression_cfg())
    set_seed(cfg["seed"])
    cfg["_gpu_caps"] = gpu_caps()

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.preprocessing import OrdinalEncoder

    def _score(y, p):
        return score_regression(y, p, cfg["metric"])

    # --- data ---
    train = read_table(cfg["train_path"])
    test = read_table(cfg["test_path"])
    target_col = cfg["target_col"] or guess_col(
        train.columns, ["target", "y", "value", "price"])
    if target_col is None or target_col not in train.columns:
        raise ValueError("target_col not found; set cfg['target_col'].")
    id_col = cfg["id_col"]
    drop_cols = list(cfg["drop_cols"] or [])

    ids = test[id_col].tolist() if (id_col and id_col in test.columns) \
        else list(range(len(test)))

    y_orig = train[target_col].astype(float).values

    # --- feature columns + alignment ---
    not_feat = set([target_col] + drop_cols + ([id_col] if id_col else []))
    feat_cols = [c for c in train.columns if c not in not_feat]
    for c in feat_cols:
        if c not in test.columns:
            test[c] = np.nan

    # --- dates ---
    date_cols = detect_date_cols(train, feat_cols)
    new_num = parse_dates(train, test, date_cols) if date_cols else []
    feat_cols = [c for c in feat_cols if c not in date_cols] + new_num

    # --- column types ---
    num_cols, cat_cols = [], []
    for c in feat_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    skew = _auto_eda(train, target_col, feat_cols, num_cols, cat_cols, date_cols)

    # --- target transform decision ---
    can_log = float(np.nanmin(y_orig)) >= 0
    if cfg["log_target"] == "auto":
        log_target = can_log and (cfg["metric"] == "rmsle"
                                  or abs(skew) > cfg["skew_threshold"])
    else:
        log_target = bool(cfg["log_target"]) and can_log
    if cfg["metric"] == "rmsle" and not can_log:
        print("[warning] RMSLE is undefined for negative targets; log transform off.")
    y_work = np.log1p(y_orig) if log_target else y_orig.copy()
    print("[info] log1p target transform:", log_target, "| metric:", cfg["metric"])

    # --- can predictions be negative? ---
    non_negative = can_log if cfg["non_negative"] == "auto" else bool(cfg["non_negative"])

    # --- imputation ---
    medians = {c: float(train[c].median()) if train[c].notna().any() else 0.0
               for c in num_cols}
    for c in num_cols:
        train[c] = train[c].fillna(medians[c]).astype("float32")
        test[c] = pd.to_numeric(test[c], errors="coerce").fillna(
            medians[c]).astype("float32")
    for c in cat_cols:
        train[c] = train[c].astype(str).fillna("missing").replace("nan", "missing")
        test[c] = test[c].astype(str).fillna("missing").replace("nan", "missing")

    low_card = [c for c in cat_cols if train[c].nunique() <= cfg["high_card_threshold"]]
    high_card = [c for c in cat_cols if train[c].nunique() > cfg["high_card_threshold"]]

    # --- ordinal encoding (low cardinality; LGB/XGB) ---
    ord_cols = low_card[:]
    if ord_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                            encoded_missing_value=-1)
        oe.fit(train[ord_cols])
        train_ord = oe.transform(train[ord_cols]).astype("float32")
        test_ord = oe.transform(test[ord_cols]).astype("float32")
    else:
        train_ord = np.empty((len(train), 0), dtype="float32")
        test_ord = np.empty((len(test), 0), dtype="float32")

    base_num_cols = num_cols + ord_cols
    Xnum_tr = np.hstack([train[num_cols].values.astype("float32"), train_ord])
    Xnum_te = np.hstack([test[num_cols].values.astype("float32"), test_ord])

    # CatBoost raw frame
    cat_all = low_card + high_card
    cb_feat = num_cols + cat_all
    cb_train = train[cb_feat].copy()
    cb_test = test[cb_feat].copy()
    cat_idx = [cb_feat.index(c) for c in cat_all]

    print("[info] numeric:", len(num_cols), "| low-card:", len(low_card),
          "| high-card:", len(high_card), "| non_negative:", non_negative)

    # --- K-fold, optionally stratified on binned target ---
    folds = max(2, min(cfg["n_folds"], len(train)))
    use_strat = cfg["stratify_bins"] and len(train) >= folds * 5
    if use_strat:
        try:
            q = min(10, max(2, len(np.unique(y_orig)) - 1))
            bins = pd.qcut(pd.Series(y_orig), q=q, duplicates="drop",
                           labels=False).fillna(0).astype(int).values
            splits = list(StratifiedKFold(n_splits=folds, shuffle=True,
                                          random_state=cfg["seed"]).split(Xnum_tr, bins))
        except Exception:
            splits = list(KFold(n_splits=folds, shuffle=True,
                                random_state=cfg["seed"]).split(Xnum_tr))
    else:
        splits = list(KFold(n_splits=folds, shuffle=True,
                            random_state=cfg["seed"]).split(Xnum_tr))

    l1 = _use_l1(cfg["metric"])
    model_names = ["lgbm", "xgb", "catboost"]
    oof = {m: np.zeros(len(train), dtype="float32") for m in model_names}   # original space
    test_pred = {m: np.zeros(len(test), dtype="float32") for m in model_names}
    importances = np.zeros(len(base_num_cols)) if base_num_cols else None
    done = 0

    def _to_orig(p_work):
        p = np.expm1(p_work) if log_target else p_work
        if non_negative:
            p = np.clip(p, 0, None)
        return p

    for fold, (tr_i, va_i) in enumerate(splits):
        if done >= 1 and (time.time() - t0) > cfg["time_budget_sec"]:
            print("[info] time budget spent — continuing with %d fold(s)." % done)
            break
        print("===== FOLD %d/%d =====" % (fold, folds))
        ytr_w, yva_w = y_work[tr_i], y_work[va_i]
        yva_orig = y_orig[va_i]

        # --- leakage-safe target encoding for high-cardinality columns ---
        Xtr = Xnum_tr[tr_i].copy(); Xva = Xnum_tr[va_i].copy(); Xte = Xnum_te.copy()
        if high_card:
            te_tr, te_va, te_te = [], [], []
            for c in high_card:
                a, b, d = fold_safe_target_encode(
                    train[c].values[tr_i], ytr_w, train[c].values[va_i],
                    test[c].values, cfg["te_smoothing"])
                te_tr.append(a); te_va.append(b); te_te.append(d)
            Xtr = np.hstack([Xtr] + [c.reshape(-1, 1) for c in te_tr])
            Xva = np.hstack([Xva] + [c.reshape(-1, 1) for c in te_va])
            Xte = np.hstack([Xte] + [c.reshape(-1, 1) for c in te_te])

        # --- LightGBM ---
        m_lgb = fit_lgbm(Xtr, ytr_w, Xva, yva_w, cfg, "regression", l1=l1)
        oof["lgbm"][va_i] = _to_orig(predict(m_lgb, Xva))
        test_pred["lgbm"] += _to_orig(predict(m_lgb, Xte))
        if importances is not None and len(m_lgb.feature_importances_) >= len(base_num_cols):
            importances += m_lgb.feature_importances_[:len(base_num_cols)]

        # --- XGBoost ---
        m_xgb = fit_xgb(Xtr, ytr_w, Xva, yva_w, cfg, "regression", l1=l1)
        oof["xgb"][va_i] = _to_orig(predict(m_xgb, Xva))
        test_pred["xgb"] += _to_orig(predict(m_xgb, Xte))

        # --- CatBoost (raw categoricals) ---
        m_cb = fit_catboost(cb_train.iloc[tr_i], ytr_w, cb_train.iloc[va_i], yva_w,
                            cat_idx, cfg, "regression", l1=l1)
        oof["catboost"][va_i] = _to_orig(
            predict(m_cb, cb_train.iloc[va_i], is_cat=True, cat_idx=cat_idx))
        test_pred["catboost"] += _to_orig(
            predict(m_cb, cb_test, is_cat=True, cat_idx=cat_idx))

        done += 1
        for s in model_names:
            print("   [%s] fold %d  %s=%.5f  (%.0fs)"
                  % (s, fold, cfg["metric"],
                     report_value(cfg["metric"], _score(yva_orig, oof[s][va_i])),
                     time.time() - t0))

        del m_lgb, m_xgb, m_cb
        gc.collect()

    for m in model_names:
        test_pred[m] /= max(done, 1)

    # --- OOF blend (original space) ---
    mask = np.zeros(len(train), dtype=bool)
    for _, va_i in splits[:done]:
        mask[va_i] = True
    oof_list = [oof[m][mask] for m in model_names]
    w, _ = blend_weights(oof_list, y_orig[mask], _score, cfg["blend_iters"], cfg["seed"])
    blend_w = {m: float(wi) for m, wi in zip(model_names, w)}

    blend_test = weighted([test_pred[m] for m in model_names], w)
    if non_negative:
        blend_test = np.clip(blend_test, 0, None)

    for m in model_names:
        print("[score] %-9s OOF %s = %.5f"
              % (m, cfg["metric"],
                 report_value(cfg["metric"], _score(y_orig[mask], oof[m][mask]))))
    blend_oof = weighted([oof[m][mask] for m in model_names], w)
    cv_score = report_value(cfg["metric"], _score(y_orig[mask], blend_oof))

    # --- feature importance ---
    fi_df = None
    if importances is not None and base_num_cols:
        order = np.argsort(importances)[::-1]
        fi_df = pd.DataFrame({"feature": [base_num_cols[i] for i in order],
                              "importance": importances[order]})
        print("[importance] top 8:", ", ".join(fi_df["feature"].head(8).tolist()))

    # --- submission ---
    sub = build_submission(cfg, "regression", ids, pred=blend_test)
    elapsed = time.time() - t0

    print("\n================== SUMMARY (regression) ==================")
    print("blend weights : %s" % blend_w)
    print("log1p target  : %s | non_negative: %s" % (log_target, non_negative))
    print("CV %-10s : %.5f  (%d folds, blended)" % (cfg["metric"], cv_score, done))
    print("elapsed       : %.1f s" % elapsed)
    print("files written : %s" % ", ".join(sub["files"]))
    print("==========================================================\n")

    try_download([sub["files"][0]], cfg["download"])

    return {
        "submission": sub["primary"],
        "submission_files": sub["files"],
        "cv_score": cv_score,
        "metric": cfg["metric"],
        "time_sec": elapsed,
        "blend_weights": blend_w,
        "log_target": log_target,
        "test_probs": sub["prob_df"],
        "feature_importance": fi_df,
        "n_folds_done": done,
    }
