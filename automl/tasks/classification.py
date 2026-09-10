# ==========================================================================
#  Tabular classification engine.
#  Interface: run_classification(cfg) -> dict
#
#  LightGBM + XGBoost + CatBoost, stratified K-fold, OOF-weighted blend.
#  Target encoding is fitted inside each fold; CatBoost sees raw categoricals.
#  Binary and multiclass are detected from the data. Writes a submission on
#  its own. Fixed seed, wall-clock budget, lazy imports.
# ==========================================================================

import gc
import time
import warnings

from ..core.blend import blend_weights, weighted
from ..core.boosters import (
    fit_catboost,
    fit_lgbm,
    fit_xgb,
    gpu_caps,
    predict_proba,
)
from ..core.config import merge_defaults
from ..core.io import read_table, try_download
from ..core.metrics import best_threshold, score_classification
from ..core.prep import (
    detect_date_cols,
    fold_safe_target_encode,
    guess_col,
    parse_dates,
    positive_index,
    set_seed,
)
from ..core.submission import build_submission


# --------------------------------------------------------------------------
# 1) DEFAULTS
# --------------------------------------------------------------------------
def default_classification_cfg():
    return {
        "train_path": None,        # training CSV/parquet path OR DataFrame
        "test_path": None,         # test CSV/parquet path OR DataFrame
        "target_col": None,        # label column — required
        "id_col": None,            # submission key (falls back to row index)
        "drop_cols": None,         # columns to ignore (list)
        "metric": "logloss",       # "logloss" | "accuracy" | "f1" | "auc"
        "n_folds": 5,
        "seed": 42,
        "high_card_threshold": 20, # above this = high cardinality (target enc.)
        "te_smoothing": 10.0,      # target-encoding smoothing
        "n_estimators": 10000,
        "learning_rate": 0.01,
        "early_stopping_rounds": 300,
        "time_budget_sec": 5000,   # wall-clock cap; early stopping usually hits first
        "blend_iters": 600,        # random draws in the OOF blend weight search
        # --- output ---
        "submission_example": None,
        "out_dir": ".",
        "download": True,
        "positive_class": None,    # binary: name of the positive class
        "result_as_proba": True,   # probability for logloss/auc, label otherwise
    }


# --------------------------------------------------------------------------
# 2) QUICK EDA
# --------------------------------------------------------------------------
def _auto_eda(df, target_col, feat_cols, num_cols, cat_cols, date_cols):
    print("---------- QUICK EDA ----------")
    print("rows: %d | features: %d  (numeric %d, categorical %d, date %d)"
          % (len(df), len(feat_cols), len(num_cols), len(cat_cols), len(date_cols)))
    vc = df[target_col].value_counts(dropna=False)
    dist = ", ".join("%s:%d" % (str(k), int(v)) for k, v in vc.head(8).items())
    print("target distribution: %s" % dist)
    miss = df[feat_cols].isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        print("missing (top 6):",
              ", ".join("%s=%d" % (k, int(v)) for k, v in miss.head(6).items()))
    hc = [c for c in cat_cols if df[c].nunique() > 20]
    if hc:
        print("high-cardinality categoricals:", hc[:8])
    print("-------------------------------")


# --------------------------------------------------------------------------
# 3) ENGINE
# --------------------------------------------------------------------------
def run_classification(cfg=None):
    """
    Tabular classification. Interface: run_classification(cfg) -> dict.

    Returns {submission, submission_files, cv_score, metric, time_sec,
             class_names, blend_weights, threshold, test_probs (DataFrame),
             feature_importance (DataFrame), n_folds_done}
    """
    warnings.filterwarnings("ignore")
    t0 = time.time()
    cfg = merge_defaults(cfg, default_classification_cfg())
    set_seed(cfg["seed"])
    cfg["_gpu_caps"] = gpu_caps()

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import OrdinalEncoder

    def _score(y, p, n_classes):
        return score_classification(y, p, cfg["metric"], n_classes)

    # --- data ---
    train = read_table(cfg["train_path"])
    test = read_table(cfg["test_path"])
    target_col = cfg["target_col"] or guess_col(
        train.columns, ["target", "label", "class", "y"])
    if target_col is None or target_col not in train.columns:
        raise ValueError("target_col not found; set cfg['target_col'].")
    id_col = cfg["id_col"]
    drop_cols = list(cfg["drop_cols"] or [])

    if id_col and id_col in test.columns:
        ids = test[id_col].tolist()
    else:
        ids = list(range(len(test)))

    # --- label encoding ---
    class_names = sorted([str(v) for v in train[target_col].dropna().unique()])
    cls2idx = {c: i for i, c in enumerate(class_names)}
    y = train[target_col].astype(str).map(cls2idx).values
    n_classes = len(class_names)

    # --- feature columns ---
    not_feat = set([target_col] + drop_cols + ([id_col] if id_col else []))
    feat_cols = [c for c in train.columns if c not in not_feat]
    for c in feat_cols:                     # align: columns missing from test
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

    _auto_eda(train, target_col, feat_cols, num_cols, cat_cols, date_cols)

    # --- imputation (medians from train; categoricals -> "missing") ---
    medians = {c: float(train[c].median()) if train[c].notna().any() else 0.0
               for c in num_cols}
    for c in num_cols:
        train[c] = train[c].fillna(medians[c]).astype("float32")
        test[c] = pd.to_numeric(test[c], errors="coerce").fillna(
            medians[c]).astype("float32")
    for c in cat_cols:
        train[c] = train[c].astype(str).fillna("missing").replace("nan", "missing")
        test[c] = test[c].astype(str).fillna("missing").replace("nan", "missing")

    # --- split categoricals by cardinality ---
    low_card = [c for c in cat_cols if train[c].nunique() <= cfg["high_card_threshold"]]
    high_card = [c for c in cat_cols if train[c].nunique() > cfg["high_card_threshold"]]

    # --- ordinal encoding for the low-cardinality ones (LGB/XGB) ---
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

    # LGB/XGB base matrix: numeric + ordinal [+ per-fold target encoding]
    base_num_cols = num_cols + ord_cols
    Xnum_tr = np.hstack([train[num_cols].values.astype("float32"), train_ord])
    Xnum_te = np.hstack([test[num_cols].values.astype("float32"), test_ord])

    # CatBoost gets the raw frame; all categoricals stay strings
    cat_all = low_card + high_card
    cb_feat = num_cols + cat_all
    cb_train = train[cb_feat].copy()
    cb_test = test[cb_feat].copy()
    cat_idx = [cb_feat.index(c) for c in cat_all]

    # --- class imbalance ---
    counts = np.bincount(y, minlength=n_classes).astype(float)
    ratio = counts.max() / max(counts.min(), 1)
    imbalanced = ratio > 1.3
    spw = None
    if imbalanced and n_classes == 2:
        spw = float(counts[0] / max(counts[1], 1))   # negative / positive
    if imbalanced:
        print("[info] class imbalance (ratio %.2f) — weighting enabled." % ratio)
    print("[info] classes:", class_names, "| metric:", cfg["metric"],
          "| numeric:", len(num_cols), "| low-card:", len(low_card),
          "| high-card:", len(high_card))

    # --- K-fold ---
    folds = max(2, min(cfg["n_folds"], int(counts.min())))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg["seed"])

    model_names = ["lgbm", "xgb", "catboost"]
    oof = {m: np.zeros((len(train), n_classes), dtype="float32") for m in model_names}
    test_pred = {m: np.zeros((len(test), n_classes), dtype="float32") for m in model_names}
    importances = np.zeros(len(base_num_cols)) if base_num_cols else None
    done = 0

    for fold, (tr_i, va_i) in enumerate(skf.split(Xnum_tr, y)):
        if done >= 1 and (time.time() - t0) > cfg["time_budget_sec"]:
            print("[info] time budget spent — continuing with %d fold(s)." % done)
            break
        print("===== FOLD %d/%d =====" % (fold, folds))
        ytr, yva = y[tr_i], y[va_i]

        # --- LGB/XGB matrix: leakage-safe target encoding for high cardinality ---
        Xtr = Xnum_tr[tr_i].copy()
        Xva = Xnum_tr[va_i].copy()
        Xte = Xnum_te.copy()
        if high_card and n_classes == 2:
            te_tr_cols, te_va_cols, te_te_cols = [], [], []
            ybin = (ytr == positive_index(class_names, cfg["positive_class"])).astype(int)
            for c in high_card:
                a, b, d = fold_safe_target_encode(
                    train[c].values[tr_i], ybin, train[c].values[va_i],
                    test[c].values, cfg["te_smoothing"])
                te_tr_cols.append(a); te_va_cols.append(b); te_te_cols.append(d)
            Xtr = np.hstack([Xtr] + [c.reshape(-1, 1) for c in te_tr_cols])
            Xva = np.hstack([Xva] + [c.reshape(-1, 1) for c in te_va_cols])
            Xte = np.hstack([Xte] + [c.reshape(-1, 1) for c in te_te_cols])
        elif high_card:
            # multiclass: fall back to ordinal, still fitted on fold-train only
            oe2 = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            oe2.fit(train[high_card].iloc[tr_i])
            Xtr = np.hstack([Xtr, oe2.transform(train[high_card].iloc[tr_i]).astype("float32")])
            Xva = np.hstack([Xva, oe2.transform(train[high_card].iloc[va_i]).astype("float32")])
            Xte = np.hstack([Xte, oe2.transform(test[high_card]).astype("float32")])

        # --- LightGBM ---
        m_lgb = fit_lgbm(Xtr, ytr, Xva, yva, cfg, "classification",
                         n_classes=n_classes, spw=spw)
        oof["lgbm"][va_i] = predict_proba(m_lgb, Xva, n_classes)
        test_pred["lgbm"] += predict_proba(m_lgb, Xte, n_classes)
        if importances is not None and len(m_lgb.feature_importances_) >= len(base_num_cols):
            importances += m_lgb.feature_importances_[:len(base_num_cols)]

        # --- XGBoost ---
        m_xgb = fit_xgb(Xtr, ytr, Xva, yva, cfg, "classification",
                        n_classes=n_classes, spw=spw)
        oof["xgb"][va_i] = predict_proba(m_xgb, Xva, n_classes)
        test_pred["xgb"] += predict_proba(m_xgb, Xte, n_classes)

        # --- CatBoost (raw categoricals) ---
        m_cb = fit_catboost(cb_train.iloc[tr_i], ytr, cb_train.iloc[va_i], yva,
                            cat_idx, cfg, "classification",
                            n_classes=n_classes, imbalanced=imbalanced)
        oof["catboost"][va_i] = predict_proba(m_cb, cb_train.iloc[va_i], n_classes,
                                              is_cat=True, cat_idx=cat_idx)
        test_pred["catboost"] += predict_proba(m_cb, cb_test, n_classes,
                                               is_cat=True, cat_idx=cat_idx)

        done += 1
        for s in model_names:
            print("   [%s] fold %d  %s=%.5f  (%.0fs)"
                  % (s, fold, cfg["metric"],
                     _score(yva, oof[s][va_i], n_classes), time.time() - t0))

        del m_lgb, m_xgb, m_cb
        gc.collect()

    for m in model_names:
        test_pred[m] /= max(done, 1)

    # --- OOF blend ---
    mask = oof["lgbm"].sum(1) > 0          # rows that were actually scored
    oof_list = [oof[m][mask] for m in model_names]
    w, _ = blend_weights(oof_list, y[mask],
                         lambda yy, pp: _score(yy, pp, n_classes),
                         cfg["blend_iters"], cfg["seed"])
    blend_w = {m: float(wi) for m, wi in zip(model_names, w)}

    blend_oof = weighted([oof[m] for m in model_names], w)
    blend_test = weighted([test_pred[m] for m in model_names], w)

    for m in model_names:
        print("[score] %-9s OOF %s = %.5f"
              % (m, cfg["metric"], _score(y[mask], oof[m][mask], n_classes)))
    cv_score = _score(y[mask], blend_oof[mask], n_classes)

    # --- decision threshold (accuracy/f1, binary only) ---
    threshold = 0.5
    if n_classes == 2 and cfg["metric"] in ("accuracy", "f1"):
        pos_idx = positive_index(class_names, cfg["positive_class"])
        ypos = (y[mask] == pos_idx).astype(int)
        threshold = best_threshold(ypos, blend_oof[mask][:, pos_idx], cfg["metric"])
        print("[info] best OOF threshold (%s): %.3f" % (cfg["metric"], threshold))

    # --- feature importance ---
    fi_df = None
    if importances is not None and base_num_cols:
        order = np.argsort(importances)[::-1]
        fi_df = pd.DataFrame({"feature": [base_num_cols[i] for i in order],
                              "importance": importances[order]})
        print("[importance] top 8:", ", ".join(fi_df["feature"].head(8).tolist()))

    # --- submission ---
    sub = build_submission(cfg, "classification", ids, proba=blend_test,
                           class_names=class_names, threshold=threshold)
    elapsed = time.time() - t0

    print("\n================ SUMMARY (classification) ================")
    print("blend weights : %s" % blend_w)
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
        "class_names": class_names,
        "blend_weights": blend_w,
        "threshold": threshold,
        "test_probs": sub["prob_df"],
        "feature_importance": fi_df,
        "n_folds_done": done,
    }
