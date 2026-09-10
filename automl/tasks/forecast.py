# ==========================================================================
#  Time-series forecasting engine.
#  Interface: run_forecast(cfg) -> dict
#
#  Use this when the data has a date/order column and the test rows are in the
#  FUTURE. Without a time axis the problem belongs to the regression or the
#  classification engine.
#
#  Leakage rules this engine is built around:
#    - rows are never shuffled; validation is always a block of the latest dates
#    - lag and rolling features are computed inside each series and shifted,
#      so a row can only ever see its own past
#    - the recursive strategy feeds its own predictions forward, never the
#      true future values
#
#  Global LightGBM (main model) + seasonal-naive baseline; whichever wins on
#  the time-based validation block is the one used for the final forecast.
#  Single series or many (group_cols) is detected automatically.
# ==========================================================================

import time
import warnings

from ..core.config import merge_defaults
from ..core.io import read_table, try_download
from ..core.metrics import report_value, score_forecast
from ..core.prep import guess_col, set_seed
from ..core.submission import build_submission


# --------------------------------------------------------------------------
# 1) DEFAULTS
# --------------------------------------------------------------------------
def default_forecast_cfg():
    return {
        "train_path": None,        # CSV/parquet OR DataFrame
        "test_path": None,
        "date_col": None,          # required (date/time column)
        "target_col": None,        # required
        "id_col": None,            # submission key
        "group_cols": None,        # e.g. ["store", "item"]; None = one series
        "horizon": None,           # future steps; None = distinct dates in test
        "metric": "rmse",          # "rmse" | "mae" | "rmsle" | "smape"
        "seed": 42,
        # --- features ---
        "lags": [1, 2, 3, 7, 14, 28],
        "windows": [7, 14, 28, 56],
        "season_period": 7,        # period of the seasonal-naive baseline
        # --- model / CV ---
        "n_estimators": 6000,
        "learning_rate": 0.01,
        "early_stopping_rounds": 300,
        "strategy": "auto",        # "auto" | "recursive" | "direct"
        "cv_scheme": "holdout",    # "holdout" | "expanding"
        "cv_folds": 3,             # for expanding
        "use_prophet": False,      # optional, only if prophet is installed
        "non_negative": "auto",    # "auto" | True | False
        "time_budget_sec": 5000,
        # --- output ---
        "submission_example": None,
        "out_dir": ".",
        "download": True,
    }


# --------------------------------------------------------------------------
# 2) DATE + LAG/ROLLING FEATURES (within a series, past-only)
# --------------------------------------------------------------------------
def _date_feats(dt):
    import pandas as pd

    iso = dt.dt.isocalendar()
    return pd.DataFrame({
        "f_year": dt.dt.year, "f_month": dt.dt.month, "f_day": dt.dt.day,
        "f_dow": dt.dt.dayofweek, "f_week": iso["week"].astype(int),
        "f_quarter": dt.dt.quarter, "f_doy": dt.dt.dayofyear,
    }, index=dt.index)


def _lag_roll(df, target_col, gk, lags, windows):
    """Lags plus rolling stats of the shifted series, computed per group.

    The rolling windows are built on `shift(1)`, so the window for row t ends
    at t-1 and the current value can never leak into its own feature.
    """
    import pandas as pd

    out = {}
    g = df.groupby(gk, sort=False)[target_col]
    for k in lags:
        out["lag_%d" % k] = g.shift(k).values
    tmp = pd.DataFrame({"_sh": g.shift(1).values}, index=df.index)
    tmp[gk] = df[gk].values if isinstance(gk, str) else df[gk].values
    gg = tmp.groupby(gk, sort=False)["_sh"]
    for w in windows:
        out["rmean_%d" % w] = gg.transform(lambda s: s.rolling(w, min_periods=1).mean()).values
        out["rstd_%d" % w] = gg.transform(lambda s: s.rolling(w, min_periods=2).std()).values
        out["rmax_%d" % w] = gg.transform(lambda s: s.rolling(w, min_periods=1).max()).values
        out["rmin_%d" % w] = gg.transform(lambda s: s.rolling(w, min_periods=1).min()).values
    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------
# 3) GLOBAL LIGHTGBM FORECAST (recursive / direct)
# --------------------------------------------------------------------------
def _forecast_lgb(hist_df, future_df, cfg, gk, date_col, target_col, recursive):
    """
    hist_df: rows whose target is known. future_df: rows to predict (target NaN).
    Returns (predictions in future_df order, feature_importance_df | None).
    """
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder

    lags, windows = cfg["lags"], cfg["windows"]
    hist = hist_df.copy(); fut = future_df.copy()
    hist["_isfut"] = 0; fut["_isfut"] = 1
    if target_col not in fut.columns:
        fut[target_col] = np.nan
    comb = pd.concat([hist, fut], ignore_index=True)
    comb = comb.sort_values(gk + [date_col]).reset_index(drop=True)

    # --- static features: group encoding + calendar ---
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    grp_enc = oe.fit_transform(comb[gk].astype(str)).astype("float32")
    grp_cols = ["g_%d" % i for i in range(grp_enc.shape[1])]
    static = pd.DataFrame(grp_enc, columns=grp_cols, index=comb.index)
    static = pd.concat([static, _date_feats(comb[date_col])], axis=1)

    feat_dyn = ["lag_%d" % k for k in lags]
    for w in windows:
        feat_dyn += ["rmean_%d" % w, "rstd_%d" % w, "rmax_%d" % w, "rmin_%d" % w]
    feat_cols = grp_cols + list(_date_feats(comb[date_col]).columns) + feat_dyn

    # --- training features (true targets in the past, NaN in the future) ---
    work = comb[target_col].astype(float).values.copy()
    comb["_t"] = work
    dyn0 = _lag_roll(comb, "_t", gk, lags, windows)
    Xall = pd.concat([static, dyn0], axis=1)
    train_mask = (comb["_isfut"].values == 0) & (~np.isnan(work))
    model = lgb.LGBMRegressor(
        n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
        num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=cfg["seed"], n_jobs=-1, verbose=-1,
        objective="mae" if cfg["metric"] in ("mae", "smape") else "regression")
    model.fit(Xall.loc[train_mask, feat_cols], work[train_mask])

    fi = pd.DataFrame({"feature": feat_cols, "importance": model.feature_importances_}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)

    # --- predict ---
    fut_dates = np.sort(comb.loc[comb["_isfut"] == 1, date_col].unique())
    if recursive:
        for d in fut_dates:                     # in date order, feeding predictions back
            comb["_t"] = work
            dyn = _lag_roll(comb, "_t", gk, lags, windows)
            X = pd.concat([static, dyn], axis=1)[feat_cols]
            rows = np.where((comb["_isfut"].values == 1) & (comb[date_col].values == d))[0]
            work[rows] = model.predict(X.iloc[rows])
    else:                                       # direct / one shot
        comb["_t"] = work
        dyn = _lag_roll(comb, "_t", gk, lags, windows)
        X = pd.concat([static, dyn], axis=1)[feat_cols]
        rows = np.where(comb["_isfut"].values == 1)[0]
        work[rows] = model.predict(X.iloc[rows])

    # map back to future_df order
    comb["_pred"] = work
    fut_rows = comb[comb["_isfut"] == 1].copy()
    key_cols = gk + [date_col]
    merged = future_df.merge(fut_rows[key_cols + ["_pred"]], on=key_cols, how="left")
    return merged["_pred"].values, fi


# --------------------------------------------------------------------------
# 4) SEASONAL-NAIVE BASELINE
# --------------------------------------------------------------------------
def _seasonal_naive_seq(hist_vals, n_future, season):
    import numpy as np

    h = np.asarray(hist_vals, float)
    h = h[~np.isnan(h)]
    if len(h) == 0:
        return np.zeros(n_future)
    if len(h) < season:
        return np.full(n_future, h[-1])
    block = h[-season:]
    return np.array([block[i % season] for i in range(n_future)])


def _baseline_forecast(hist_df, future_df, gk, date_col, target_col, season):
    import numpy as np

    preds = np.zeros(len(future_df))
    fut = future_df.copy()
    fut["_oi"] = np.arange(len(fut))          # positional index, not the frame's index
    hist_sorted = hist_df.sort_values(gk + [date_col])
    hg = {k: v[target_col].values for k, v in hist_sorted.groupby(gk, sort=False)}
    for key, sub in fut.sort_values(gk + [date_col]).groupby(gk, sort=False):
        hv = hg.get(key, np.array([]))
        seq = _seasonal_naive_seq(hv, len(sub), season)
        preds[sub["_oi"].values] = seq
    return preds


# --------------------------------------------------------------------------
# 5) OPTIONAL PROPHET (only if installed; fitted on history only)
# --------------------------------------------------------------------------
def _prophet_forecast(hist_df, future_df, gk, date_col, target_col):
    import numpy as np
    import pandas as pd

    try:
        from prophet import Prophet
    except Exception:
        return None
    preds = np.full(len(future_df), np.nan)
    fut = future_df.reset_index(drop=False).rename(columns={"index": "_oi"})
    try:
        for key, sub in fut.groupby(gk, sort=False):
            hsub = hist_df[(hist_df[gk] == key).all(axis=1) if len(gk) > 1
                           else (hist_df[gk[0]] == key)]
            if len(hsub) < 10:
                continue
            dfp = pd.DataFrame({"ds": pd.to_datetime(hsub[date_col]),
                                "y": hsub[target_col].astype(float)})
            m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                        daily_seasonality=False)
            m.fit(dfp)
            fdf = pd.DataFrame({"ds": pd.to_datetime(sub[date_col])})
            preds[sub["_oi"].values] = m.predict(fdf)["yhat"].values
        return preds
    except Exception:
        return None


# --------------------------------------------------------------------------
# 6) ENGINE
# --------------------------------------------------------------------------
def run_forecast(cfg=None):
    """
    Time-series forecasting. Interface: run_forecast(cfg) -> dict.

    Returns {submission, submission_files, cv_score, metric, time_sec,
             chosen_model, strategy, horizon, test_probs (DataFrame),
             feature_importance, n_series}
    """
    warnings.filterwarnings("ignore")
    t0 = time.time()
    cfg = merge_defaults(cfg, default_forecast_cfg())
    set_seed(cfg["seed"])

    import numpy as np
    import pandas as pd

    train = read_table(cfg["train_path"])
    test = read_table(cfg["test_path"])
    # "tarih"/"zaman" are here for the same reason prep._DATE_KW has them:
    # cheap support for non-English date column names.
    date_col = cfg["date_col"] or guess_col(
        train.columns, ["date", "ds", "time", "timestamp", "tarih", "zaman"])
    target_col = cfg["target_col"] or guess_col(
        train.columns, ["target", "sales", "y", "demand", "value"])
    if date_col is None or target_col is None:
        raise ValueError("date_col and target_col are required (auto-detection failed).")
    id_col = cfg["id_col"]

    # one series vs many
    gk = list(cfg["group_cols"]) if cfg["group_cols"] else ["_grp"]
    if gk == ["_grp"]:
        train["_grp"] = "s0"; test["_grp"] = "s0"

    # parse dates and SORT (never shuffle)
    train[date_col] = pd.to_datetime(train[date_col])
    test[date_col] = pd.to_datetime(test[date_col])
    train = train.sort_values(gk + [date_col]).reset_index(drop=True)
    test = test.sort_values(gk + [date_col]).reset_index(drop=True)

    # submission key: keep the ORIGINAL test row order, not the sorted one
    test_out = read_table(cfg["test_path"])
    test_out[date_col] = pd.to_datetime(test_out[date_col])
    if gk == ["_grp"]:
        test_out["_grp"] = "s0"
    ids = test_out[id_col].tolist() if (id_col and id_col in test_out.columns) \
        else list(range(len(test_out)))

    n_series = test[gk].drop_duplicates().shape[0]
    horizon = cfg["horizon"] or int(test[date_col].nunique())
    y_min = float(train[target_col].min())
    non_negative = (y_min >= 0) if cfg["non_negative"] == "auto" \
        else bool(cfg["non_negative"])

    # strategy
    if cfg["strategy"] == "auto":
        recursive = horizon > 1
    else:
        recursive = (cfg["strategy"] == "recursive")
    strat = "recursive" if recursive else "direct"
    print("[info] series: %d | horizon: %d | strategy: %s | metric: %s | non_negative: %s"
          % (n_series, horizon, strat, cfg["metric"], non_negative))
    if recursive:
        print("   -> recursive: each period is predicted and written back, so the "
              "next period's lags use the prediction.")
    else:
        print("   -> direct: horizon=1, every future row is predicted from history "
              "in one pass.")

    # ---------- TIME-BASED CV (holdout / expanding) ----------
    uniq_dates = np.sort(train[date_col].unique())
    season = cfg["season_period"]

    def _eval_block(val_dates):
        val_start = val_dates.min()
        hist = train[train[date_col] < val_start]
        valdf = train[train[date_col].isin(val_dates)]
        if len(hist) < season + 1 or len(valdf) == 0:
            return None
        ytrue = valdf[target_col].astype(float).values
        rec = len(val_dates) > 1
        lgb_pred, _ = _forecast_lgb(hist, valdf.drop(columns=[target_col]),
                                    cfg, gk, date_col, target_col, rec)
        if non_negative:
            lgb_pred = np.clip(lgb_pred, 0, None)
        base_pred = _baseline_forecast(hist, valdf, gk, date_col, target_col, season)
        if non_negative:
            base_pred = np.clip(base_pred, 0, None)
        scores = {"lgb": score_forecast(ytrue, lgb_pred, cfg["metric"]),
                  "baseline": score_forecast(ytrue, base_pred, cfg["metric"])}
        if cfg["use_prophet"]:
            pp = _prophet_forecast(hist, valdf, gk, date_col, target_col)
            if pp is not None and not np.all(np.isnan(pp)):
                pp = np.where(np.isnan(pp), base_pred, pp)
                if non_negative:
                    pp = np.clip(pp, 0, None)
                scores["prophet"] = score_forecast(ytrue, pp, cfg["metric"])
        return scores

    if cfg["cv_scheme"] == "expanding":
        folds = max(1, cfg["cv_folds"])
        blocks = [uniq_dates[-(i + 1) * horizon: len(uniq_dates) - i * horizon]
                  for i in range(folds)]
        blocks = [b for b in blocks if len(b) > 0]
    else:
        blocks = [uniq_dates[-horizon:]]

    agg = {}
    for b in blocks:
        if time.time() - t0 > cfg["time_budget_sec"]:
            break
        s = _eval_block(b)
        if s is None:
            continue
        for k, v in s.items():
            agg.setdefault(k, []).append(v)
    cv_scores = {k: float(np.mean(v)) for k, v in agg.items()} if agg \
        else {"lgb": float("nan")}
    chosen = max(cv_scores, key=cv_scores.get)
    for k in cv_scores:
        print("[CV] %-9s %s = %.5f"
              % (k, cfg["metric"], report_value(cfg["metric"], cv_scores[k])))
    print("[info] chosen model:", chosen)

    # ---------- FINAL FORECAST (fitted on all of train) ----------
    fi_df = None
    if chosen == "baseline":
        final = _baseline_forecast(train, test, gk, date_col, target_col, season)
    elif chosen == "prophet":
        final = _prophet_forecast(train, test, gk, date_col, target_col)
        base = _baseline_forecast(train, test, gk, date_col, target_col, season)
        if final is None or np.all(np.isnan(final)):
            final = base
        else:
            final = np.where(np.isnan(final), base, final)
    else:
        final, fi_df = _forecast_lgb(
            train, test.drop(columns=[target_col], errors="ignore"),
            cfg, gk, date_col, target_col, recursive)
    if non_negative:
        final = np.clip(final, 0, None)

    # map the sorted-test prediction back onto the original test row order
    test_pred_df = test[gk + [date_col]].copy()
    test_pred_df["_pred"] = final
    key_cols = gk + [date_col]
    out_pred = test_out.merge(test_pred_df, on=key_cols, how="left")["_pred"].values
    out_pred = np.where(np.isnan(out_pred), float(np.nanmedian(final)), out_pred)

    if fi_df is not None:
        print("[importance] top 8:", ", ".join(fi_df["feature"].head(8).tolist()))

    sub = build_submission(cfg, "forecast", ids, pred=out_pred)
    elapsed = time.time() - t0
    cv_val = report_value(cfg["metric"], cv_scores.get(chosen, float("nan")))

    print("\n=================== SUMMARY (forecast) ===================")
    print("chosen model  : %s" % chosen)
    print("CV %-10s : %.5f" % (cfg["metric"], cv_val))
    print("strategy/hzn  : %s / %d | series: %d" % (strat, horizon, n_series))
    print("elapsed       : %.1f s" % elapsed)
    print("files written : %s" % ", ".join(sub["files"]))
    print("==========================================================\n")

    try_download([sub["files"][0]], cfg["download"])

    return {
        "submission": sub["primary"],
        "submission_files": sub["files"],
        "cv_score": cv_val,
        "metric": cfg["metric"],
        "time_sec": elapsed,
        "chosen_model": chosen,
        "strategy": strat,
        "horizon": horizon,
        "test_probs": sub["prob_df"],
        "feature_importance": fi_df,
        "n_series": n_series,
    }
