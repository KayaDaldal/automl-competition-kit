#!/usr/bin/env python3
"""Benchmark the automl engines on real public datasets.

What this measures, precisely:

  For every dataset we hold out 20% of the rows (the last H dates for time
  series), train only on the rest, and score the predictions against that
  held-out truth. The pipeline never sees it. Alongside that we record the
  CV score the pipeline reported, so the gap between "what it promised" and
  "what it delivered" is visible.

  Three systems are compared on identical splits:
    ours-fast   the fast preset
    ours-full   the full preset (capped by --full-budget seconds)
    baseline    a single LightGBM with default parameters, ordinal-encoded
                categoricals, median-imputed numerics
    leaky       the same single LightGBM, but with target encoding fitted on
                the whole training set — the ordinary shortcut. Its CV score
                and its held-out score are both recorded, and the gap between
                them is the thing this repo is built to avoid.

Usage:
    python benchmark/run_benchmark.py --check        # environment report only
    python benchmark/run_benchmark.py --offline      # no network, sklearn data
    python benchmark/run_benchmark.py                # everything in datasets.yaml
    python benchmark/run_benchmark.py --only telco-churn pima-diabetes
    python benchmark/run_benchmark.py --force        # redo datasets already done

Results are appended to benchmark/results.json after every dataset, so the run
is resumable: stop it with Ctrl-C, start it again, it picks up where it left.
"""

import argparse
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
RESULTS_JSON = os.path.join(HERE, "results.json")
RESULTS_MD = os.path.join(HERE, "RESULTS.md")

if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ==========================================================================
# environment
# ==========================================================================
def check_env():
    info = {"python": sys.version.split()[0], "platform": sys.platform, "packages": {},
            "problems": []}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost", "yaml"):
        try:
            m = __import__(mod)
            info["packages"][mod] = getattr(m, "__version__", "?")
        except Exception as e:
            info["packages"][mod] = "MISSING (%s)" % type(e).__name__
            info["problems"].append("package %s is not importable" % mod)
    try:
        import automl

        info["automl"] = automl.__version__
    except Exception as e:
        info["automl"] = "MISSING"
        info["problems"].append("automl is not importable (%s) — run `pip install -e .` "
                                "from the repo root" % e)

    kj = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    info["kaggle_json"] = os.path.exists(kj)
    info["kaggle_env"] = bool(os.environ.get("KAGGLE_USERNAME") and
                              os.environ.get("KAGGLE_KEY"))
    try:
        import kaggle  # noqa: F401

        info["kaggle_package"] = True
    except Exception:
        info["kaggle_package"] = False
    if not info["kaggle_package"]:
        info["problems"].append("kaggle package missing — `pip install kaggle` "
                                "(only needed for the kaggle: datasets)")
    if not (info["kaggle_json"] or info["kaggle_env"]):
        info["problems"].append("no Kaggle credentials — put kaggle.json in "
                                "%USERPROFILE%\\.kaggle\\ (only needed for kaggle: datasets)")
    return info


def print_env(info):
    print("python           :", info["python"], "on", info["platform"])
    for k, v in info["packages"].items():
        print("  %-12s %s" % (k, v))
    print("automl           :", info.get("automl"))
    print("kaggle package   :", info.get("kaggle_package"))
    print("kaggle creds     :", "kaggle.json" if info["kaggle_json"]
          else ("env vars" if info["kaggle_env"] else "NONE"))
    if info["problems"]:
        print("\nproblems:")
        for p in info["problems"]:
            print("  -", p)
    else:
        print("\nno problems found.")


# ==========================================================================
# dataset-specific fixes, kept explicit rather than hidden in a generic path
# ==========================================================================
def _hook_product_demand(df, spec):
    """Order rows -> one row per (category, day). The raw column is text with
    thousands separators and parenthesised negatives."""
    import pandas as pd

    s = (df["Order_Demand"].astype(str).str.strip()
         .str.replace(",", "", regex=False)
         .str.replace(r"^\((.*)\)$", r"-\1", regex=True))
    df = df.assign(Order_Demand=pd.to_numeric(s, errors="coerce"))
    df = df.dropna(subset=["Order_Demand", "Date", "Product_Category"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    out = (df.groupby(["Product_Category", "Date"], as_index=False)["Order_Demand"].sum())
    # keep the last two years so the run stays inside a laptop's patience
    cutoff = out["Date"].max() - pd.Timedelta(days=730)
    return out[out["Date"] >= cutoff].reset_index(drop=True)


def _hook_air_passengers(df, spec):
    import pandas as pd

    df = df.copy()
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    return df.dropna(subset=["Month"]).reset_index(drop=True)


HOOKS = {
    "product-demand": _hook_product_demand,
    "air-passengers": _hook_air_passengers,
}


# ==========================================================================
# acquiring data
# ==========================================================================
def acquire(spec):
    """Return the raw DataFrame for a spec, downloading if needed."""
    import pandas as pd

    source = spec["source"]
    kind, ref = source.split(":", 1)

    if kind == "sklearn":
        from sklearn import datasets as sk

        loaders = {
            "breast_cancer": sk.load_breast_cancer,
            "wine": sk.load_wine,
            "diabetes": sk.load_diabetes,
            "california_housing": sk.fetch_california_housing,
        }
        if ref not in loaders:
            raise ValueError("unknown sklearn loader %r" % ref)
        return loaders[ref](as_frame=True).frame

    if kind == "openml":
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(name=ref, version=1, as_frame=True, parser="auto")
        frame = bunch.frame.copy()
        if spec["target"] not in frame.columns and bunch.target is not None:
            frame[spec["target"]] = bunch.target
        return frame

    if kind == "kaggle":
        dest = os.path.join(DATA, "raw", spec["name"])
        wanted = os.path.join(dest, spec["file"])
        if not os.path.exists(wanted):
            os.makedirs(dest, exist_ok=True)
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            print("   downloading %s ..." % ref)
            api.dataset_download_files(ref, path=dest, unzip=True, quiet=True)
        if not os.path.exists(wanted):
            have = []
            for root, _, files in os.walk(dest):
                have += [os.path.relpath(os.path.join(root, f), dest) for f in files]
            raise FileNotFoundError(
                "%s downloaded but %r is not in it. Files present: %s"
                % (ref, spec["file"], ", ".join(have[:15]) or "(none)"))
        return pd.read_csv(wanted, low_memory=False)

    raise ValueError("unknown source kind %r" % kind)


# ==========================================================================
# splitting
# ==========================================================================
def prepare(df, spec):
    """Split into train/test with an id column, and keep the truth aside."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    name, task, target = spec["name"], spec["task"], spec["target"]
    df = df.copy()

    hook = HOOKS.get(name)
    if hook is not None:
        df = hook(df, spec)

    for c in spec.get("drop", []) or []:
        if c in df.columns:
            df = df.drop(columns=[c])

    if target not in df.columns:
        raise KeyError("target %r not in columns: %s"
                       % (target, ", ".join(map(str, df.columns[:25]))))
    df = df.dropna(subset=[target]).reset_index(drop=True)

    outdir = os.path.join(DATA, "prepared", name)
    os.makedirs(outdir, exist_ok=True)

    if task == "forecast":
        date_col = spec["date"]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        gk = spec.get("group") or []
        df = df.sort_values(gk + [date_col]).reset_index(drop=True)
        dates = np.sort(df[date_col].unique())
        h = int(spec.get("horizon", 12))
        h = min(h, max(1, len(dates) // 5))
        cut = dates[-h]
        train = df[df[date_col] < cut].reset_index(drop=True)
        test_full = df[df[date_col] >= cut].reset_index(drop=True)
        test_full.insert(0, "id", ["r%06d" % i for i in range(len(test_full))])
        test = test_full.drop(columns=[target])
        truth = test_full[["id", target]].rename(columns={target: "_truth"})
        meta = {"horizon": int(h), "n_series": int(df[gk].drop_duplicates().shape[0])
                if gk else 1}
    else:
        strat = df[target] if task == "classification" else None
        if strat is not None and strat.value_counts().min() < 2:
            strat = None
        train, test_full = train_test_split(
            df, test_size=0.2, random_state=42, shuffle=True, stratify=strat)
        train = train.reset_index(drop=True)
        test_full = test_full.reset_index(drop=True)
        test_full.insert(0, "id", ["r%06d" % i for i in range(len(test_full))])
        test = test_full.drop(columns=[target])
        truth = test_full[["id", target]].rename(columns={target: "_truth"})
        meta = {}

    train_path = os.path.join(outdir, "train.csv")
    test_path = os.path.join(outdir, "test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    truth.to_csv(os.path.join(outdir, "truth.csv"), index=False)

    meta.update({"n_train": int(len(train)), "n_test": int(len(test)),
                 "n_features": int(len(train.columns) - 1)})
    return train_path, test_path, train, test, truth, meta


# ==========================================================================
# scoring
# ==========================================================================
def classes_of(truth):
    return sorted({str(v) for v in truth["_truth"].dropna().unique()})


def score_predictions(spec, truth, probs_df):
    """probs_df: id + one column per class (classification) or id + pred."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (accuracy_score, log_loss, mean_absolute_error,
                                 roc_auc_score)

    metric, task = spec["metric"], spec["task"]
    merged = truth.merge(probs_df, on="id", how="left")
    if merged[probs_df.columns[1]].isna().any():
        raise ValueError("some test rows got no prediction")

    if task == "classification":
        cls = classes_of(truth)
        cols = [c for c in probs_df.columns if c != "id"]
        missing = [c for c in cls if c not in cols]
        if missing:
            raise ValueError("prediction is missing class column(s): %s" % missing)
        P = merged[cls].values.astype(float)
        P = np.clip(P, 1e-15, 1 - 1e-15)
        P = P / P.sum(axis=1, keepdims=True)
        y = merged["_truth"].astype(str).values
        if metric == "auc":
            if len(cls) != 2:
                return float(roc_auc_score(
                    pd.Categorical(y, categories=cls).codes, P, multi_class="ovr"))
            pos = cls[-1]                       # fixed, identical for every system
            return float(roc_auc_score((y == pos).astype(int), P[:, cls.index(pos)]))
        if metric == "accuracy":
            pred = [cls[i] for i in P.argmax(1)]
            return float(accuracy_score(y, pred))
        idx = pd.Categorical(y, categories=cls).codes
        return float(log_loss(idx, P, labels=list(range(len(cls)))))

    y = merged["_truth"].astype(float).values
    p = merged["pred"].astype(float).values
    if metric == "mae":
        return float(mean_absolute_error(y, p))
    return float(np.sqrt(np.mean((y - p) ** 2)))


def better(metric, a, b):
    """True when a is a better score than b."""
    if a is None:
        return False
    if b is None:
        return True
    return a > b if metric in ("auc", "accuracy") else a < b


# ==========================================================================
# the three systems
# ==========================================================================
def run_ours(spec, train_path, test_path, preset, budget):
    from automl import run

    out_dir = os.path.join(OUT, spec["name"], preset)
    cfg = {
        "task": spec["task"],
        "data": {"train": train_path, "test": test_path,
                 "target": spec["target"], "id": "id"},
        "run": {"preset": preset, "metric": spec["metric"], "seed": 42,
                "out_dir": out_dir, "download": False,
                "time_budget_sec": int(budget)},
    }
    if spec["task"] == "forecast":
        cfg["data"]["date"] = spec["date"]
        if spec.get("group"):
            cfg["data"]["group"] = spec["group"]
        cfg["run"]["horizon"] = spec.get("horizon")
        cfg["run"]["season_period"] = spec.get("season_period", 7)
    if spec["task"] == "classification" and spec["metric"] == "auc":
        cfg["run"]["positive_class"] = None      # engine picks; scoring is fixed anyway

    t0 = time.time()
    result = run(cfg)
    elapsed = time.time() - t0

    import pandas as pd

    probs = pd.read_csv(os.path.join(out_dir, "test_probs_%s.csv" % spec["task"]))
    probs["id"] = probs["id"].astype(str)
    probs.columns = [str(c) for c in probs.columns]

    # The classification engine reports scores higher-is-better, so its log loss
    # comes back negated. Flip it so cv and holdout are on the same scale here.
    cv = float(result["cv_score"])
    if spec["task"] == "classification" and spec["metric"] == "logloss":
        cv = -cv
    return probs, elapsed, cv, result


def _encode(train, test, target, leaky_target_encoding=False):
    """Shared preprocessing for the two single-model comparisons."""
    import numpy as np
    import pandas as pd

    feats = [c for c in train.columns if c not in (target, "id")]
    Xtr, Xte = train[feats].copy(), test[[c for c in feats if c in test.columns]].copy()
    for c in feats:
        if c not in Xte.columns:
            Xte[c] = np.nan
    Xte = Xte[feats]

    y = train[target]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(Xtr[c])]
    cat = [c for c in feats if c not in num]

    for c in num:
        med = Xtr[c].median()
        med = 0.0 if pd.isna(med) else float(med)
        Xtr[c] = Xtr[c].fillna(med).astype("float32")
        Xte[c] = pd.to_numeric(Xte[c], errors="coerce").fillna(med).astype("float32")

    if leaky_target_encoding and cat:
        # The shortcut: the map is built from every training row, including the
        # ones this same model is about to be validated on.
        ynum = pd.factorize(y)[0].astype(float) if not pd.api.types.is_numeric_dtype(y) \
            else y.astype(float)
        for c in cat:
            m = ynum.groupby(Xtr[c].astype(str)).mean() if hasattr(ynum, "groupby") \
                else pd.Series(ynum).groupby(Xtr[c].astype(str).values).mean()
            g = float(pd.Series(ynum).mean())
            Xtr[c] = Xtr[c].astype(str).map(m).fillna(g).astype("float32")
            Xte[c] = Xte[c].astype(str).map(m).fillna(g).astype("float32")
    else:
        for c in cat:
            codes = pd.Categorical(Xtr[c].astype(str))
            mapping = {v: i for i, v in enumerate(codes.categories)}
            Xtr[c] = Xtr[c].astype(str).map(mapping).fillna(-1).astype("float32")
            Xte[c] = Xte[c].astype(str).map(mapping).fillna(-1).astype("float32")

    return Xtr, Xte, feats


def run_baseline(spec, train, test, truth, leaky=False):
    """One LightGBM with default parameters. Returns (probs_df, elapsed, cv)."""
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, StratifiedKFold

    target, task = spec["target"], spec["task"]

    if leaky:
        # The leaky variant only differs from the baseline when there is
        # something to target-encode. Saying so beats printing the same number
        # twice and looking like a bug.
        feats = [c for c in train.columns if c not in (target, "id")]
        has_cat = any(not pd.api.types.is_numeric_dtype(train[c]) for c in feats)
        if not has_cat:
            raise ValueError("no categorical columns — identical to baseline")

    if task == "forecast":
        # A single global model with no lag features would be a strawman, so the
        # forecast comparison uses the seasonal-naive baseline instead.
        return run_seasonal_naive(spec, train, test), 0.0, None

    Xtr, Xte, _ = _encode(train, test, target, leaky_target_encoding=leaky)
    t0 = time.time()

    if task == "classification":
        cls = sorted({str(v) for v in train[target].dropna().unique()})
        y = pd.Categorical(train[target].astype(str), categories=cls).codes
        mk = lambda: lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        model = mk().fit(Xtr, y)
        P = np.asarray(model.predict_proba(Xte))
        if P.ndim == 1:
            P = np.vstack([1 - P, P]).T
        probs = pd.DataFrame({"id": test["id"].astype(str)})
        for i, c in enumerate(cls):
            probs[c] = P[:, i]

        cv = None
        try:                                    # the optimistic in-sample estimate
            from sklearn.metrics import log_loss, roc_auc_score

            skf = StratifiedKFold(n_splits=min(5, int(np.bincount(y).min())),
                                  shuffle=True, random_state=42)
            oof = np.zeros((len(Xtr), len(cls)))
            for tr_i, va_i in skf.split(Xtr, y):
                m = mk().fit(Xtr.iloc[tr_i], y[tr_i])
                p = np.asarray(m.predict_proba(Xtr.iloc[va_i]))
                oof[va_i] = np.vstack([1 - p, p]).T if p.ndim == 1 else p
            if spec["metric"] == "auc" and len(cls) == 2:
                cv = float(roc_auc_score(y, oof[:, 1]))
            elif spec["metric"] == "auc":
                cv = float(roc_auc_score(y, oof, multi_class="ovr"))
            elif spec["metric"] == "accuracy":
                cv = float((oof.argmax(1) == y).mean())
            else:
                cv = float(log_loss(y, np.clip(oof, 1e-15, 1 - 1e-15),
                                    labels=list(range(len(cls)))))
        except Exception:
            pass
        return probs, time.time() - t0, cv

    y = train[target].astype(float).values
    mk = lambda: lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    model = mk().fit(Xtr, y)
    probs = pd.DataFrame({"id": test["id"].astype(str),
                          "pred": model.predict(Xte).astype(float)})
    cv = None
    try:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(Xtr))
        for tr_i, va_i in kf.split(Xtr):
            oof[va_i] = mk().fit(Xtr.iloc[tr_i], y[tr_i]).predict(Xtr.iloc[va_i])
        if spec["metric"] == "mae":
            cv = float(np.mean(np.abs(y - oof)))
        else:
            cv = float(np.sqrt(np.mean((y - oof) ** 2)))
    except Exception:
        pass
    return probs, time.time() - t0, cv


def run_seasonal_naive(spec, train, test):
    import numpy as np
    import pandas as pd

    target, date_col = spec["target"], spec["date"]
    gk = spec.get("group") or []
    season = int(spec.get("season_period", 7))
    tr = train.copy()
    tr[date_col] = pd.to_datetime(tr[date_col])
    preds = np.zeros(len(test))

    def seq(hist, n):
        h = np.asarray(hist, float)
        h = h[~np.isnan(h)]
        if len(h) == 0:
            return np.zeros(n)
        if len(h) < season:
            return np.full(n, h[-1])
        block = h[-season:]
        return np.array([block[i % season] for i in range(n)])

    te = test.copy()
    te["_oi"] = np.arange(len(te))
    te[date_col] = pd.to_datetime(te[date_col])
    if gk:
        hist = {k: v.sort_values(date_col)[target].values
                for k, v in tr.groupby(gk, sort=False)}
        for key, sub in te.sort_values(gk + [date_col]).groupby(gk, sort=False):
            preds[sub["_oi"].values] = seq(hist.get(key, np.array([])), len(sub))
    else:
        h = tr.sort_values(date_col)[target].values
        sub = te.sort_values(date_col)
        preds[sub["_oi"].values] = seq(h, len(sub))
    return pd.DataFrame({"id": test["id"].astype(str), "pred": preds})


# ==========================================================================
# driving one dataset
# ==========================================================================
def run_one(spec, args):
    print("\n" + "=" * 74)
    print("DATASET %s  (%s, %s)" % (spec["name"], spec["task"], spec["metric"]))
    print("=" * 74)

    rec = {"name": spec["name"], "task": spec["task"], "metric": spec["metric"],
           "source": spec["source"], "note": spec.get("note", "")}

    df = acquire(spec)
    train_path, test_path, train, test, truth, meta = prepare(df, spec)
    rec.update(meta)
    print("   rows: %d train / %d test | features: %d"
          % (meta["n_train"], meta["n_test"], meta["n_features"]))

    for preset, budget in (("fast", args.fast_budget), ("full", args.full_budget)):
        if preset == "full" and args.skip_full:
            continue
        try:
            probs, elapsed, cv, _ = run_ours(spec, train_path, test_path, preset, budget)
            hold = score_predictions(spec, truth, probs)
            rec["ours_%s" % preset] = {"holdout": hold, "cv": cv, "sec": elapsed}
            print("   ours-%-5s holdout %s=%.5f  (cv %.5f, %.1fs)"
                  % (preset, spec["metric"], hold, cv, elapsed))
        except Exception as e:
            rec["ours_%s" % preset] = {"error": "%s: %s" % (type(e).__name__, e)}
            print("   ours-%-5s FAILED: %s" % (preset, e))
            if args.traceback:
                traceback.print_exc()

    for label, leaky in (("baseline", False), ("leaky", True)):
        if label == "leaky" and spec["task"] == "forecast":
            continue
        try:
            probs, elapsed, cv = run_baseline(spec, train, test, truth, leaky=leaky)
            hold = score_predictions(spec, truth, probs)
            rec[label] = {"holdout": hold, "cv": cv, "sec": elapsed}
            print("   %-10s holdout %s=%.5f  (cv %s, %.1fs)"
                  % (label, spec["metric"], hold,
                     "%.5f" % cv if cv is not None else "n/a", elapsed))
        except Exception as e:
            rec[label] = {"error": "%s: %s" % (type(e).__name__, e)}
            print("   %-10s FAILED: %s" % (label, e))
            if args.traceback:
                traceback.print_exc()

    return rec


# ==========================================================================
# reporting
# ==========================================================================
def load_results():
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r") as f:
            return json.load(f)
    return {}


def save_results(res):
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)


def fmt(v, nd=4):
    if v is None:
        return "—"
    if abs(v) >= 1000:
        return "%.1f" % v
    return ("%%.%df" % nd) % v


def write_markdown(res):
    rows = [r for r in res.values() if "error" not in r]
    lines = ["# Benchmark results", "",
             "Held-out scores on data no model saw during training: a 20% random",
             "split for tabular datasets, the last H dates for time series.",
             "`cv` is what each system reported before seeing that data — the gap",
             "between `cv` and `holdout` is the number worth looking at.", "",
             "`=base` in the leaky column means the dataset has no categorical",
             "columns, so the leaky variant is identical to the baseline.",
             "For log loss and RMSE lower is better; for AUC and accuracy higher is.", "",
             "Generated by `benchmark/run_benchmark.py`.", "",
             "| dataset | task | metric | rows | ours-fast | ours-full | baseline | leaky | fast sec | full sec |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def cell(key):
            d = r.get(key) or {}
            if "error" in d:
                return "=base" if "no categorical" in d["error"] else "err"
            if d.get("holdout") is None:
                return "—"
            return fmt(d["holdout"])
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["name"], r["task"], r["metric"], r.get("n_train", "?"),
            cell("ours_fast"), cell("ours_full"), cell("baseline"), cell("leaky"),
            fmt((r.get("ours_fast") or {}).get("sec"), 1),
            fmt((r.get("ours_full") or {}).get("sec"), 1)))

    lines += ["", "## CV vs held-out gap", "",
              "How far each system's own cross-validated estimate was from the truth.",
              "A large positive gap means the score it reported was optimistic.", "",
              "| dataset | metric | ours-full cv | ours-full holdout | gap | leaky cv | leaky holdout | gap |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        o = r.get("ours_full") or r.get("ours_fast") or {}
        lk = r.get("leaky") or {}
        if "error" in o or o.get("cv") is None:
            continue
        higher = r["metric"] in ("auc", "accuracy")
        def gap(d):
            if not d or d.get("cv") is None or d.get("holdout") is None:
                return None
            return (d["cv"] - d["holdout"]) if higher else (d["holdout"] - d["cv"])
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["name"], r["metric"], fmt(o.get("cv")), fmt(o.get("holdout")),
            fmt(gap(o)), fmt(lk.get("cv")), fmt(lk.get("holdout")), fmt(gap(lk))))

    failed = [(n, r) for n, r in res.items() if "error" in r]
    if failed:
        lines += ["", "## Datasets that did not run", ""]
        for n, r in failed:
            lines.append("- **%s** — %s" % (n, r["error"]))

    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote", RESULTS_MD)


# ==========================================================================
# slug verification
# ==========================================================================
def verify_slugs(specs):
    """Check every kaggle: entry resolves and contains the file we expect.

    Downloads nothing — it only lists the dataset's files. A dataset that has
    been renamed or removed shows up here in seconds instead of an hour into
    the run.
    """
    kag = [s for s in specs if s["source"].startswith("kaggle:")]
    if not kag:
        print("no kaggle datasets in the list.")
        return 0
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print("cannot authenticate with Kaggle: %s" % e)
        return 2

    bad = 0
    print("\nverifying %d kaggle datasets ...\n" % len(kag))
    for spec in kag:
        ref = spec["source"].split(":", 1)[1]
        try:
            files = api.dataset_list_files(ref).files
            names = [getattr(f, "name", str(f)) for f in files]
        except Exception as e:
            print("  MISSING  %-20s %s  (%s)" % (spec["name"], ref, type(e).__name__))
            bad += 1
            continue
        want = spec.get("file")
        if want and not any(n == want or n.endswith("/" + want) for n in names):
            print("  FILE?    %-20s %s" % (spec["name"], ref))
            print("           expected %r, dataset has: %s"
                  % (want, ", ".join(names[:8])))
            bad += 1
        else:
            print("  ok       %-20s %s" % (spec["name"], ref))

    print("\n%d of %d kaggle entries need fixing." % (bad, len(kag))
          if bad else "\nall %d kaggle entries look good." % len(kag))
    return 0


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="environment report only")
    ap.add_argument("--verify", action="store_true",
                    help="check every kaggle slug and file name without downloading")
    ap.add_argument("--offline", action="store_true",
                    help="only datasets marked offline (no network needed)")
    ap.add_argument("--only", nargs="+", help="run just these dataset names")
    ap.add_argument("--force", action="store_true", help="redo finished datasets")
    ap.add_argument("--skip-full", action="store_true", help="fast preset only")
    ap.add_argument("--fast-budget", type=int, default=120,
                    help="seconds for the fast preset (default 120)")
    ap.add_argument("--full-budget", type=int, default=600,
                    help="seconds for the full preset (default 600)")
    ap.add_argument("--traceback", action="store_true", help="print full tracebacks")
    args = ap.parse_args()

    info = check_env()
    print_env(info)
    with open(os.path.join(HERE, "_env.json"), "w") as f:
        json.dump(info, f, indent=2)
    if args.check:
        return 0
    if info.get("automl") == "MISSING":
        print("\nstopping: automl is not importable.")
        return 2

    import yaml

    with open(os.path.join(HERE, "datasets.yaml"), "r", encoding="utf-8") as f:
        specs = yaml.safe_load(f)["datasets"]

    if args.verify:
        return verify_slugs(specs)

    if args.offline:
        specs = [s for s in specs if s.get("offline")]
    if args.only:
        specs = [s for s in specs if s["name"] in args.only]

    res = load_results()
    started = time.time()
    for spec in specs:
        if spec["name"] in res and not args.force:
            print("skip %s (already in results.json — use --force to redo)" % spec["name"])
            continue
        try:
            res[spec["name"]] = run_one(spec, args)
        except KeyboardInterrupt:
            print("\ninterrupted — results so far are saved.")
            break
        except Exception as e:
            res[spec["name"]] = {"name": spec["name"], "task": spec.get("task"),
                                 "error": "%s: %s" % (type(e).__name__, e)}
            print("   DATASET FAILED: %s: %s" % (type(e).__name__, e))
            if args.traceback:
                traceback.print_exc()
        save_results(res)

    write_markdown(res)
    print("total wall clock: %.1f min" % ((time.time() - started) / 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
