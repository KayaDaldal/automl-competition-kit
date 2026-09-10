#!/usr/bin/env python3
"""The leakage claim, in thirty seconds and with no downloads.

Builds a dataset with a real but weak signal (age, and one city) plus a
high-cardinality id column that is pure noise. Then fits the same LightGBM
twice:

  honest  categoricals ordinal-encoded — no target information in the features
  leaky   categoricals target-encoded on the whole training set, the ordinary
          shortcut, then cross-validated on top of it

Both report a cross-validated AUC and are then scored on rows neither of them
ever saw. The honest one reports roughly what it delivers. The leaky one
reports a much better number and delivers close to a coin flip, because the
noise column's encoding memorised the labels of the rows it is scored on.

    python benchmark/leakage_demo.py
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 0


def make_data(n=3000):
    rng = np.random.RandomState(SEED)
    uid = ["u%04d" % i for i in rng.randint(0, n // 2, n)]   # noise, ~2 rows per level
    city = rng.choice(list("ABCDE"), n)
    age = rng.normal(40, 12, n)
    logit = 0.05 * (age - 40) + (city == "A") * 0.7 - 0.3    # the only real signal
    y = (rng.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"uid": uid, "city": city, "age": age, "y": y})


def encode(train, test, cols, y, leaky):
    tr, te = train.copy(), test.copy()
    if leaky:
        gmean = float(y.mean())
        for c in cols:
            m = y.groupby(tr[c].astype(str)).mean()          # fitted on ALL of train
            tr[c] = tr[c].astype(str).map(m).fillna(gmean).astype(float)
            te[c] = te[c].astype(str).map(m).fillna(gmean).astype(float)
    else:
        for c in cols:
            cats = pd.Categorical(tr[c].astype(str)).categories
            mapping = {v: i for i, v in enumerate(cats)}
            tr[c] = tr[c].astype(str).map(mapping).fillna(-1).astype(float)
            te[c] = te[c].astype(str).map(mapping).fillna(-1).astype(float)
    return tr, te


def evaluate(leaky, df):
    train = df.iloc[:2400].reset_index(drop=True)
    test = df.iloc[2400:].reset_index(drop=True)
    feats = ["uid", "city", "age"]
    y = train["y"]

    Xtr, Xte = encode(train[feats], test[feats], ["uid", "city"], y, leaky)

    oof = np.zeros(len(Xtr))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_i, va_i in skf.split(Xtr, y):
        m = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(Xtr.iloc[tr_i], y.iloc[tr_i])
        oof[va_i] = m.predict_proba(Xtr.iloc[va_i])[:, 1]
    cv = roc_auc_score(y, oof)

    m = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1).fit(Xtr, y)
    holdout = roc_auc_score(test["y"], m.predict_proba(Xte)[:, 1])
    return cv, holdout


if __name__ == "__main__":
    df = make_data()
    print("%-8s %-10s %-10s %s" % ("", "cv auc", "holdout", "gap"))
    for label, leaky in (("honest", False), ("leaky", True)):
        cv, hold = evaluate(leaky, df)
        print("%-8s %-10.4f %-10.4f %+.4f" % (label, cv, hold, cv - hold))
    print("\nThe leaky row is what a fast pipeline reports when target encoding is")
    print("fitted before the split. This repo fits it inside each fold instead;")
    print("see automl/core/prep.py:fold_safe_target_encode and its callers.")
