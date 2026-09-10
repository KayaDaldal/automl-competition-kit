"""Shared preprocessing helpers: seeding, column guessing, dates, target encoding."""

_DATE_KW = ("date", "tarih", "time", "zaman", "datetime", "timestamp")

# Heuristic for naming the positive class of a binary problem when the caller
# did not set `positive_class`. These are simply common positive-class names
# across public datasets, not domain knowledge about any particular one.
POSITIVE_KEYWORDS = (
    "1", "yes", "true", "positive", "pos", "fraud", "churn", "default",
    "malignant", "tumor", "abnormal", "disease", "sick",
)
NEGATIVE_PREFIXES = ("non", "no-", "no_", "neg", "healthy", "normal")


def set_seed(seed):
    """Seed python and numpy. Torch is seeded separately by the image engine."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def guess_col(cols, candidates, fallback_first=False):
    """Find the first column matching a candidate name (exact, then substring).

    `cols` may be a DataFrame column index or any iterable of names.
    Returns None when nothing matches, unless `fallback_first` is set.
    """
    cols = list(cols)
    low = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for c in cols:
        for cand in candidates:
            if cand in str(c).lower():
                return c
    if fallback_first and cols:
        return cols[0]
    return None


def positive_index(class_names, positive_class):
    """Index of the positive class among `class_names`.

    Explicit `positive_class` wins. Otherwise fall back to the keyword
    heuristic above, and finally to the last class alphabetically.
    """
    lower = [str(c).lower().strip() for c in class_names]
    if positive_class is not None:
        pc = str(positive_class).lower().strip()
        if pc in lower:
            return lower.index(pc)
    for kw in POSITIVE_KEYWORDS:              # exact name match first
        if kw in lower:
            return lower.index(kw)
    for i, c in enumerate(lower):             # then substring, skipping negations
        if any(c.startswith(n) for n in NEGATIVE_PREFIXES):
            continue
        if any(kw in c for kw in POSITIVE_KEYWORDS):
            return i
    return len(class_names) - 1


def detect_date_cols(df, cols):
    """Columns that are datetimes, or object columns whose name looks like a date
    and that parse cleanly for more than 70% of rows."""
    import pandas as pd

    out = []
    for c in cols:
        # pandas 3 reads text columns as StringDtype rather than object, so both
        # have to be accepted here.
        texty = pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            out.append(c)
        elif texty and any(k in str(c).lower() for k in _DATE_KW):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.7:
                out.append(c)
    return out


def parse_dates(train, test, date_cols):
    """Expand each date column into year/month/day/dow and drop the original.

    Mutates both frames in place; returns the names of the new numeric columns.
    """
    import pandas as pd

    new_num = []
    for c in date_cols:
        for d in (train, test):
            dt = pd.to_datetime(d[c], errors="coerce")
            d[c + "_year"] = dt.dt.year
            d[c + "_month"] = dt.dt.month
            d[c + "_day"] = dt.dt.day
            d[c + "_dow"] = dt.dt.dayofweek
        new_num += [c + "_year", c + "_month", c + "_day", c + "_dow"]
        train.drop(columns=[c], inplace=True)
        test.drop(columns=[c], inplace=True)
    return new_num


def fold_safe_target_encode(tr_s, y_tr, val_s, te_s, smoothing):
    """Smoothed target encoding fitted on the fold-train rows only.

    This is the leakage-critical helper: the encoder never sees the validation
    fold's targets, so the encoded column carries no information about the rows
    it is scored on. Works for a binary target (pass 0/1) and for a continuous
    one (pass the raw values) — the arithmetic is the same.
    """
    import numpy as np
    import pandas as pd

    gmean = float(np.mean(y_tr))
    dfa = pd.DataFrame({"c": np.asarray(tr_s), "y": np.asarray(y_tr)})
    agg = dfa.groupby("c")["y"].agg(["mean", "count"])
    enc = (agg["count"] * agg["mean"] + smoothing * gmean) / (agg["count"] + smoothing)
    m = enc.to_dict()

    def _map(s):
        return pd.Series(np.asarray(s)).map(m).fillna(gmean).values.astype("float32")

    return _map(tr_s), _map(val_s), _map(te_s)
