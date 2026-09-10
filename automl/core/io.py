"""Reading tabular sources and submission-example files.

Extracted from the four engines, where the same four helpers were duplicated
almost verbatim (`_read_table`, `_load_example`, `_json_keys`, `_try_download`).
"""

import json
import os

# Keys that look like a row identifier in a submission example, in priority
# order. The union of the per-engine lists that used to live in each file.
ID_KEY_CANDIDATES = (
    "id", "image", "img", "image_id", "filename", "file", "fname",
    "row_id", "index", "key", "name", "date",
)

# Keys that look like the prediction field in a submission example.
RESULT_KEY_CANDIDATES = (
    "result", "label", "target", "pred", "prediction", "class", "y",
    "proba", "prob", "score", "value", "sales", "demand", "amount", "price",
)


def read_table(src):
    """Read a CSV/parquet path, or copy a DataFrame that is already in memory."""
    import pandas as pd

    if hasattr(src, "columns"):
        return src.copy()
    if isinstance(src, str):
        if src.lower().endswith(".parquet"):
            return pd.read_parquet(src)
        return pd.read_csv(src)
    raise ValueError(
        "A table source must be a CSV/parquet path or a DataFrame: %r" % (src,)
    )


def load_example(ex):
    """Normalise a submission example into ('json_list', list) | ('csv', df) | (None, None).

    Accepts a path, a raw JSON string, a list/dict, or a DataFrame.
    """
    import pandas as pd

    if ex is None:
        return (None, None)
    if isinstance(ex, list):
        return ("json_list", ex)
    if isinstance(ex, dict):
        return ("json_list", [ex])
    if hasattr(ex, "columns"):
        return ("csv", ex)
    if isinstance(ex, str):
        if os.path.exists(ex):
            if ex.lower().endswith(".json"):
                with open(ex, "r") as f:
                    return ("json_list", json.load(f))
            if ex.lower().endswith((".csv", ".tsv")):
                return ("csv", pd.read_csv(ex))
        try:
            obj = json.loads(ex)
            return ("json_list", obj if isinstance(obj, list) else [obj])
        except Exception:
            return (None, None)
    return (None, None)


def json_keys(example_list):
    """Pick the id key and the prediction key out of a JSON submission example."""
    keys = list(example_list[0].keys())
    id_key = None
    for cand in ID_KEY_CANDIDATES:
        for k in keys:
            if k.lower() == cand:
                id_key = k
                break
        if id_key is not None:
            break
    res_key = None
    for cand in RESULT_KEY_CANDIDATES:
        for k in keys:
            if k == id_key:
                continue
            if k.lower() == cand:
                res_key = k
                break
        if res_key is not None:
            break
    if id_key is None:
        id_key = keys[0]
    if res_key is None:
        res_key = keys[-1] if keys[-1] != id_key else keys[0]
    return id_key, res_key


def try_download(paths, enabled):
    """Trigger a browser download for each path when running inside Colab.

    Silently does nothing anywhere else.
    """
    if not enabled:
        return
    try:
        from google.colab import files as colab_files

        for p in paths:
            colab_files.download(p)
    except Exception:
        pass
