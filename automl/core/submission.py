"""Submission writing.

One function for all four engines. Which branch runs is decided by `task`
(numeric prediction vs. class probabilities) and by whether the caller supplied
a `submission_example` to copy the shape of.

Output shape rules, in order:
  1. A JSON example  -> a JSON list with the example's id/result key names.
  2. A CSV example   -> a CSV with the example's exact column names and order.
  3. No example      -> a plain CSV: an id column plus `target` (or one column
                        per class for multiclass probability metrics).
There is no format that appears without an example asking for it.
"""

import json
import os

from .io import json_keys, load_example
from .prep import guess_col, positive_index

_PROBA_METRICS = ("logloss", "auc")


def build_submission(
    cfg,
    task,
    ids,
    pred=None,
    proba=None,
    class_names=None,
    threshold=0.5,
    json_ids=None,
    proba_clip=1e-15,
    write_alt_json=False,
):
    """Write submission files and return a dict describing them.

    Numeric tasks ('regression', 'forecast') pass `pred`; classification tasks
    ('classification', 'image') pass `proba` and `class_names`.

    Returns {files, primary, kind, prob_df, ...}.
    """
    import pandas as pd

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ex_kind, ex_obj = load_example(cfg.get("submission_example"))

    if proba is None:
        result = _numeric_submission(cfg, ids, pred, ex_kind, ex_obj, out_dir)
        prob_df = pd.DataFrame({"id": ids, "pred": [float(v) for v in pred]})
    else:
        result = _proba_submission(
            cfg,
            ids,
            proba,
            class_names,
            threshold,
            ex_kind,
            ex_obj,
            out_dir,
            json_ids=json_ids,
            proba_clip=proba_clip,
            write_alt_json=write_alt_json,
        )
        prob_df = pd.DataFrame({"id": ids})
        for c in class_names:
            prob_df[str(c)] = proba[:, class_names.index(c)]

    # Per-task probability dump with a fixed column order, so several runs can
    # be ensembled afterwards without guessing which column is which class.
    p_probs = os.path.join(out_dir, "test_probs_%s.csv" % task)
    prob_df.to_csv(p_probs, index=False)
    result["files"].append(p_probs)
    result["prob_df"] = prob_df
    return result


# --------------------------------------------------------------------------
# numeric target (regression, forecast)
# --------------------------------------------------------------------------
def _numeric_submission(cfg, ids, pred, ex_kind, ex_obj, out_dir):
    import pandas as pd

    pred = [float(v) for v in pred]
    files = []

    if ex_kind == "json_list":
        id_key, res_key = json_keys(ex_obj)
        primary = [{id_key: ids[i], res_key: pred[i]} for i in range(len(ids))]
        p_main = os.path.join(out_dir, "submission.json")
        with open(p_main, "w") as f:
            json.dump(primary, f, ensure_ascii=False)
        files.append(p_main)
        return {"files": files, "primary": primary, "kind": "json_list"}

    if ex_kind == "csv":
        cols = list(ex_obj.columns)
        id_name = guess_col(cols, ["id", "image", "row_id", "index", "key"]) or cols[0]
        tcol = [c for c in cols if c != id_name]
        tcol = tcol[0] if tcol else "target"
        df = pd.DataFrame({id_name: ids, tcol: pred})
        df = df[[c for c in cols if c in df.columns]]
    else:
        df = pd.DataFrame({(cfg.get("id_col") or "id"): ids, "target": pred})

    p_main = os.path.join(out_dir, "submission.csv")
    df.to_csv(p_main, index=False)
    files.append(p_main)
    return {"files": files, "primary": df, "kind": "csv"}


# --------------------------------------------------------------------------
# class probabilities (classification, image)
# --------------------------------------------------------------------------
def _proba_submission(
    cfg,
    ids,
    proba,
    class_names,
    threshold,
    ex_kind,
    ex_obj,
    out_dir,
    json_ids=None,
    proba_clip=1e-15,
    write_alt_json=False,
):
    import numpy as np
    import pandas as pd

    n_classes = len(class_names)
    is_binary = n_classes == 2
    pos_idx = positive_index(class_names, cfg.get("positive_class"))
    proba_metric = cfg.get("metric") in _PROBA_METRICS
    files = []

    p1 = np.clip(proba[:, pos_idx], proba_clip, 1 - proba_clip)
    if is_binary:
        labels = [
            class_names[pos_idx] if v >= threshold else class_names[1 - pos_idx]
            for v in p1
        ]
    else:
        labels = [class_names[i] for i in proba.argmax(1)]

    if ex_kind == "json_list":
        id_key, res_key = json_keys(ex_obj)
        keys = json_ids if json_ids is not None else ids

        def make_list(as_proba):
            out = []
            for i, idv in enumerate(keys):
                if as_proba and is_binary:
                    val = float(p1[i])
                elif as_proba and not is_binary:
                    val = labels[i]          # one field, many classes -> label
                else:
                    val = int(p1[i] >= threshold) if is_binary else labels[i]
                out.append({id_key: idv, res_key: val})
            return out

        as_proba = bool(cfg.get("result_as_proba", True)) and proba_metric
        primary = make_list(as_proba)
        p_main = os.path.join(out_dir, "submission.json")
        with open(p_main, "w") as f:
            json.dump(primary, f, ensure_ascii=False)
        files.append(p_main)

        # Some scoring platforms reject floats in a field the example showed as
        # an int (or vice versa). When asked, write the other shape as well so
        # the fallback is already on disk.
        if write_alt_json and is_binary:
            alt = make_list(not as_proba)
            alt_name = "submission_int.json" if as_proba else "submission_proba.json"
            p_alt = os.path.join(out_dir, alt_name)
            with open(p_alt, "w") as f:
                json.dump(alt, f, ensure_ascii=False)
            files.append(p_alt)

        return {
            "files": files,
            "primary": primary,
            "kind": "json_list",
            "pos_idx": pos_idx,
            "pos_class": class_names[pos_idx],
        }

    if ex_kind == "csv":
        cols = list(ex_obj.columns)
        id_name = (
            guess_col(cols, ["id", "image", "img", "filename", "file", "row_id", "key"])
            or cols[0]
        )
        name2idx = {str(c): i for i, c in enumerate(class_names)}
        class_cols = [c for c in cols if str(c) in name2idx]
        df = pd.DataFrame({id_name: ids})
        if class_cols and proba_metric:
            for c in cols:
                if c == id_name:
                    continue
                df[c] = proba[:, name2idx[str(c)]] if str(c) in name2idx else 0.0
            df = df[cols]
        else:
            tcol = [c for c in cols if c != id_name]
            tcol = tcol[0] if tcol else "target"
            df[tcol] = p1 if (proba_metric and is_binary) else labels
            df = df[[c for c in cols if c in df.columns]]
    else:
        df = pd.DataFrame({(cfg.get("id_col") or "id"): ids})
        if proba_metric:
            if is_binary:
                df["target"] = p1
            else:
                for c in class_names:
                    df[str(c)] = proba[:, class_names.index(c)]
        else:
            df["target"] = labels

    p_main = os.path.join(out_dir, "submission.csv")
    df.to_csv(p_main, index=False)
    files.append(p_main)
    return {
        "files": files,
        "primary": df,
        "kind": "csv",
        "pos_idx": pos_idx,
        "pos_class": class_names[pos_idx],
    }
