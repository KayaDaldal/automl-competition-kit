#!/usr/bin/env python3
"""Benchmark the image engine on a real Kaggle dataset. GPU strongly advised.

Kept separate from run_benchmark.py because it needs different dependencies
(torch, timm) and different hardware. Same idea though: hold out 20% of the
images, train on the rest, score against the held-out truth.

Three systems on the same split:
    ours-fast   the fast preset (small backbone, few epochs, 2 folds)
    ours-full   the full preset, capped by --full-budget
    baseline    one fold, one fine-tune of the same backbone, no TTA, no
                fold averaging — what you would write in twenty lines

    python benchmark/image_benchmark.py --check
    python benchmark/image_benchmark.py --per-class 300
    python benchmark/image_benchmark.py --slug owner/dataset --per-class 300
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
RESULTS_JSON = os.path.join(HERE, "results.json")

if REPO not in sys.path:
    sys.path.insert(0, REPO)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_SLUG = "puneet6060/intel-image-classification"


# --------------------------------------------------------------------------
# acquiring and laying out the data
# --------------------------------------------------------------------------
def download(slug, name):
    dest = os.path.join(DATA, "raw", name)
    marker = os.path.join(dest, ".downloaded")
    if os.path.exists(marker):
        print("already downloaded:", dest)
        return dest
    os.makedirs(dest, exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("downloading %s (this can take a few minutes) ..." % slug)
    api.dataset_download_files(slug, path=dest, unzip=True, quiet=False)
    open(marker, "w").close()
    return dest


def find_class_tree(root):
    """Find the directory whose sub-directories are classes full of images.

    Kaggle image datasets nest differently every time — seg_train/seg_train/,
    training_set/, images/train/. Rather than hard-coding one layout, this
    picks the candidate with the most images.
    """
    best, best_count = None, 0
    for dirpath, dirnames, _ in os.walk(root):
        if not dirnames:
            continue
        counts = []
        for d in dirnames:
            p = os.path.join(dirpath, d)
            try:
                n = sum(1 for f in os.listdir(p) if f.lower().endswith(IMG_EXTS))
            except OSError:
                n = 0
            counts.append(n)
        classes = [c for c in counts if c > 0]
        if len(classes) >= 2 and sum(classes) > best_count:
            best, best_count = dirpath, sum(classes)
    if best is None:
        raise RuntimeError(
            "no class sub-folders with images found under %s — check the layout" % root)
    print("class tree: %s  (%d images)" % (best, best_count))
    return best


def build_split(tree, name, per_class, seed=42):
    """Copy a sample into train/<class>/ and a flat test/, and write truth.csv."""
    import pandas as pd

    out = os.path.join(DATA, "prepared", name)
    if os.path.exists(out):
        shutil.rmtree(out)
    train_dir = os.path.join(out, "train")
    test_dir = os.path.join(out, "test")
    os.makedirs(test_dir, exist_ok=True)

    rng = random.Random(seed)
    truth = []
    classes = sorted(d for d in os.listdir(tree)
                     if os.path.isdir(os.path.join(tree, d)))
    for cls in classes:
        files = [f for f in sorted(os.listdir(os.path.join(tree, cls)))
                 if f.lower().endswith(IMG_EXTS)]
        if not files:
            continue
        rng.shuffle(files)
        if per_class:
            files = files[:per_class]
        cut = max(1, int(len(files) * 0.8))
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        for f in files[:cut]:
            shutil.copy2(os.path.join(tree, cls, f), os.path.join(train_dir, cls, f))
        for f in files[cut:]:
            # flat test folder with a unique name, so the label is not in the path
            newname = "%s__%s" % (cls, f)
            shutil.copy2(os.path.join(tree, cls, f), os.path.join(test_dir, newname))
            truth.append({"id": newname, "_truth": cls})

    pd.DataFrame(truth).to_csv(os.path.join(out, "truth.csv"), index=False)
    n_train = sum(len(fs) for _, _, fs in os.walk(train_dir))
    print("split: %d train / %d test across %d classes" % (n_train, len(truth), len(classes)))
    return train_dir, test_dir, os.path.join(out, "truth.csv"), classes


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score(truth_csv, probs_csv, classes, metric):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, log_loss

    truth = pd.read_csv(truth_csv)
    probs = pd.read_csv(probs_csv)
    probs["id"] = probs["id"].astype(str)
    probs.columns = [str(c) for c in probs.columns]
    m = truth.merge(probs, on="id", how="left")
    if m[classes[0]].isna().any():
        raise ValueError("some test images got no prediction")
    P = np.clip(m[classes].values.astype(float), 1e-15, 1 - 1e-15)
    P = P / P.sum(axis=1, keepdims=True)
    y = pd.Categorical(m["_truth"].astype(str), categories=classes).codes
    acc = float(accuracy_score(y, P.argmax(1)))
    ll = float(log_loss(y, P, labels=list(range(len(classes)))))
    return {"accuracy": acc, "logloss": ll, "primary": acc if metric == "accuracy" else ll}


# --------------------------------------------------------------------------
# the systems
# --------------------------------------------------------------------------
def run_ours(train_dir, test_dir, name, preset, metric, budget, extra=None):
    from automl import run

    out_dir = os.path.join(OUT, name, preset)
    cfg = {
        "task": "image",
        "data": {"train": train_dir, "test": test_dir},
        "run": {"preset": preset, "metric": metric, "seed": 42,
                "out_dir": out_dir, "download": False,
                "time_budget_sec": int(budget)},
    }
    if extra:
        cfg.setdefault("model", {}).update(extra)
    t0 = time.time()
    res = run(cfg)
    return (os.path.join(out_dir, "test_probs_image.csv"), time.time() - t0,
            float(res["cv_score"]), res)


def run_baseline(train_dir, test_dir, name, epochs, img_size, backbone, batch_size,
                 pretrained=True, num_workers=2):
    """One backbone, one fold, no TTA, no averaging — the twenty-line version.

    Reuses the engine's own pieces so the comparison is about the pipeline
    around the model, not about two different training loops.
    """
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    from automl.tasks.image import (_amp_tools, _build_transforms, _detect_data,
                                    _make_dataset_cls, _train_one_fold, _infer)

    cfg = {
        "data_path": train_dir, "test_path": test_dir,
        "image_col": None, "label_col": None, "id_col": None,
        "image_root": None, "test_image_root": None,
        "metric": "logloss", "img_size": img_size, "batch_size": batch_size,
        "epochs": epochs, "freeze_epochs": 1, "image_model": backbone,
        "seed": 42, "lr_head": 1e-3, "lr_backbone": 1e-4, "weight_decay": 1e-4,
        "label_smoothing": 0.05, "patience": 5, "tta": False,
        "num_workers": num_workers, "pretrained": pretrained,
        "time_budget_sec": 10 ** 6,
    }
    train_items, test_items, classes = _detect_data(cfg)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_idx[l] for _, l in train_items])

    ImgDS = _make_dataset_cls()
    tf_tr = _build_transforms(img_size, train=True)
    tf_va = _build_transforms(img_size, train=False)
    tr_i, va_i = train_test_split(np.arange(len(train_items)), test_size=0.2,
                                  random_state=42, stratify=y)
    dl = lambda items, tf, sh: DataLoader(
        ImgDS(items, tf, label_to_idx), batch_size=batch_size, shuffle=sh,
        num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    model, _ = _train_one_fold(dl([train_items[i] for i in tr_i], tf_tr, True),
                               dl([train_items[i] for i in va_i], tf_va, False),
                               len(classes), None, cfg, device, 0, t0)
    ac, _ = _amp_tools(torch.cuda.is_available())
    test_loader = DataLoader(ImgDS([(it["path"], None) for it in test_items], tf_va, None),
                             batch_size=batch_size, shuffle=False,
                             num_workers=cfg["num_workers"])
    P = _infer(model, test_loader, device, len(classes), ac, tta=False, with_label=False)

    out = os.path.join(OUT, name, "baseline")
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame({"id": [it["id"] for it in test_items]})
    for i, c in enumerate(classes):
        df[c] = P[:, i]
    path = os.path.join(out, "test_probs_image.csv")
    df.to_csv(path, index=False)
    return path, time.time() - t0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", default=DEFAULT_SLUG, help="Kaggle dataset slug")
    ap.add_argument("--name", default=None, help="short name used for folders")
    ap.add_argument("--per-class", type=int, default=300,
                    help="images per class to use (0 = all). Keeps the run bounded.")
    ap.add_argument("--metric", default="accuracy", choices=["accuracy", "logloss"])
    ap.add_argument("--fast-budget", type=int, default=900)
    ap.add_argument("--full-budget", type=int, default=2700)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--check", action="store_true", help="report hardware and stop")
    args = ap.parse_args()

    name = args.name or args.slug.split("/")[-1]

    try:
        import timm
        import torch

        gpu = torch.cuda.is_available()
        print("torch %s | timm %s | GPU: %s"
              % (torch.__version__, timm.__version__,
                 torch.cuda.get_device_name(0) if gpu else "NONE (this will be slow)"))
    except Exception as e:
        print("image dependencies missing: %s" % e)
        print("install them with: pip install -r requirements-image.txt")
        return 2
    if args.check:
        return 0

    root = download(args.slug, name)
    tree = find_class_tree(root)
    train_dir, test_dir, truth_csv, classes = build_split(tree, name, args.per_class)

    rec = {"name": "image:" + name, "task": "image", "metric": args.metric,
           "source": "kaggle:" + args.slug, "n_classes": len(classes),
           "per_class": args.per_class}

    for preset, budget in (("fast", args.fast_budget), ("full", args.full_budget)):
        if preset == "full" and args.skip_full:
            continue
        try:
            probs, sec, cv, _ = run_ours(train_dir, test_dir, name, preset,
                                         args.metric, budget)
            s = score(truth_csv, probs, classes, args.metric)
            if args.metric == "logloss":
                cv = cv                       # the image engine already reports it positive
            rec["ours_%s" % preset] = {"holdout": s["primary"], "cv": cv, "sec": sec,
                                       "accuracy": s["accuracy"], "logloss": s["logloss"]}
            print("ours-%-5s holdout acc=%.4f logloss=%.4f  (cv %.4f, %.0fs)"
                  % (preset, s["accuracy"], s["logloss"], cv, sec))
        except Exception as e:
            rec["ours_%s" % preset] = {"error": "%s: %s" % (type(e).__name__, e)}
            print("ours-%-5s FAILED: %s" % (preset, e))

    if not args.skip_baseline:
        try:
            probs, sec = run_baseline(train_dir, test_dir, name, epochs=3,
                                      img_size=224, backbone="tf_efficientnet_b0_ns",
                                      batch_size=32)
            s = score(truth_csv, probs, classes, args.metric)
            rec["baseline"] = {"holdout": s["primary"], "cv": None, "sec": sec,
                               "accuracy": s["accuracy"], "logloss": s["logloss"]}
            print("baseline  holdout acc=%.4f logloss=%.4f  (%.0fs)"
                  % (s["accuracy"], s["logloss"], sec))
        except Exception as e:
            rec["baseline"] = {"error": "%s: %s" % (type(e).__name__, e)}
            print("baseline  FAILED: %s" % e)

    res = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            res = json.load(f)
    res[rec["name"]] = rec
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print("\nwrote", RESULTS_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
