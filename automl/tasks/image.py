# ==========================================================================
#  Image classification engine.
#  Interface: run_image(cfg) -> dict
#
#  timm backbone + transfer learning and fine-tuning, stratified K-fold, AMP,
#  horizontal-flip TTA. Reads either a folder tree (one sub-folder per class)
#  or a CSV of file names and labels. Unreadable images are skipped. Falls
#  back to CPU with a warning, and writes a valid partial submission as soon
#  as the first fold finishes, so a run that hits the time budget still leaves
#  something usable on disk.
#
#  This engine needs the extra dependencies in requirements-image.txt, and in
#  practice it needs a GPU.
# ==========================================================================

import gc
import math
import os
import random
import time
import warnings

from ..core.config import merge_defaults
from ..core.io import try_download
from ..core.metrics import accuracy as _accuracy
from ..core.metrics import logloss as _logloss
from ..core.prep import guess_col
from ..core.submission import build_submission

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# 1) DEFAULTS
# --------------------------------------------------------------------------
def default_image_cfg():
    return {
        # --- data ---
        "data_path": None,         # training: folder with one sub-folder per class, OR a CSV
        "test_path": None,         # test: image folder OR a CSV
        "image_col": None,         # CSV mode: column holding the file name/path
        "label_col": None,         # CSV mode: label column
        "id_col": None,            # CSV mode: submission key
        "image_root": None,        # CSV mode: root folder for training images
        "test_image_root": None,   # CSV mode: root folder for test images
        # --- problem ---
        "num_classes": None,       # None = infer from the data
        "metric": "logloss",       # "logloss" | "accuracy"
        # --- model / training ---
        "img_size": 300,
        "batch_size": 16,
        "epochs": 8,
        "freeze_epochs": 1,        # backbone frozen for the first N epochs
        "image_model": "auto",     # "auto" | any timm model name
        "folds": 5,
        "seed": 42,
        "lr_head": 1e-3,           # discriminative LR: head
        "lr_backbone": 1e-4,       # discriminative LR: backbone
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "patience": 5,             # early-stopping patience
        "tta": True,               # horizontal-flip TTA at test time
        "num_workers": 2,
        "pretrained": True,        # set False to run without downloading weights
        "time_budget_sec": 6000,
        # --- output ---
        "submission_example": None,
        "out_dir": ".",
        "download": True,
        "positive_class": None,    # binary: name of the positive class
        "result_as_proba": True,   # probability vs. 0/1 in a JSON submission
    }


# --------------------------------------------------------------------------
# 2) SMALL HELPERS
# --------------------------------------------------------------------------
def _set_seed(seed):
    """Seed python, numpy and torch. cudnn.benchmark stays on: full determinism
    costs more than it is worth here, and every run is seeded anyway."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _read_image(path):
    """Read an image as RGB uint8. Returns None when the file is unreadable."""
    import cv2
    import numpy as np

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        try:
            from PIL import Image

            img = np.array(Image.open(path).convert("RGB"))
        except Exception:
            return None
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _is_readable(path):
    return _read_image(path) is not None


def _list_images(folder):
    out = []
    for root, _, fnames in os.walk(folder):
        for fn in fnames:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, fn))
    return sorted(out)


# --------------------------------------------------------------------------
# 3) DATA DETECTION — folder tree or CSV
# --------------------------------------------------------------------------
def _detect_data(cfg):
    """
    Returns:
      train_items: list[(path, label_str)]
      test_items:  list[dict(path, basename, id)]
      class_names: sorted list[str]
    """
    import pandas as pd

    data_path = cfg["data_path"]
    test_path = cfg["test_path"]
    csv_mode = isinstance(data_path, str) and data_path.lower().endswith((".csv", ".tsv"))

    train_items = []
    if csv_mode:
        df = pd.read_csv(data_path)
        img_col = cfg["image_col"] or guess_col(
            df.columns, ["image", "img", "filename", "file", "path", "id"],
            fallback_first=True)
        lab_col = cfg["label_col"] or guess_col(
            df.columns, ["label", "target", "class", "y", "result", "diagnosis"],
            fallback_first=True)
        root = cfg["image_root"] or os.path.dirname(data_path)
        for _, r in df.iterrows():
            raw = str(r[img_col])
            p = raw if os.path.isabs(raw) or os.path.exists(raw) else os.path.join(root, raw)
            train_items.append((p, str(r[lab_col])))
    else:
        subdirs = [d for d in sorted(os.listdir(data_path))
                   if os.path.isdir(os.path.join(data_path, d))]
        if not subdirs:
            raise ValueError(
                "folder mode expects one sub-folder per class under: %s" % data_path)
        for cls in subdirs:
            for p in _list_images(os.path.join(data_path, cls)):
                train_items.append((p, cls))

    class_names = sorted({lab for _, lab in train_items})

    test_items = []
    test_csv = isinstance(test_path, str) and test_path.lower().endswith((".csv", ".tsv"))
    if test_csv:
        dft = pd.read_csv(test_path)
        img_col = cfg["image_col"] or guess_col(
            dft.columns, ["image", "img", "filename", "file", "path", "id"],
            fallback_first=True)
        id_col = cfg["id_col"]
        root = cfg["test_image_root"] or os.path.dirname(test_path)
        for _, r in dft.iterrows():
            raw = str(r[img_col])
            p = raw if os.path.isabs(raw) or os.path.exists(raw) else os.path.join(root, raw)
            base = os.path.basename(p)
            key = str(r[id_col]) if id_col and id_col in dft.columns else base
            test_items.append({"path": p, "basename": base, "id": key})
    else:
        for p in _list_images(test_path):
            base = os.path.basename(p)
            test_items.append({"path": p, "basename": base, "id": base})

    return train_items, test_items, class_names


# --------------------------------------------------------------------------
# 4) AUGMENTATION
# --------------------------------------------------------------------------
def _build_transforms(img_size, train):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if train:
        aug = [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.10,
                               rotate_limit=12, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15,
                                       contrast_limit=0.15, p=0.5),
        ]
    else:
        aug = [A.Resize(img_size, img_size)]
    aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(aug)


# --------------------------------------------------------------------------
# 5) DATASET (factory, so torch is imported lazily)
# --------------------------------------------------------------------------
def _make_dataset_cls():
    import numpy as np
    from torch.utils.data import Dataset

    class ImgDS(Dataset):
        def __init__(self, items, transform, label_to_idx=None):
            self.items = items                # list[(path, label_str_or_None)]
            self.transform = transform
            self.label_to_idx = label_to_idx

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, lab = self.items[i]
            img = _read_image(path)
            if img is None:                   # unreadable -> black square
                img = np.zeros((32, 32, 3), dtype=np.uint8)
            img = self.transform(image=img)["image"]
            if self.label_to_idx is not None and lab is not None:
                return img, self.label_to_idx[lab]
            return img, 0                     # test: label unused

    return ImgDS


# --------------------------------------------------------------------------
# 6) MODEL
# --------------------------------------------------------------------------
_MODEL_ALIASES = {
    "tf_efficientnet_b3_ns": ["tf_efficientnet_b3_ns",
                              "tf_efficientnet_b3.ns_jft_in1k",
                              "tf_efficientnet_b3"],
    "tf_efficientnet_b0_ns": ["tf_efficientnet_b0_ns",
                              "tf_efficientnet_b0.ns_jft_in1k",
                              "tf_efficientnet_b0"],
    "convnext_tiny": ["convnext_tiny",
                      "convnext_tiny.fb_in22k_ft_in1k",
                      "convnext_tiny.in12k_ft_in1k"],
}


def _safe_create_model(name, num_classes, pretrained):
    """Try the given name and its known aliases across timm versions."""
    import timm

    tried = []
    names = _MODEL_ALIASES.get(name, [name])
    for nm in names:
        try:
            return timm.create_model(nm, pretrained=pretrained,
                                     num_classes=num_classes), nm
        except Exception as e:
            tried.append("%s (%s)" % (nm, type(e).__name__))
    for nm in names:                          # last resort: no pretrained weights
        try:
            return (timm.create_model(nm, pretrained=False, num_classes=num_classes),
                    nm + " [pretrained=False]")
        except Exception as e:
            tried.append("%s pf (%s)" % (nm, type(e).__name__))
    raise RuntimeError("could not build a model. Tried: " + ", ".join(tried))


def _auto_model(n_train, has_gpu, time_budget):
    """Pick a backbone from data size, hardware and the time budget."""
    if (not has_gpu) or time_budget < 1200 or n_train < 800:
        return "tf_efficientnet_b0_ns"
    if n_train > 8000:
        return "convnext_tiny"
    return "tf_efficientnet_b3_ns"


def _classifier_param_ids(model):
    try:
        return set(id(p) for p in model.get_classifier().parameters())
    except Exception:
        return set()


def _set_backbone_grad(model, requires, clf_ids):
    for p in model.parameters():
        if id(p) not in clf_ids:
            p.requires_grad = requires


def _make_optimizer(model, lr_head, lr_backbone, wd, clf_ids):
    """Discriminative learning rates: high for the head, low for the backbone."""
    import torch

    head, backbone = [], []
    for p in model.parameters():
        (head if id(p) in clf_ids else backbone).append(p)
    groups = [{"params": backbone, "lr": lr_backbone},
              {"params": head, "lr": lr_head}]
    return torch.optim.AdamW(groups, weight_decay=wd)


def _amp_tools(use_amp):
    """AMP autocast + GradScaler across torch versions."""
    try:
        from torch.amp import GradScaler, autocast

        def ac():
            return autocast("cuda", enabled=use_amp)

        scaler = GradScaler("cuda", enabled=use_amp)
    except Exception:
        from torch.cuda.amp import GradScaler, autocast

        def ac():
            return autocast(enabled=use_amp)

        scaler = GradScaler(enabled=use_amp)
    return ac, scaler


# --------------------------------------------------------------------------
# 7) TRAIN / PREDICT
# --------------------------------------------------------------------------
def _train_one_fold(train_loader, valid_loader, n_classes, class_weights,
                    cfg, device, fold, t_start):
    import torch
    import torch.nn as nn

    use_amp = (device.type == "cuda")
    model, used_name = _safe_create_model(cfg["image_model"], n_classes, cfg["pretrained"])
    model = model.to(device)
    if fold == 0:
        print("   [model] timm model:", used_name)

    clf_ids = _classifier_param_ids(model)
    optimizer = _make_optimizer(model, cfg["lr_head"], cfg["lr_backbone"],
                                cfg["weight_decay"], clf_ids)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg["epochs"], 1))
    ac, scaler = _amp_tools(use_amp)

    weight_t = None
    if class_weights is not None:
        weight_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_t,
                                    label_smoothing=cfg["label_smoothing"])

    freeze_e = min(cfg["freeze_epochs"], max(cfg["epochs"] - 1, 0))
    best_metric = math.inf if cfg["metric"] == "logloss" else -math.inf
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad = 0

    for epoch in range(cfg["epochs"]):
        if epoch < freeze_e:
            _set_backbone_grad(model, False, clf_ids)
        elif epoch == freeze_e:
            _set_backbone_grad(model, True, clf_ids)

        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with ac():
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        val_probs, val_true = _infer(model, valid_loader, device, n_classes, ac,
                                     tta=False, with_label=True)
        if cfg["metric"] == "logloss":
            m = _logloss(val_true, val_probs, n_classes)
            improved = m < best_metric - 1e-6
        else:
            m = _accuracy(val_true, val_probs)
            improved = m > best_metric + 1e-6

        print("   [fold %d] epoch %d/%d  %s=%.5f  (%.0fs)"
              % (fold, epoch + 1, cfg["epochs"], cfg["metric"], m, time.time() - t_start))

        if improved:
            best_metric = m
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print("   [fold %d] early stopping (epoch %d)" % (fold, epoch + 1))
                break

        if time.time() - t_start > cfg["time_budget_sec"]:
            print("   [fold %d] time budget spent, ending this fold." % fold)
            break

    model.load_state_dict(best_state)
    return model, best_metric


def _infer(model, loader, device, n_classes, ac, tta, with_label):
    """Softmax probabilities; with TTA, the mean of the image and its mirror."""
    import numpy as np
    import torch

    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with ac():
                p = torch.softmax(model(x).float(), dim=1)
                if tta:
                    p = (p + torch.softmax(model(torch.flip(x, dims=[3])).float(),
                                           dim=1)) / 2.0
            probs_all.append(p.cpu().numpy())
            if with_label:
                labels_all.append(y.numpy())
    probs = np.concatenate(probs_all, axis=0)
    if with_label:
        return probs, np.concatenate(labels_all, axis=0)
    return probs


# --------------------------------------------------------------------------
# 8) ENGINE
# --------------------------------------------------------------------------
def run_image(cfg=None):
    """
    Image classification. Interface: run_image(cfg) -> dict.

    Returns {submission, submission_files, cv_score, metric, time_sec,
             class_names, model_name, test_probs (DataFrame), n_folds_done}
    """
    warnings.filterwarnings("ignore")
    t_start = time.time()
    cfg = merge_defaults(cfg, default_image_cfg())
    _set_seed(cfg["seed"])

    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader

    # --- hardware ---
    has_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if has_gpu else "cpu")
    if not has_gpu:
        print("[warning] no GPU found — running on CPU. This will be slow.")
    else:
        print("[info] GPU:", torch.cuda.get_device_name(0))

    # --- data ---
    train_items, test_items, class_names = _detect_data(cfg)
    n_classes = cfg["num_classes"] or len(class_names)
    if n_classes != len(class_names):
        print("[warning] cfg num_classes (%d) differs from the %d classes found; "
              "using the data." % (n_classes, len(class_names)))
        n_classes = len(class_names)
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    # --- drop unreadable training images ---
    clean, dropped = [], 0
    for p, lab in train_items:
        if _is_readable(p):
            clean.append((p, lab))
        else:
            dropped += 1
    train_items = clean
    if dropped:
        print("[info] skipped %d unreadable training image(s)." % dropped)

    # --- backbone ---
    if cfg["image_model"] == "auto":
        cfg["image_model"] = _auto_model(len(train_items), has_gpu, cfg["time_budget_sec"])
    print("[info] classes:", class_names, "| metric:", cfg["metric"],
          "| model:", cfg["image_model"], "| train:", len(train_items),
          "| test:", len(test_items))

    # --- shrink fold count to what the data supports ---
    y_all = np.array([label_to_idx[lab] for _, lab in train_items])
    min_count = int(np.bincount(y_all, minlength=n_classes).min())
    folds = max(2, min(cfg["folds"], min_count)) if min_count >= 2 else 1
    if folds != cfg["folds"]:
        print("[info] fold count reduced to %d to fit the data." % folds)

    # --- class weights when imbalanced ---
    counts = np.bincount(y_all, minlength=n_classes).astype(float)
    ratio = counts.max() / max(counts.min(), 1)
    class_weights = None
    if ratio > 1.3:
        class_weights = counts.sum() / (n_classes * np.maximum(counts, 1))
        print("[info] class imbalance (ratio %.2f) — applying class weights." % ratio)

    ImgDS = _make_dataset_cls()
    tf_train = _build_transforms(cfg["img_size"], train=True)
    tf_valid = _build_transforms(cfg["img_size"], train=False)

    test_ds = ImgDS([(it["path"], None) for it in test_items], tf_valid, label_to_idx=None)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=has_gpu)

    oof = np.zeros((len(train_items), n_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_items), n_classes), dtype=np.float32)
    done_folds = 0
    fold_metrics = []

    def make_ac():
        ac, _ = _amp_tools(has_gpu)
        return ac

    if folds >= 2:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg["seed"])
        split_iter = skf.split(np.zeros(len(train_items)), y_all)
    else:
        # too little data to fold: train on everything, validate on a slice
        idx = np.arange(len(train_items))
        split_iter = [(idx, idx[: max(1, len(idx) // 5)])]

    for fold, (tr_idx, va_idx) in enumerate(split_iter):
        if done_folds >= 1 and (time.time() - t_start) > cfg["time_budget_sec"]:
            print("[info] time budget spent — continuing with %d fold(s)." % done_folds)
            break

        print("===== FOLD %d/%d =====" % (fold, folds))
        tr = [train_items[i] for i in tr_idx]
        va = [train_items[i] for i in va_idx]
        tr_loader = DataLoader(ImgDS(tr, tf_train, label_to_idx),
                               batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], pin_memory=has_gpu,
                               drop_last=False)
        va_loader = DataLoader(ImgDS(va, tf_valid, label_to_idx),
                               batch_size=cfg["batch_size"], shuffle=False,
                               num_workers=cfg["num_workers"], pin_memory=has_gpu)

        model, best_m = _train_one_fold(tr_loader, va_loader, n_classes, class_weights,
                                        cfg, device, fold, t_start)
        fold_metrics.append(best_m)

        ac = make_ac()
        oof[va_idx] = _infer(model, va_loader, device, n_classes, ac,
                             tta=False, with_label=False)
        test_probs += _infer(model, test_loader, device, n_classes, ac,
                             tta=cfg["tta"], with_label=False)
        done_folds += 1

        # Write a valid submission as soon as fold 0 is done, so a run that
        # dies or runs out of budget still leaves something submittable.
        if done_folds == 1:
            interim = _write_submission(cfg, test_items, test_probs / done_folds,
                                        class_names)
            print("[info] interim submission written:", interim["files"][0])

        del model
        gc.collect()
        if has_gpu:
            torch.cuda.empty_cache()

    test_probs /= max(done_folds, 1)

    # --- CV score from the out-of-fold predictions ---
    if folds >= 2 and done_folds >= 1:
        mask = oof.sum(1) > 0
        if cfg["metric"] == "logloss":
            cv = _logloss(y_all[mask], oof[mask], n_classes)
        else:
            cv = _accuracy(y_all[mask], oof[mask])
    else:
        cv = float(np.mean(fold_metrics)) if fold_metrics else float("nan")

    sub = _write_submission(cfg, test_items, test_probs, class_names)
    elapsed = time.time() - t_start

    print("\n==================== SUMMARY (image) =====================")
    print("CV %-10s : %.5f  (%d folds)" % (cfg["metric"], cv, done_folds))
    print("elapsed       : %.1f s" % elapsed)
    print("positive class: '%s' (index %d)" % (sub["pos_class"], sub["pos_idx"]))
    print("files written : %s" % ", ".join(sub["files"]))
    print("==========================================================\n")

    try_download([sub["files"][0]], cfg["download"])

    return {
        "submission": sub["primary"],
        "submission_files": sub["files"],
        "cv_score": cv,
        "metric": cfg["metric"],
        "time_sec": elapsed,
        "class_names": class_names,
        "model_name": cfg["image_model"],
        "test_probs": sub["prob_df"],
        "n_folds_done": done_folds,
    }


def _write_submission(cfg, test_items, probs, class_names):
    """JSON submissions key rows by file name; CSV ones by the id column."""
    return build_submission(
        cfg, "image",
        ids=[it["id"] for it in test_items],
        proba=probs,
        class_names=class_names,
        json_ids=[it["basename"] for it in test_items],
        proba_clip=1e-3,
        write_alt_json=True,
    )
