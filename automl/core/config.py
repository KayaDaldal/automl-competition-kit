"""The single config schema: parsing, validation, presets and task routing.

One YAML file (or one dict) describes any of the four tasks:

    task: classification | regression | forecast | image | auto
    data:  where the data is and which columns matter
    run:   how to score and how long to spend
    model: model-specific knobs

The engines themselves still take a flat dict of their own keys — this module
is the translation layer. It also refuses unknown keys instead of silently
ignoring them, which is the failure mode the original `base.update(cfg)` had:
a typo in a key name changed nothing and reported nothing.
"""

import copy
import difflib
import os

TASKS = ("classification", "regression", "forecast", "image")
SECTIONS = ("data", "run", "model")

# Keys whose values are paths and are resolved relative to the config file.
_PATH_KEYS = {
    ("data", "train"),
    ("data", "test"),
    ("data", "image_root"),
    ("data", "test_image_root"),
    ("run", "submission_example"),
}

# --------------------------------------------------------------------------
# schema: yaml key -> engine cfg key, per task and section
# --------------------------------------------------------------------------
_RUN_COMMON = {
    "metric": "metric",
    "seed": "seed",
    "time_budget_sec": "time_budget_sec",
    "out_dir": "out_dir",
    "download": "download",
    "submission_example": "submission_example",
    "preset": None,            # consumed here, never forwarded to an engine
}

_DATA_TABULAR = {
    "train": "train_path",
    "test": "test_path",
    "target": "target_col",
    "id": "id_col",
    "drop": "drop_cols",
}

_MODEL_BOOSTED = {
    "n_estimators": "n_estimators",
    "learning_rate": "learning_rate",
    "early_stopping": "early_stopping_rounds",
    "high_card_threshold": "high_card_threshold",
    "te_smoothing": "te_smoothing",
    "blend_iters": "blend_iters",
}

SCHEMA = {
    "classification": {
        "data": dict(_DATA_TABULAR),
        "run": dict(
            _RUN_COMMON,
            folds="n_folds",
            positive_class="positive_class",
            result_as_proba="result_as_proba",
        ),
        "model": dict(_MODEL_BOOSTED),
    },
    "regression": {
        "data": dict(_DATA_TABULAR),
        "run": dict(
            _RUN_COMMON,
            folds="n_folds",
            log_target="log_target",
            skew_threshold="skew_threshold",
            non_negative="non_negative",
            stratify_bins="stratify_bins",
        ),
        "model": dict(_MODEL_BOOSTED),
    },
    "forecast": {
        "data": dict(
            _DATA_TABULAR,
            date="date_col",
            group="group_cols",
        ),
        "run": dict(
            _RUN_COMMON,
            horizon="horizon",
            strategy="strategy",
            cv_scheme="cv_scheme",
            cv_folds="cv_folds",
            season_period="season_period",
            non_negative="non_negative",
            use_prophet="use_prophet",
        ),
        "model": {
            "n_estimators": "n_estimators",
            "learning_rate": "learning_rate",
            "early_stopping": "early_stopping_rounds",
            "lags": "lags",
            "windows": "windows",
        },
    },
    "image": {
        "data": {
            "train": "data_path",
            "test": "test_path",
            "id": "id_col",
            "image_col": "image_col",
            "label_col": "label_col",
            "image_root": "image_root",
            "test_image_root": "test_image_root",
        },
        "run": dict(
            _RUN_COMMON,
            folds="folds",
            positive_class="positive_class",
            result_as_proba="result_as_proba",
        ),
        "model": {
            "backbone": "image_model",
            "img_size": "img_size",
            "batch_size": "batch_size",
            "epochs": "epochs",
            "freeze_epochs": "freeze_epochs",
            "patience": "patience",
            "lr_head": "lr_head",
            "lr_backbone": "lr_backbone",
            "weight_decay": "weight_decay",
            "label_smoothing": "label_smoothing",
            "tta": "tta",
            "num_workers": "num_workers",
            "pretrained": "pretrained",
            "num_classes": "num_classes",
        },
    },
}

# --------------------------------------------------------------------------
# required fields, as (section, key, example value for the error message)
# --------------------------------------------------------------------------
REQUIRED = {
    "classification": [
        ("data", "train", "data/train.csv"),
        ("data", "test", "data/test.csv"),
        ("data", "target", "label"),
    ],
    "regression": [
        ("data", "train", "data/train.csv"),
        ("data", "test", "data/test.csv"),
        ("data", "target", "price"),
    ],
    "forecast": [
        ("data", "train", "data/train.csv"),
        ("data", "test", "data/test.csv"),
        ("data", "target", "sales"),
        ("data", "date", "date"),
    ],
    "image": [
        ("data", "train", "data/train/  (one sub-folder per class)"),
        ("data", "test", "data/test/"),
    ],
}

# --------------------------------------------------------------------------
# presets — engine-level keys, applied under the user's own values
# --------------------------------------------------------------------------
_FAST_BOOSTED = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "early_stopping_rounds": 50,
    "n_folds": 3,
    "blend_iters": 150,
    "time_budget_sec": 120,
}

PRESETS = {
    "classification": {"fast": dict(_FAST_BOOSTED), "full": {}},
    "regression": {"fast": dict(_FAST_BOOSTED), "full": {}},
    "forecast": {
        "fast": {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "early_stopping_rounds": 50,
            "lags": [1, 2, 3, 7, 14],
            "windows": [7, 14],
            "cv_scheme": "holdout",
            "time_budget_sec": 120,
        },
        "full": {},
    },
    "image": {
        "fast": {
            "image_model": "tf_efficientnet_b0_ns",
            "img_size": 224,
            "batch_size": 32,
            "epochs": 3,
            "folds": 2,
            "patience": 2,
            "time_budget_sec": 300,
        },
        "full": {},
    },
}

DEFAULT_PRESET = "fast"


class ConfigError(ValueError):
    """Raised for anything wrong with the config, with a message meant to be read."""


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def load_config(source, overrides=None):
    """Read a YAML path / dict and return (task, flat_engine_cfg).

    `overrides` is a flat dict of already-resolved schema values, e.g.
    {"run": {"preset": "full"}}, applied on top of the file.
    """
    raw, base_dir = _read_source(source)
    if overrides:
        raw = _deep_merge(raw, overrides)
    _check_top_level(raw)

    task = raw.get("task", "auto")
    if not isinstance(task, str):
        raise ConfigError("task must be a string, got %r" % (task,))
    task = task.strip().lower()
    if task not in TASKS + ("auto",):
        raise ConfigError(
            "unknown task %r. Valid values: %s"
            % (task, ", ".join(TASKS + ("auto",)))
        )

    raw = _resolve_paths(raw, base_dir)
    if task == "auto":
        task = detect_task(raw)

    # Key spelling first, then missing fields: a typo'd key would otherwise be
    # reported as a missing required field, which points at the wrong problem.
    _check_keys(task, raw)
    _check_required(task, raw)
    cfg = _flatten(task, raw)
    return task, cfg


def default_cfg(task):
    """The engine's own defaults for a task."""
    from ..tasks import forecast as _fc

    if task == "classification":
        from ..tasks.classification import default_classification_cfg

        return default_classification_cfg()
    if task == "regression":
        from ..tasks.regression import default_regression_cfg

        return default_regression_cfg()
    if task == "forecast":
        return _fc.default_forecast_cfg()
    if task == "image":
        from ..tasks.image import default_image_cfg

        return default_image_cfg()
    raise ConfigError("unknown task %r" % (task,))


def merge_defaults(cfg, base):
    """Layer a partial engine cfg over the engine defaults (used by the engines)."""
    out = dict(base)
    if cfg:
        out.update(cfg)
    return out


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------
def _read_source(source):
    if isinstance(source, dict):
        return copy.deepcopy(source), os.getcwd()
    if isinstance(source, str):
        if not os.path.exists(source):
            raise ConfigError("config file not found: %s" % source)
        import yaml

        with open(source, "r") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raise ConfigError("config file is empty: %s" % source)
        if not isinstance(raw, dict):
            raise ConfigError(
                "config file must be a YAML mapping (key: value), got %s"
                % type(raw).__name__
            )
        return raw, os.path.dirname(os.path.abspath(source))
    raise ConfigError(
        "config must be a path to a YAML file or a dict, got %s" % type(source).__name__
    )


def _deep_merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out


def _suggest(key, valid):
    close = difflib.get_close_matches(str(key), list(valid), n=1, cutoff=0.6)
    if close:
        return "  Did you mean %r?" % close[0]
    return "  Valid keys here: %s" % ", ".join(sorted(valid))


def _check_top_level(raw):
    allowed = ("task",) + SECTIONS
    for k in raw:
        if k not in allowed:
            raise ConfigError(
                "unknown top-level key %r.\n%s" % (k, _suggest(k, allowed))
            )
    for s in SECTIONS:
        if s in raw and raw[s] is not None and not isinstance(raw[s], dict):
            raise ConfigError(
                "section %r must be a mapping of key: value, got %s"
                % (s, type(raw[s]).__name__)
            )


def _section(raw, name):
    v = raw.get(name) or {}
    return v if isinstance(v, dict) else {}


def _resolve_paths(raw, base_dir):
    raw = copy.deepcopy(raw)
    for section, key in _PATH_KEYS:
        sec = raw.get(section)
        if not isinstance(sec, dict):
            continue
        val = sec.get(key)
        if not isinstance(val, str) or os.path.isabs(val):
            continue
        if os.path.exists(val):
            continue
        candidate = os.path.join(base_dir, val)
        if os.path.exists(candidate):
            sec[key] = candidate
    return raw


def detect_task(raw):
    """Guess the task from the data, and say out loud why."""
    import pandas as pd

    data = _section(raw, "data")
    train = data.get("train")
    if train is None:
        raise ConfigError(
            "task: auto needs data.train to look at.\n"
            "  Set the task explicitly, or point data.train at your training data."
        )

    if isinstance(train, str) and os.path.isdir(train):
        _announce("image", "data.train is a directory of images")
        return "image"
    if data.get("image_root") or data.get("image_col"):
        _announce("image", "data.image_root / data.image_col is set")
        return "image"

    if data.get("date"):
        _announce("forecast", "data.date is set, so rows are ordered in time")
        return "forecast"

    target = data.get("target")
    if not target:
        raise ConfigError(
            "task: auto needs data.target to decide between classification and regression.\n"
            "  Set data.target, or set task explicitly."
        )
    try:
        head = pd.read_csv(train, nrows=5000) if isinstance(train, str) else train
    except Exception as e:
        raise ConfigError("could not read data.train for task detection: %s" % e)
    if target not in head.columns:
        raise ConfigError(
            "data.target %r is not a column of data.train.\n  Columns: %s"
            % (target, ", ".join(str(c) for c in list(head.columns)[:20]))
        )

    col = head[target].dropna()
    n_unique = col.nunique()
    if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        _announce("classification", "the target is non-numeric")
        return "classification"
    try:
        whole = float(col.mod(1).abs().max()) == 0.0
    except Exception:
        whole = False
    if n_unique <= 20 and whole:
        _announce(
            "classification",
            "the target is whole-numbered with only %d distinct values" % n_unique,
        )
        return "classification"
    _announce("regression", "the target is numeric and continuous, and no date column is set")
    return "regression"


def _announce(task, reason):
    print("[auto] task=%s — %s. Set `task:` explicitly to override." % (task, reason))


def _check_required(task, raw):
    for section, key, example in REQUIRED[task]:
        val = _section(raw, section).get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            raise ConfigError(
                "missing required field %s.%s for task %r.\n  Example:\n    %s:\n      %s: %s"
                % (section, key, task, section, key, example)
            )


def _check_keys(task, raw):
    """Reject unknown keys, and say where a misplaced known key belongs."""
    schema = SCHEMA[task]
    every = set()
    for s in SECTIONS:
        every |= set(schema[s])
    for section in SECTIONS:
        allowed = schema[section]
        for key in _section(raw, section):
            if key in allowed:
                continue
            if key in every:
                right = [s for s in SECTIONS if key in schema[s]][0]
                raise ConfigError(
                    "key %r is not valid under %r — it belongs under %r."
                    % (key, section, right)
                )
            raise ConfigError(
                "unknown key %r under %r for task %r.\n%s"
                % (key, section, task, _suggest(key, allowed))
            )


def _flatten(task, raw):
    """Engine defaults -> preset -> user values. Keys are already validated."""
    schema = SCHEMA[task]
    cfg = default_cfg(task)

    preset = _section(raw, "run").get("preset", DEFAULT_PRESET)
    if preset is None:
        preset = DEFAULT_PRESET
    if preset not in PRESETS[task]:
        raise ConfigError(
            "unknown preset %r for task %r. Valid presets: %s"
            % (preset, task, ", ".join(sorted(PRESETS[task])))
        )
    cfg.update(PRESETS[task][preset])

    for section in SECTIONS:
        allowed = schema[section]
        for key, value in _section(raw, section).items():
            engine_key = allowed[key]
            if engine_key is None:      # consumed by this module (e.g. preset)
                continue
            cfg[engine_key] = value

    cfg["_preset"] = preset
    return cfg
