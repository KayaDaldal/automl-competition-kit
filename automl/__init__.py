"""automl — one config in, a submission file and a CV score out.

    from automl import run
    result = run("config.yaml")        # or run({...}) with the same structure

The CLI (`automl run config.yaml`) goes through this same function, so a
notebook and a terminal do exactly the same thing.
"""

from .core.config import ConfigError, load_config

__version__ = "0.1.0"
__all__ = ["run", "load_config", "ConfigError", "__version__"]

_ENGINES = {
    "classification": ("automl.tasks.classification", "run_classification"),
    "regression": ("automl.tasks.regression", "run_regression"),
    "forecast": ("automl.tasks.forecast", "run_forecast"),
    "image": ("automl.tasks.image", "run_image"),
}


def run(config, **overrides):
    """Run whichever engine the config asks for.

    `config` is a path to a YAML file or a dict with the same structure.
    `overrides` are schema sections, e.g. run={"preset": "full"}.

    Returns the engine's result dict, which always contains at least
    `cv_score`, `metric` and `submission_files`.
    """
    task, cfg = load_config(config, overrides or None)
    module_name, func_name = _ENGINES[task]
    import importlib

    engine = getattr(importlib.import_module(module_name), func_name)
    return engine(cfg)
