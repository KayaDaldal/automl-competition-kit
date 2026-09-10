"""The four engines. Each one is standalone: given a flat config dict it runs
cross-validation and writes a submission.

They are imported lazily by `automl.run` so that, for example, a tabular run
never imports torch.
"""

__all__ = ["classification", "regression", "forecast", "image"]
