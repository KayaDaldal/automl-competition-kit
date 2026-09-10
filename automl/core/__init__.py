"""Shared building blocks used by all four engines.

Nothing in here is task-specific: reading tables and submission examples,
column guessing and date handling, leakage-safe target encoding, metrics,
blend-weight search, booster fitting, submission writing, and the config
schema.
"""

from . import blend, boosters, config, io, metrics, prep, submission  # noqa: F401

__all__ = ["blend", "boosters", "config", "io", "metrics", "prep", "submission"]
