"""Command line interface.

    automl run config.yaml [--preset fast|full] [--out-dir DIR] [--task TASK]
    automl validate config.yaml

`run` is a thin wrapper around `automl.run`, so the CLI and the Python API
take exactly the same path through the code.
"""

import argparse
import sys

from . import __version__, run as run_config
from .core.config import ConfigError, load_config


def _build_parser():
    p = argparse.ArgumentParser(
        prog="automl",
        description="Run a competition ML pipeline from a single config file.",
    )
    p.add_argument("--version", action="version", version="automl %s" % __version__)
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="run the pipeline described by a config file")
    r.add_argument("config", help="path to the YAML config")
    r.add_argument("--preset", choices=["fast", "full"],
                   help="override run.preset")
    r.add_argument("--out-dir", help="override run.out_dir")
    r.add_argument("--task", help="override the task (classification, regression, "
                                  "forecast, image, auto)")

    v = sub.add_parser("validate", help="check a config file without running anything")
    v.add_argument("config", help="path to the YAML config")
    v.add_argument("--preset", choices=["fast", "full"])
    v.add_argument("--out-dir")
    v.add_argument("--task")
    return p


def _overrides(args):
    run_over = {}
    if getattr(args, "preset", None):
        run_over["preset"] = args.preset
    if getattr(args, "out_dir", None):
        run_over["out_dir"] = args.out_dir
    over = {}
    if run_over:
        over["run"] = run_over
    if getattr(args, "task", None):
        over["task"] = args.task
    return over


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "validate":
            task, cfg = load_config(args.config, _overrides(args) or None)
            print("config OK — task=%s, preset=%s, metric=%s, out_dir=%s"
                  % (task, cfg.get("_preset"), cfg.get("metric"), cfg.get("out_dir")))
            return 0

        result = run_config(args.config, **_overrides(args))
        print("CV %s = %.5f" % (result["metric"], result["cv_score"]))
        for f in result["submission_files"]:
            print("wrote", f)
        return 0
    except ConfigError as e:
        print("config error: %s" % e, file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print("file not found: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
