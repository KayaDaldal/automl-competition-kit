# Benchmark

Measuring this pipeline on real public datasets instead of the synthetic
examples, so the README's numbers come from somewhere.

## What is measured

For every dataset, 20% of the rows are held out (the last H dates for time
series). Nothing is trained on them. Four systems are scored on identical
splits:

| | |
|---|---|
| `ours-fast` | the `fast` preset |
| `ours-full` | the `full` preset, capped by `--full-budget` seconds |
| `baseline` | one LightGBM with default parameters, ordinal-encoded categoricals |
| `leaky` | the same LightGBM, but target encoding fitted on the whole training set |

Each system's own cross-validated estimate is recorded next to its held-out
score. The gap between the two is the point: a pipeline that reports a number
it cannot reproduce on unseen data is worse than useless, because you pick your
submission with it.

This is a comparison against a plain baseline on a handful of datasets. It is
not a benchmark against AutoGluon, FLAML or anything else, and no claim of
that kind is made anywhere in this repo.

## Running it

```bash
python benchmark/leakage_demo.py            # 30 seconds, no downloads, no credentials
python benchmark/run_benchmark.py --check   # environment report
python benchmark/run_benchmark.py --verify  # check every Kaggle slug, downloads nothing
python benchmark/run_benchmark.py --offline # scikit-learn's bundled datasets only
python benchmark/run_benchmark.py           # everything in datasets.yaml
```

The full run is resumable: results are written to `results.json` after every
dataset, and a second run skips whatever is already in there. `--force` redoes
them, `--only NAME ...` picks specific ones, `--skip-full` halves the time.

Expect the full run to take one to three hours on a laptop CPU. `--fast-budget`
and `--full-budget` cap the per-dataset time budget; the pipeline stops after
the current fold when a budget runs out and still writes a valid submission.

### Kaggle datasets

The `kaggle:` entries in `datasets.yaml` need:

```bash
pip install kaggle
```

plus an API token at `~/.kaggle/kaggle.json` (Windows:
`%USERPROFILE%\.kaggle\kaggle.json`), created from your Kaggle account page
under Settings → API → Create New Token. `--check` reports whether it found one.

Kaggle *datasets* download without extra steps. Kaggle *competitions* require
accepting that competition's rules on the website first, which is why the list
here uses datasets. If a slug has been renamed or removed, that entry fails,
gets recorded in `RESULTS.md` under "Datasets that did not run", and the rest of
the run continues.

The `openml:` and `sklearn:` entries need no credentials at all.

## Output

- `results.json` — every number, machine-readable, appended as the run proceeds
- `RESULTS.md` — the tables, regenerated at the end of every run
- `_env.json` — package versions the run used
- `data/` and `out/` — downloaded data and per-run outputs, both gitignored

## Image classification

Separate script, because it needs different dependencies and different
hardware:

```bash
python benchmark/image_benchmark.py --check              # report the GPU and stop
python benchmark/image_benchmark.py --per-class 300      # a Kaggle image dataset
python benchmark/image_benchmark.py --slug owner/name --per-class 400
```

It downloads a Kaggle dataset laid out as one folder per class, samples
`--per-class` images from each, holds out 20%, and compares the fast preset,
the full preset and a single-fold fine-tune of the same backbone with no TTA
and no fold averaging. Results land in the same `results.json`.

On a CPU this measures the hardware rather than the code.
`benchmark/colab_benchmark.ipynb` runs the whole thing on a free Colab T4:
open it in Colab, set the runtime to GPU, and work down the cells.
