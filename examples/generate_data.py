"""Generate the synthetic datasets the example configs point at.

The three tabular datasets are small enough to be committed to the repo, so
`automl run examples/classification.yaml` works straight after a clone. Run
this script only if you want to regenerate them, or to create the image
dataset (which is not committed):

    python examples/generate_data.py            # tabular only, rewrites the CSVs
    python examples/generate_data.py --image    # also builds examples/data/image/

Every dataset has a known signal planted in it, so a run that reports a
sensible CV score is actually learning something rather than getting lucky.
"""

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _out(name):
    os.makedirs(DATA, exist_ok=True)
    return os.path.join(DATA, name)


def make_classification():
    """Binary target driven by age, income and one city, plus a high-cardinality
    id column and a date column, so the whole preprocessing path gets used."""
    rng = np.random.RandomState(42)
    n = 1200
    cities = ["ist", "ank", "izm", "bur", "ant"]
    user_ids = ["u%04d" % i for i in rng.randint(0, 400, n)]      # high cardinality
    age = rng.randint(18, 70, n).astype(float)
    income = rng.normal(50, 15, n)
    city = rng.choice(cities, n)
    dt = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.randint(0, 400, n), unit="D")

    logit = 0.04 * (age - 40) + 0.03 * (income - 50) + (city == "ist") * 0.8 - 0.5
    target = (rng.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)
    age[rng.rand(n) < 0.05] = np.nan                              # missing values

    df = pd.DataFrame({"age": age, "income": income, "city": city,
                       "user_id": user_ids, "signup_date": dt, "target": target})
    train = df.iloc[:900].reset_index(drop=True)
    test = df.iloc[900:].drop(columns=["target"]).reset_index(drop=True)
    test.insert(0, "id", ["t%04d" % i for i in range(len(test))])

    train.to_csv(_out("classification_train.csv"), index=False)
    test.to_csv(_out("classification_test.csv"), index=False)
    pd.DataFrame({"id": [test["id"].iloc[0]], "target": [0]}).to_csv(
        _out("classification_example.csv"), index=False)
    print("classification: %d train / %d test rows" % (len(train), len(test)))


def make_regression():
    """Log-normal house-price style target: skewed, so the log1p path kicks in."""
    rng = np.random.RandomState(0)
    n = 1500
    area = rng.normal(120, 40, n).clip(30, 400)
    rooms = rng.randint(1, 7, n).astype(float)
    district = rng.choice(["A", "B", "C", "D", "E"], n)
    agent = ["ag%03d" % i for i in rng.randint(0, 300, n)]        # high cardinality
    base = (1000 * area + 25000 * rooms
            + (district == "A") * 200000 - (district == "E") * 80000)
    price = np.exp(np.log(base.clip(1)) + rng.normal(0, 0.25, n))
    area[rng.rand(n) < 0.05] = np.nan

    df = pd.DataFrame({"area": area, "rooms": rooms, "district": district,
                       "agent": agent, "price": price})
    train = df.iloc[:1100].reset_index(drop=True)
    test = df.iloc[1100:].drop(columns=["price"]).reset_index(drop=True)
    test.insert(0, "id", ["h%04d" % i for i in range(len(test))])

    train.to_csv(_out("regression_train.csv"), index=False)
    test.to_csv(_out("regression_test.csv"), index=False)
    pd.DataFrame({"id": ["h0000"], "price": [0.0]}).to_csv(
        _out("regression_example.csv"), index=False)
    print("regression: %d train / %d test rows" % (len(train), len(test)))


def make_forecast(horizon=14):
    """Four daily series with a level, a trend and weekly seasonality. The last
    `horizon` days are held out as the test set."""
    rng = np.random.RandomState(0)
    stores = ["A", "B", "C", "D"]
    days = pd.date_range("2023-01-01", periods=220, freq="D")
    rows = []
    for si, s in enumerate(stores):
        level = 100 + si * 40
        for t, d in enumerate(days):
            weekly = 15 * np.sin(2 * np.pi * d.dayofweek / 7)
            rows.append({"store": s, "date": d,
                         "sales": max(0, level + 0.25 * t + weekly + rng.normal(0, 6))})
    full = pd.DataFrame(rows)

    cut = days[-horizon]
    train = full[full["date"] < cut].reset_index(drop=True)
    test_full = full[full["date"] >= cut].reset_index(drop=True)
    test = test_full.drop(columns=["sales"]).reset_index(drop=True)
    test.insert(0, "id", ["%s_%s" % (r.store, r.date.date()) for r in test.itertuples()])

    train.to_csv(_out("forecast_train.csv"), index=False)
    test.to_csv(_out("forecast_test.csv"), index=False)
    # The true future values, so you can check the forecast against reality.
    test_full.assign(id=test["id"]).to_csv(_out("forecast_truth.csv"), index=False)
    pd.DataFrame({"id": [test["id"].iloc[0]], "sales": [0.0]}).to_csv(
        _out("forecast_example.csv"), index=False)
    print("forecast: %d train rows, %d series, horizon %d"
          % (len(train), len(stores), horizon))


def make_image():
    """Two visually separable classes: one bright, one dark, plus noise.
    Not committed to the repo — run this script with --image to build it."""
    from PIL import Image

    rng = np.random.RandomState(0)
    root = os.path.join(DATA, "image")
    for sub in ("train/bright", "train/dark", "test"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    def save(path, bright):
        base = 170 if bright else 70
        arr = np.clip(rng.randn(48, 48, 3) * 25 + base, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path)

    for i in range(40):
        save(os.path.join(root, "train/bright/b%02d.jpg" % i), True)
        save(os.path.join(root, "train/dark/d%02d.jpg" % i), False)
    for i in range(12):
        save(os.path.join(root, "test/image%02d.jpg" % i), i % 2 == 0)
    print("image: 80 train / 12 test images under %s" % root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", action="store_true",
                    help="also build the synthetic image dataset")
    args = ap.parse_args()

    make_classification()
    make_regression()
    make_forecast()
    if args.image:
        make_image()
