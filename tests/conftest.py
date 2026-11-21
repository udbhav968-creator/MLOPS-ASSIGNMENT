"""Shared test data. Built inside the tests: the CSVs are in DVC, not in git."""

import numpy as np
import pandas as pd
import pytest

NUMERIC = ["hour", "nearby_offices", "temp_c", "rain_mm", "feeder_buses", "ticket_price_rs"]
CATEGORICAL = ["day_type", "line"]
LINES = ["Red Line", "Blue Line", "Yellow Line"]


@pytest.fixture
def small_train() -> pd.DataFrame:
    """60 tidy rows that look like train.csv: entries ~= gate count, no problems."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "record_id": [f"R{i:05d}" for i in range(n)],
        "date": pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "hour": rng.integers(6, 24, n),
        "nearby_offices": rng.integers(0, 80, n),
        "temp_c": rng.uniform(8, 45, n).round(1),
        "rain_mm": rng.uniform(0, 60, n).round(1),
        "feeder_buses": rng.integers(0, 40, n),
        "ticket_price_rs": rng.integers(10, 60, n),
        "day_type": rng.choice(["weekday", "weekend", "holiday"], n),
        "line": [LINES[i % 3] for i in range(n)],
    })
    df["entries_per_hour"] = (1000 + 25 * df.nearby_offices + 50 * df.hour).astype(int)
    df["gate_count_end_of_hour"] = df["entries_per_hour"] + rng.integers(-20, 20, n)
    return df


@pytest.fixture
def holdout_columns() -> list[str]:
    return ["record_id", "date"] + NUMERIC + CATEGORICAL
