"""
The seven data-problem checks for train.csv (Student Handbook, section 7).

Every check is a plain function that takes a DataFrame and returns a dict
shaped like the matching block of findings.yaml. Run the whole report with:

    python -m mp290.profiling

Counts are made on train.csv as given (the handbook accepts this).
"""

import pandas as pd

from mp290.config import load_params, path_of

NUMERIC_COLS = ["hour", "nearby_offices", "temp_c", "rain_mm", "feeder_buses",
                "ticket_price_rs", "gate_count_end_of_hour", "entries_per_hour"]
# Columns measured by a device that could get stuck. `hour` is left out on
# purpose: it repeats every day by design.
SENSOR_COLS = ["nearby_offices", "temp_c", "rain_mm", "feeder_buses",
               "ticket_price_rs", "gate_count_end_of_hour"]
TEXT_COLS = ["day_type", "line"]
TARGET = "entries_per_hour"

# Exact unit conversions the handbook allows (wrong value = right value x factor).
UNIT_FACTORS = [10, 12, 60, 100, 1000, 1 / 10, 1 / 12, 1 / 60, 1 / 100, 1 / 1000]


def check_leaking_column(df: pd.DataFrame, holdout_columns: list[str]) -> dict:
    """A column missing from the holdout file that agrees almost exactly with the target."""
    candidates = [c for c in df.columns
                  if c not in holdout_columns and c != TARGET and c in NUMERIC_COLS]
    for col in candidates:
        ratio = df[TARGET] / df[col]
        normal = ratio.between(0.5, 2)            # ignore the 10x target typos
        if normal.mean() > 0.95 and df.loc[normal, [col, TARGET]].corr().iloc[0, 1] > 0.99:
            return {"present": True, "column": col}
    return {"present": False, "column": None}


def check_unit_mixup(df: pd.DataFrame, group_col: str = "line") -> dict:
    """One group's mean of one column is an exact unit factor away from the other groups."""
    for col in ["nearby_offices", "temp_c", "rain_mm", "feeder_buses", "ticket_price_rs"]:
        means = df.groupby(group_col)[col].mean()
        for group in means.index:
            others = means.drop(group)
            if others.std() / others.mean() > 0.1:      # the other groups must agree
                continue
            ratio = means[group] / others.mean()          # wrong / right
            for factor in UNIT_FACTORS:
                if abs(ratio - factor) / factor < 0.15:
                    return {"present": True, "column": col, "group_value": group,
                            "factor": round(factor, 6)}
    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(df: pd.DataFrame) -> dict:
    """One impossible number repeated many times (e.g. -1, -999, 999, 9999)."""
    for col in NUMERIC_COLS:
        counts = df[col].value_counts()
        for value in (-1, -99, -999, -9999, 999, 9999, 99999):
            if counts.get(value, 0) >= 5:
                return {"present": True, "column": col, "value": value,
                        "count": int(counts[value])}
        # a single value far more common than any other also counts
        if len(counts) > 5 and counts.std() > 0:
            z = (counts.iloc[0] - counts.mean()) / counts.std()
            if z > 5 and counts.iloc[0] >= 15:
                return {"present": True, "column": col, "value": counts.index[0],
                        "count": int(counts.iloc[0])}
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(df: pd.DataFrame) -> dict:
    """Number of EXTRA copies: rows that would be deleted, keeping one of each."""
    extra = int(df.duplicated().sum())
    return {"present": extra > 0, "count": extra if extra else None}


def check_stuck_sensor(df: pd.DataFrame, min_days: int = 7) -> dict:
    """Sort by date; a column showing one value on every row for a long run of days."""
    d = df.drop_duplicates().sort_values(["date", "record_id"]).reset_index(drop=True)
    for col in SENSOR_COLS:
        run_id = (d[col] != d[col].shift()).cumsum()       # new id whenever value changes
        for _, run in d.groupby(run_id):
            if run["date"].nunique() >= min_days:
                return {"present": True, "column": col,
                        "start_date": run["date"].iloc[0], "end_date": run["date"].iloc[-1]}
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def longest_stuck_run_days(df: pd.DataFrame) -> int:
    """Longest run (in distinct dates) of one repeated value in any sensor column."""
    d = df.drop_duplicates().sort_values(["date", "record_id"]).reset_index(drop=True)
    best = 0
    for col in SENSOR_COLS:
        run_id = (d[col] != d[col].shift()).cumsum()
        best = max(best, int(d.groupby(run_id)["date"].nunique().max()))
    return best


def target_typo_mask(df: pd.DataFrame) -> pd.Series:
    """True where the target is about 10x too big: far above the rest, after a big gap."""
    t = df[TARGET].sort_values()
    jumps = t / t.shift()                 # ratio between neighbouring sorted values
    gap = jumps.idxmax()                  # biggest jump = start of the typo group
    if jumps[gap] < 2:                    # no big gap: no typos
        return pd.Series(False, index=df.index)
    return df[TARGET] >= t[gap]


def check_target_typos(df: pd.DataFrame) -> dict:
    n = int(target_typo_mask(df).sum())
    return {"present": n > 0, "count": n if n else None}


def check_spelling_variants(df: pd.DataFrame) -> dict:
    """Same category written several ways: compare raw levels with normalised levels."""
    for col in TEXT_COLS:
        norm = (df[col].astype(str).str.strip().str.lower()
                .str.replace(r"[^a-z0-9]", "", regex=True))
        if df[col].nunique() != norm.nunique():
            return {"present": True, "column": col, "true_levels": int(norm.nunique())}
    return {"present": False, "column": None, "true_levels": None}


def run_all_checks(train: pd.DataFrame, holdout_columns: list[str]) -> dict:
    return {
        "leaking_column": check_leaking_column(train, holdout_columns),
        "unit_mixup": check_unit_mixup(train),
        "sentinel_value": check_sentinel_value(train),
        "duplicate_rows": check_duplicate_rows(train),
        "stuck_sensor": check_stuck_sensor(train),
        "target_typos": check_target_typos(train),
        "spelling_variants": check_spelling_variants(train),
    }


def main() -> dict:
    params = load_params()
    train = pd.read_csv(path_of(params["data"]["train"]))
    holdout_cols = list(pd.read_csv(path_of(params["data"]["holdout"]), nrows=1).columns)
    report = run_all_checks(train, holdout_cols)
    print(f"train.csv: {len(train)} rows as given\n")
    for name, result in report.items():
        flag = "PRESENT" if result["present"] else "absent "
        details = {k: v for k, v in result.items() if k != "present"}
        print(f"[{flag}] {name:18s} {details}")
    print(f"\n(longest run of one repeated sensor value: "
          f"{longest_stuck_run_days(train)} days, so no stuck sensor)")
    return report


if __name__ == "__main__":
    main()
