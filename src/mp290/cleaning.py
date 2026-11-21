"""
Cleaning for train.csv. Fixes only what profiling.py found present:

  1. duplicate_rows  -> drop the exact duplicate rows
  2. unit_mixup      -> divide the affected group's column by the factor
                        (Yellow Line rain_mm was recorded x 0.1, so this multiplies by 10)
  3. target_typos    -> divide the 10x-too-big targets by 10
  4. leaking_column  -> drop it, so it can never become a model feature

The fixes are driven by the check results, not hard-coded numbers, so the same
function works on any file the checks have been run on.
"""

import pandas as pd

from mp290.profiling import (check_leaking_column, check_unit_mixup,
                             target_typo_mask, TARGET)


def clean_train(df: pd.DataFrame, holdout_columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    # 1. duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. unit mix-up
    unit = check_unit_mixup(df)
    if unit["present"]:
        rows = df["line"] == unit["group_value"]
        df.loc[rows, unit["column"]] = (df.loc[rows, unit["column"]] / unit["factor"]).round(2)

    # 3. target typos
    typos = target_typo_mask(df)
    df.loc[typos, TARGET] = (df.loc[typos, TARGET] // 10).astype(int)

    # 4. leaking column
    leak = check_leaking_column(df, holdout_columns)
    if leak["present"]:
        df = df.drop(columns=[leak["column"]])

    return df


def load_clean_train(params: dict) -> pd.DataFrame:
    from mp290.config import path_of
    raw = pd.read_csv(path_of(params["data"]["train"]))
    holdout_cols = list(pd.read_csv(path_of(params["data"]["holdout"]), nrows=1).columns)
    return clean_train(raw, holdout_cols)
