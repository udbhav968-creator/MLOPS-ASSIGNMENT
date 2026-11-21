"""The seven data checks and the cleaning that follows them."""

from mp290.cleaning import clean_train
from mp290.profiling import (check_duplicate_rows, check_leaking_column,
                             check_spelling_variants, check_stuck_sensor,
                             check_target_typos, check_unit_mixup, run_all_checks)
import pandas as pd


def test_clean_data_has_no_problems_except_the_leak(small_train, holdout_columns):
    report = run_all_checks(small_train, holdout_columns)
    assert report["leaking_column"] == {"present": True, "column": "gate_count_end_of_hour"}
    for name in ["unit_mixup", "sentinel_value", "duplicate_rows",
                 "stuck_sensor", "target_typos", "spelling_variants"]:
        assert report[name]["present"] is False, name


def test_finds_duplicates_typos_and_unit_mixup(small_train):
    df = pd.concat([small_train, small_train.iloc[:3]], ignore_index=True)   # 3 extra copies
    df.loc[5, "entries_per_hour"] *= 10                                        # 1 typo
    df["rain_mm"] = 30.0 + (df.index % 5)                                     # similar on every line
    yellow = df["line"] == "Yellow Line"
    df.loc[yellow, "rain_mm"] = df.loc[yellow, "rain_mm"] / 10                 # Yellow typed in cm
    assert check_duplicate_rows(df)["count"] == 3
    assert check_target_typos(df)["count"] == 1
    unit = check_unit_mixup(df)
    assert (unit["column"], unit["group_value"], unit["factor"]) == ("rain_mm", "Yellow Line", 0.1)


def test_finds_stuck_sensor_and_spelling(small_train):
    df = small_train.copy()
    df.loc[10:20, "temp_c"] = 21.5                                             # 11 days stuck
    stuck = check_stuck_sensor(df)
    assert stuck["present"] and stuck["column"] == "temp_c"
    assert (stuck["start_date"], stuck["end_date"]) == (df.date[10], df.date[20])
    df.loc[0, "line"] = "red line "
    assert check_spelling_variants(df) == {"present": True, "column": "line", "true_levels": 3}


def test_cleaning_fixes_everything(small_train, holdout_columns):
    df = pd.concat([small_train, small_train.iloc[:2]], ignore_index=True)
    true_target = df.loc[7, "entries_per_hour"]
    df.loc[7, "entries_per_hour"] *= 10
    clean = clean_train(df, holdout_columns)
    assert len(clean) == len(small_train)
    assert "gate_count_end_of_hour" not in clean.columns
    assert clean.loc[clean.record_id == df.loc[7, "record_id"], "entries_per_hour"].item() == true_target
