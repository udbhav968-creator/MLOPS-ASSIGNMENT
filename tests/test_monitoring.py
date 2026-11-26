"""PSI and the client's promotion rule."""

import numpy as np
import pandas as pd

from mp290.monitor import psi
from mp290.retrain import should_promote, split_later


def test_psi_is_small_for_same_data_and_big_for_shift():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(40, 10, 2000))
    assert psi(a, pd.Series(rng.normal(40, 10, 2000))) < 0.1
    assert psi(a, pd.Series(rng.normal(80, 10, 2000))) > 0.25


def test_promotion_rule_needs_8_percent():
    assert should_promote(100.0, 92.0, 0.08)        # exactly 8% lower: promote
    assert not should_promote(100.0, 93.0, 0.08)    # 7% lower: keep
    assert not should_promote(100.0, 120.0, 0.08)   # worse: keep


def test_split_later_is_by_date():
    df = pd.DataFrame({"record_id": list("abcdefghij"),
                       "date": [f"2026-07-{d:02d}" for d in [10, 1, 9, 2, 8, 3, 7, 4, 6, 5]]})
    first, last = split_later(df, 0.7)
    assert len(first) == 7 and len(last) == 3
    assert first["date"].max() < last["date"].min()
