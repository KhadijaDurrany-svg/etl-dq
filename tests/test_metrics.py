from __future__ import annotations

import numpy as np
import pytest

from dqguard.checks import Finding
from dqguard.metrics import (
    Score,
    aggregate,
    bootstrap_ci,
    paired_permutation_test,
    score_findings,
)


def _f(kind, col, row):
    return Finding(kind, col, row)


def test_score_perfect():
    gt = {_f("not_null", "a", 0), _f("range", "b", 1)}
    s = score_findings(gt, gt)
    assert (s.tp, s.fp, s.fn) == (2, 0, 0)
    assert s.f1 == 1.0


def test_score_partial():
    gt = {_f("not_null", "a", 0), _f("range", "b", 1), _f("unique", "c", 2)}
    pred = {_f("not_null", "a", 0), _f("range", "b", 99)}  # 1 TP, 1 FP, 2 FN
    s = score_findings(pred, gt)
    assert (s.tp, s.fp, s.fn) == (1, 1, 2)
    assert s.precision == 0.5
    assert s.recall == pytest.approx(1 / 3)


def test_score_empty_sets():
    s = score_findings([], [])
    assert s.precision == 1.0 and s.recall == 1.0
    assert s.f1 == 1.0


def test_aggregate():
    s = aggregate([Score(2, 1, 0), Score(3, 0, 2)])
    assert (s.tp, s.fp, s.fn) == (5, 1, 2)


def test_bootstrap_ci_bounds_and_determinism():
    vals = [0.8, 0.9, 0.85, 0.95, 0.7]
    lo1, hi1 = bootstrap_ci(vals, np.random.default_rng(0), iterations=1000)
    lo2, hi2 = bootstrap_ci(vals, np.random.default_rng(0), iterations=1000)
    assert (lo1, hi1) == (lo2, hi2)
    assert 0.0 <= lo1 <= hi1 <= 1.0
    assert lo1 <= np.mean(vals) <= hi1


def test_bootstrap_empty():
    assert bootstrap_ci([], np.random.default_rng(0)) == (0.0, 0.0)


def test_permutation_identical_gives_p_one():
    res = paired_permutation_test([0.5, 0.6], [0.5, 0.6], np.random.default_rng(0), iterations=500)
    assert res["observed_diff"] == 0.0
    assert res["p_value"] == 1.0


def test_permutation_large_difference_small_p():
    a = [1.0] * 12
    b = [0.0] * 12
    res = paired_permutation_test(a, b, np.random.default_rng(1), iterations=5000)
    assert res["observed_diff"] == 1.0
    assert 0.0 < res["p_value"] < 0.05


def test_permutation_shape_mismatch():
    with pytest.raises(ValueError):
        paired_permutation_test([1.0], [1.0, 2.0], np.random.default_rng(0))


def test_permutation_min_resolution():
    a = [1.0] * 8
    b = [0.0] * 8
    res = paired_permutation_test(a, b, np.random.default_rng(2), iterations=1000)
    assert res["p_value"] >= 1 / (1000 + 1)
