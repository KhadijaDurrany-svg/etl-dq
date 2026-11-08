"""Scoring and statistics for the benchmark.

Findings are matched against ground truth as *sets* of ``Finding`` keys, so a
detector is rewarded only for the exact issues it names (right kind, right
column, right row). On top of per-case precision/recall/F1 we provide
non-parametric uncertainty: a bootstrap 95% CI over cases and a paired
sign-flip permutation test between two detectors.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .checks import Finding


@dataclass(frozen=True)
class Score:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score_findings(predicted: Iterable[Finding], truth: Iterable[Finding]) -> Score:
    """Compare predicted findings against ground truth by exact key match."""
    pred = {f.key() for f in predicted}
    gold = {f.key() for f in truth}
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return Score(tp=tp, fp=fp, fn=fn)


def aggregate(scores: Sequence[Score]) -> Score:
    """Micro-average: sum tp/fp/fn across cases, then derive P/R/F1."""
    return Score(
        tp=sum(s.tp for s in scores),
        fp=sum(s.fp for s in scores),
        fn=sum(s.fn for s in scores),
    )


def bootstrap_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    iterations: int = 5000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Resamples cases (the unit of analysis) with replacement.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    n = arr.size
    idx = rng.integers(0, n, size=(iterations, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    rng: np.random.Generator,
    iterations: int = 10000,
) -> dict[str, float]:
    """Two-sided paired sign-flip permutation test on per-case differences.

    ``a`` and ``b`` are paired per-case metrics (e.g. F1 of two detectors on the
    same cases). Returns the observed mean difference and a p-value. The
    smallest reportable p-value is ``1 / (iterations + 1)``.
    """
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if av.shape != bv.shape:
        raise ValueError("inputs must be the same length")
    diff = av - bv
    n = diff.size
    observed = float(diff.mean())
    if n == 0:
        return {"observed_diff": 0.0, "p_value": 1.0, "iterations": float(iterations)}
    signs = rng.integers(0, 2, size=(iterations, n)) * 2 - 1
    perm_means = (signs * diff).mean(axis=1)
    count = int(np.sum(np.abs(perm_means) >= abs(observed) - 1e-12))
    p = (count + 1) / (iterations + 1)
    return {"observed_diff": observed, "p_value": float(p), "iterations": float(iterations)}
