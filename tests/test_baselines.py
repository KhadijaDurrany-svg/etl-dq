from __future__ import annotations

import numpy as np

from dqguard.baselines import (
    DETECTORS,
    full_detector,
    null_only_detector,
    random_detector,
)
from dqguard.checks import KIND_NOT_NULL
from dqguard.generate import generate_case


def test_detector_registry():
    assert set(DETECTORS) == {"full", "null_only", "random"}


def test_null_only_only_null_kinds():
    case = generate_case(0)
    findings = null_only_detector(case, np.random.default_rng(0))
    assert findings
    assert all(f.kind == KIND_NOT_NULL for f in findings)


def test_full_superset_of_null_only():
    case = generate_case(1)
    full = {f.key() for f in full_detector(case, np.random.default_rng(0))}
    nulls = {f.key() for f in null_only_detector(case, np.random.default_rng(0))}
    assert nulls <= full


def test_random_reproducible():
    case = generate_case(2)
    a = random_detector(case, np.random.default_rng(5))
    b = random_detector(case, np.random.default_rng(5))
    assert [f.key() for f in a] == [f.key() for f in b]


def test_random_differs_across_seeds():
    case = generate_case(2)
    a = {f.key() for f in random_detector(case, np.random.default_rng(1))}
    b = {f.key() for f in random_detector(case, np.random.default_rng(2))}
    assert a != b


def test_full_detector_strong_recall():
    case = generate_case(3)
    full = {f.key() for f in full_detector(case, np.random.default_rng(0))}
    gt = {f.key() for f in case.ground_truth}
    # should recover most of the ground truth
    assert len(full & gt) / len(gt) > 0.7
