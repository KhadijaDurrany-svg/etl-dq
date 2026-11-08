from __future__ import annotations

import numpy as np
import pytest

from dqguard.baselines import full_detector
from dqguard.checks import DQ_KINDS
from dqguard.drift import KIND_DRIFT_ADD, KIND_DRIFT_DROP, KIND_DRIFT_TYPE
from dqguard.generate import DatasetSpec, generate_case


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_clean_base_has_zero_findings(seed):
    case = generate_case(seed)
    assert case.suite.run(case.clean_orders) == []


def test_determinism():
    a = generate_case(5)
    b = generate_case(5)
    assert a.candidate == b.candidate
    assert a.ground_truth == b.ground_truth
    assert a.clean_orders == b.clean_orders


def test_ground_truth_counts():
    spec = DatasetSpec()
    case = generate_case(3, spec)
    gt = case.ground_truth
    cell = sum(1 for f in gt if f.kind in DQ_KINDS)
    assert cell == spec.total_cell_defects
    add = sum(1 for f in gt if f.kind == KIND_DRIFT_ADD)
    drop = sum(1 for f in gt if f.kind == KIND_DRIFT_DROP)
    typ = sum(1 for f in gt if f.kind == KIND_DRIFT_TYPE)
    assert (add, drop, typ) == (2, 2, 6)


def test_candidate_schema_changes():
    case = generate_case(9)
    cols = set(case.candidate.column_names())
    assert "promo_flag" in cols and "coupon" in cols
    assert "ext_ref" not in cols and "legacy_id" not in cols


def test_cell_defects_distinct_rows_per_column():
    case = generate_case(11)
    by_col: dict[str, list[int]] = {}
    for f in case.ground_truth:
        if f.row is not None:
            by_col.setdefault(f.column, []).append(f.row)
    for rows in by_col.values():
        assert len(rows) == len(set(rows))


def test_full_detector_catches_all_cell_defects():
    case = generate_case(2)
    predicted = {f.key() for f in full_detector(case, np.random.default_rng(0))}
    cell_gt = {f.key() for f in case.ground_truth if f.kind in DQ_KINDS}
    # Exact declarative checks must recover every injected cell defect.
    assert cell_gt <= predicted


def test_customers_keys_match():
    case = generate_case(4)
    assert set(case.customers.column("customer_id")) == set(case.keys)
