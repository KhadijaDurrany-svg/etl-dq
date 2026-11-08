from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from dqguard.checks import Finding, NotNull, Range, TypeCheck, Unique
from dqguard.generate import generate_case
from dqguard.metrics import score_findings
from dqguard.schema import Fingerprint, infer_column_type, value_type
from dqguard.table import Table

_finite = st.floats(allow_nan=False, allow_infinity=False, width=32)


@given(st.lists(st.one_of(st.none(), st.integers())))
def test_not_null_flags_exactly_none(values):
    t = Table({"a": values})
    flagged = {f.row for f in NotNull("a").run(t)}
    expected = {i for i, v in enumerate(values) if v is None}
    assert flagged == expected


@given(st.lists(_finite, min_size=1), _finite, _finite)
def test_range_flags_exactly_out_of_bounds(values, b1, b2):
    lo, hi = (b1, b2) if b1 <= b2 else (b2, b1)
    t = Table({"a": values})
    flagged = {f.row for f in Range("a", lo, hi).run(t)}
    expected = {i for i, v in enumerate(values) if v < lo or v > hi}
    assert flagged == expected


@given(st.lists(st.integers(min_value=0, max_value=20)))
def test_unique_flags_exactly_later_occurrences(values):
    t = Table({"a": values})
    flagged = sorted(f.row for f in Unique("a").run(t))
    seen: set[int] = set()
    expected = []
    for i, v in enumerate(values):
        if v in seen:
            expected.append(i)
        else:
            seen.add(v)
    assert flagged == expected


@given(st.lists(st.one_of(st.none(), st.integers(), st.text(max_size=3))))
def test_type_check_matches_value_type(values):
    t = Table({"a": values})
    flagged = {f.row for f in TypeCheck("a", "int").run(t)}
    expected = {i for i, v in enumerate(values) if v is not None and value_type(v) != "int"}
    assert flagged == expected


@given(
    st.lists(st.sampled_from(["not_null", "range", "unique"]), min_size=0, max_size=6),
)
def test_score_f1_in_unit_interval(kinds):
    gold = {Finding(k, "c", i) for i, k in enumerate(kinds)}
    pred = {Finding(k, "c", i) for i, k in enumerate(kinds) if i % 2 == 0}
    s = score_findings(pred, gold)
    assert 0.0 <= s.f1 <= 1.0
    assert 0.0 <= s.precision <= 1.0
    assert 0.0 <= s.recall <= 1.0


@given(gold=st.sets(st.tuples(st.sampled_from(["a", "b"]), st.integers(0, 5)), min_size=1))
def test_score_identical_is_perfect(gold):
    findings = {Finding("range", c, r) for c, r in gold}
    s = score_findings(findings, findings)
    assert s.f1 == 1.0


@given(
    names=st.lists(st.text(min_size=1, max_size=4), min_size=1, max_size=6, unique=True),
    types=st.lists(st.sampled_from(["int", "float", "str"]), min_size=1, max_size=6),
)
def test_fingerprint_roundtrip_property(names, types):
    n = min(len(names), len(types))
    names, types = names[:n], types[:n]
    fp = Fingerprint(types=dict(zip(names, types, strict=True)), order=tuple(names))
    assert Fingerprint.from_dict(fp.to_dict()) == fp


@given(st.lists(st.integers(), min_size=1, max_size=200))
def test_infer_homogeneous_int(values):
    assert infer_column_type(values, sample_size=200, rng=np.random.default_rng(0)) == "int"


@settings(max_examples=20, deadline=None)
@given(st.integers(min_value=0, max_value=5000))
def test_clean_generated_dataset_is_defect_free(seed):
    case = generate_case(seed)
    assert case.suite.run(case.clean_orders) == []
