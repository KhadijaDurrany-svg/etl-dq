from __future__ import annotations

from dqguard.checks import (
    KIND_NOT_NULL,
    AllowedValues,
    CheckSuite,
    Finding,
    NotNull,
    Range,
    Referential,
    TypeCheck,
    Unique,
    summarise,
)
from dqguard.table import Table


def test_not_null():
    t = Table({"a": [1, None, 3, None]})
    rows = [f.row for f in NotNull("a").run(t)]
    assert rows == [1, 3]


def test_range_inclusive_bounds():
    t = Table({"a": [0, 1, 5, 10, 11, -1]})
    rows = sorted(f.row for f in Range("a", 1, 10).run(t))
    # 0 (<1), 11 (>10), -1 (<1) flagged; 1 and 10 are inclusive
    assert rows == [0, 4, 5]


def test_range_ignores_null_and_nonnumeric():
    t = Table({"a": [None, "big", 500]})
    rows = [f.row for f in Range("a", 0, 100).run(t)]
    assert rows == [2]


def test_unique_flags_later_occurrences():
    t = Table({"a": [1, 2, 1, 3, 2, 1]})
    rows = sorted(f.row for f in Unique("a").run(t))
    assert rows == [2, 4, 5]


def test_unique_skips_null():
    t = Table({"a": [None, None, 1, 1]})
    rows = [f.row for f in Unique("a").run(t)]
    assert rows == [3]


def test_referential():
    t = Table({"a": [1, 2, 9, None]})
    rows = [f.row for f in Referential("a", frozenset({1, 2, 3})).run(t)]
    assert rows == [2]


def test_referential_from_values():
    chk = Referential.from_values("a", [1, 2, None, 2])
    assert chk.keys == frozenset({1, 2})


def test_type_check_strict():
    t = Table({"a": [1, 2.0, "x", None, True]})
    rows = sorted(f.row for f in TypeCheck("a", "int").run(t))
    # 2.0 float, "x" str, True bool all != int; None skipped
    assert rows == [1, 2, 4]


def test_allowed_values():
    t = Table({"a": ["x", "y", "z", None]})
    rows = [f.row for f in AllowedValues("a", frozenset({"x", "y"})).run(t)]
    assert rows == [2]


def test_check_suite_runs_all():
    t = Table({"a": [1, None], "b": [5, 5]})
    suite = CheckSuite([NotNull("a"), Unique("b")])
    findings = suite.run(t)
    assert len(findings) == 2
    assert suite.columns() == ["a", "b"]


def test_finding_key_and_hash():
    f1 = Finding(KIND_NOT_NULL, "a", 1)
    f2 = Finding(KIND_NOT_NULL, "a", 1)
    assert f1 == f2
    assert f1.key() == (KIND_NOT_NULL, "a", 1)
    assert len({f1, f2}) == 1


def test_summarise():
    findings = [NotNull("a").run(Table({"a": [None, None]}))][0]
    assert summarise(findings) == {KIND_NOT_NULL: 2}
    assert summarise([]) == {}


def test_clean_column_no_findings():
    t = Table({"a": [1, 2, 3]})
    assert NotNull("a").run(t) == []
    assert Unique("a").run(t) == []
    assert Range("a", 0, 10).run(t) == []
    assert TypeCheck("a", "int").run(t) == []
