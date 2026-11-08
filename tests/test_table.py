from __future__ import annotations

import pytest

from dqguard.table import Table


def test_construction_dims():
    t = Table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert t.n_rows == 3
    assert t.n_cols == 2
    assert len(t) == 3
    assert t.column_names() == ["a", "b"]


def test_unequal_length_raises():
    with pytest.raises(ValueError):
        Table({"a": [1, 2], "b": [1]})


def test_non_string_name_raises():
    with pytest.raises(TypeError):
        Table({1: [1, 2]})  # type: ignore[dict-item]


def test_from_rows_missing_keys_become_none():
    t = Table.from_rows([{"a": 1}, {"a": 2, "b": 9}])
    assert t.column("a") == [1, 2]
    assert t.column("b") == [None, 9]


def test_from_csv(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,x\n,y\n3,\n", encoding="utf-8")
    t = Table.from_csv(str(p))
    assert t.column("a") == ["1", None, "3"]
    assert t.column("b") == ["x", "y", None]


def test_column_returns_copy():
    t = Table({"a": [1, 2]})
    col = t.column("a")
    col.append(99)
    assert t.column("a") == [1, 2]


def test_missing_column_raises():
    t = Table({"a": [1]})
    with pytest.raises(KeyError):
        t.column("nope")


def test_with_column_non_mutating():
    t = Table({"a": [1, 2]})
    t2 = t.with_column("b", [3, 4])
    assert not t.has_column("b")
    assert t2.column("b") == [3, 4]


def test_with_column_wrong_length():
    t = Table({"a": [1, 2]})
    with pytest.raises(ValueError):
        t.with_column("b", [1])


def test_drop_rename_select_map():
    t = Table({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    assert t.drop_column("b").column_names() == ["a", "c"]
    assert t.rename({"a": "A"}).column_names() == ["A", "b", "c"]
    assert t.select(["c", "a"]).column_names() == ["c", "a"]
    assert t.map_column("a", lambda x: x * 10).column("a") == [10, 20]


def test_select_unknown_raises():
    t = Table({"a": [1]})
    with pytest.raises(KeyError):
        t.select(["a", "z"])


def test_rows_and_to_rows():
    t = Table({"a": [1, 2], "b": ["x", "y"]})
    assert list(t.rows()) == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    assert t.to_rows() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_equality_and_copy():
    t = Table({"a": [1, 2]})
    assert t == t.copy()
    assert t != Table({"a": [1, 3]})
    assert (t == 5) is False


def test_cell_access():
    t = Table({"a": [10, 20, 30]})
    assert t.cell("a", 1) == 20


def test_empty_table():
    t = Table({})
    assert t.n_rows == 0
    assert t.n_cols == 0
