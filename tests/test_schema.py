from __future__ import annotations

import numpy as np

from dqguard.schema import (
    TYPE_BOOL,
    TYPE_FLOAT,
    TYPE_INT,
    TYPE_NONE,
    TYPE_STR,
    Fingerprint,
    fingerprint_table,
    infer_column_type,
    value_type,
)
from dqguard.table import Table


def test_value_type_scalars():
    assert value_type(None) == TYPE_NONE
    assert value_type(True) == TYPE_BOOL  # bool checked before int
    assert value_type(3) == TYPE_INT
    assert value_type(3.5) == TYPE_FLOAT
    assert value_type("x") == TYPE_STR


def test_infer_homogeneous():
    rng = np.random.default_rng(0)
    assert infer_column_type([1, 2, 3, 4], rng=rng) == TYPE_INT
    assert infer_column_type([1.0, 2.0], rng=rng) == TYPE_FLOAT
    assert infer_column_type(["a", "b"], rng=rng) == TYPE_STR


def test_infer_all_null():
    assert infer_column_type([None, None]) == TYPE_NONE
    assert infer_column_type([]) == TYPE_NONE


def test_infer_majority():
    rng = np.random.default_rng(1)
    # 9 ints, 1 str -> majority int regardless of sample
    assert infer_column_type([1, 2, 3, 4, 5, 6, 7, 8, 9, "x"], sample_size=100, rng=rng) == TYPE_INT


def test_fingerprint_roundtrip():
    fp = Fingerprint(types={"a": "int", "b": "str"}, order=("a", "b"))
    fp2 = Fingerprint.from_dict(fp.to_dict())
    assert fp == fp2
    assert fp.checksum() == fp2.checksum()


def test_fingerprint_checksum_changes():
    fp1 = Fingerprint(types={"a": "int"}, order=("a",))
    fp2 = Fingerprint(types={"a": "str"}, order=("a",))
    assert fp1.checksum() != fp2.checksum()


def test_fingerprint_table():
    t = Table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    fp = fingerprint_table(t)
    assert fp.order == ("a", "b")
    assert fp.types == {"a": "int", "b": "str"}


def test_fingerprint_eq_non_fingerprint():
    fp = Fingerprint(types={}, order=())
    assert (fp == 3) is False
