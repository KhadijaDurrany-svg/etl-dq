from __future__ import annotations

import numpy as np

from dqguard.checks import Finding
from dqguard.drift import (
    KIND_DRIFT_ADD,
    KIND_DRIFT_DROP,
    KIND_DRIFT_TYPE,
    detect_drift,
    diff_fingerprints,
)
from dqguard.schema import Fingerprint, fingerprint_table
from dqguard.table import Table


def _fp(types, order):
    return Fingerprint(types=types, order=tuple(order))


def test_diff_detects_add():
    ref = _fp({"a": "int"}, ["a"])
    cand = _fp({"a": "int", "b": "str"}, ["a", "b"])
    f = diff_fingerprints(ref, cand)
    assert f == [Finding(KIND_DRIFT_ADD, "b", None)]


def test_diff_detects_drop():
    ref = _fp({"a": "int", "b": "str"}, ["a", "b"])
    cand = _fp({"a": "int"}, ["a"])
    kinds = [(x.kind, x.column) for x in diff_fingerprints(ref, cand)]
    assert kinds == [(KIND_DRIFT_DROP, "b")]


def test_diff_detects_type():
    ref = _fp({"a": "int"}, ["a"])
    cand = _fp({"a": "str"}, ["a"])
    kinds = [(x.kind, x.column) for x in diff_fingerprints(ref, cand)]
    assert kinds == [(KIND_DRIFT_TYPE, "a")]


def test_no_drift_when_identical():
    ref = _fp({"a": "int", "b": "str"}, ["a", "b"])
    assert diff_fingerprints(ref, ref) == []


def test_detect_drift_on_table():
    clean = Table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    ref = fingerprint_table(clean)
    candidate = Table({"a": [1, 2, 3], "c": ["p", "q", "r"]})
    out = {(f.kind, f.column) for f in detect_drift(ref, candidate, rng=np.random.default_rng(0))}
    assert (KIND_DRIFT_ADD, "c") in out
    assert (KIND_DRIFT_DROP, "b") in out


def test_combined_add_drop_type():
    ref = _fp({"a": "int", "b": "str", "c": "float"}, ["a", "b", "c"])
    cand = _fp({"a": "str", "c": "float", "d": "int"}, ["a", "c", "d"])
    out = {(x.kind, x.column) for x in diff_fingerprints(ref, cand)}
    assert out == {
        (KIND_DRIFT_TYPE, "a"),
        (KIND_DRIFT_DROP, "b"),
        (KIND_DRIFT_ADD, "d"),
    }
