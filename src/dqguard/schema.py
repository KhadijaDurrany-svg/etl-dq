"""Schema inference and fingerprinting.

A *fingerprint* is a compact, serialisable snapshot of a table's structure:
the ordered column names and an inferred scalar type per column. Fingerprints
are compared across runs to detect schema drift (see :mod:`dqguard.drift`).

Type inference is deliberately *sample-based*: for a column we draw a bounded
random sample of its non-null cells and take the majority Python type. This
mirrors how real systems profile wide tables without scanning every row, and
it is the one genuinely approximate step in the library -- small samples on
heterogeneous columns can disagree between runs, which is exactly the honest
imperfection the benchmark measures.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Canonical scalar type names used throughout the library.
TYPE_NONE = "none"
TYPE_BOOL = "bool"
TYPE_INT = "int"
TYPE_FLOAT = "float"
TYPE_STR = "str"

# Tie-break priority when a sample splits evenly across types. Lower index wins.
_TYPE_PRIORITY = [TYPE_STR, TYPE_FLOAT, TYPE_INT, TYPE_BOOL, TYPE_NONE]


def value_type(value: Any) -> str:
    """Return the canonical scalar type name for a single Python value.

    ``bool`` is checked before ``int`` because ``bool`` subclasses ``int``.
    """
    if value is None:
        return TYPE_NONE
    if isinstance(value, bool):
        return TYPE_BOOL
    if isinstance(value, int):
        return TYPE_INT
    if isinstance(value, float):
        return TYPE_FLOAT
    if isinstance(value, str):
        return TYPE_STR
    # Fall back to the Python class name for anything exotic.
    return type(value).__name__


def infer_column_type(
    values: Sequence[Any],
    sample_size: int = 24,
    rng: np.random.Generator | None = None,
) -> str:
    """Infer a column's scalar type from a bounded sample of non-null cells.

    Returns :data:`TYPE_NONE` for an all-null (or empty) column. When more than
    ``sample_size`` non-null cells are present a random subset is drawn using
    ``rng`` (a fresh deterministic generator is created when ``rng`` is None),
    so two runs that pass different generators may disagree on a heterogeneous
    column -- this is intentional.
    """
    non_null = [v for v in values if v is not None]
    if not non_null:
        return TYPE_NONE
    if rng is None:
        rng = np.random.default_rng(0)
    if len(non_null) > sample_size:
        idx = rng.choice(len(non_null), size=sample_size, replace=False)
        sample = [non_null[int(i)] for i in idx]
    else:
        sample = non_null
    counts: dict[str, int] = {}
    for v in sample:
        t = value_type(v)
        counts[t] = counts.get(t, 0) + 1
    best = max(counts.values())
    tied = [t for t, c in counts.items() if c == best]
    if len(tied) == 1:
        return tied[0]
    for candidate in _TYPE_PRIORITY:
        if candidate in tied:
            return candidate
    return sorted(tied)[0]


@dataclass(frozen=True)
class Fingerprint:
    """An ordered column-name -> inferred-type snapshot of a table."""

    types: dict[str, str]
    order: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"order": list(self.order), "types": dict(self.types)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fingerprint:
        order = tuple(data["order"])
        types = dict(data["types"])
        return cls(types=types, order=order)

    def checksum(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fingerprint):
            return NotImplemented
        return self.types == other.types and self.order == other.order


def fingerprint_table(
    table: Any,
    sample_size: int = 24,
    rng: np.random.Generator | None = None,
) -> Fingerprint:
    """Compute a :class:`Fingerprint` for a :class:`~dqguard.table.Table`."""
    order = tuple(table.column_names())
    types = {
        name: infer_column_type(table.column(name), sample_size=sample_size, rng=rng)
        for name in order
    }
    return Fingerprint(types=types, order=order)
