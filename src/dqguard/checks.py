"""Declarative data-quality checks.

Each check is a small, exact rule over one column (or, for referential
integrity, one column against a key set). Running a check yields zero or more
:class:`Finding` objects. A :class:`Finding` is the atomic unit the benchmark
scores against ground truth, so its identity (``kind``, ``column``, ``row``)
must be stable and hashable.

The cell-level checks are deterministic and exact by construction: on a table
that is genuinely clean they emit no findings, and on an injected defect they
emit exactly the matching finding. That is deliberate -- the interesting,
approximate behaviour lives in schema-drift detection, not here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .schema import (
    TYPE_FLOAT,
    TYPE_INT,
    value_type,
)
from .table import Table

# Finding kinds.
KIND_NOT_NULL = "not_null"
KIND_RANGE = "range"
KIND_UNIQUE = "unique"
KIND_REFERENTIAL = "referential"
KIND_TYPE = "type"
KIND_ALLOWED = "allowed_values"

DQ_KINDS = (
    KIND_NOT_NULL,
    KIND_RANGE,
    KIND_UNIQUE,
    KIND_REFERENTIAL,
    KIND_TYPE,
    KIND_ALLOWED,
)


@dataclass(frozen=True)
class Finding:
    """A single detected (or ground-truth) data-quality issue.

    ``row`` is ``None`` for table/column-level issues (e.g. schema drift) and
    an integer row index for cell-level issues.
    """

    kind: str
    column: str
    row: int | None = None

    def key(self) -> tuple[str, str, int | None]:
        return (self.kind, self.column, self.row)


class Check(Protocol):
    """A runnable check."""

    column: str

    def run(self, table: Table) -> list[Finding]:  # pragma: no cover - protocol
        ...


# Numeric scalar types eligible for range comparison.
_NUMERIC = {TYPE_INT, TYPE_FLOAT}


@dataclass(frozen=True)
class NotNull:
    """Flag every ``None`` cell in ``column``."""

    column: str

    def run(self, table: Table) -> list[Finding]:
        return [
            Finding(KIND_NOT_NULL, self.column, i)
            for i, v in enumerate(table.column(self.column))
            if v is None
        ]


@dataclass(frozen=True)
class Range:
    """Flag numeric cells outside the inclusive ``[low, high]`` interval.

    Null cells are ignored (that is the job of :class:`NotNull`). Non-numeric
    cells are ignored here too (that is the job of :class:`TypeCheck`).
    """

    column: str
    low: float
    high: float

    def run(self, table: Table) -> list[Finding]:
        out: list[Finding] = []
        for i, v in enumerate(table.column(self.column)):
            if v is None or value_type(v) not in _NUMERIC:
                continue
            if v < self.low or v > self.high:
                out.append(Finding(KIND_RANGE, self.column, i))
        return out


@dataclass(frozen=True)
class Unique:
    """Flag the second and later occurrence of each repeated value.

    The *first* occurrence of a value is treated as canonical and is not
    flagged; every subsequent occurrence is. Null cells are skipped. This
    "flag the extras" convention keeps scoring unambiguous: injecting a
    duplicate by overwriting a later row with an earlier row's value produces
    exactly one finding, on the overwritten row.
    """

    column: str

    def run(self, table: Table) -> list[Finding]:
        seen: set[Any] = set()
        out: list[Finding] = []
        for i, v in enumerate(table.column(self.column)):
            if v is None:
                continue
            if v in seen:
                out.append(Finding(KIND_UNIQUE, self.column, i))
            else:
                seen.add(v)
        return out


@dataclass(frozen=True)
class Referential:
    """Flag cells whose value is absent from a reference key set.

    The key set is supplied at construction (typically the unique keys of a
    parent table). Null cells are skipped.
    """

    column: str
    keys: frozenset[Any]

    @classmethod
    def from_values(cls, column: str, values: Iterable[Any]) -> Referential:
        return cls(column=column, keys=frozenset(v for v in values if v is not None))

    def run(self, table: Table) -> list[Finding]:
        return [
            Finding(KIND_REFERENTIAL, self.column, i)
            for i, v in enumerate(table.column(self.column))
            if v is not None and v not in self.keys
        ]


@dataclass(frozen=True)
class TypeCheck:
    """Flag cells whose scalar type is not ``expected``.

    Null cells are skipped. ``expected`` is a canonical type name from
    :mod:`dqguard.schema` (e.g. ``"int"``). A ``float`` is *not* accepted for an
    ``int`` column and vice versa -- the check is strict.
    """

    column: str
    expected: str

    def run(self, table: Table) -> list[Finding]:
        out: list[Finding] = []
        for i, v in enumerate(table.column(self.column)):
            if v is None:
                continue
            if value_type(v) != self.expected:
                out.append(Finding(KIND_TYPE, self.column, i))
        return out


@dataclass(frozen=True)
class AllowedValues:
    """Flag cells whose value is not in an allowed set (categorical domain)."""

    column: str
    allowed: frozenset[Any]

    def run(self, table: Table) -> list[Finding]:
        return [
            Finding(KIND_ALLOWED, self.column, i)
            for i, v in enumerate(table.column(self.column))
            if v is not None and v not in self.allowed
        ]


@dataclass
class CheckSuite:
    """An ordered collection of checks run together."""

    checks: list[Check]

    def run(self, table: Table) -> list[Finding]:
        out: list[Finding] = []
        for check in self.checks:
            out.extend(check.run(table))
        return out

    def columns(self) -> list[str]:
        seen: list[str] = []
        for c in self.checks:
            if c.column not in seen:
                seen.append(c.column)
        return seen


def summarise(findings: Sequence[Finding]) -> dict[str, int]:
    """Count findings by kind."""
    out: dict[str, int] = {}
    for f in findings:
        out[f.kind] = out.get(f.kind, 0) + 1
    return out
