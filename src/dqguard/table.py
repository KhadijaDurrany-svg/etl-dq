"""In-memory columnar table.

A deliberately small tabular container built on the standard library. Columns
are stored as plain Python lists so that cells can hold mixed types (``None``,
``int``, ``float``, ``bool``, ``str``) -- which is what makes type and null
checks meaningful. No pandas, no numpy here; numeric work lives in
:mod:`dqguard.metrics`.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any


class Table:
    """An ordered collection of equal-length named columns.

    The table is intentionally immutable-by-convention: the mutating helpers
    (:meth:`with_column`, :meth:`drop_column`, :meth:`select`) return new
    tables and never alter ``self``.
    """

    __slots__ = ("_columns", "_n_rows")

    def __init__(self, columns: Mapping[str, Sequence[Any]]):
        cols: dict[str, list[Any]] = {}
        n_rows: int | None = None
        for name, values in columns.items():
            if not isinstance(name, str):
                raise TypeError(f"column name must be str, got {type(name)!r}")
            as_list = list(values)
            if n_rows is None:
                n_rows = len(as_list)
            elif len(as_list) != n_rows:
                raise ValueError(
                    f"column {name!r} has {len(as_list)} rows, expected {n_rows}"
                )
            cols[name] = as_list
        self._columns = cols
        self._n_rows = n_rows or 0

    # -- construction ----------------------------------------------------
    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> Table:
        """Build a table from an iterable of row dicts.

        Column order follows first appearance. Missing keys become ``None``.
        """
        rows = list(rows)
        names: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    names.append(key)
        columns: dict[str, list[Any]] = {name: [] for name in names}
        for row in rows:
            for name in names:
                columns[name].append(row.get(name))
        return cls(columns)

    @classmethod
    def from_csv(cls, path: str) -> Table:
        """Read a CSV file into a table of strings (empty cells -> ``None``)."""
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            columns: dict[str, list[Any]] = {name: [] for name in fieldnames}
            for row in reader:
                for name in fieldnames:
                    cell = row.get(name)
                    columns[name].append(None if cell is None or cell == "" else cell)
        return cls(columns)

    # -- introspection ---------------------------------------------------
    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def n_cols(self) -> int:
        return len(self._columns)

    def __len__(self) -> int:
        return self._n_rows

    def column_names(self) -> list[str]:
        return list(self._columns)

    def has_column(self, name: str) -> bool:
        return name in self._columns

    def column(self, name: str) -> list[Any]:
        """Return a *copy* of a column's values."""
        try:
            return list(self._columns[name])
        except KeyError:
            raise KeyError(f"no column named {name!r}") from None

    def cell(self, name: str, row: int) -> Any:
        return self._columns[name][row]

    def rows(self) -> Iterator[dict[str, Any]]:
        names = self.column_names()
        for i in range(self._n_rows):
            yield {name: self._columns[name][i] for name in names}

    # -- transformation (all non-mutating) -------------------------------
    def with_column(self, name: str, values: Sequence[Any]) -> Table:
        values = list(values)
        if self._columns and len(values) != self._n_rows:
            raise ValueError(
                f"new column {name!r} has {len(values)} rows, expected {self._n_rows}"
            )
        new = dict(self._columns)
        new[name] = values
        return Table(new)

    def drop_column(self, name: str) -> Table:
        if name not in self._columns:
            raise KeyError(f"no column named {name!r}")
        return Table({k: v for k, v in self._columns.items() if k != name})

    def rename(self, mapping: Mapping[str, str]) -> Table:
        return Table({mapping.get(k, k): v for k, v in self._columns.items()})

    def select(self, names: Sequence[str]) -> Table:
        missing = [n for n in names if n not in self._columns]
        if missing:
            raise KeyError(f"unknown columns: {missing}")
        return Table({name: self._columns[name] for name in names})

    def map_column(self, name: str, fn: Any) -> Table:
        if name not in self._columns:
            raise KeyError(f"no column named {name!r}")
        return self.with_column(name, [fn(v) for v in self._columns[name]])

    def copy(self) -> Table:
        return Table({k: list(v) for k, v in self._columns.items()})

    def to_rows(self) -> list[dict[str, Any]]:
        return list(self.rows())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Table):
            return NotImplemented
        return self._columns == other._columns

    def __repr__(self) -> str:
        return f"Table(n_rows={self._n_rows}, columns={self.column_names()})"
