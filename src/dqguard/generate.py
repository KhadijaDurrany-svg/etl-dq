"""Seeded synthetic data generation and defect injection.

Everything here is offline and reproducible: given a seed, the same customers
and orders tables, the same injected defects, and the same ground-truth label
set come out every time.

The design goal is an *honest and non-trivial* benchmark, so:

1. A freshly generated clean ``orders`` table produces **zero** findings from
   the default check suite. Every cell-level finding the benchmark scores is one
   we deliberately injected (or a genuine detector mistake).
2. Each injected cell-level defect lands on a distinct row and is constructed
   so that exactly one check fires for it.
3. Schema drift is deliberately pushed toward the hard regime. Type-drift is
   injected by converting only a *fraction* (near one half) of a column's cells
   to another type, so the sample-based inference in :mod:`dqguard.schema`
   genuinely misses some of them (false negatives). Two untouched columns are
   heterogeneous (mixed str/int) so that independent reference/candidate samples
   occasionally disagree on their type (false positives). Both effects are real
   and measured -- the full detector is not expected to be perfect.

Column roles
------------
Core columns carry data-quality constraints: ``order_id`` (unique / int),
``customer_id`` (referential / int), ``amount`` (float / range), ``quantity``
(int / range), ``status`` (categorical). Peripheral columns carry no
constraints and exist to exercise schema drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .checks import (
    KIND_ALLOWED,
    KIND_NOT_NULL,
    KIND_RANGE,
    KIND_REFERENTIAL,
    KIND_TYPE,
    KIND_UNIQUE,
    AllowedValues,
    CheckSuite,
    Finding,
    NotNull,
    Range,
    Referential,
    TypeCheck,
    Unique,
)
from .drift import KIND_DRIFT_ADD, KIND_DRIFT_DROP, KIND_DRIFT_TYPE
from .schema import Fingerprint, fingerprint_table
from .table import Table

_STATUS = ("new", "paid", "shipped", "cancelled")
_REGIONS = ("north", "south", "east", "west")
_CHANNEL_STR = ("web", "store", "phone")
_MIX_STR = ("a", "b", "c", "d")
_BUCKET = ("low", "mid", "high")

# Peripheral columns targeted by each kind of schema drift.
_DRIFT_TYPE_TARGETS = ("score", "rating", "latency", "code", "tag", "bucket")
_DRIFT_DROP_TARGETS = ("ext_ref", "legacy_id")
_DRIFT_ADD_COLS = ("promo_flag", "coupon")


@dataclass(frozen=True)
class DatasetSpec:
    """Knobs controlling dataset size, defect counts, and drift difficulty."""

    n_rows: int = 500
    n_customers: int = 60
    n_null: int = 4
    n_range: int = 4
    n_type: int = 4
    n_unique: int = 4
    n_referential: int = 4
    n_allowed: int = 4
    drift_frac_low: float = 0.42
    drift_frac_high: float = 0.60
    channel_hetero_fraction: float = 0.30
    mix_hetero_fraction: float = 0.45
    sample_size: int = 16

    @property
    def total_cell_defects(self) -> int:
        return (
            self.n_null
            + self.n_range
            + self.n_type
            + self.n_unique
            + self.n_referential
            + self.n_allowed
        )


@dataclass
class Case:
    """A fully materialised benchmark case."""

    customers: Table
    clean_orders: Table
    candidate: Table
    suite: CheckSuite
    reference: Fingerprint
    ground_truth: frozenset[Finding]
    keys: frozenset[int]
    spec: DatasetSpec
    seed: int = 0
    _audit: dict[str, int] = field(default_factory=dict)


def make_customers(
    rng: np.random.Generator, spec: DatasetSpec
) -> tuple[Table, frozenset[int]]:
    ids = rng.choice(
        np.arange(1000, 1000 + spec.n_customers * 4), size=spec.n_customers, replace=False
    )
    ids = sorted(int(x) for x in ids)
    return Table({"customer_id": ids}), frozenset(ids)


def make_clean_orders(
    rng: np.random.Generator, spec: DatasetSpec, keys: frozenset[int]
) -> Table:
    n = spec.n_rows
    key_list = sorted(keys)
    order_id = [int(x) for x in rng.permutation(np.arange(5000, 5000 + n))]
    customer_id = [int(key_list[int(i)]) for i in rng.integers(0, len(key_list), size=n)]
    amount = [round(float(x), 2) for x in rng.uniform(1.0, 9999.0, size=n)]
    quantity = [int(x) for x in rng.integers(1, 101, size=n)]
    status = [str(_STATUS[int(i)]) for i in rng.integers(0, len(_STATUS), size=n)]
    region = [str(_REGIONS[int(i)]) for i in rng.integers(0, len(_REGIONS), size=n)]

    def heterogeneous(frac: float, strs: tuple[str, ...]) -> list[object]:
        pick_int = rng.random(size=n) < frac
        out: list[object] = []
        for i in range(n):
            if pick_int[i]:
                out.append(int(rng.integers(0, 5)))
            else:
                out.append(str(strs[int(rng.integers(0, len(strs)))]))
        return out

    channel = heterogeneous(spec.channel_hetero_fraction, _CHANNEL_STR)
    mix1 = heterogeneous(spec.mix_hetero_fraction, _MIX_STR)
    score = [round(float(x), 3) for x in rng.uniform(0.0, 1.0, size=n)]
    rating = [round(float(x), 2) for x in rng.uniform(1.0, 5.0, size=n)]
    latency = [round(float(x), 1) for x in rng.uniform(10.0, 900.0, size=n)]
    code = [int(x) for x in rng.integers(0, 1000, size=n)]
    tag = [f"T{int(x)}" for x in rng.integers(1, 50, size=n)]
    bucket = [str(_BUCKET[int(i)]) for i in rng.integers(0, len(_BUCKET), size=n)]
    ext_ref = [f"R{int(x)}" for x in rng.integers(1000, 9999, size=n)]
    legacy_id = [int(x) for x in rng.integers(100000, 999999, size=n)]

    return Table(
        {
            "order_id": order_id,
            "customer_id": customer_id,
            "amount": amount,
            "quantity": quantity,
            "status": status,
            "region": region,
            "channel": channel,
            "mix1": mix1,
            "score": score,
            "rating": rating,
            "latency": latency,
            "code": code,
            "tag": tag,
            "bucket": bucket,
            "ext_ref": ext_ref,
            "legacy_id": legacy_id,
        }
    )


def default_suite(keys: frozenset[int]) -> CheckSuite:
    return CheckSuite(
        checks=[
            NotNull("order_id"),
            TypeCheck("order_id", "int"),
            Unique("order_id"),
            NotNull("customer_id"),
            TypeCheck("customer_id", "int"),
            Referential("customer_id", keys),
            NotNull("amount"),
            TypeCheck("amount", "float"),
            Range("amount", 0.0, 10000.0),
            NotNull("quantity"),
            TypeCheck("quantity", "int"),
            Range("quantity", 1, 100),
            NotNull("status"),
            TypeCheck("status", "str"),
            AllowedValues("status", frozenset(_STATUS)),
        ]
    )


def _convert_cell(column: str, value: object, rng: np.random.Generator) -> object:
    """Convert a cell to a *different* scalar type for type-drift injection."""
    if column in ("score", "rating", "latency", "code"):
        return str(value)  # numeric -> string
    if column == "tag":
        return int(rng.integers(1, 99))  # string -> int
    if column == "bucket":
        return round(float(rng.uniform(0.0, 1.0)), 3)  # string -> float
    return str(value)


def inject(
    table: Table, rng: np.random.Generator, spec: DatasetSpec, keys: frozenset[int]
) -> tuple[Table, frozenset[Finding]]:
    """Inject defects into a clean table; return the candidate and ground truth."""
    n = table.n_rows
    need = spec.total_cell_defects + spec.n_unique  # +sources for unique dups
    pool = [int(i) for i in rng.choice(n, size=min(need, n), replace=False)]
    cursor = 0

    def take(k: int) -> list[int]:
        nonlocal cursor
        chunk = pool[cursor : cursor + k]
        cursor += k
        return chunk

    cols = {name: table.column(name) for name in table.column_names()}
    gt: set[Finding] = set()

    for r in take(spec.n_null):  # not_null on amount
        cols["amount"][r] = None
        gt.add(Finding(KIND_NOT_NULL, "amount", r))

    for r in take(spec.n_range):  # range on quantity (out-of-range int)
        cols["quantity"][r] = int(100 + rng.integers(5, 500))
        gt.add(Finding(KIND_RANGE, "quantity", r))

    for r in take(spec.n_type):  # type on order_id (non-null, non-dup string)
        cols["order_id"][r] = f"BAD-{r}"
        gt.add(Finding(KIND_TYPE, "order_id", r))

    max_key = max(keys)
    for r in take(spec.n_referential):  # referential on customer_id
        cols["customer_id"][r] = int(max_key + 100 + rng.integers(1, 1000))
        gt.add(Finding(KIND_REFERENTIAL, "customer_id", r))

    for r in take(spec.n_allowed):  # allowed_values on status
        cols["status"][r] = "unknown_status"
        gt.add(Finding(KIND_ALLOWED, "status", r))

    targets = take(spec.n_unique)  # unique on order_id
    sources = take(spec.n_unique)
    for tgt, src in zip(targets, sources, strict=True):
        lo, hi = (src, tgt) if src < tgt else (tgt, src)
        cols["order_id"][hi] = cols["order_id"][lo]
        gt.add(Finding(KIND_UNIQUE, "order_id", hi))

    candidate = Table(cols)

    # -- schema drift -----------------------------------------------------
    for col in _DRIFT_ADD_COLS:
        if col == "promo_flag":
            values: list[object] = [int(x) for x in rng.integers(0, 2, size=n)]
        else:
            values = [f"C{int(x)}" for x in rng.integers(0, 50, size=n)]
        candidate = candidate.with_column(col, values)
        gt.add(Finding(KIND_DRIFT_ADD, col, None))

    for col in _DRIFT_DROP_TARGETS:
        candidate = candidate.drop_column(col)
        gt.add(Finding(KIND_DRIFT_DROP, col, None))

    for col in _DRIFT_TYPE_TARGETS:
        vals = candidate.column(col)
        frac = float(rng.uniform(spec.drift_frac_low, spec.drift_frac_high))
        k = int(round(frac * n))
        flip_idx = rng.choice(n, size=k, replace=False)
        for i in flip_idx:
            i = int(i)
            vals[i] = _convert_cell(col, vals[i], rng)
        candidate = candidate.with_column(col, vals)
        gt.add(Finding(KIND_DRIFT_TYPE, col, None))

    return candidate, frozenset(gt)


def generate_case(seed: int = 0, spec: DatasetSpec | None = None) -> Case:
    """Produce a complete, reproducible benchmark case for ``seed``."""
    spec = spec or DatasetSpec()
    rng = np.random.default_rng(seed)
    customers, keys = make_customers(rng, spec)
    clean = make_clean_orders(rng, spec, keys)
    suite = default_suite(keys)
    # Reference fingerprint: profiled on the clean baseline with its own sampler.
    ref_rng = np.random.default_rng(seed * 2 + 1)
    reference = fingerprint_table(clean, sample_size=spec.sample_size, rng=ref_rng)
    inj_rng = np.random.default_rng(seed * 3 + 7)
    candidate, gt = inject(clean.copy(), inj_rng, spec, keys)
    return Case(
        customers=customers,
        clean_orders=clean,
        candidate=candidate,
        suite=suite,
        reference=reference,
        ground_truth=gt,
        keys=keys,
        spec=spec,
        seed=seed,
    )
