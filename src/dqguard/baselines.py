"""Detectors evaluated by the benchmark.

* ``full``       -- the whole library: the declarative check suite plus
  sample-based schema-drift detection.
* ``null_only``  -- a common real-world shortcut: only null checks run. It can
  never find range, type, uniqueness, referential, categorical, or drift
  issues, so its recall is capped at the share of defects that are nulls.
* ``random``     -- an uninformed control that flags positions at random with a
  budget matched to the true defect density. Establishes the chance floor.

Each detector has signature ``(case, rng) -> list[Finding]``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .checks import (
    KIND_ALLOWED,
    KIND_NOT_NULL,
    KIND_RANGE,
    KIND_REFERENTIAL,
    KIND_TYPE,
    KIND_UNIQUE,
    AllowedValues,
    Finding,
    NotNull,
    Range,
    Referential,
    TypeCheck,
    Unique,
)
from .drift import KIND_DRIFT_ADD, KIND_DRIFT_DROP, KIND_DRIFT_TYPE, detect_drift
from .generate import Case

_KIND_OF = {
    NotNull: KIND_NOT_NULL,
    Range: KIND_RANGE,
    TypeCheck: KIND_TYPE,
    Unique: KIND_UNIQUE,
    Referential: KIND_REFERENTIAL,
    AllowedValues: KIND_ALLOWED,
}


def full_detector(case: Case, rng: np.random.Generator) -> list[Finding]:
    findings = list(case.suite.run(case.candidate))
    findings.extend(
        detect_drift(
            case.reference,
            case.candidate,
            sample_size=case.spec.sample_size,
            rng=rng,
        )
    )
    return findings


def null_only_detector(case: Case, rng: np.random.Generator) -> list[Finding]:
    out: list[Finding] = []
    for check in case.suite.checks:
        if isinstance(check, NotNull):
            out.extend(check.run(case.candidate))
    return out


def random_detector(case: Case, rng: np.random.Generator) -> list[Finding]:
    cand = case.candidate
    n = cand.n_rows
    pairs = [(_KIND_OF[type(c)], c.column) for c in case.suite.checks]
    total_positions = max(len(pairs) * n, 1)
    p_cell = case.spec.total_cell_defects / total_positions

    out: list[Finding] = []
    for kind, col in pairs:
        mask = rng.random(n) < p_cell
        for i in np.nonzero(mask)[0]:
            out.append(Finding(kind, col, int(i)))

    ref_cols = list(case.reference.order)
    cand_cols = cand.column_names()
    shared = [c for c in ref_cols if c in set(cand_cols)]
    universe = max(len(cand_cols) + len(ref_cols) + len(shared), 1)
    p_drift = 3.0 / universe
    for c in cand_cols:
        if rng.random() < p_drift:
            out.append(Finding(KIND_DRIFT_ADD, c, None))
    for c in ref_cols:
        if rng.random() < p_drift:
            out.append(Finding(KIND_DRIFT_DROP, c, None))
    for c in shared:
        if rng.random() < p_drift:
            out.append(Finding(KIND_DRIFT_TYPE, c, None))
    return out


Detector = Callable[[Case, "np.random.Generator"], list[Finding]]

DETECTORS: dict[str, Detector] = {
    "full": full_detector,
    "null_only": null_only_detector,
    "random": random_detector,
}
