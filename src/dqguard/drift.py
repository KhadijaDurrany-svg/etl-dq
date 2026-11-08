"""Schema-drift detection between a reference fingerprint and a candidate table.

Three drift kinds are reported, all at column granularity (``row is None``):

* ``drift_add``  -- a column present in the candidate but not the reference.
* ``drift_drop`` -- a column present in the reference but not the candidate.
* ``drift_type`` -- a column present in both whose inferred type differs.

``drift_add`` and ``drift_drop`` are exact set differences. ``drift_type``
depends on sample-based type inference (:func:`dqguard.schema.infer_column_type`)
and is therefore the one approximate detector in the library: on a wide or
heterogeneous column a bounded sample can flip the inferred type between the
reference run and the candidate run, producing honest false positives and
false negatives that the benchmark quantifies.
"""

from __future__ import annotations

import numpy as np

from .checks import Finding
from .schema import Fingerprint, fingerprint_table
from .table import Table

KIND_DRIFT_ADD = "drift_add"
KIND_DRIFT_DROP = "drift_drop"
KIND_DRIFT_TYPE = "drift_type"

DRIFT_KINDS = (KIND_DRIFT_ADD, KIND_DRIFT_DROP, KIND_DRIFT_TYPE)


def detect_drift(
    reference: Fingerprint,
    candidate: Table,
    sample_size: int = 24,
    rng: np.random.Generator | None = None,
) -> list[Finding]:
    """Compare ``reference`` against ``candidate`` and return drift findings.

    ``rng`` seeds the candidate's type inference. Passing a generator distinct
    from the one used to build ``reference`` reproduces the realistic case where
    the two profiling runs sampled different rows.
    """
    cand_fp = fingerprint_table(candidate, sample_size=sample_size, rng=rng)
    return diff_fingerprints(reference, cand_fp)


def diff_fingerprints(reference: Fingerprint, candidate: Fingerprint) -> list[Finding]:
    """Return drift findings from two precomputed fingerprints."""
    ref_cols = set(reference.order)
    cand_cols = set(candidate.order)
    out: list[Finding] = []
    for col in candidate.order:
        if col not in ref_cols:
            out.append(Finding(KIND_DRIFT_ADD, col, None))
    for col in reference.order:
        if col not in cand_cols:
            out.append(Finding(KIND_DRIFT_DROP, col, None))
    for col in reference.order:
        if col in cand_cols and reference.types[col] != candidate.types[col]:
            out.append(Finding(KIND_DRIFT_TYPE, col, None))
    return out
