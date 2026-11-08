"""dqguard -- a small deterministic ETL framework with a data-quality and
schema-drift detection layer.

Define extract/transform/load steps over in-memory tabular data, attach a
declarative check suite (nulls, ranges, uniqueness, referential integrity,
types, categorical domains), and detect schema drift between runs. Everything
is offline, seeded, and reproducible.
"""

from __future__ import annotations

from .checks import (
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
from .drift import detect_drift, diff_fingerprints
from .metrics import Score, bootstrap_ci, paired_permutation_test, score_findings
from .pipeline import (
    Extract,
    Load,
    Pipeline,
    PipelineError,
    PipelineResult,
    Transform,
    ValidationStage,
)
from .schema import Fingerprint, fingerprint_table, infer_column_type, value_type
from .table import Table

__version__ = "0.1.0"

__all__ = [
    "AllowedValues",
    "CheckSuite",
    "Extract",
    "Finding",
    "Fingerprint",
    "Load",
    "NotNull",
    "Pipeline",
    "PipelineError",
    "PipelineResult",
    "Range",
    "Referential",
    "Score",
    "Table",
    "Transform",
    "TypeCheck",
    "Unique",
    "ValidationStage",
    "__version__",
    "bootstrap_ci",
    "detect_drift",
    "diff_fingerprints",
    "fingerprint_table",
    "infer_column_type",
    "paired_permutation_test",
    "score_findings",
    "summarise",
    "value_type",
]
