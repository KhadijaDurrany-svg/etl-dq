"""A small deterministic ETL pipeline with a built-in validation gate.

The pipeline threads a :class:`~dqguard.table.Table` through an ordered list of
steps. Three step flavours are provided for readability -- :class:`Extract`,
:class:`Transform`, :class:`Load` -- but they share one contract: given a table
(or ``None`` for a source extract), return a table.

A :class:`ValidationStage` is a special step that does not alter the data; it
runs a :class:`~dqguard.checks.CheckSuite` and, optionally, schema-drift
detection against a reference fingerprint, recording any findings on the
:class:`PipelineResult`. The stage can be configured to raise on findings
(a quality *gate*) or to pass them through for inspection.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .checks import CheckSuite, Finding
from .drift import detect_drift
from .schema import Fingerprint
from .table import Table


class PipelineError(RuntimeError):
    """Raised when a gating :class:`ValidationStage` finds issues."""

    def __init__(self, stage: str, findings: list[Finding]):
        self.stage = stage
        self.findings = findings
        super().__init__(f"stage {stage!r} found {len(findings)} issue(s)")


@dataclass
class Step:
    """A named callable ``Table | None -> Table``."""

    name: str
    fn: Callable[[Table | None], Table]
    kind: str = "step"

    def apply(self, table: Table | None) -> Table:
        return self.fn(table)


def Extract(name: str, fn: Callable[[], Table]) -> Step:
    """A source step that ignores its input and produces a table."""
    return Step(name=name, fn=lambda _=None: fn(), kind="extract")


def Transform(name: str, fn: Callable[[Table], Table]) -> Step:
    """A step that maps a table to a table."""

    def _wrapped(table: Table | None) -> Table:
        if table is None:
            raise PipelineError(name, [])
        return fn(table)

    return Step(name=name, fn=_wrapped, kind="transform")


def Load(name: str, fn: Callable[[Table], Table]) -> Step:
    """A sink step; ``fn`` typically has a side effect and returns the table."""

    def _wrapped(table: Table | None) -> Table:
        if table is None:
            raise PipelineError(name, [])
        return fn(table)

    return Step(name=name, fn=_wrapped, kind="load")


@dataclass
class ValidationStage:
    """A non-mutating step that records data-quality and drift findings."""

    name: str
    suite: CheckSuite | None = None
    reference: Fingerprint | None = None
    sample_size: int = 24
    gate: bool = False
    rng: np.random.Generator | None = None
    kind: str = "validate"

    def run(self, table: Table) -> list[Finding]:
        findings: list[Finding] = []
        if self.suite is not None:
            findings.extend(self.suite.run(table))
        if self.reference is not None:
            findings.extend(
                detect_drift(
                    self.reference,
                    table,
                    sample_size=self.sample_size,
                    rng=self.rng,
                )
            )
        return findings


@dataclass
class PipelineResult:
    """The outcome of running a pipeline."""

    table: Table | None
    findings: dict[str, list[Finding]] = field(default_factory=dict)

    @property
    def all_findings(self) -> list[Finding]:
        out: list[Finding] = []
        for group in self.findings.values():
            out.extend(group)
        return out

    @property
    def ok(self) -> bool:
        return not self.all_findings


@dataclass
class Pipeline:
    """An ordered sequence of steps and validation stages."""

    steps: list[Step | ValidationStage]

    def run(self) -> PipelineResult:
        table: Table | None = None
        result = PipelineResult(table=None)
        for step in self.steps:
            if isinstance(step, ValidationStage):
                if table is None:
                    raise PipelineError(step.name, [])
                findings = step.run(table)
                result.findings[step.name] = findings
                if step.gate and findings:
                    raise PipelineError(step.name, findings)
            else:
                table = step.apply(table)
        result.table = table
        return result


def constant_source(name: str, table: Table) -> Step:
    """Convenience: an :func:`Extract` that always yields ``table``."""
    return Extract(name, lambda: table.copy())


def rename_step(name: str, mapping: dict[str, str]) -> Step:
    return Transform(name, lambda t: t.rename(mapping))


def drop_step(name: str, column: str) -> Step:
    return Transform(name, lambda t: t.drop_column(column))


def collect_sink(name: str, into: list[Any]) -> Step:
    """A :func:`Load` that appends the table to ``into`` and returns it."""

    def _sink(t: Table) -> Table:
        into.append(t)
        return t

    return Load(name, _sink)
