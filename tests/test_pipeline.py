from __future__ import annotations

import pytest

from dqguard.checks import CheckSuite, NotNull, Unique
from dqguard.pipeline import (
    Extract,
    Load,
    Pipeline,
    PipelineError,
    Transform,
    ValidationStage,
    collect_sink,
    constant_source,
    drop_step,
    rename_step,
)
from dqguard.schema import fingerprint_table
from dqguard.table import Table


def _src():
    return Table({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_steps_run_in_order():
    pipe = Pipeline(
        steps=[
            Extract("src", _src),
            Transform("double_a", lambda t: t.map_column("a", lambda x: x * 2)),
        ]
    )
    res = pipe.run()
    assert res.table.column("a") == [2, 4, 6]


def test_validation_stage_collects():
    bad = Table({"a": [1, None], "b": [5, 5]})
    pipe = Pipeline(
        steps=[
            Extract("src", lambda: bad),
            ValidationStage("vq", suite=CheckSuite([NotNull("a"), Unique("b")])),
        ]
    )
    res = pipe.run()
    assert len(res.findings["vq"]) == 2
    assert not res.ok
    assert len(res.all_findings) == 2


def test_validation_gate_raises():
    bad = Table({"a": [None]})
    pipe = Pipeline(
        steps=[
            Extract("src", lambda: bad),
            ValidationStage("gate", suite=CheckSuite([NotNull("a")]), gate=True),
        ]
    )
    with pytest.raises(PipelineError) as exc:
        pipe.run()
    assert exc.value.stage == "gate"
    assert len(exc.value.findings) == 1


def test_clean_pipeline_ok():
    pipe = Pipeline(
        steps=[
            Extract("src", _src),
            ValidationStage("vq", suite=CheckSuite([NotNull("a"), Unique("a")])),
        ]
    )
    res = pipe.run()
    assert res.ok


def test_validation_drift_stage():
    clean = _src()
    ref = fingerprint_table(clean)
    candidate = clean.drop_column("b").with_column("c", [9, 9, 9])
    pipe = Pipeline(
        steps=[
            Extract("src", lambda: candidate),
            ValidationStage("drift", reference=ref),
        ]
    )
    res = pipe.run()
    kinds = {f.kind for f in res.findings["drift"]}
    assert "drift_add" in kinds and "drift_drop" in kinds


def test_transform_on_none_raises():
    pipe = Pipeline(steps=[Transform("t", lambda t: t)])
    with pytest.raises(PipelineError):
        pipe.run()


def test_load_and_helpers():
    sink: list[Table] = []
    pipe = Pipeline(
        steps=[
            constant_source("src", _src()),
            rename_step("rn", {"a": "A"}),
            drop_step("dr", "b"),
            collect_sink("out", sink),
        ]
    )
    res = pipe.run()
    assert res.table.column_names() == ["A"]
    assert sink and sink[0].column_names() == ["A"]


def test_load_step_direct():
    captured = {}

    def _save(t):
        captured["n"] = t.n_rows
        return t

    pipe = Pipeline(steps=[Extract("s", _src), Load("save", _save)])
    pipe.run()
    assert captured["n"] == 3
