from __future__ import annotations

import json

from dqguard.bench import (
    ALL_KINDS,
    build_manifest,
    dataset_checksum,
    run_benchmark,
    write_outputs,
)
from dqguard.generate import DatasetSpec, generate_case


def _small():
    return run_benchmark(n_datasets=5, bootstrap_iters=300, perm_iters=500)


def test_run_benchmark_wellformed():
    r = _small()
    assert r["config"]["n_datasets"] == 5
    for name in ("full", "null_only", "random"):
        m = r["methods"][name]
        for key in ("micro_precision", "micro_recall", "micro_f1"):
            assert 0.0 <= m[key] <= 1.0
        assert m["f1_ci95"][0] <= m["f1_ci95"][1]


def test_full_beats_null_only():
    r = _small()
    assert r["methods"]["full"]["micro_f1"] > r["methods"]["null_only"]["micro_f1"]
    assert r["methods"]["full"]["micro_f1"] > r["methods"]["random"]["micro_f1"]


def test_comparisons_present():
    r = _small()
    for key in ("full_vs_null_only", "full_vs_random"):
        comp = r["comparisons"][key]
        assert 0.0 < comp["p_value"] <= 1.0
        assert comp["observed_f1_diff"] > 0


def test_per_kind_present():
    r = _small()
    assert set(r["full_detector_by_kind"]) == set(ALL_KINDS)


def test_dataset_checksum_deterministic():
    spec = DatasetSpec()
    a = [generate_case(i, spec) for i in range(3)]
    b = [generate_case(i, spec) for i in range(3)]
    assert dataset_checksum(a) == dataset_checksum(b)


def test_manifest_fields():
    r = _small()
    man = build_manifest(r)
    for key in ("git_sha", "python", "numpy", "dataset_checksum", "config"):
        assert key in man
    assert isinstance(man["dataset_checksum"], str) and len(man["dataset_checksum"]) == 64


def test_write_outputs(tmp_path):
    r = _small()
    rp, mp = write_outputs(r, tmp_path)
    assert rp.exists() and mp.exists()
    loaded = json.loads(rp.read_text())
    assert loaded["methods"]["full"]["micro_f1"] == r["methods"]["full"]["micro_f1"]


def test_results_reproducible():
    a = run_benchmark(n_datasets=4, bootstrap_iters=200, perm_iters=200)
    b = run_benchmark(n_datasets=4, bootstrap_iters=200, perm_iters=200)
    assert a["methods"]["full"] == b["methods"]["full"]
    assert a["methods"]["random"] == b["methods"]["random"]
