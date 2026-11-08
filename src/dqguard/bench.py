"""The benchmark harness: generate cases, run detectors, score honestly.

All numbers written to ``benchmarks/results.json`` come from a real run over
freshly generated cases. Nothing is hard-coded. The companion
``benchmarks/manifest.json`` records enough to reproduce the run: the git SHA
(or ``"n/a"``), the Python and numpy versions, the full config, and a checksum
of the generated data.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import DETECTORS
from .checks import DQ_KINDS, Finding
from .drift import DRIFT_KINDS
from .generate import Case, DatasetSpec, generate_case
from .metrics import (
    Score,
    aggregate,
    bootstrap_ci,
    paired_permutation_test,
    score_findings,
)

ALL_KINDS = (*DQ_KINDS, *DRIFT_KINDS)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else "n/a"
    except Exception:
        return "n/a"


def _case_checksum(case: Case) -> str:
    payload = {
        "seed": case.seed,
        "customers": case.customers.to_rows(),
        "clean_orders": case.clean_orders.to_rows(),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def dataset_checksum(cases: list[Case]) -> str:
    h = hashlib.sha256()
    for case in cases:
        h.update(_case_checksum(case).encode("utf-8"))
    return h.hexdigest()


def _score_by_kind(predicted: list[Finding], truth: frozenset[Finding]) -> dict[str, Score]:
    pred_keys = {f.key() for f in predicted}
    gold_keys = {f.key() for f in truth}
    out: dict[str, Score] = {}
    for kind in ALL_KINDS:
        p = {k for k in pred_keys if k[0] == kind}
        g = {k for k in gold_keys if k[0] == kind}
        out[kind] = Score(tp=len(p & g), fp=len(p - g), fn=len(g - p))
    return out


def run_benchmark(
    n_datasets: int = 30,
    base_seed: int = 20260101,
    spec: DatasetSpec | None = None,
    bootstrap_iters: int = 5000,
    perm_iters: int = 10000,
) -> dict[str, Any]:
    """Run the benchmark and return a results dictionary."""
    spec = spec or DatasetSpec()
    cases = [generate_case(base_seed + i, spec) for i in range(n_datasets)]

    # per-case findings and scores
    per_case_f1: dict[str, list[float]] = {name: [] for name in DETECTORS}
    per_case_prec: dict[str, list[float]] = {name: [] for name in DETECTORS}
    per_case_rec: dict[str, list[float]] = {name: [] for name in DETECTORS}
    micro: dict[str, list[Score]] = {name: [] for name in DETECTORS}
    kind_scores: dict[str, list[Score]] = {kind: [] for kind in ALL_KINDS}

    # Stable per-detector offsets keep every run byte-for-byte reproducible
    # (unlike the salted builtin hash of the name).
    name_offset = {name: idx for idx, name in enumerate(sorted(DETECTORS))}

    for i, case in enumerate(cases):
        for name, detector in DETECTORS.items():
            det_rng = np.random.default_rng(base_seed + 1009 * (i + 1) + name_offset[name])
            predicted = detector(case, det_rng)
            score = score_findings(predicted, case.ground_truth)
            per_case_f1[name].append(score.f1)
            per_case_prec[name].append(score.precision)
            per_case_rec[name].append(score.recall)
            micro[name].append(score)
            if name == "full":
                for kind, ks in _score_by_kind(predicted, case.ground_truth).items():
                    kind_scores[kind].append(ks)

    boot_rng = np.random.default_rng(base_seed + 1)
    perm_rng = np.random.default_rng(base_seed + 2)

    methods: dict[str, Any] = {}
    for name in DETECTORS:
        agg = aggregate(micro[name])
        lo, hi = bootstrap_ci(per_case_f1[name], boot_rng, iterations=bootstrap_iters)
        methods[name] = {
            "micro_precision": round(agg.precision, 6),
            "micro_recall": round(agg.recall, 6),
            "micro_f1": round(agg.f1, 6),
            "mean_f1": round(float(np.mean(per_case_f1[name])), 6),
            "f1_ci95": [round(lo, 6), round(hi, 6)],
            "mean_precision": round(float(np.mean(per_case_prec[name])), 6),
            "mean_recall": round(float(np.mean(per_case_rec[name])), 6),
            "tp": agg.tp,
            "fp": agg.fp,
            "fn": agg.fn,
        }

    comparisons: dict[str, Any] = {}
    for other in ("null_only", "random"):
        res = paired_permutation_test(
            per_case_f1["full"], per_case_f1[other], perm_rng, iterations=perm_iters
        )
        comparisons[f"full_vs_{other}"] = {
            "observed_f1_diff": round(res["observed_diff"], 6),
            "p_value": res["p_value"],
            "permutations": int(res["iterations"]),
        }

    per_kind = {}
    for kind, scores in kind_scores.items():
        agg = aggregate(scores)
        per_kind[kind] = {
            "precision": round(agg.precision, 6),
            "recall": round(agg.recall, 6),
            "f1": round(agg.f1, 6),
            "tp": agg.tp,
            "fp": agg.fp,
            "fn": agg.fn,
        }

    return {
        "config": {
            "n_datasets": n_datasets,
            "base_seed": base_seed,
            "bootstrap_iters": bootstrap_iters,
            "perm_iters": perm_iters,
            "spec": asdict(spec),
        },
        "methods": methods,
        "comparisons": comparisons,
        "full_detector_by_kind": per_kind,
    }


def build_manifest(results: dict[str, Any]) -> dict[str, Any]:
    cfg = results["config"]
    spec = DatasetSpec(**cfg["spec"])
    cases = [generate_case(cfg["base_seed"] + i, spec) for i in range(cfg["n_datasets"])]
    return {
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "dataset_checksum": dataset_checksum(cases),
        "config": cfg,
    }


def write_outputs(results: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(results)
    results_path = out / "results.json"
    manifest_path = out / "manifest.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results_path, manifest_path
