"""Command-line interface for dqguard.

Two subcommands:

``bench``
    Run the offline benchmark. By default it writes ``results.json`` and
    ``manifest.json`` into ``benchmarks/``; ``--no-write`` prints the summary
    only.

``validate``
    Profile a CSV file and, optionally, run a declarative check suite and/or
    schema-drift detection against a saved fingerprint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .checks import (
    AllowedValues,
    CheckSuite,
    NotNull,
    Range,
    Referential,
    TypeCheck,
    Unique,
    summarise,
)
from .drift import diff_fingerprints
from .generate import DatasetSpec
from .schema import Fingerprint, fingerprint_table
from .table import Table


def _coerce_numeric(table: Table) -> Table:
    """Best-effort: turn numeric-looking string cells into int/float."""
    cols: dict[str, list[Any]] = {}
    for name in table.column_names():
        new: list[Any] = []
        for v in table.column(name):
            if isinstance(v, str):
                try:
                    new.append(int(v))
                    continue
                except ValueError:
                    pass
                try:
                    new.append(float(v))
                    continue
                except ValueError:
                    pass
            new.append(v)
        cols[name] = new
    return Table(cols)


def _build_suite(spec: dict[str, Any]) -> CheckSuite:
    checks: list[Any] = []
    for col in spec.get("not_null", []):
        checks.append(NotNull(col))
    for col in spec.get("unique", []):
        checks.append(Unique(col))
    for col, allowed in spec.get("allowed_values", {}).items():
        checks.append(AllowedValues(col, frozenset(allowed)))
    for col, bounds in spec.get("range", {}).items():
        checks.append(Range(col, float(bounds[0]), float(bounds[1])))
    for col, typename in spec.get("type", {}).items():
        checks.append(TypeCheck(col, typename))
    for col, keys in spec.get("referential", {}).items():
        checks.append(Referential(col, frozenset(keys)))
    return CheckSuite(checks=checks)


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_benchmark, write_outputs

    spec = DatasetSpec(n_rows=args.rows)
    results = run_benchmark(
        n_datasets=args.datasets,
        base_seed=args.seed,
        spec=spec,
        bootstrap_iters=args.bootstrap_iters,
        perm_iters=args.perm_iters,
    )
    m = results["methods"]
    print(f"cases: {results['config']['n_datasets']}  rows/case: {spec.n_rows}")
    print(f"{'method':<12} {'precision':>10} {'recall':>8} {'f1':>8}  {'f1 95% CI':>18}")
    for name in ("full", "null_only", "random"):
        row = m[name]
        ci = row["f1_ci95"]
        print(
            f"{name:<12} {row['micro_precision']:>10.3f} {row['micro_recall']:>8.3f} "
            f"{row['micro_f1']:>8.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]"
        )
    for key, comp in results["comparisons"].items():
        print(
            f"{key}: dF1={comp['observed_f1_diff']:+.3f}  "
            f"p={comp['p_value']:.2e}  (perms={comp['permutations']})"
        )
    if args.no_write:
        print("(--no-write: nothing written)")
        return 0
    out_dir = args.out_dir or (Path(__file__).resolve().parents[2] / "benchmarks")
    results_path, manifest_path = write_outputs(results, out_dir)
    print(f"wrote {results_path}")
    print(f"wrote {manifest_path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    table = Table.from_csv(args.datafile)
    if args.coerce:
        table = _coerce_numeric(table)

    fp = fingerprint_table(table, sample_size=args.sample_size, rng=np.random.default_rng(args.seed))

    print(f"file: {args.datafile}  rows: {table.n_rows}  columns: {table.n_cols}")
    print("profile:")
    for name in table.column_names():
        values = table.column(name)
        nulls = sum(1 for v in values if v is None)
        distinct = len({v for v in values if v is not None})
        print(f"  {name:<16} type={fp.types[name]:<6} nulls={nulls:<4} distinct={distinct}")

    findings = []
    if args.schema:
        spec = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        suite = _build_suite(spec)
        findings = suite.run(table)
        print("\ncheck findings:", summarise(findings) or "none")
        for f in findings[: args.max_show]:
            print(f"  {f.kind:<14} {f.column}[{f.row}]")
        if len(findings) > args.max_show:
            print(f"  ... {len(findings) - args.max_show} more")

    if args.reference:
        ref = Fingerprint.from_dict(json.loads(Path(args.reference).read_text(encoding="utf-8")))
        drift = diff_fingerprints(ref, fp)
        print("\nschema drift:", summarise(drift) or "none")
        for f in drift:
            print(f"  {f.kind:<12} {f.column} ({ref.types.get(f.column, '-')} -> {fp.types.get(f.column, '-')})")

    if args.save_fingerprint:
        Path(args.save_fingerprint).write_text(
            json.dumps(fp.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote fingerprint {args.save_fingerprint}")

    return 1 if findings else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dqguard",
        description="Deterministic ETL data-quality + schema-drift toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="run the offline benchmark")
    b.add_argument("--datasets", type=int, default=30, help="number of synthetic cases")
    b.add_argument("--rows", type=int, default=500, help="rows per case")
    b.add_argument("--seed", type=int, default=20260101, help="base RNG seed")
    b.add_argument("--bootstrap-iters", type=int, default=5000)
    b.add_argument("--perm-iters", type=int, default=10000)
    b.add_argument("--out-dir", type=str, default=None, help="output directory")
    b.add_argument("--no-write", action="store_true", help="print summary, write nothing")
    b.set_defaults(func=cmd_bench)

    v = sub.add_parser("validate", help="profile and validate a CSV file")
    v.add_argument("datafile", help="path to a CSV file")
    v.add_argument("--schema", help="JSON file describing checks")
    v.add_argument("--reference", help="JSON fingerprint to detect drift against")
    v.add_argument("--save-fingerprint", help="write this file's fingerprint to a path")
    v.add_argument("--coerce", action="store_true", help="parse numeric-looking strings")
    v.add_argument("--sample-size", type=int, default=24)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--max-show", type=int, default=20)
    v.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
