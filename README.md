# dqguard

A small, deterministic ETL framework with a **data-quality validation** and
**schema-drift detection** layer, plus an **honest offline benchmark** of how
well it catches injected issues.

You define extract / transform / load steps over in-memory tabular data, attach
a declarative check suite (nulls, ranges, uniqueness, referential integrity,
types, categorical domains), and detect schema drift between two runs. The
benchmark injects a known mix of data-quality defects and schema drift into
seeded synthetic data and measures detection precision / recall / F1 against
weak baselines, with bootstrap confidence intervals and a paired permutation
test.

Everything runs offline with no network, no credentials, and numpy as the only
runtime dependency. Tables are plain Python lists of scalars; numpy is used only
for the RNG and the statistics. There is **no pandas**.

## How this differs

This repository was written to be distinct from a lower-level, single-job ETL
script (one that runs a cleaning query against a warehouse and writes a CSV to
object storage). Instead of being a one-shot extract-clean-load job, `dqguard`
is a **framework plus a measured quality layer**:

- The unit of value is a reusable **pipeline of steps** with a built-in
  **validation gate**, not a single hard-coded query and destination.
- It ships a **declarative check suite** (six check kinds) and **schema-drift
  detection** (add / drop / type), which a plain cleaning job does not have.
- It treats detection as a **benchmarked task**: ground-truth defects are
  injected and scored, with baselines, CIs, and a significance test — not just
  "it ran and uploaded".
- It is **fully offline and reproducible** by construction, with synthetic data
  rather than a live warehouse connection.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"      # dev extras: pytest, hypothesis, ruff
```

Requires Python >= 3.10.

## Use

Run the benchmark (writes `benchmarks/results.json` and `benchmarks/manifest.json`):

```bash
dqguard bench                 # 30 synthetic cases, 500 rows each
dqguard bench --no-write      # print the summary only, write nothing
dqguard bench --datasets 50 --rows 1000 --seed 7
```

Profile and validate a CSV file:

```bash
# profile columns (inferred type, null count, distinct count) + save a fingerprint
dqguard validate data.csv --coerce --save-fingerprint fp.json

# run declarative checks from a schema file, and detect drift vs the fingerprint
dqguard validate later.csv --coerce --schema schema.json --reference fp.json
```

A `schema.json` is a small declarative spec, e.g.:

```json
{
  "not_null": ["order_id", "amount"],
  "unique": ["order_id"],
  "range": {"amount": [0, 10000]},
  "allowed_values": {"status": ["new", "paid", "shipped", "cancelled"]},
  "type": {"order_id": "int"}
}
```

Programmatic use:

```python
from dqguard import (
    Table, Pipeline, Extract, Transform, ValidationStage,
    CheckSuite, NotNull, Range, Unique, fingerprint_table,
)

src = Table({"order_id": [1, 2, 2], "amount": [10.0, 20.0, -5.0]})
pipe = Pipeline(steps=[
    Extract("src", lambda: src),
    ValidationStage("quality", suite=CheckSuite([
        Unique("order_id"), Range("amount", 0, 1000),
    ])),
])
result = pipe.run()
print(result.findings["quality"])   # [Finding(unique, order_id, 2), Finding(range, amount, 2)]
```

## Benchmark results

Real numbers from `dqguard bench` (30 cases x 500 rows, base seed 20260101,
numpy 2.5.4). Reproduce with `dqguard bench`; the run is byte-for-byte stable.

Each case injects 24 cell-level data-quality defects (4 each of null, range,
type, uniqueness, referential, categorical) and 10 schema-drift defects (2
added columns, 2 dropped columns, 6 type-drifted columns), i.e. 34 ground-truth
issues per case.

| detector    | precision | recall |    F1 | F1 95% CI (bootstrap) |
|-------------|----------:|-------:|------:|-----------------------|
| **full**    |     0.986 |  0.920 | 0.952 | [0.942, 0.960]        |
| null_only   |     1.000 |  0.118 | 0.211 | [0.211, 0.211]        |
| random      |     0.022 |  0.018 | 0.019 | [0.012, 0.029]        |

`null_only` runs only the null check (a common real-world shortcut) and so can
never exceed the share of defects that are nulls. `random` flags positions at
random with a budget matched to the true defect density, establishing the
chance floor.

Paired sign-flip permutation test on per-case F1 (10,000 permutations; the
smallest reportable p-value is `1 / (10000 + 1) ≈ 9.999e-5`):

| comparison            | mean F1 difference | p-value   |
|-----------------------|-------------------:|-----------|
| full vs null_only     |             +0.741 | 9.999e-5  |
| full vs random        |             +0.932 | 9.999e-5  |

### Where the full detector succeeds and fails (by issue kind)

| kind            | precision | recall |    F1 | TP | FP | FN |
|-----------------|----------:|-------:|------:|---:|---:|---:|
| not_null        |     1.000 |  1.000 | 1.000 |120 |  0 |  0 |
| range           |     1.000 |  1.000 | 1.000 |120 |  0 |  0 |
| unique          |     1.000 |  1.000 | 1.000 |120 |  0 |  0 |
| referential     |     1.000 |  1.000 | 1.000 |120 |  0 |  0 |
| type            |     1.000 |  1.000 | 1.000 |120 |  0 |  0 |
| allowed_values  |     1.000 |  1.000 | 1.000 |120 |  0 |  0 |
| drift_add       |     1.000 |  1.000 | 1.000 | 60 |  0 |  0 |
| drift_drop      |     1.000 |  1.000 | 1.000 | 60 |  0 |  0 |
| **drift_type**  |     0.883 |  0.544 | 0.674 | 98 | 13 | 82 |

The exact declarative checks and the structural drift kinds (`drift_add`,
`drift_drop`) are perfect — they are exact set/comparison operations, so on a
provably clean base they neither miss an injected defect nor flag a clean cell.
**All of the error lives in `drift_type`**, which relies on sample-based type
inference (ADR 0002): it misses type drifts that converted only part of a column
(the sampled majority never flips → 82 false negatives) and occasionally flags
untouched heterogeneous columns whose reference and candidate samples disagree
(13 false positives). This is a genuine, measured limitation, not a tuned
number.

## Known gaps

- **Type-drift detection is approximate and the weakest link** (R≈0.54). It is
  sample-based by design; raising `sample_size` or widening the injected
  conversion fraction improves recall but is less realistic. We did not tune it
  toward 1.0.
- **Scoring is all-or-nothing per finding.** A near-miss on a row index counts
  as both a false positive and a false negative. There is no partial credit.
- **Referential integrity is single-column.** Composite foreign keys are not
  modelled.
- **Type checking is strict**: an `int` column rejects `float` cells and vice
  versa. There is no numeric-tower coercion.
- **Synthetic data only.** The generator covers the defect taxonomy the checks
  target; it is not a model of any specific production dataset, and the headline
  numbers are properties of that generator, not a universal accuracy claim.
- **Baselines are intentionally weak.** They bound the problem from below; they
  are not competing production validators.
- **`manifest.json` records `git_sha: "n/a"`** when run outside a git checkout.

## Project layout

```
src/dqguard/
  table.py       # in-memory columnar Table (stdlib only)
  schema.py      # type inference + serialisable Fingerprint
  checks.py      # Finding + declarative checks + CheckSuite
  drift.py       # schema-drift detection from fingerprints
  pipeline.py    # Extract/Transform/Load steps + ValidationStage gate
  generate.py    # seeded synthetic data + defect injection + ground truth
  baselines.py   # full / null_only / random detectors
  metrics.py     # precision/recall/F1, bootstrap CI, permutation test
  bench.py       # benchmark harness + results/manifest writer
  cli.py         # argparse CLI: `bench`, `validate`
tests/           # 100+ tests (pytest + Hypothesis)
docs/adr/        # architecture decision records
benchmarks/      # results.json + manifest.json (regenerated by `dqguard bench`)
```

## Development

```bash
ruff check .
pytest -q
dqguard bench --no-write
```

A non-root Docker image is provided (`docker build -t dqguard .`; runs as uid
10001). CI runs lint, tests, and a benchmark smoke run on Python 3.10 / 3.11 /
3.12, and builds the container.

## License

MIT — see [LICENSE](LICENSE).
