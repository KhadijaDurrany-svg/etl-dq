# ADR 0002: Schema-drift type inference is sample-based (and therefore fallible)

## Status
Accepted.

## Context
Detecting type drift requires knowing each column's type in two runs. Scanning
every cell of every column on every run is both expensive on wide tables and
unrealistic -- real profilers sample. We wanted the benchmark to reflect a
genuine detection problem, not a trivially solvable one.

## Decision
`infer_column_type` draws a bounded random sample (default 16 non-null cells)
and takes the majority Python type, with a fixed tie-break order. The reference
fingerprint and the candidate are profiled with *different* RNG streams, as two
real runs would be. `drift_add` and `drift_drop` remain exact set differences;
only `drift_type` depends on inference.

## Consequences
- The detector is honestly imperfect on `drift_type`. In the shipped benchmark
  it scores roughly P≈0.88 / R≈0.54 on that kind, while every exact check and
  the structural drift kinds score 1.0. Overall F1 lands near 0.95, not 1.0.
- Two real error sources appear and are measured: partially converted columns
  whose sampled majority never flips (false negatives), and untouched
  heterogeneous columns whose two samples disagree (false positives).
- The gap is tunable from the dataset spec (`sample_size`, conversion fraction,
  heterogeneity) rather than hard-coded, so the benchmark can be made harder or
  easier without touching detector code. We deliberately did not "fix" the
  inference to reach 1.0, because the imperfection is the point.
