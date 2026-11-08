# ADR 0001: Findings are the atomic unit of truth and scoring

## Status
Accepted.

## Context
The library needs one representation that works for three jobs: what a check
emits, what ground truth records, and what the benchmark scores. If these used
different shapes we would need lossy conversions and the benchmark numbers
would be hard to trust.

## Decision
A single immutable `Finding(kind, column, row)` is used everywhere. `row` is an
integer for cell-level issues and `None` for column/table-level issues such as
schema drift. Scoring is pure set algebra on finding *keys*: a predicted
finding is a true positive only if its exact `(kind, column, row)` triple
appears in ground truth.

## Consequences
- Precision/recall/F1 are unambiguous and trivially auditable: TP, FP and FN are
  set intersections and differences.
- Checks must be deterministic and must agree with the injector on row indices.
  This is enforced by the "clean base has zero findings" test and by injecting
  each defect so that exactly one check fires.
- Partial credit is impossible (a finding either matches or it does not). We
  accept this; it keeps the metric honest and prevents score inflation from
  "close enough" matches.
