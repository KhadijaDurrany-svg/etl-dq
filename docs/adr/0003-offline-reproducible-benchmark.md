# ADR 0003: The benchmark is offline, seeded, and reproducible by construction

## Status
Accepted.

## Context
A benchmark that cannot be re-run to the same numbers is not evidence. The
project also must run with no network, no credentials, and minimal
dependencies (numpy only; no pandas).

## Decision
- All data is synthetic and generated from an integer seed via
  `numpy.random.Generator`. No files are read, no services are contacted.
- Every stochastic step derives its generator deterministically from the base
  seed, including each detector's RNG (via a stable per-name offset, never the
  salted builtin `hash`).
- `results.json` contains only measured numbers. `manifest.json` records the
  git SHA (falling back to `"n/a"`), the Python and numpy versions, the full
  config, and a SHA-256 checksum of the generated data.
- Uncertainty is reported non-parametrically: a percentile bootstrap 95% CI
  over cases, and a paired sign-flip permutation test between detectors (whose
  smallest reportable p-value is `1 / (iterations + 1)`).

## Consequences
- `dqguard bench` is byte-for-byte reproducible across runs on the same
  interpreter; this is asserted by a test.
- Tables are plain Python lists of scalars, so numpy is used only for RNG and
  statistics -- keeping the dependency surface tiny and the data model explicit.
- The permutation p-value floor means a huge, real effect is reported as e.g.
  `1.0e-4` at 10,000 permutations rather than `0`; the README says so plainly.
