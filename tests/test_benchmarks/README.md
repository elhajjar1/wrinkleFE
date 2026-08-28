# Performance benchmarks

Micro-benchmarks for the hot kernels, driven by
[`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/). Every
benchmark is marked `benchmark` **and** `slow`, so it is excluded from a
bare `pytest` run (the default `addopts` carry `-m 'not benchmark'`) and
from the `-m "not slow"` fast lane. Run them explicitly:

```bash
pytest tests/test_benchmarks -m benchmark
```

Each benchmark runs on a small, deterministic input and asserts a
correctness invariant (finite / bounded / expected shape) so a timing
harness can never silently pass on a broken kernel.

## The committed baseline

`baseline/Linux-CPython-3.12-64bit/` holds the run the CI compare step
measures against. pytest-benchmark keys storage by machine id
(`{system}-{impl}-{major.minor}-{bits}`), so that directory name must
match the `benchmarks` job's interpreter — Python 3.12 on Linux.

The current baseline is **container-generated**, not captured from a
GitHub runner, so its absolute timings are not comparable to CI hardware.
The compare step therefore runs under `continue-on-error: true`: it
executes and reports on every run — the gate is armed and visible — but
does not fail the build. Promoting it to blocking needs a runner-captured
baseline; see the "Benchmarks" section of `../../CONTRIBUTING.md` for the
bootstrap and refresh procedure.

Refreshing the baseline is a deliberate, reviewed act in its own PR —
never an automatic or incidental update.
