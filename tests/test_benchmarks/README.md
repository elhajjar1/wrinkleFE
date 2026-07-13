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

There is **no committed baseline** — a baseline captured inside an
ephemeral container is meaningless on CI runners. See the "Benchmarks"
section of `../../CONTRIBUTING.md` for how to bootstrap and refresh the
`baseline/` directory from a green `main` run, and how the CI comparison
gate becomes active once that directory exists.
