# Contributing to WrinkleFE

Thank you for your interest in contributing to WrinkleFE! This document provides guidelines for contributing to the project.

## Reporting Issues

Open a new issue and pick the form that fits — the
[**New issue**](https://github.com/elhajjar1/wrinkleFE/issues/new/choose)
chooser offers three:

- **Bug report** — for incorrect behavior or crashes (asks for version
  and a reproduction).
- **Enhancement** — pre-seeded with the project's
  Where / What / Suggested approach / Acceptance criteria skeleton.
- **Validation / physics discrepancy** — when a predicted knockdown or
  strength disagrees with experiment or another model (material, layup,
  wrinkle geometry, measured vs. predicted, data source).

Blank issues stay enabled for anything that doesn't fit a form. A
results JSON's `provenance` block carries the version and environment
fields a bug report needs.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/elhajjar1/wrinklefe.git
cd wrinklefe

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install in development mode with all dependencies
pip install -e ".[all,dev]"

# Run tests
pytest
```

## Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run the test suite (`pytest`) and ensure all tests pass
5. For user-visible changes, add an entry to the `[Unreleased]` section
   of [`CHANGELOG.md`](CHANGELOG.md) — and call out anything that shifts
   predictions under its **Numerical results** category
6. Submit a pull request with a clear description

## Pre-commit hooks

`.pre-commit-config.yaml` mirrors the `lint` CI job so you catch lint and
type errors before pushing rather than after:

```bash
pip install pre-commit
pre-commit install                        # ruff on every commit
pre-commit install --hook-type pre-push   # + mypy before a push
```

| Hook   | Stage      | What runs                                              |
|--------|------------|--------------------------------------------------------|
| `ruff` | commit     | `ruff check` — the full `pyproject.toml` ruleset        |
| `mypy` | pre-push   | `python -m mypy src/wrinklefe app.py streamlit_viz.py`  |

Both hooks are `language: system`: they invoke the ruff and mypy already
in your environment rather than installing pinned copies in an isolated
venv, so they cannot report a different answer than CI does. That does
mean the mypy hook needs the full development install — `pip install -e
".[all,dev]"` — because it type-checks `app.py` / `streamlit_viz.py`
against the real streamlit and plotly packages, exactly as the lint job
does. mypy is whole-tree and runs on the **pre-push** stage only; it is
too slow to sit on every commit.

Run everything against the whole tree at once with `pre-commit run
--all-files` (add `--hook-stage pre-push` to include mypy). There is
deliberately no formatter hook — this repo does not use `ruff format` —
and no commit-stage pytest hook.

## Test coverage floor

`pyproject.toml`'s `[tool.coverage.report]` sets a `fail_under` floor, so
a drop in coverage fails the build instead of quietly uploading a worse
number to Codecov.

The floor gates the `test-full` job — the complete suite, which is what
the floor was measured against. The OS/Python matrix job also passes
`--cov`, but it deselects the slow integration solves, so its coverage is
structurally lower; it opts out with an explicit `--cov-fail-under=0`
rather than being held to a number its lane cannot reach.

The floor sits a couple of points below the measured total so ordinary
run-to-run variation does not flake the build. It is a **ratchet**: when
coverage rises durably, raise the floor in the same PR. Never lower it to
make a build pass — add the missing tests instead.

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Add docstrings to public functions and classes
- Keep functions focused and modular

## Testing

- Add tests for new functionality
- Maintain or improve test coverage
- Tests are in the `tests/` directory, organized by module

### Test markers and the fast path

The suite carries four registered pytest markers (declared in
`pyproject.toml`; `--strict-markers` is on, so a typo'd or undeclared
marker is a hard collection error rather than a silent no-op):

- `slow` — FE/CZM solves taking more than ~5 s per test.
- `integration` — end-to-end pipeline tests (`tests/integration/` and
  `tests/test_integration/`).
- `viz` — needs the streamlit/plotly extras.
- `benchmark` — the pytest-benchmark performance suite, excluded from a
  bare `pytest` run by the default `-m 'not benchmark'` in `addopts`.

For the inner development loop, skip the slow integration solves:

```bash
pytest -m "not slow"
```

This deselects every `tests/integration/` file and the handful of
individually-slow tests elsewhere, running in a small fraction of the
full-suite time while still exercising the unit and fast-integration
layers. CI mirrors this: the OS/Python matrix job runs `-m "not slow"`,
and a dedicated `test-full` job runs the complete suite (`-m "not
benchmark"`) once and owns the coverage upload. Run the full suite
locally (`pytest`) before opening a PR when your change touches the FE,
CZM, or analysis paths.

### Doctests

The `>>>` examples in the source docstrings are executed as tests so a
drift between the documented output and real behaviour is caught. They
run via a dedicated invocation (and a matching `doctests` CI job), kept
**out** of the default `addopts` so a doc example can never block the
core suite:

```bash
pytest --doctest-modules src/wrinklefe -q
```

`doctest_optionflags = "NORMALIZE_WHITESPACE ELLIPSIS"` (in
`pyproject.toml`) is applied, so `...` may stand in for volatile float
digits and array/line-wrap whitespace is tolerated.

When you add or edit a docstring example, triage it into one of:

- **Make it exact** — include the setup lines and the *real* expected
  output (run the snippet and paste the actual repr; never guess). This
  is preferred for scalar/short-array/pure-function examples, which then
  act as regression guards. Wrap NumPy scalar results in `bool(...)` /
  `float(...)` so the repr is stable across NumPy versions.
- **Mark `# doctest: +SKIP`** when a faithful example would need a heavy
  object (a generated mesh, a full FE solve, a fitted result) or be
  slow. An illustrative snippet is fine, but it **must** be explicitly
  `+SKIP` — non-runnable examples are marked, never silently unrun.
- **Convert to a plain code block** (strip the `>>>`) only if it was
  never meant as an executable transcript.

Keep the doctests fast (target < ~1 min); never let a "make exact"
example trigger a real FE solve.

### Benchmarks

Performance micro-benchmarks for the hot kernels live in
`tests/test_benchmarks/` and use
[`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/) (in the
`dev` extra). Each is marked `benchmark` and `slow`, so it is excluded
from both a bare `pytest` run and the `-m "not slow"` fast lane. Run and
save timings explicitly:

```bash
# run the suite
pytest tests/test_benchmarks -m benchmark

# capture before/after numbers around a change
pytest tests/test_benchmarks -m benchmark --benchmark-autosave
```

CI runs these in a dedicated `benchmarks` job that autosaves the timings
as a downloadable artifact and compares them against the committed
baseline in `tests/test_benchmarks/baseline/`, flagging a median
regression worse than 2x (`--benchmark-compare-fail=median:100%`). The 2x
threshold is deliberate: shared CI runners have large run-to-run CPU
variance and a tighter bound would flake.

**The compare step is currently non-blocking** (`continue-on-error:
true`). The committed baseline was generated inside a development
container rather than captured from a GitHub runner — on Python 3.12, so
its pytest-benchmark machine id (`Linux-CPython-3.12-64bit`) matches the
job's storage key, but on different hardware. Absolute timings are
therefore not comparable to a runner's, and a hard failure would be noise.
The step still runs and reports on every build, so the gate is armed and
visible rather than silently skipped.

**To promote it to a blocking gate:**

1. Download the `benchmark-timings` artifact from a green `main` run of
   the `benchmarks` job. (Note: before #376 this artifact was always
   empty — `.benchmarks/` is a dot directory and
   `actions/upload-artifact` skips hidden files unless
   `include-hidden-files: true` is set, which it now is.)
2. Replace the `*.json` run under
   `tests/test_benchmarks/baseline/<machine-id>/` with the downloaded
   one, keeping the machine-id directory that matches the `benchmarks`
   job's interpreter.
3. In the same PR, drop `continue-on-error: true` from the "Compare
   against committed baseline" step in `.github/workflows/ci.yml`.

**Refreshing the baseline** is a deliberate, reviewed act — never
automatic. When an intentional performance change (or a runner upgrade)
shifts the numbers, capture a fresh artifact from a green `main` run and
replace the file under `tests/test_benchmarks/baseline/` in its own PR,
so the reviewer can see exactly which kernels moved and why.

## Adding Materials

To add a new material to the built-in library, add an entry in `src/wrinklefe/core/material.py` in the `_load_builtins()` method. Include all elastic constants, strength allowables, and a literature reference.

For one-off or ad-hoc materials you do not need to modify source — turn on
**Expert mode** in the Streamlit app and pick **Custom…** from the
**Material** selectbox. The inline editor exposes E1/E2/E3, G12/G13/G23,
ν12/ν13/ν23 and the Xt/Xc/Yt/Yc/Zt/Zc/S12/S13/S23 allowables, seeded from
IM7/8552. Custom materials are scoped to the current Streamlit session and
do not persist; use the source-level workflow above for anything you want
to keep. See [`DEPLOYMENT_STREAMLIT.md`](docs/internal/DEPLOYMENT_STREAMLIT.md) for a
full feature tour.

## Questions

For questions about the science or implementation, open a GitHub issue with the "question" label.
