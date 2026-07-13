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
