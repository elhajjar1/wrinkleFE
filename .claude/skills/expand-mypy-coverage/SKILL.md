---
name: expand-mypy-coverage
description: Use when typing a NEW module in wrinkleFE, fixing mypy
  errors, or tightening mypy settings. The whole-tree scope migration is
  already finished — do not narrow it. Triggers on "add types",
  "fix mypy", "expand mypy scope", "address mypy follow-up to #87".
---

# mypy in wrinkleFE

## Current state — the scope migration is DONE

**Read this before touching any mypy config or CI file.** The
incremental module-by-module widening this skill used to describe was
completed. CI checks the entire tree and it is at **zero errors**:

- `.github/workflows/lint.yml` runs, verbatim:
  `mypy src/wrinklefe app.py streamlit_viz.py` — the whole `src/wrinklefe`
  package *plus* both repo-root Streamlit entry points.
- That job installs `".[dev,streamlit]"`, so `app.py` / `streamlit_viz.py`
  are checked against **real** streamlit and plotly imports rather than an
  Any-typed fallback. Installing only `[dev]` would leave CI green while
  the documented local gate fails (issue #374) — keep the extra.
- `[tool.mypy]` in `pyproject.toml`: `python_version = "3.10"`,
  `warn_return_any = true`, `warn_unused_configs = true`, and
  `check_untyped_defs = true` (so unannotated function bodies are checked
  too, not skipped).
- The only `[[tool.mypy.overrides]]` are `ignore_missing_imports` scoped
  to genuinely unstubbed third-party packages: `streamlit`/`plotly` (no
  stubs exist on PyPI) and `gspread`/`google.oauth2.*` (lazily imported
  optional usage-logging deps, absent from the lint env). `scipy-stubs`
  is a real `dev` dependency — scipy is *not* ignored.
- CLAUDE.md documents `python -m mypy src/wrinklefe app.py streamlit_viz.py`
  as a local gate, and `.pre-commit-config.yaml` runs that exact command
  as a `pre-push` hook.

**This skill's job is now (a) guarding that scope against regression and
(b) typing new code well.** It is no longer a migration plan.

## Do not regress the scope

Treat all of these as bugs, not shortcuts:

- Narrowing `lint.yml`'s mypy invocation to a subset of paths, or dropping
  `app.py` / `streamlit_viz.py` from it.
- Dropping `streamlit` from the lint job's install extras.
- Adding a `[[tool.mypy.overrides]]` block that silences a *first-party*
  module (`ignore_errors`, or a blanket `ignore_missing_imports` on
  `wrinklefe.*`). The only legitimate override is
  `ignore_missing_imports` for an unstubbed third-party package, scoped to
  that package.
- Turning off `warn_return_any` or `check_untyped_defs` to make an error
  go away.

If a change genuinely cannot be typed, the answer is a narrow,
error-coded `# type: ignore[code]` with a comment saying why — never a
config-level rollback.

## Workflow — adding or typing a new module

1. Write the annotations as you write the module; whole-tree mypy means a
   new file is checked the moment it lands. There is no "bring it under
   mypy later" step any more.
2. Run the real gate (not a narrowed one):
   ```
   python -m mypy src/wrinklefe app.py streamlit_viz.py
   ```
   Checking just your file can both miss errors and invent them — mypy's
   results depend on the full import graph.
3. Fix errors in this priority order:
   - **Add real annotations** to public functions and dataclass fields.
   - **For `numpy` returns**: `npt.NDArray[np.float64]`
     (`import numpy.typing as npt`) rather than `Any` or bare
     `np.ndarray`. `warn_return_any` will flag the `Any` anyway.
   - **For an unstubbed third-party import**: check PyPI for a `types-*`
     or `*-stubs` package first (that is how `scipy-stubs` got added to
     the `dev` extra). Only if none exists, add a package-scoped
     `[[tool.mypy.overrides]] ignore_missing_imports` with a comment
     explaining why, following the existing blocks' style.
   - **Never `# type: ignore` without a code** — an uncoded ignore hides
     future, unrelated regressions on that line.
4. Run `pytest -m "not slow"` to confirm nothing behavioural changed.
5. No CI edit is needed. The invocation is already whole-tree; a new
   module under `src/wrinklefe/` is picked up automatically.

## Gotchas

- **`check_untyped_defs = true`** means an unannotated helper is still
  type-checked internally — you cannot dodge an error by dropping the
  signature.
- **`warn_return_any` is on globally** — a function returning a numpy
  expression trips it until the return type is explicit.
- The package ships a `py.typed` marker (`[tool.setuptools.package-data]`),
  so these types are part of the public API. A sloppy `Any` leaks to
  downstream consumers.
- `app.py` and `streamlit_viz.py` are at the repo root, not under `src/` —
  they need their own path arguments, which is why the invocation lists
  three paths rather than one.

## Done check

- `python -m mypy src/wrinklefe app.py streamlit_viz.py` reports
  **zero** errors (`Success: no issues found`).
- `pytest -m "not slow"` passes.
- `lint.yml`'s mypy scope and install extras are **unchanged** (or
  genuinely widened — never narrowed).
- No new uncoded `# type: ignore`, and no new first-party override block.
