---
name: expand-mypy-coverage
description: Use when widening mypy's checked scope beyond
  src/wrinklefe/__init__.py, or fixing type errors uncovered when
  enabling mypy on a new module. Triggers on "add types", "fix mypy",
  "expand mypy scope", "address mypy follow-up to #87".
---

# Expanding mypy coverage in wrinkleFE

## Current state

- mypy config in `pyproject.toml` under `[tool.mypy]`:
  `python_version = "3.10"`, `warn_return_any = true`,
  `warn_unused_configs = true`.
- **CI only checks the package init file:**
  `.github/workflows/lint.yml` runs `mypy src/wrinklefe/__init__.py`.
- The broader `src/wrinklefe/` tree has ~102 errors today — mostly
  `numpy` `Any` returns and missing third-party stubs. Tracked as a
  follow-up to **issue #87**.
- Goal: walk the scope outward module-by-module until CI checks the
  whole package.

## Workflow

1. **Pick the next module** to bring under mypy. Suggested order
   (leaves first, then aggregators):
   1. `core/` (dataclasses, transforms) — least dependency surface
   2. `elements/` (hex8, hex8i)
   3. `failure/` (criteria modules)
   4. `solver/`
   5. `viz/`, `sweep/`, `io/`
   6. `analysis.py`, `cli.py`
   7. `app.py`, `streamlit_viz.py` (these live at repo root, not under `src/`)
2. Run mypy locally on just that module:
   ```
   mypy src/wrinklefe/<module>
   ```
3. Fix errors in this priority order:
   - **Add real annotations** to public functions and dataclass fields.
   - **For `numpy` returns**: prefer `np.ndarray` or
     `npt.NDArray[np.float64]` (import as `import numpy.typing as npt`)
     over `Any`.
   - **For SciPy / meshio / pyvista / streamlit / plotly**: install
     stubs if available (`types-*` on PyPI), otherwise add a targeted
     `# type: ignore[import-untyped]` on the import line with a comment.
   - **Avoid `# type: ignore` without a code** — always specify the
     error code so future widening doesn't silently hide regressions.
4. Run `pytest` to confirm no behavior changed.
5. **Add the module to `lint.yml`** — append it to the mypy invocation:
   ```
   mypy src/wrinklefe/__init__.py src/wrinklefe/<module>
   ```
   (Or switch to `mypy src/wrinklefe/` once enough modules are clean.)
   Update the comment on lines 36-38 of `lint.yml` to reflect new scope.

## Gotchas

- **`warn_return_any` is on globally** — any function returning a
  `numpy` operation result will trip this until annotated. Annotate
  the return type explicitly.
- **Don't add a per-module `[[tool.mypy.overrides]]` section to
  silence whole files.** The point is to fix them, not hide them. The
  only legitimate override is `ignore_missing_imports` for genuinely
  unstubbed third-party packages, scoped to that package only.
- The package ships a `py.typed` marker (see `[tool.setuptools.package-data]`),
  so downstream consumers will see our types. Sloppy `Any` here leaks
  to users.
- `app.py` and `streamlit_viz.py` are NOT under `src/wrinklefe/` —
  they're at repo root. Adding them to the mypy invocation needs a
  separate path argument.

## Done check

- `mypy <new scope>` exits clean.
- `pytest` passes.
- `.github/workflows/lint.yml` mypy invocation includes the new module.
- No new blanket `ignore` directives added.
