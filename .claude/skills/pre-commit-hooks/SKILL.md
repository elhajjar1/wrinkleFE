---
name: pre-commit-hooks
description: Use when setting up, modifying, or troubleshooting
  pre-commit hooks for wrinkleFE. Triggers on "add pre-commit",
  "set up git hooks", "catch lint locally", or requests to run
  ruff/mypy/pytest before commits.
---

# Pre-commit hooks for wrinkleFE

## Current state

- **No pre-commit infrastructure exists today.** No
  `.pre-commit-config.yaml`, no `.husky`, nothing in `.git/hooks` beyond
  Git defaults.
- CI runs Ruff (starter scope) and mypy (init only) in
  `.github/workflows/lint.yml`, plus pytest in `ci.yml`. The point of
  this skill is to mirror those locally so contributors catch issues
  before pushing.

## Goal

Add a `.pre-commit-config.yaml` that runs the **same checks CI runs**,
no stricter. Don't introduce drift between local and CI enforcement —
that frustrates contributors.

## Workflow

1. Create `.pre-commit-config.yaml` at repo root:
   ```yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.6.9  # pin to a current release; bump deliberately
       hooks:
         - id: ruff
           args: [--select, "E9,F63,F7,F82"]  # match lint.yml scope exactly
     - repo: https://github.com/pre-commit/mirrors-mypy
       rev: v1.11.2
       hooks:
         - id: mypy
           files: ^src/wrinklefe/__init__\.py$  # match lint.yml scope exactly
           additional_dependencies: [numpy, scipy]
   ```
2. **Keep scopes in sync with `lint.yml`.** When `fix-ruff-violations`
   or `expand-mypy-coverage` widens CI scope, update this file in the
   same PR.
3. Do NOT add a pytest hook by default — the test suite is large
   enough that running it on every commit would be painful. If the
   user wants test-on-push, use `pre-push` stage instead of
   `pre-commit`:
   ```yaml
   - repo: local
     hooks:
       - id: pytest
         name: pytest
         entry: pytest
         language: system
         pass_filenames: false
         stages: [pre-push]
   ```
4. Add a brief installation note to `CONTRIBUTING.md`:
   ```
   pip install pre-commit
   pre-commit install            # for pre-commit stage
   pre-commit install --hook-type pre-push  # if pytest hook enabled
   ```
5. Test it: `pre-commit run --all-files`. Expect it to pass on a clean
   checkout of `main`. If it fails, the hook scope is wider than CI's —
   tighten it.

## Gotchas

- **`mirrors-mypy` runs mypy in an isolated venv** without the project's
  deps. List anything imported by the checked files in
  `additional_dependencies`. Currently `__init__.py` only re-exports,
  but widening scope will require adding `meshio`, `pyvista`, etc.
- **Pin `rev:` to a release tag**, not `main`. Auto-updates via
  `pre-commit autoupdate` are fine but should be deliberate PRs.
- **Don't enable hooks contributors will skip.** A hook that fails
  every commit on unrelated files trains people to use `--no-verify`,
  which defeats the purpose.

## Done check

- `.pre-commit-config.yaml` exists and matches CI scope exactly.
- `pre-commit run --all-files` passes on `main`.
- `CONTRIBUTING.md` documents the install step.
- CI behavior unchanged (this is a local-only addition).
