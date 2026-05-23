---
name: fix-ruff-violations
description: Use when expanding Ruff lint enforcement in CI or fixing
  batches of existing Ruff violations in src/wrinklefe. Triggers on
  requests like "fix ruff", "clean up lint", "expand lint scope",
  "address issue #87", or "turn on rule X in CI".
---

# Fixing Ruff violations in wrinkleFE

## Current state

- Full Ruff config lives in `pyproject.toml` under `[tool.ruff.lint]`:
  `select = ["E", "F", "W", "I", "N", "UP"]`, line-length 100, target
  `py310`.
- **CI does NOT enforce that full set yet.** `.github/workflows/lint.yml`
  runs only the starter scope: `ruff check --select E9,F63,F7,F82 .`
- ~840 violations across the repo are deferred. Tracked in **issue #87**.
- The goal is to incrementally widen CI's `--select` list until it
  matches `pyproject.toml`.

## Workflow

1. **Pick ONE rule family** to expand next (e.g. `I` imports, `UP`
   pyupgrade, `W` whitespace). Smaller families first reduces blast
   radius per PR.
2. Scope the work:
   ```
   ruff check --select <FAMILY> src/wrinklefe tests app.py streamlit_viz.py
   ```
3. Apply autofixes where safe:
   ```
   ruff check --select <FAMILY> --fix src/wrinklefe tests app.py streamlit_viz.py
   ```
4. Hand-fix anything left. Do NOT silence with blanket `# noqa` — use
   per-line `# noqa: <CODE>` only when the rule is genuinely wrong for
   that line.
5. Run the full test suite (`pytest`) — autofixes occasionally break
   semantics (e.g. `UP` rewrites can shift behavior on edge cases).
6. Once the family is clean, **add it to `lint.yml`'s `--select` list**
   so CI starts enforcing it. Update the comment on lines 30-32 of
   `lint.yml` to reflect the new scope.

## Gotchas

- **`N` (naming) hits scientific math hard.** Identifiers like `E`, `K`,
  `sigma_x`, `D_matrix`, single-letter Voigt indices match real domain
  conventions. Prefer per-file `# noqa: N806` (or `N803` for arg names)
  over renaming domain terms. Discuss before renaming anything that
  appears in `core/transforms.py`, `elements/`, or `failure/`.
- **`I` (isort) can reorder past `# type: ignore` comments** — verify
  mypy still passes after autofix.
- Keep diffs minimal: don't reformat unrelated lines just because Ruff
  also flagged them in passing.
- `app.py` and `streamlit_viz.py` live at the repo root, not under
  `src/`. Make sure your scope includes them.

## Done check

- `ruff check --select <FAMILY> .` exits clean.
- `pytest` passes.
- `.github/workflows/lint.yml` `--select` list updated.
- PR references issue #87.
