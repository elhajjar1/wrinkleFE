# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this
repository.

## Project

**WrinkleFE** — a Python finite-element package for predicting strength
and stiffness knockdown in composite laminates containing fibre-waviness
defects. Layout: source in `src/wrinklefe/`, tests in `tests/`, docs in
`docs/`, validation drivers in `validation/`, the Streamlit app in
`app.py` / `streamlit_viz.py`.

## Local checks before pushing

Run the same gates CI enforces, whenever the change touches the
relevant area:

- `ruff check .`
- `python -m mypy src/wrinklefe app.py streamlit_viz.py`
- `pytest`
- For docs changes: `sphinx-build -W docs docs/_build` (warnings are
  errors).

Library code uses module-level `logging` (never `print()`); match the
surrounding style and keep changes scoped.

## Merging to main (automatic)

When a unit of work is complete, take it all the way to `main`
automatically — do **not** wait for the user to ask to merge. Procedure:

1. **Push** the feature branch (never commit directly to `main`).
2. **Open the pull request as ready for review** (not a draft) so it is
   mergeable.
3. **Wait for all required CI checks to pass.** Never merge a pull
   request with failing or still-pending checks. If a check fails,
   diagnose and re-push the fix rather than merging.
4. **Post the PR link with a one-line summary, then pause for a brief
   review window** (the time CI takes is normally enough) so the user
   can object before the merge lands.
5. If no objection by the time CI is green, **merge the PR into `main`**
   and delete the branch.

Hold and ask first only when the change is risky, ambiguous, explicitly
flagged to wait, or the user has said to pause — otherwise the merge is
automatic once CI is green.
