<!-- Thanks for contributing to WrinkleFE. -->

## Linked issue

Fixes #<!-- issue number, or "Refs #N" for partial work -->

## Summary

<!-- What changed and why. Note any behavior change to numeric outputs. -->

## Checklist

- [ ] Tests added or updated for the change
- [ ] `pytest` green (use `-m "not slow"` to skip FE integration solves)
- [ ] `ruff check .` clean
- [ ] `mypy src/wrinklefe` clean
- [ ] Docs/README updated if user-facing
- [ ] `CHANGELOG.md` `[Unreleased]` updated if the change is user-visible (note prediction-shifting changes under **Numerical results**)
- [ ] `python scripts/validate.py` shows no validation-ledger drift (or the change is intentional and re-pinned)
