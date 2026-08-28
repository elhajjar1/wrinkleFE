---
name: pre-commit-hooks
description: Use when modifying or troubleshooting wrinkleFE's
  pre-commit hooks. The config already exists at .pre-commit-config.yaml
  and mirrors CI. Triggers on "add pre-commit", "set up git hooks",
  "catch lint locally", "pre-commit is failing", or requests to run
  ruff/mypy/pytest before commits.
---

# Pre-commit hooks in wrinkleFE

## Current state — the config EXISTS

**Read `.pre-commit-config.yaml` before changing anything.** It was added
in issue #376 and already mirrors CI. Do not recreate it, and do not
follow a from-scratch recipe.

What it configures today:

| Hook   | Stage        | Command                                          | Mirrors |
|--------|--------------|--------------------------------------------------|---------|
| `ruff` | commit       | `ruff check` (full pyproject ruleset)            | `lint.yml` "Ruff (full pyproject config)" |
| `mypy` | **pre-push** | `python -m mypy src/wrinklefe app.py streamlit_viz.py` | `lint.yml` "Mypy (whole tree)" |

Both are `repo: local` / `language: system`. That is a deliberate design
choice, not an oversight — see below.

Install:

```bash
pip install pre-commit
pre-commit install                        # ruff on every commit
pre-commit install --hook-type pre-push   # + mypy before a push
```

Verify: `pre-commit run --all-files` (and
`pre-commit run --all-files --hook-stage pre-push` for the mypy hook).
Both pass on a clean `main`.

## Why local/system hooks, not upstream mirrors

The old recipe pinned `astral-sh/ruff-pre-commit` and
`pre-commit/mirrors-mypy` with a `rev:`. **Do not switch back.** A pinned
mirror installs a *second*, differently-versioned ruff/mypy in an
isolated venv, which then:

- reports findings CI does not (or misses findings CI has) whenever the
  pinned `rev` and the `dev` extra's floors diverge; and
- for mypy, runs without the project's dependencies, so every import
  needs duplicating into `additional_dependencies` — including
  `streamlit` and `plotly`, whose absence is exactly the failure mode
  issue #374 fixed in CI.

`language: system` runs the ruff and mypy from your `pip install -e
".[all,dev]"` environment, so the hooks cannot drift from CI's tools:
they *are* CI's tools.

## Keeping it in sync with CI

The scopes are already in sync and both CI scopes are final (see the
`fix-ruff-violations` and `expand-mypy-coverage` skills — both migrations
are complete). Ruff's rules live in `pyproject.toml`, which the hook and
CI both read, so widening `select` needs no change here at all.

Change `.pre-commit-config.yaml` only when `lint.yml`'s *commands*
change — and then in the same PR, so the mirror stays exact.

## Gotchas

- **The mypy hook needs streamlit + plotly importable.** It runs whole-tree
  against `app.py` / `streamlit_viz.py`; in a bare `[dev]` environment
  those fall back to Any-typed imports and the hook can pass where CI
  fails. Install `".[all,dev]"` (or at minimum `".[dev,streamlit]"`).
- **mypy is `pass_filenames: false` and whole-tree on purpose.** mypy's
  results depend on the full import graph, so checking only the changed
  files would report a different answer than CI. That is also why it sits
  on `pre-push` rather than `pre-commit` — it is too slow per commit.
- **No pytest hook, and no formatter hook.** The suite is minutes long,
  and this repo deliberately does not use `ruff format` — adding either
  would create noise contributors route around with `--no-verify`.
- **`pre-commit run` alone only runs staged files.** Use `--all-files` to
  reproduce CI's whole-tree result.
- A hook failing on files you did not touch usually means the tree is
  genuinely dirty (or your env is missing an extra) — fix that rather
  than narrowing the hook.

## Done check

- `.pre-commit-config.yaml` still mirrors `lint.yml`'s commands exactly.
- `pre-commit run --all-files` and
  `pre-commit run --all-files --hook-stage pre-push` both pass.
- CONTRIBUTING.md's "Pre-commit hooks" section still matches the config.
- CI behaviour unchanged — this is a local-only convenience.
