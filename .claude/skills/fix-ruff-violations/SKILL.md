---
name: fix-ruff-violations
description: Use when fixing Ruff findings in wrinkleFE or enabling an
  ADDITIONAL rule family (B, SIM, C4, ...) on top of the enforced set.
  The E/F/W/I/N/UP scope migration is already finished and CI is clean —
  do not narrow it. Triggers on "fix ruff", "clean up lint",
  "expand lint scope", "address issue #87", "turn on rule X in CI".
---

# Ruff in wrinkleFE

## Current state — the scope migration is DONE

**Read this before touching any ruff config or CI file.** The incremental
`--select` widening this skill used to describe was completed as part of
issue #87. CI enforces the **full** pyproject ruleset and the tree is
clean:

- `.github/workflows/lint.yml` runs, verbatim: `ruff check .` — no
  `--select` override, no path restriction. It lints the whole repo
  (`src/`, `tests/`, `validation/`, `examples/`, `scripts/`, `app.py`,
  `streamlit_viz.py`).
- `[tool.ruff.lint]` in `pyproject.toml`: `select = ["E", "F", "W", "I",
  "N", "UP"]`, `line-length = 100`, `target-version = "py310"`.
- Project-wide `ignore = ["N802", "N803", "N806", "N815", "N816"]` — the
  pep8-naming codes that clash with composite-mechanics notation (`E1`,
  `G12`, `Q11`, `Ke`, `sigma_x`, `gamma_Y`, `GIc`, `eta_BK`, Voigt
  indices, and unit-suffixed fields like `analytical_strength_MPa`).
  These are deliberate, documented ignores — chosen over hundreds of
  per-line `noqa` or renaming domain terms. **N801 (class names) stays
  enforced**; it caught a genuinely off-convention test class.
- One `per-file-ignores` entry: `"validation/*" = ["E402"]`, because each
  validation runner inserts the in-repo `src` onto `sys.path` before
  importing wrinklefe — the canonical reason E402 exists to be silenced.
- `.pre-commit-config.yaml` runs `ruff check` as a commit-stage hook
  against the same pyproject config.

**This skill's job is now (a) guarding that scope against regression,
(b) fixing findings in new code, and (c) adding NEW rule families.** It
is no longer a migration plan, and there is no backlog of ~840 deferred
violations.

## Do not regress the scope

Treat all of these as bugs, not shortcuts:

- Reintroducing a `--select` (or `--ignore`) override on the CI
  invocation — CI must read the pyproject config, so local and CI agree.
- Restricting `ruff check .` to a subset of paths.
- Removing a rule family from `select` to make a finding go away.
- Adding a code to the project-wide `ignore` list for anything other than
  a genuine, repo-wide domain-notation clash. A single awkward line wants
  a per-line `# noqa: <CODE>`; a single awkward file wants a
  `per-file-ignores` entry with a comment.

## Workflow — enabling an ADDITIONAL rule family

The natural next step is widening `select`, e.g. `B` (bugbear), `SIM`
(simplify), `C4` (comprehensions), `RUF`. One family per PR:

1. **Measure the blast radius first:**
   ```
   ruff check --select <FAMILY> --statistics .
   ```
2. Apply autofixes where safe, then read the diff:
   ```
   ruff check --select <FAMILY> --fix .
   ```
3. Hand-fix the remainder. Do **not** blanket-silence: use a per-line
   `# noqa: <CODE>` with a reason only where the rule is genuinely wrong
   for that line.
4. Run `pytest -m "not slow"` — autofixes occasionally shift semantics
   (`UP` and `SIM` rewrites especially).
5. **Add the family to `select` in `pyproject.toml`** — that is the whole
   change CI needs, because `lint.yml` reads the config. Do not touch
   `lint.yml`. Document any new ignores next to the existing ones.
6. `ruff check .` must be clean before the PR opens.

## Workflow — a finding in new code

Just fix it. `ruff check .` is the gate; `ruff check --fix .` handles most
of E/W/I/UP mechanically. Verify with `ruff check .` and the pre-commit
hook.

## Gotchas

- **`N` (naming) hits scientific math hard** — but the codes that matter
  are already ignored project-wide. If you hit a *new* N violation,
  prefer fixing the name unless it is genuine domain notation; do not
  widen the global ignore list casually.
- **`I` (isort) can reorder past `# type: ignore` comments** — rerun
  `python -m mypy src/wrinklefe app.py streamlit_viz.py` after an `I`
  autofix.
- Keep diffs minimal: don't reformat unrelated lines Ruff flagged in
  passing. This repo has **no formatter** — `ruff format` is deliberately
  not adopted, and running it would create tree-wide diff noise.
- `ruff check .` already covers `app.py` and `streamlit_viz.py` at the
  repo root; no extra path argument is needed.

## Done check

- `ruff check .` exits clean.
- `pytest -m "not slow"` passes.
- `lint.yml` is **unchanged** (config lives in `pyproject.toml`).
- Any new `select` entry and any new ignore is documented with a comment.
