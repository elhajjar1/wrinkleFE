"""Regression tests for issue #186: remove dead buckling / Monte Carlo
configuration fields from :class:`AnalysisConfig`.

The fields ``run_buckling``, ``n_buckling_modes``, ``run_montecarlo``,
``mc_samples``, and ``mc_seed`` (along with the matching CLI flags
``--buckling``, ``--montecarlo``, ``--mc-samples``) were declared,
validated, and plumbed through the CLI but never read by
:meth:`WrinkleAnalysis.run`.  The docstring promised optional buckling
and Monte Carlo steps that did not exist.

Option 1 of the issue (remove the dead fields) was chosen over option 2
(wire them up), as wiring them up would require a much larger scope
(buckling extraction, MC distribution choices, Jensen gap analysis,
testing).  This file pins the removal so the silent no-op cannot
regress.
"""

from __future__ import annotations

import pytest

from wrinklefe.analysis import AnalysisConfig

# --------------------------------------------------------------------------- #
# AnalysisConfig: removed kwargs now raise TypeError
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwarg, value",
    [
        ("run_buckling", True),
        ("n_buckling_modes", 5),
        ("run_montecarlo", True),
        ("mc_samples", 1000),
        ("mc_seed", 42),
    ],
)
def test_removed_kwargs_raise_type_error(kwarg, value):
    """Constructing ``AnalysisConfig`` with any of the five removed
    kwargs must raise ``TypeError`` — not silently succeed.

    This is the core acceptance criterion of issue #186: the fields are
    gone, so a user who still passes them gets a loud failure rather
    than a paid-for-but-ignored construction-time validator."""
    with pytest.raises(TypeError):
        AnalysisConfig(**{kwarg: value})


def test_removed_fields_absent_from_dataclass():
    """Belt-and-braces: the dataclass must not carry the removed fields
    as attributes either (so reflection / `dataclasses.fields` users
    don't see them)."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(AnalysisConfig)}
    for dead in (
        "run_buckling",
        "n_buckling_modes",
        "run_montecarlo",
        "mc_samples",
        "mc_seed",
    ):
        assert dead not in field_names, (
            f"AnalysisConfig still declares dead field {dead!r}"
        )


# --------------------------------------------------------------------------- #
# CLI: removed argparse flags no longer parse
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("flag", ["--buckling", "--montecarlo", "--mc-samples"])
def test_removed_cli_flags_rejected(flag):
    """The CLI no longer advertises ``--buckling``, ``--montecarlo``, or
    ``--mc-samples``; argparse must therefore reject them with the
    standard exit code 2 (CLI usage error)."""
    from wrinklefe.cli import main as cli_main

    # ``--mc-samples`` takes a value; the rest are flags.
    args = ["analyze", flag]
    if flag == "--mc-samples":
        args.append("100")

    with pytest.raises(SystemExit) as exc_info:
        cli_main(args)
    assert exc_info.value.code == 2


# --------------------------------------------------------------------------- #
# WrinkleAnalysis.run() docstring no longer promises buckling/MC steps
# --------------------------------------------------------------------------- #


def test_run_docstring_does_not_mention_buckling_or_montecarlo():
    """The ``run()`` docstring used to list optional buckling and Monte
    Carlo steps that ``run()`` never actually performed.  Removing the
    promise is part of issue #186."""
    from wrinklefe.analysis import WrinkleAnalysis

    doc = WrinkleAnalysis.run.__doc__ or ""
    lower = doc.lower()
    assert "buckling" not in lower, (
        "run() docstring still mentions buckling after #186"
    )
    assert "monte carlo" not in lower, (
        "run() docstring still mentions Monte Carlo after #186"
    )
