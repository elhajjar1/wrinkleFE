"""Failure-mode breakdown plot (issue #269)."""

import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from wrinklefe.viz.plots_2d import (  # noqa: E402
    _governing_mode_shares,
    failure_mode_family,
    plot_failure_mode_breakdown,
)


def _synthetic_results():
    """Four elements, two criteria; governing = per-element max-FI criterion.

    elem0 larc05 0.9 (fiber_kinking), elem1 max_stress 0.7 (matrix_compression),
    elem2 larc05 0.8 (fiber_kinking), elem3 max_stress 0.6 (matrix_tension).
    """
    fi = {
        "larc05": np.array([[0.9], [0.2], [0.8], [0.1]]),
        "max_stress": np.array([[0.3], [0.7], [0.5], [0.6]]),
    }
    modes = {
        "larc05": np.array(
            [["fiber_kinking"], ["matrix_tension"], ["fiber_kinking"], ["shear"]]
        ),
        "max_stress": np.array(
            [["matrix_compression"], ["matrix_compression"], ["shear"],
             ["matrix_tension"]]
        ),
    }
    return SimpleNamespace(failure_indices=fi, failure_modes=modes)


def test_family_mapping():
    assert failure_mode_family("fiber_kinking") == "fiber"
    assert failure_mode_family("kink_band") == "fiber"
    assert failure_mode_family("matrix_transverse_compression") == "matrix"
    assert failure_mode_family("shear") == "shear"
    assert failure_mode_family("delamination") == "delamination"
    assert failure_mode_family("") == "other"


def test_governing_shares_count():
    shares, top, max_fi = _governing_mode_shares(_synthetic_results(), "count")
    assert top == "fiber_kinking"
    assert max_fi == pytest.approx(0.9)
    assert shares["fiber_kinking"] == pytest.approx(0.5)
    assert shares["matrix_compression"] == pytest.approx(0.25)
    assert shares["matrix_tension"] == pytest.approx(0.25)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_governing_shares_fi_weighted():
    shares, top, _ = _governing_mode_shares(_synthetic_results(), "fi")
    assert top == "fiber_kinking"
    assert shares["fiber_kinking"] == pytest.approx(1.7 / 3.0)
    assert shares["matrix_compression"] == pytest.approx(0.7 / 3.0)
    assert shares["matrix_tension"] == pytest.approx(0.6 / 3.0)


@pytest.mark.parametrize("weight", ["count", "fi"])
def test_plot_renders_one_bar_per_mode(weight):
    ax = plot_failure_mode_breakdown(_synthetic_results(), weight=weight)
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 3  # three distinct governing modes
    plt.close(ax.figure)


def test_invalid_weight_raises():
    with pytest.raises(ValueError, match="weight"):
        plot_failure_mode_breakdown(_synthetic_results(), weight="bogus")


def test_empty_field_raises():
    empty = SimpleNamespace(failure_indices=None, failure_modes=None)
    with pytest.raises(ValueError, match="FE failure field"):
        plot_failure_mode_breakdown(empty, weight="count")
