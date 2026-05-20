"""Regression tests for issue #190: interface_1/interface_2 defaults.

Previously ``AnalysisConfig.interface_1`` / ``interface_2`` defaulted to
the literals ``11`` / ``12``, which are only valid for the default
24-ply quasi-isotropic layup.  Any user-supplied ``angles`` shorter
than 13 plies (e.g. the canonical ``[0, 90]`` smoke-test layup) raised
``ValueError`` during ``__post_init__`` even though the user had not
touched the interface fields at all.

The fix uses the same sentinel pattern as ``domain_length``: the
defaults are ``None`` and resolve to the midplane pair
``((n_plies // 2) - 1, n_plies // 2)`` once ``angles`` is known.
Explicit out-of-range integers must still raise.
"""

from __future__ import annotations

import pytest

from wrinklefe.analysis import AnalysisConfig


def test_short_layup_constructs_without_explicit_interfaces():
    """Two-ply layup must construct cleanly and pick midplane interfaces."""
    cfg = AnalysisConfig(angles=[0, 90])
    # For n_plies=2: (2 // 2) - 1 = 0, 2 // 2 = 1 → straddle the midplane.
    assert cfg.interface_1 == 0
    assert cfg.interface_2 == 1


def test_eight_ply_ud_layup_picks_midplane_defaults():
    """UD [0]*8 must construct and place wrinkles around the midplane."""
    cfg = AnalysisConfig(angles=[0] * 8)
    assert cfg.interface_1 == 3
    assert cfg.interface_2 == 4


def test_default_24_ply_layup_keeps_11_and_12():
    """The default 24-ply layup must keep the legacy (11, 12) defaults so
    existing fixtures and saved results don't shift."""
    cfg = AnalysisConfig()
    assert len(cfg.angles) == 24
    assert cfg.interface_1 == 11
    assert cfg.interface_2 == 12


def test_explicit_interface_values_still_honoured():
    """User-supplied in-range values must override the midplane defaults."""
    cfg = AnalysisConfig(angles=[0, 90, 0, 90], interface_1=1, interface_2=2)
    assert cfg.interface_1 == 1
    assert cfg.interface_2 == 2


def test_explicit_out_of_range_interface_still_raises():
    """Explicit out-of-range integers must still fail validation."""
    with pytest.raises(ValueError, match=r"interface_1 must be in \[0, 2\)"):
        AnalysisConfig(angles=[0, 90], interface_1=11)
    with pytest.raises(ValueError, match=r"interface_2 must be in \[0, 2\)"):
        AnalysisConfig(angles=[0, 90], interface_2=12)


def test_explicit_negative_interface_still_raises():
    """Negative explicit indices must still fail validation."""
    with pytest.raises(ValueError, match=r"interface_1 must be in \[0, 4\)"):
        AnalysisConfig(angles=[0, 90, 0, 90], interface_1=-1)
