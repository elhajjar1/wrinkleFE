"""FE solve for multi-wrinkle configurations (issue #252, Stage 1).

Stage 1 supports N wrinkles whose longitudinal supports do not overlap
(the Li 2025 specimen layout): the composed displacement / fiber-angle
fields come from ``WrinkleConfiguration.apply_to_nodes`` /
``fiber_angles_at_nodes``, which already superpose N placements.
Overlapping wrinkles and the CZM pathway remain rejected with precise
messages.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis, WrinkleSpec

_ANGLES_8 = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]


def _two_wrinkle_config(**overrides):
    """Two identical wrinkles centred at x = L/2 ± 8 mm (disjoint supports).

    Each spec: lambda = 8, width = 2 -> support = center ± 6 mm.
    phase_offset = ±2*pi shifts the centre by ±lambda = ±8 mm.
    """
    spec = dict(amplitude=0.15, wavelength=8.0, width=2.0, ply_interface=3)
    defaults = dict(
        amplitude=0.15, wavelength=8.0, width=2.0,
        morphology="graded", loading="compression",
        angles=list(_ANGLES_8),
        wrinkles=[
            WrinkleSpec(**spec, phase_offset=-2.0 * np.pi),
            WrinkleSpec(**spec, phase_offset=+2.0 * np.pi),
        ],
        nx=32, ny=2, nz_per_ply=1,
    )
    defaults.update(overrides)
    return AnalysisConfig(**defaults)


def _element_x_centroids(mesh) -> np.ndarray:
    return mesh.nodes[mesh.elements].mean(axis=1)[:, 0]


class TestMultiWrinkleFE:

    def test_two_wrinkles_fe_runs_with_local_fi_peaks(self):
        """FE solve runs for 2 non-overlapping wrinkles and produces a
        local max-FI peak near each wrinkle centre."""
        cfg = _two_wrinkle_config()
        result = WrinkleAnalysis(cfg).run()

        assert result.field_results is not None
        assert result.failure_indices, "failure evaluation must run"

        # Combined per-element max FI across criteria and Gauss points.
        fi = np.max(
            np.stack([
                np.asarray(arr).max(axis=1)
                for arr in result.failure_indices.values()
            ]),
            axis=0,
        )
        x_c = _element_x_centroids(result.mesh)
        L = cfg.domain_length
        centers = (L / 2.0 - 8.0, L / 2.0 + 8.0)

        # The global FI maximum must sit inside one of the two wrinkle
        # supports (the wrinkles, not the far field, govern).
        x_peak = x_c[int(np.argmax(fi))]
        assert any(abs(x_peak - c) < 6.0 for c in centers), (
            f"global FI peak at x={x_peak:.1f} mm is outside both "
            f"wrinkle supports {centers}"
        )

        # Each wrinkle produces a local elevation above the far field.
        # (LaRC05 carries a uniform baseline FI under compression, so
        # the peaks are elevations on that baseline, not zero-to-one.)
        far = np.ones_like(x_c, dtype=bool)
        for center in centers:
            far &= np.abs(x_c - center) > 6.0
        assert far.any()
        peaks = []
        for center in centers:
            near = np.abs(x_c - center) < 4.0      # within lambda/2
            assert near.any()
            assert fi[near].max() > 1.1 * fi[far].max(), (
                f"expected a local FI peak near x={center:.1f} mm"
            )
            peaks.append(fi[near].max())
        # Identical wrinkles -> the two local peaks agree closely.
        assert peaks[0] == pytest.approx(peaks[1], rel=0.05)

    def test_domain_sized_from_union_of_extents(self):
        cfg = _two_wrinkle_config()
        # half-span = |dx| + 3*width = 8 + 6 = 14 -> L = 28 (> 3*lambda).
        assert cfg.domain_length == pytest.approx(28.0)

    def test_single_spec_matches_scalar_graded_path(self):
        """A one-entry wrinkles list is tolerance-equal to the scalar
        graded config it denotes (same placement, same mesh density,
        same domain)."""
        common = dict(
            amplitude=0.15, wavelength=16.0, width=8.0,
            morphology="graded", loading="compression",
            angles=list(_ANGLES_8),
            nx=16, ny=2, nz_per_ply=1,
            domain_length=48.0,
        )
        scalar_cfg = AnalysisConfig(**common, interface_1=3)
        multi_cfg = AnalysisConfig(
            **common,
            wrinkles=[
                WrinkleSpec(
                    amplitude=0.15, wavelength=16.0, width=8.0,
                    ply_interface=3, phase_offset=0.0,
                )
            ],
        )
        r_scalar = WrinkleAnalysis(scalar_cfg).run()
        r_multi = WrinkleAnalysis(multi_cfg).run()

        assert r_multi.modulus_retention == pytest.approx(
            r_scalar.modulus_retention, rel=1e-9
        )
        for crit, arr in r_scalar.failure_indices.items():
            np.testing.assert_allclose(
                r_multi.failure_indices[crit], arr, rtol=1e-9,
                err_msg=f"{crit} FI field diverged between the scalar "
                "graded path and the one-entry wrinkles path",
            )

    def test_overlapping_wrinkles_rejected_with_precise_message(self):
        cfg = _two_wrinkle_config()
        # Pull the second wrinkle onto the first: supports overlap.
        cfg.wrinkles[1].phase_offset = 0.5 * np.pi  # dx = +2 mm
        with pytest.raises(NotImplementedError, match="overlap longitudinally"):
            WrinkleAnalysis(cfg).run()

    def test_overlapping_wrinkles_still_run_analytically(self):
        cfg = _two_wrinkle_config()
        cfg.wrinkles[1].phase_offset = 0.5 * np.pi
        r = WrinkleAnalysis(cfg).run(analytical_only=True)
        assert 0.0 < r.analytical_knockdown <= 1.0

    def test_multi_wrinkle_czm_rejected_with_precise_message(self):
        cfg = _two_wrinkle_config(enable_czm=True)
        with pytest.raises(NotImplementedError, match="cohesive"):
            WrinkleAnalysis(cfg).run()
