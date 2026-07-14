"""Save / load round-trips for :class:`AnalysisConfig` (issue #259).

Covers the canonical serialisation contract:

* ``from_dict(to_dict(cfg))`` reproduces ``cfg`` for defaults, a custom
  material, a CZM config, a multi-wrinkle config, and a penetration-gate
  preset.
* ``save_json`` / ``load_json`` (and the extension-dispatching
  ``save`` / ``load``) round-trip through a file.
* Unknown keys and a ``config_version`` mismatch are rejected loudly,
  naming the offender.
* The ``analyze --config`` CLI path reaches the same config as the
  equivalent explicit flags, and an explicit flag overrides the file.
* A completeness guard walks ``dataclasses.fields(AnalysisConfig)`` so a
  silently-dropped field fails the suite (the #345-style drift guard).
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from wrinklefe.analysis import (
    CONFIG_VERSION,
    AnalysisConfig,
    WrinkleSpec,
)
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.penetration_gate import (
    GATE_LI2024_MOULDED,
    GATE_LI2025_VACBAG,
)

# --------------------------------------------------------------------------- #
# Sample configs
# --------------------------------------------------------------------------- #

_SMALL_LAYUP = [0.0, 45.0, -45.0, 90.0]
_UD_LAYUP = [0.0, 0.0, 0.0, 0.0]


def _sample_configs() -> dict[str, AnalysisConfig]:
    return {
        "defaults": AnalysisConfig(),
        "custom_material": AnalysisConfig(
            material=OrthotropicMaterial(name="mine", E1=101_000.0, E2=8_000.0),
            angles=list(_SMALL_LAYUP),
        ),
        "czm": AnalysisConfig(
            enable_czm=True,
            czm_interfaces=[1, 2],
            czm_GIc=0.3,
            angles=[0.0, 90.0, 0.0, 90.0],
            analytical_only=False,
        ),
        "multi_wrinkle": AnalysisConfig(
            wrinkles=[
                WrinkleSpec(0.3, 16.0, 12.0, 1),
                WrinkleSpec(0.2, 10.0, 8.0, 2, phase_offset=0.5),
            ],
            angles=list(_SMALL_LAYUP),
            analytical_only=True,
        ),
        "penetration_gate": AnalysisConfig(
            penetration_gate=GATE_LI2024_MOULDED,
            angles=list(_UD_LAYUP),
        ),
    }


# --------------------------------------------------------------------------- #
# Round-trips
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(_sample_configs()))
def test_dict_round_trip(name):
    """``from_dict(to_dict(cfg))`` reproduces the config (dict + object eq)."""
    cfg = _sample_configs()[name]
    d = cfg.to_dict()
    assert d["config_version"] == CONFIG_VERSION
    restored = AnalysisConfig.from_dict(d)
    # Idempotent serialisation and full dataclass equality.
    assert restored.to_dict() == d
    assert restored == cfg


def test_material_preset_serialised_by_name():
    """A library material serialises as a preset reference, not inline."""
    cfg = AnalysisConfig()  # default material is the IM7_8552 preset
    assert cfg.to_dict()["material"] == {"preset": "IM7_8552"}


def test_custom_material_serialised_inline():
    """A material that differs from the like-named preset is inlined."""
    cfg = AnalysisConfig(
        material=OrthotropicMaterial(name="IM7_8552", E1=1.0e5),
        angles=list(_SMALL_LAYUP),
    )
    mat = cfg.to_dict()["material"]
    assert set(mat) == {"custom"}
    assert mat["custom"]["E1"] == pytest.approx(1.0e5)


def test_penetration_gate_preset_reference():
    cfg = AnalysisConfig(
        penetration_gate=GATE_LI2025_VACBAG, angles=list(_UD_LAYUP)
    )
    assert cfg.to_dict()["penetration_gate"] == {
        "preset": "AC318_S6C10_vacbag"
    }


def test_unregistered_gate_raises_on_to_dict():
    from wrinklefe.core.penetration_gate import GateParameters

    cfg = AnalysisConfig(
        penetration_gate=GateParameters(0.2, 0.1, 3.0, name="homebrew"),
        angles=list(_UD_LAYUP),
    )
    with pytest.raises(ValueError, match="homebrew"):
        cfg.to_dict()


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #


def test_save_load_json_file(tmp_path):
    cfg = _sample_configs()["multi_wrinkle"]
    path = tmp_path / "case.json"
    cfg.save_json(path)
    assert path.is_file()
    assert AnalysisConfig.load_json(path).to_dict() == cfg.to_dict()


def test_save_load_dispatch_by_extension(tmp_path):
    """``save``/``load`` pick JSON for a .json path."""
    cfg = AnalysisConfig(amplitude=0.42)
    path = tmp_path / "cfg.json"
    cfg.save(path)
    assert AnalysisConfig.load(path).amplitude == pytest.approx(0.42)


def test_save_load_yaml_file(tmp_path):
    """YAML round-trips when PyYAML is installed; skipped otherwise."""
    pytest.importorskip("yaml")
    cfg = _sample_configs()["custom_material"]
    path = tmp_path / "case.yaml"
    cfg.save(path)
    assert AnalysisConfig.load(path).to_dict() == cfg.to_dict()


# --------------------------------------------------------------------------- #
# Loud failures
# --------------------------------------------------------------------------- #


def test_unknown_key_rejected_by_name():
    d = AnalysisConfig().to_dict()
    d["totally_bogus"] = 123
    with pytest.raises(ValueError, match="totally_bogus"):
        AnalysisConfig.from_dict(d)


def test_version_mismatch_rejected():
    d = AnalysisConfig().to_dict()
    d["config_version"] = CONFIG_VERSION + 1
    with pytest.raises(ValueError, match="config_version"):
        AnalysisConfig.from_dict(d)


def test_missing_version_rejected():
    d = AnalysisConfig().to_dict()
    del d["config_version"]
    with pytest.raises(ValueError, match="config_version"):
        AnalysisConfig.from_dict(d)


def test_unknown_material_preset_rejected():
    d = AnalysisConfig().to_dict()
    d["material"] = {"preset": "no_such_material"}
    with pytest.raises(ValueError, match="no_such_material"):
        AnalysisConfig.from_dict(d)


# --------------------------------------------------------------------------- #
# Completeness guard (#345-style drift guard)
# --------------------------------------------------------------------------- #

#: Fields intentionally not exported by ``to_dict``. Empty today: every
#: field must round-trip. A new field silently dropped from ``to_dict``
#: is the failure mode this guard exists to catch.
_TO_DICT_ALLOWLIST: frozenset[str] = frozenset()


def test_to_dict_exports_every_field():
    """Every ``AnalysisConfig`` field is exported (or explicitly allowlisted)."""
    exported = set(AnalysisConfig().to_dict())
    for f in dataclasses.fields(AnalysisConfig):
        assert f.name in exported or f.name in _TO_DICT_ALLOWLIST, (
            f"AnalysisConfig field {f.name!r} is neither exported by "
            f"to_dict() nor in the allowlist — it would be silently "
            f"dropped on save."
        )


def test_from_dict_accepts_all_exported_keys():
    """The full exported dict loads without an unknown-key rejection."""
    for cfg in _sample_configs().values():
        AnalysisConfig.from_dict(cfg.to_dict())


# --------------------------------------------------------------------------- #
# CLI --config parity and override
# --------------------------------------------------------------------------- #


def _capture_cli_config():
    """Stub ``WrinkleAnalysis.run`` and capture the config it received."""
    captured: dict = {}

    def fake_run(self, analytical_only=None):
        captured["config"] = self.config
        captured["analytical_only"] = analytical_only
        result = MagicMock()
        result.summary.return_value = "<stub>"
        result.czm_damage = None
        return result

    return captured, patch(
        "wrinklefe.analysis.WrinkleAnalysis.run", new=fake_run
    )


def test_cli_config_parity_with_flags(tmp_path):
    """``analyze --config file`` reaches the same config as equivalent flags."""
    from wrinklefe.cli import main as cli_main

    cfg = AnalysisConfig(
        amplitude=0.4,
        wavelength=12.0,
        morphology="concave",
        angles=list(_SMALL_LAYUP),
        analytical_only=True,
    )
    path = tmp_path / "case.json"
    cfg.save_json(path)

    cap_file, patch_file = _capture_cli_config()
    with patch_file:
        cli_main(["analyze", "--config", str(path)])
    from_file = cap_file["config"]

    cap_flags, patch_flags = _capture_cli_config()
    with patch_flags:
        cli_main([
            "analyze",
            "--amplitude", "0.4",
            "--wavelength", "12.0",
            "--morphology", "concave",
            "--angles", "0,45,-45,90",
            "--analytical-only",
        ])
    from_flags = cap_flags["config"]

    assert from_file.to_dict() == from_flags.to_dict()
    # Analytical knockdowns therefore match too.
    from wrinklefe.analysis import WrinkleAnalysis
    kd_file = WrinkleAnalysis(from_file).run(
        analytical_only=True
    ).analytical_knockdown
    kd_flags = WrinkleAnalysis(from_flags).run(
        analytical_only=True
    ).analytical_knockdown
    assert kd_file == pytest.approx(kd_flags)


def test_cli_explicit_flag_overrides_config(tmp_path):
    """An explicit flag wins over the --config file value."""
    from wrinklefe.cli import main as cli_main

    cfg = AnalysisConfig(
        amplitude=0.4, angles=list(_SMALL_LAYUP), analytical_only=True
    )
    path = tmp_path / "case.json"
    cfg.save_json(path)

    captured, patcher = _capture_cli_config()
    with patcher:
        cli_main([
            "analyze", "--config", str(path), "--amplitude", "0.9",
        ])
    assert captured["config"].amplitude == pytest.approx(0.9)
    # Un-overridden file values are preserved.
    assert captured["config"].angles == _SMALL_LAYUP
    assert captured["config"].analytical_only is True


def test_cli_save_config_writes_effective(tmp_path):
    """``--save-config`` writes the post-override effective config."""
    from wrinklefe.cli import main as cli_main

    out = tmp_path / "effective.json"
    captured, patcher = _capture_cli_config()
    with patcher:
        cli_main([
            "analyze", "--analytical-only",
            "--amplitude", "0.55", "--wavelength", "9.0",
            "--save-config", str(out),
        ])
    assert out.is_file()
    loaded = AnalysisConfig.load_json(out)
    assert loaded.amplitude == pytest.approx(0.55)
    assert loaded.wavelength == pytest.approx(9.0)
    assert loaded.analytical_only is True
