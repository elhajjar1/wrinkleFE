"""Mesh configuration panel."""

from __future__ import annotations

from PyQt6.QtWidgets import QGroupBox, QFormLayout, QSpinBox


class MeshPanel(QGroupBox):
    """Panel for mesh resolution (nx, ny, nz_per_ply).

    Exposes
    -------
    get_mesh_config() -> dict
        ``{"nx": int, "ny": int, "nz_per_ply": int}``
    """

    def __init__(self, parent=None) -> None:
        super().__init__("Mesh", parent)
        layout = QFormLayout(self)

        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(2, 200)
        self.nx_spin.setValue(20)
        layout.addRow("nx:", self.nx_spin)

        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(2, 100)
        self.ny_spin.setValue(6)
        layout.addRow("ny:", self.ny_spin)

        self.nz_per_ply_spin = QSpinBox()
        self.nz_per_ply_spin.setRange(1, 10)
        self.nz_per_ply_spin.setValue(3)
        layout.addRow("nz/ply:", self.nz_per_ply_spin)

    # -- public API --

    def get_mesh_config(self) -> dict:
        return {
            "nx": self.nx_spin.value(),
            "ny": self.ny_spin.value(),
            "nz_per_ply": self.nz_per_ply_spin.value(),
        }

    def reset_defaults(self) -> None:
        """Restore factory defaults."""
        self.nx_spin.setValue(20)
        self.ny_spin.setValue(6)
        self.nz_per_ply_spin.setValue(3)
