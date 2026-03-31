"""Material and laminate configuration panel."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QGroupBox, QFormLayout, QComboBox, QLineEdit, QDoubleSpinBox,
)


class MaterialPanel(QGroupBox):
    """Panel for material selection, layup, and ply thickness.

    Exposes
    -------
    get_material() -> str
        Currently selected material name.
    get_layup_text() -> str
        Raw layup string entered by the user.
    get_ply_thickness() -> float
        Ply thickness in mm.
    """

    def __init__(self, parent=None) -> None:
        super().__init__("Material && Laminate", parent)
        layout = QFormLayout(self)

        self.material_combo = QComboBox()
        self.material_combo.addItems([
            "IM7_8552", "AS4_3501_6",
            "T300_914", "T700_2510",
            "AC318_S6C10",
        ])
        layout.addRow("Material:", self.material_combo)

        self.layup_edit = QLineEdit("[0/45/-45/90]_3s")
        self.layup_edit.setToolTip(
            "Enter ply angles separated by /. Use _Ns for N repeats, "
            "s for symmetric."
        )
        layout.addRow("Layup:", self.layup_edit)

        self.ply_thickness_spin = QDoubleSpinBox()
        self.ply_thickness_spin.setRange(0.05, 0.50)
        self.ply_thickness_spin.setValue(0.183)
        self.ply_thickness_spin.setSingleStep(0.01)
        self.ply_thickness_spin.setSuffix(" mm")
        layout.addRow("Ply thickness:", self.ply_thickness_spin)

    # -- public API --

    def get_material(self) -> str:
        return self.material_combo.currentText()

    def get_layup_text(self) -> str:
        return self.layup_edit.text()

    def get_ply_thickness(self) -> float:
        return self.ply_thickness_spin.value()

    def reset_defaults(self) -> None:
        """Restore factory defaults."""
        self.material_combo.setCurrentIndex(0)
        self.layup_edit.setText("[0/45/-45/90]_3s")
        self.ply_thickness_spin.setValue(0.183)
