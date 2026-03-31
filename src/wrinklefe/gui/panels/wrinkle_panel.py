"""Wrinkle geometry configuration panel."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QLabel,
)


class WrinklePanel(QGroupBox):
    """Panel for wrinkle amplitude, wavelength, width, morphology, and loading.

    Exposes
    -------
    get_config() -> dict
        All wrinkle parameters as a flat dictionary.
    """

    def __init__(self, parent=None) -> None:
        super().__init__("Wrinkle Geometry", parent)
        layout = QFormLayout(self)

        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0.001, 3.0)
        self.amplitude_spin.setDecimals(3)
        self.amplitude_spin.setValue(0.366)
        self.amplitude_spin.setSingleStep(0.001)
        self.amplitude_spin.setSuffix(" mm")
        layout.addRow("Amplitude:", self.amplitude_spin)

        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(1.0, 100.0)
        self.wavelength_spin.setDecimals(3)
        self.wavelength_spin.setValue(16.0)
        self.wavelength_spin.setSingleStep(0.001)
        self.wavelength_spin.setSuffix(" mm")
        self.wavelength_spin.setToolTip(
            "Wavelength \u03bb: oscillation period of the sinusoidal wrinkle.\n"
            "Controls wrinkle steepness \u2014 shorter \u03bb = steeper angle.\n"
            "\u03b8 \u2248 arctan(2\u03c0A/\u03bb)"
        )
        layout.addRow("Wavelength (\u03bb):", self.wavelength_spin)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1.0, 100.0)
        self.width_spin.setDecimals(3)
        self.width_spin.setValue(12.0)
        self.width_spin.setSingleStep(0.001)
        self.width_spin.setSuffix(" mm")
        self.width_spin.setToolTip(
            "Gaussian envelope width w: controls how far the wrinkle\n"
            "extends before decaying to zero.\n"
            "Small w = localized defect (1\u20132 undulations visible).\n"
            "Large w = extended waviness (many undulations).\n"
            "w \u2248 0.75\u03bb for typical manufacturing wrinkles."
        )
        layout.addRow("Envelope width (w):", self.width_spin)

        self.morphology_combo = QComboBox()
        self.morphology_combo.addItems([
            "stack", "convex", "concave", "uniform", "graded",
        ])
        self.morphology_combo.currentTextChanged.connect(
            self._on_morphology_changed
        )
        layout.addRow("Morphology:", self.morphology_combo)

        self.decay_floor_label = QLabel("Decay floor:")
        self.decay_floor_spin = QDoubleSpinBox()
        self.decay_floor_spin.setRange(0.0, 1.0)
        self.decay_floor_spin.setSingleStep(0.05)
        self.decay_floor_spin.setValue(0.0)
        self.decay_floor_spin.setDecimals(2)
        self.decay_floor_spin.setToolTip(
            "Minimum amplitude fraction for graded mode.\n"
            "graded: decay floor at surface plies (max at midplane)\n"
            "0.0 = full decay to zero, 1.0 = no decay (uniform)"
        )
        layout.addRow(self.decay_floor_label, self.decay_floor_spin)
        # Initially hidden -- only shown for graded morphology
        self.decay_floor_label.setVisible(False)
        self.decay_floor_spin.setVisible(False)

        self.loading_combo = QComboBox()
        self.loading_combo.addItems(["compression", "tension"])
        layout.addRow("Loading:", self.loading_combo)

    # -- internal slots --

    def _on_morphology_changed(self, name: str) -> None:
        """Show/hide decay floor spinner based on morphology selection."""
        is_graded = name.lower().strip() == "graded"
        self.decay_floor_label.setVisible(is_graded)
        self.decay_floor_spin.setVisible(is_graded)

    # -- public API --

    def get_config(self) -> dict:
        """Return all wrinkle parameters as a dictionary."""
        return {
            "amplitude": self.amplitude_spin.value(),
            "wavelength": self.wavelength_spin.value(),
            "width": self.width_spin.value(),
            "morphology": self.morphology_combo.currentText(),
            "decay_floor": self.decay_floor_spin.value(),
            "loading": self.loading_combo.currentText(),
        }

    def reset_defaults(self) -> None:
        """Restore factory defaults."""
        self.amplitude_spin.setValue(0.366)
        self.wavelength_spin.setValue(16.0)
        self.width_spin.setValue(12.0)
        self.morphology_combo.setCurrentIndex(0)
        self.decay_floor_spin.setValue(0.0)
        self.loading_combo.setCurrentIndex(0)
