"""Parametric sweep panel with config controls, progress, plots, and results."""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QProgressBar,
    QTextEdit, QFileDialog, QMessageBox, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL_QT = True
except ImportError:
    HAS_MPL_QT = False


class SweepPanel(QWidget):
    """Full sweep tab: config row, progress, two matplotlib canvases, results text, export.

    Signals
    -------
    sweep_requested(str, float, float, int)
        Emitted when the user clicks *Run Sweep*.
        Args: parameter name, min value, max value, number of points.
    export_requested()
        Emitted when the user clicks *Export...*.
    """

    sweep_requested = pyqtSignal(str, float, float, int)
    export_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._sweep_results_cache: Optional[dict] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        if not HAS_MPL_QT:
            layout.addWidget(QLabel("Matplotlib Qt backend not available."))
            return

        # --- Config row ---
        sweep_row1 = QHBoxLayout()

        sweep_row1.addWidget(QLabel("Sweep:"))
        self.sweep_param_combo = QComboBox()
        self.sweep_param_combo.addItems(["amplitude", "wavelength", "width"])
        self.sweep_param_combo.setToolTip(
            "Parameter to vary across the sweep range"
        )
        sweep_row1.addWidget(self.sweep_param_combo)

        sweep_row1.addWidget(QLabel("Min:"))
        self.sweep_min_spin = QDoubleSpinBox()
        self.sweep_min_spin.setRange(0.001, 100.0)
        self.sweep_min_spin.setDecimals(3)
        self.sweep_min_spin.setValue(0.183)
        sweep_row1.addWidget(self.sweep_min_spin)

        sweep_row1.addWidget(QLabel("Max:"))
        self.sweep_max_spin = QDoubleSpinBox()
        self.sweep_max_spin.setRange(0.001, 100.0)
        self.sweep_max_spin.setDecimals(3)
        self.sweep_max_spin.setValue(0.549)
        sweep_row1.addWidget(self.sweep_max_spin)

        sweep_row1.addWidget(QLabel("Points:"))
        self.sweep_points_spin = QSpinBox()
        self.sweep_points_spin.setRange(2, 20)
        self.sweep_points_spin.setValue(5)
        sweep_row1.addWidget(self.sweep_points_spin)

        self.sweep_run_btn = QPushButton("Run Sweep")
        self.sweep_run_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 6px 16px; }"
        )
        self.sweep_run_btn.clicked.connect(self._on_run_sweep)
        sweep_row1.addWidget(self.sweep_run_btn)

        self.sweep_export_btn = QPushButton("Export...")
        self.sweep_export_btn.setEnabled(False)
        self.sweep_export_btn.clicked.connect(self._on_sweep_export)
        sweep_row1.addWidget(self.sweep_export_btn)

        sweep_row1.addStretch()
        layout.addLayout(sweep_row1)

        # --- Progress bar + time label ---
        sweep_status_layout = QHBoxLayout()
        self.sweep_progress_bar = QProgressBar()
        self.sweep_progress_bar.setRange(0, 100)
        self.sweep_progress_bar.setVisible(False)
        self.sweep_progress_bar.setMaximumHeight(18)
        sweep_status_layout.addWidget(self.sweep_progress_bar, stretch=1)

        self.sweep_time_label = QLabel("")
        self.sweep_time_label.setStyleSheet(
            "QLabel { color: #555; font-size: 11px; }"
        )
        sweep_status_layout.addWidget(self.sweep_time_label)
        layout.addLayout(sweep_status_layout)

        # Update defaults when parameter changes
        self.sweep_param_combo.currentTextChanged.connect(
            self._on_sweep_param_changed
        )

        # --- Two side-by-side canvases ---
        sweep_plots_layout = QHBoxLayout()
        self.sweep_strength_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.sweep_stiffness_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        for c in (self.sweep_strength_canvas, self.sweep_stiffness_canvas):
            c.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
        sweep_plots_layout.addWidget(self.sweep_strength_canvas)
        sweep_plots_layout.addWidget(self.sweep_stiffness_canvas)
        layout.addLayout(sweep_plots_layout, stretch=1)

        # --- Results text ---
        self.sweep_results_text = QTextEdit()
        self.sweep_results_text.setReadOnly(True)
        self.sweep_results_text.setMaximumHeight(120)
        self.sweep_results_text.setStyleSheet(
            "QTextEdit { font-family: monospace; font-size: 11px; }"
        )
        self.sweep_results_text.setPlaceholderText(
            "Sweep results table will appear here after running."
        )
        layout.addWidget(self.sweep_results_text)

        # --- Placeholder text on canvases ---
        for canvas, msg in [
            (self.sweep_strength_canvas,
             "Run a sweep to see strength retention vs parameter."),
            (self.sweep_stiffness_canvas,
             "Run a sweep to see stiffness retention vs parameter."),
        ]:
            ax = canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="0.5")
            ax.set_axis_off()
            canvas.draw()

    # ----------------------------------------------------------
    # Handlers
    # ----------------------------------------------------------

    def _on_sweep_param_changed(self, param: str) -> None:
        """Update min/max defaults when sweep parameter changes."""
        defaults = {
            'amplitude': (0.183, 0.549),
            'wavelength': (8.0, 24.0),
            'width': (5.0, 20.0),
        }
        lo, hi = defaults.get(param, (0.0, 1.0))
        self.sweep_min_spin.setValue(lo)
        self.sweep_max_spin.setValue(hi)

    def _on_run_sweep(self) -> None:
        """Validate and emit sweep_requested signal."""
        param = self.sweep_param_combo.currentText()
        vmin = self.sweep_min_spin.value()
        vmax = self.sweep_max_spin.value()
        npts = self.sweep_points_spin.value()

        if vmin >= vmax:
            QMessageBox.critical(
                self, "Invalid Range",
                f"Min ({vmin}) must be less than Max ({vmax}).",
            )
            return

        self.sweep_requested.emit(param, vmin, vmax, npts)

    def _on_sweep_export(self) -> None:
        """Export sweep results to JSON."""
        if self._sweep_results_cache is None:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Sweep Results", "sweep_results.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not filepath:
            return

        cache = self._sweep_results_cache
        export = {
            'parameter': cache['parameter'],
            'loading': cache['loading'],
            'morphology': cache['morphology'],
            'gamma_Y_eff': cache['gamma_Y_eff'],
            'data': [],
        }
        for i, val in enumerate(cache['values']):
            r = cache['results'][i]
            export['data'].append({
                cache['parameter']: val,
                'strength_retention': cache['strength_retention'][i],
                'stiffness_retention': cache['stiffness_retention'][i],
                'strength_MPa': float(r.analytical_strength_MPa),
                'theta_eff_deg': float(np.degrees(r.effective_angle_rad)),
                'morphology_factor': float(r.morphology_factor),
            })
        with open(filepath, 'w') as f:
            json.dump(export, f, indent=2)

        # Return filepath so main window can update status bar
        self._last_export_path = filepath
        self.export_requested.emit()

    # ----------------------------------------------------------
    # Public methods called by main window
    # ----------------------------------------------------------

    def set_sweep_running(self, npts: int) -> None:
        """Configure UI for an in-progress sweep."""
        self.sweep_run_btn.setEnabled(False)
        self.sweep_export_btn.setEnabled(False)
        self.sweep_progress_bar.setRange(0, npts)
        self.sweep_progress_bar.setValue(0)
        self.sweep_progress_bar.setVisible(True)
        self.sweep_time_label.setText("Starting sweep...")
        self.sweep_results_text.clear()

    def on_sweep_progress(self, current: int, total: int, elapsed: float) -> None:
        """Update progress bar and time estimate."""
        self.sweep_progress_bar.setValue(current)
        if current > 0:
            rate = elapsed / current
            remaining = rate * (total - current)
            self.sweep_time_label.setText(
                f"{current}/{total}  |  "
                f"Elapsed: {self._format_time(elapsed)}  |  "
                f"Remaining: ~{self._format_time(remaining)}"
            )

    def on_sweep_done(self, results_list, param: str, values: list) -> None:
        """Handle completed sweep: plot results and show table."""
        self.sweep_run_btn.setEnabled(True)
        self.sweep_export_btn.setEnabled(True)
        self.sweep_progress_bar.setVisible(False)

        loading = results_list[0].config.loading if results_list else "compression"
        morphology = results_list[0].config.morphology if results_list else "stack"

        strength_kd = [r.analytical_knockdown for r in results_list]
        stiffness_kd = [
            r.modulus_retention if r.modulus_retention is not None else 1.0
            for r in results_list
        ]

        # Cache for export
        self._sweep_results_cache = {
            'parameter': param,
            'values': values,
            'loading': loading,
            'morphology': morphology,
            'strength_retention': strength_kd,
            'stiffness_retention': stiffness_kd,
            'gamma_Y_eff': results_list[0].gamma_Y_eff if results_list else None,
            'results': results_list,
        }

        self.sweep_time_label.setText(
            f"Done \u2014 {len(values)} points  |  "
            f"\u03b3_Y = {results_list[0].gamma_Y_eff:.4f}  |  "
            f"{loading}, {morphology}"
        )

        labels_map = {
            'amplitude': 'Amplitude A (mm)',
            'wavelength': 'Wavelength \u03bb (mm)',
            'width': 'Envelope Width w (mm)',
        }
        xlabel = labels_map.get(param, param)

        # Strength retention plot
        fig_s = self.sweep_strength_canvas.figure
        fig_s.clear()
        ax_s = fig_s.add_subplot(111)
        ax_s.plot(values, strength_kd, '-o', color='#2196F3',
                  linewidth=2, markersize=7)
        ax_s.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax_s.set_ylabel('Strength Retention (\u03c3/\u03c3\u2080)', fontsize=11, fontweight='bold')
        ax_s.set_title(
            f'Strength Retention vs {param.capitalize()}\n'
            f'({morphology}, {loading})',
            fontsize=12, fontweight='bold',
        )
        ax_s.set_ylim(bottom=0, top=max(1.05, max(strength_kd) * 1.1))
        ax_s.grid(True, alpha=0.3)
        fig_s.tight_layout()
        self.sweep_strength_canvas.draw()

        # Stiffness retention plot
        fig_e = self.sweep_stiffness_canvas.figure
        fig_e.clear()
        ax_e = fig_e.add_subplot(111)
        ax_e.plot(values, stiffness_kd, '-s', color='#F44336',
                  linewidth=2, markersize=7)
        ax_e.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax_e.set_ylabel('Stiffness Retention (E/E\u2080)', fontsize=11, fontweight='bold')
        ax_e.set_title(
            f'Stiffness Retention vs {param.capitalize()}\n'
            f'({morphology}, {loading})',
            fontsize=12, fontweight='bold',
        )
        ax_e.set_ylim(bottom=0, top=max(1.05, max(stiffness_kd) * 1.1))
        ax_e.grid(True, alpha=0.3)
        fig_e.tight_layout()
        self.sweep_stiffness_canvas.draw()

        # Numerical results table
        theta_hdr = "\u03b8_eff (\u00b0)"
        sigma_hdr = "\u03c3 (MPa)"
        lines = [
            f"{'':>10}  {'Strength':>10}  {'Stiffness':>10}  "
            f"{theta_hdr:>10}  {'M_f':>8}  {sigma_hdr:>10}",
            "-" * 68,
        ]
        for i, val in enumerate(values):
            r = results_list[i]
            theta_deg = np.degrees(r.effective_angle_rad)
            lines.append(
                f"{val:10.3f}  {strength_kd[i]:10.4f}  {stiffness_kd[i]:10.4f}  "
                f"{theta_deg:10.2f}  {r.morphology_factor:8.3f}  "
                f"{r.analytical_strength_MPa:10.1f}"
            )
        self.sweep_results_text.setPlainText("\n".join(lines))

    def on_sweep_error(self, msg: str) -> None:
        """Handle sweep error."""
        self.sweep_run_btn.setEnabled(True)
        self.sweep_progress_bar.setVisible(False)
        self.sweep_time_label.setText("Error")
        QMessageBox.critical(self, "Sweep Error", msg)

    def get_last_export_path(self) -> Optional[str]:
        """Return the last export filepath, or None."""
        return getattr(self, '_last_export_path', None)

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            m, s = divmod(seconds, 60)
            return f"{m}:{s:02d}"
        else:
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{h}:{m:02d}:{s:02d}"
