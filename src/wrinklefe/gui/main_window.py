"""Main application window for WrinkleFE GUI.

Provides the top-level window with:
- Material and laminate configuration panel (left sidebar)
- Wrinkle geometry configuration panel (left sidebar)
- Analysis controls and progress (toolbar)
- Results visualization area (central)
- Summary and failure report (bottom)

The GUI requires PyQt6. Install with: pip install PyQt6

Usage
-----
>>> from wrinklefe.gui.main_window import launch
>>> launch()  # Opens the GUI application
"""

from __future__ import annotations

import sys
import traceback
from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QGroupBox, QLabel, QLineEdit, QComboBox, QPushButton,
        QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit, QSplitter,
        QStatusBar, QMenuBar, QMenu, QMessageBox, QFileDialog,
        QProgressBar, QFormLayout, QScrollArea, QSizePolicy,
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QElapsedTimer
    from PyQt6.QtGui import QAction
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL_QT = True
except ImportError:
    HAS_MPL_QT = False

import numpy as np


def _check_pyqt6() -> None:
    """Raise ImportError with helpful message if PyQt6 is missing."""
    if not HAS_PYQT6:
        raise ImportError(
            "WrinkleFE GUI requires PyQt6. Install with:\n"
            "  pip install PyQt6\n"
            "Or use the CLI interface: python -m wrinklefe.cli analyze ..."
        )


# ======================================================================
# Analysis worker thread
# ======================================================================

if HAS_PYQT6:

    class AnalysisWorker(QThread):
        """Background thread for running FE analysis."""

        finished = pyqtSignal(object)  # AnalysisResults
        error = pyqtSignal(str)
        progress = pyqtSignal(str)

        def __init__(self, config) -> None:
            super().__init__()
            self.config = config

        def run(self) -> None:
            try:
                from wrinklefe.analysis import WrinkleAnalysis
                self.progress.emit("Building laminate and wrinkle configuration...")
                analysis = WrinkleAnalysis(self.config)

                self.progress.emit("Running analysis...")
                result = analysis.run()

                self.progress.emit("Analysis complete.")
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ======================================================================
# Main Window
# ======================================================================

if HAS_PYQT6:

    class WrinkleFEMainWindow(QMainWindow):
        """Main application window for WrinkleFE.

        Layout
        ------
        +----------------------------------+---------------------------+
        |  Left Sidebar                    |  Central Area             |
        |  - Material Config               |  - Plot tabs              |
        |  - Laminate Config               |    (Profile, Mesh,        |
        |  - Wrinkle Config                |     Stress)               |
        |  - Analysis Controls             |                           |
        |                                  +---------------------------+
        |                                  |  Bottom: Text Output      |
        +----------------------------------+---------------------------+
        """

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("WrinkleFE - Wrinkled Composite Laminate Analysis")
            self.setMinimumSize(1200, 800)

            self._result = None
            self._worker = None

            self._setup_menu_bar()
            self._setup_central_widget()
            self._setup_status_bar()

        # ----------------------------------------------------------
        # Menu bar
        # ----------------------------------------------------------

        def _setup_menu_bar(self) -> None:
            menubar = self.menuBar()

            # File menu
            file_menu = menubar.addMenu("&File")

            new_action = QAction("&New Analysis", self)
            new_action.triggered.connect(self._on_new)
            file_menu.addAction(new_action)

            export_action = QAction("&Export Results...", self)
            export_action.triggered.connect(self._on_export)
            file_menu.addAction(export_action)

            file_menu.addSeparator()

            quit_action = QAction("&Quit", self)
            quit_action.triggered.connect(self.close)
            file_menu.addAction(quit_action)

            # Help menu
            help_menu = menubar.addMenu("&Help")
            about_action = QAction("&About", self)
            about_action.triggered.connect(self._on_about)
            help_menu.addAction(about_action)

        # ----------------------------------------------------------
        # Central widget
        # ----------------------------------------------------------

        def _setup_central_widget(self) -> None:
            central = QWidget()
            self.setCentralWidget(central)

            main_layout = QHBoxLayout(central)

            # Left sidebar: configuration panels
            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)
            left_panel.setMaximumWidth(350)

            left_layout.addWidget(self._create_material_group())
            left_layout.addWidget(self._create_wrinkle_group())
            left_layout.addWidget(self._create_mesh_group())
            left_layout.addWidget(self._create_analysis_group())
            left_layout.addStretch()

            # Right area: splitter with plots on top, text on bottom
            right_splitter = QSplitter(Qt.Orientation.Vertical)

            # Plot tabs with matplotlib canvases
            self.plot_tabs = QTabWidget()

            if HAS_MPL_QT:
                self.profile_canvas = FigureCanvas(Figure(figsize=(8, 5)))
                self.mesh_canvas = FigureCanvas(Figure(figsize=(8, 5)))
                self.stress_canvas = FigureCanvas(Figure(figsize=(8, 5)))

                for canvas in (self.profile_canvas, self.mesh_canvas,
                               self.stress_canvas):
                    canvas.setSizePolicy(
                        QSizePolicy.Policy.Expanding,
                        QSizePolicy.Policy.Expanding,
                    )

                self.plot_tabs.addTab(self.profile_canvas, "Profile")
                self.plot_tabs.addTab(self.mesh_canvas, "Mesh")

                # Stress tab: dropdown + canvas in a container widget
                self.stress_tab_widget = QWidget()
                stress_tab_layout = QVBoxLayout(self.stress_tab_widget)
                stress_tab_layout.setContentsMargins(4, 4, 4, 4)

                self.stress_component_combo = QComboBox()
                self.stress_component_combo.addItems([
                    "\u03c3\u2081\u2081 (Fiber)",
                    "\u03c3\u2082\u2082 (Transverse)",
                    "\u03c3\u2083\u2083 (Through-thickness)",
                    "\u03c4\u2082\u2083 (Transverse shear)",
                    "\u03c4\u2081\u2083 (Interlaminar shear)",
                    "\u03c4\u2081\u2082 (In-plane shear)",
                    "Von Mises",
                    "Max Principal",
                    "Interlaminar \u03c4\u2081\u2083",
                ])
                self.stress_component_combo.currentIndexChanged.connect(
                    self._on_stress_component_changed
                )

                self.stress_coord_combo = QComboBox()
                self.stress_coord_combo.addItems(["Local (material)", "Global (structural)"])
                self.stress_coord_combo.setToolTip(
                    "Local = ply material axes (1-2-3). "
                    "Global = structural axes (x-y-z).\n"
                    "Out-of-plane stresses default to Global for physical accuracy."
                )
                self.stress_coord_combo.currentIndexChanged.connect(
                    self._on_coord_changed
                )

                self.show_mesh_lines_check = QCheckBox("Mesh lines")
                self.show_mesh_lines_check.setChecked(False)
                self.show_mesh_lines_check.toggled.connect(
                    self._on_stress_component_changed
                )

                stress_controls = QHBoxLayout()
                stress_controls.addWidget(self.stress_component_combo, stretch=1)
                stress_controls.addWidget(self.stress_coord_combo)
                stress_controls.addWidget(self.show_mesh_lines_check)
                stress_tab_layout.addLayout(stress_controls)
                stress_tab_layout.addWidget(self.stress_canvas)

                self.plot_tabs.addTab(self.stress_tab_widget, "Stress")

                # Draw placeholder text on each canvas
                for canvas, msg in [
                    (self.profile_canvas, "Run an analysis to see wrinkle profile plots."),
                    (self.mesh_canvas, "Run an analysis to see mesh visualization."),
                    (self.stress_canvas, "Run an analysis to see stress results."),
                ]:
                    ax = canvas.figure.add_subplot(111)
                    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                            ha="center", va="center", fontsize=12, color="0.5")
                    ax.set_axis_off()
                    canvas.draw()
            else:
                self.profile_canvas = None
                self.mesh_canvas = None
                self.stress_canvas = None
                self.plot_tabs.addTab(QLabel("Run an analysis to see wrinkle profile plots."), "Profile")
                self.plot_tabs.addTab(QLabel("Run an analysis to see mesh visualization."), "Mesh")
                self.plot_tabs.addTab(QLabel("Run an analysis to see stress results."), "Stress")

            right_splitter.addWidget(self.plot_tabs)

            # Text output
            self.output_text = QTextEdit()
            self.output_text.setReadOnly(True)
            self.output_text.setMinimumHeight(150)
            self.output_text.setPlaceholderText(
                "Analysis results will appear here.\n"
                "Configure parameters on the left and click 'Run Analysis'."
            )
            right_splitter.addWidget(self.output_text)
            right_splitter.setSizes([500, 200])

            main_layout.addWidget(left_panel)
            main_layout.addWidget(right_splitter, stretch=1)

        # ----------------------------------------------------------
        # Configuration groups
        # ----------------------------------------------------------

        def _create_material_group(self) -> QGroupBox:
            group = QGroupBox("Material && Laminate")
            layout = QFormLayout(group)

            self.material_combo = QComboBox()
            self.material_combo.addItems([
                "IM7_8552", "AS4_3501_6",
                "T300_914", "T700_2510",
            ])
            self.material_combo.currentTextChanged.connect(self._on_material_changed)
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

            return group

        def _create_wrinkle_group(self) -> QGroupBox:
            group = QGroupBox("Wrinkle Geometry")
            layout = QFormLayout(group)

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
            # Initially hidden — only shown for graded morphology
            self.decay_floor_label.setVisible(False)
            self.decay_floor_spin.setVisible(False)

            self.loading_combo = QComboBox()
            self.loading_combo.addItems(["compression", "tension"])
            layout.addRow("Loading:", self.loading_combo)

            return group

        def _create_mesh_group(self) -> QGroupBox:
            group = QGroupBox("Mesh")
            layout = QFormLayout(group)

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

            return group

        def _create_analysis_group(self) -> QGroupBox:
            group = QGroupBox("Analysis")
            layout = QVBoxLayout(group)

            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setVisible(False)
            layout.addWidget(self.progress_bar)

            self.timer_label = QLabel("")
            self.timer_label.setStyleSheet("QLabel { color: #555; font-size: 11px; }")
            self.timer_label.setVisible(False)
            layout.addWidget(self.timer_label)

            # Countdown timer
            self._countdown_timer = QTimer(self)
            self._countdown_timer.setInterval(1000)
            self._countdown_timer.timeout.connect(self._tick_countdown)
            self._elapsed_timer = QElapsedTimer()
            self._estimated_seconds = 0
            self._last_run_seconds = {}  # cache: mesh_key -> actual seconds

            btn_layout = QHBoxLayout()

            self.run_btn = QPushButton("Run Analysis")
            self.run_btn.setStyleSheet(
                "QPushButton { background-color: #2196F3; color: white; "
                "font-weight: bold; padding: 8px; }"
            )
            self.run_btn.clicked.connect(self._on_run)
            btn_layout.addWidget(self.run_btn)


            self.stop_btn = QPushButton("Stop")
            self.stop_btn.setStyleSheet(
                "QPushButton { background-color: #F44336; color: white; "
                "font-weight: bold; padding: 8px; }"
            )
            self.stop_btn.clicked.connect(self._on_stop)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setVisible(False)
            btn_layout.addWidget(self.stop_btn)

            layout.addLayout(btn_layout)

            return group

        # ----------------------------------------------------------
        # Status bar
        # ----------------------------------------------------------

        def _setup_status_bar(self) -> None:
            self.statusBar().showMessage("Ready")

        # ----------------------------------------------------------
        # Build config from GUI
        # ----------------------------------------------------------

        def _build_config(self):
            """Build an AnalysisConfig from the current GUI state."""
            from wrinklefe.analysis import AnalysisConfig
            from wrinklefe.core.material import MaterialLibrary

            material = MaterialLibrary().get(self.material_combo.currentText())

            # Use simplified in-situ strengths (GIc/GIIc not needed for retention)
            # α₀ uses material default (~53° for all carbon/epoxy systems)
            import dataclasses
            material = dataclasses.replace(
                material,
                GIc=None,
                GIIc=None,
                beta_shear=0.0,  # linear shear assumed
            )

            # Parse layup string
            angles = self._parse_layup(self.layup_edit.text())

            # --- Physical feasibility checks ---
            amplitude = self.amplitude_spin.value()
            wavelength = self.wavelength_spin.value()
            width = self.width_spin.value()
            ply_thickness = self.ply_thickness_spin.value()
            n_plies = len(angles)
            laminate_thickness = n_plies * ply_thickness

            # 1. Amplitude cannot exceed half the laminate thickness
            max_amplitude = laminate_thickness / 2.0
            if amplitude > max_amplitude:
                raise ValueError(
                    f"Amplitude ({amplitude:.3f} mm) exceeds half the laminate "
                    f"thickness ({max_amplitude:.3f} mm = {n_plies} plies × "
                    f"{ply_thickness:.3f} mm / 2).\n\n"
                    f"Reduce amplitude to ≤ {max_amplitude:.3f} mm, "
                    f"or increase ply count / thickness."
                )

            # 2. Wavelength must be at least 2× amplitude (otherwise angle > 72°)
            min_wavelength = 2.0 * amplitude
            if wavelength < min_wavelength:
                max_angle = np.degrees(np.arctan(2 * np.pi * amplitude / wavelength))
                raise ValueError(
                    f"Wavelength ({wavelength:.1f} mm) is too short for "
                    f"amplitude ({amplitude:.3f} mm).\n"
                    f"This creates a wrinkle angle of {max_angle:.0f}°, "
                    f"which is physically impossible.\n\n"
                    f"Increase wavelength to ≥ {min_wavelength:.1f} mm."
                )

            # 3. Warn if wrinkle angle exceeds 45° (still allow, but inform)
            theta_max = np.degrees(np.arctan(2 * np.pi * amplitude / wavelength))
            if theta_max > 45.0:
                raise ValueError(
                    f"Wrinkle angle ({theta_max:.1f}°) exceeds 45°.\n"
                    f"This is physically unrealistic for manufacturing defects.\n\n"
                    f"Increase wavelength or decrease amplitude."
                )

            return AnalysisConfig(
                amplitude=amplitude,
                wavelength=wavelength,
                width=width,
                morphology=self.morphology_combo.currentText(),
                decay_floor=self.decay_floor_spin.value(),
                loading=self.loading_combo.currentText(),
                material=material,
                angles=angles,
                ply_thickness=self.ply_thickness_spin.value(),
                nx=self.nx_spin.value(),
                ny=self.ny_spin.value(),
                nz_per_ply=self.nz_per_ply_spin.value(),
                applied_strain=-0.01 if self.loading_combo.currentText() == "compression" else 0.01,
                solver="direct",
                verbose=True,
            )

        @staticmethod
        def _parse_layup(text: str) -> list:
            """Parse a layup string like '[0/45/-45/90]_3s' to angle list."""
            text = text.strip()

            # Simple fallback: try comma or slash separated numbers
            # Remove brackets
            cleaned = text.replace("[", "").replace("]", "")

            # Check for _Ns pattern (repeat N times, symmetric)
            symmetric = cleaned.endswith("s")
            if symmetric:
                cleaned = cleaned[:-1]

            # Check for _N repeat
            repeat = 1
            if "_" in cleaned:
                parts = cleaned.rsplit("_", 1)
                cleaned = parts[0]
                try:
                    repeat = int(parts[1])
                except ValueError:
                    repeat = 1

            # Parse angles
            sep = "/" if "/" in cleaned else ","
            try:
                angles = [float(a.strip()) for a in cleaned.split(sep) if a.strip()]
            except ValueError:
                # Fallback to default quasi-isotropic
                angles = [0, 45, -45, 90]
                repeat = 3
                symmetric = True

            # Apply repeat
            angles = angles * repeat

            # Apply symmetry
            if symmetric:
                angles = angles + list(reversed(angles))

            return angles

        # ----------------------------------------------------------
        # Actions
        # ----------------------------------------------------------

        def _estimate_time(self, config) -> float:
            """Estimate analysis time in seconds based on problem size."""
            n_plies = len(config.angles) if config.angles else 24

            # FE cost model
            n_elem = config.nx * config.ny * config.nz_per_ply * n_plies
            n_dof = n_elem * 3 * 2  # rough DOF estimate
            t = 0.5 + n_elem * 0.002 + (n_dof / 1000.0) ** 1.3 * 0.1

            # Check if we have a cached actual time for similar config
            mesh_key = (config.nx, config.ny, config.nz_per_ply, n_plies)
            if mesh_key in self._last_run_seconds:
                actual = self._last_run_seconds[mesh_key]
                t = 0.3 * t + 0.7 * actual

            return max(1.0, t)

        def _start_countdown(self, estimated_seconds: float) -> None:
            """Start the countdown timer display."""
            self._estimated_seconds = estimated_seconds
            self._elapsed_timer.start()
            self.progress_bar.setRange(0, int(estimated_seconds * 10))
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.timer_label.setVisible(True)
            self._update_timer_display()
            self._countdown_timer.start()

        def _stop_countdown(self) -> None:
            """Stop the countdown timer."""
            self._countdown_timer.stop()
            self.progress_bar.setVisible(False)
            self.timer_label.setVisible(False)

        def _tick_countdown(self) -> None:
            """Update countdown display every second."""
            self._update_timer_display()

        def _update_timer_display(self) -> None:
            """Refresh the timer label and progress bar."""
            elapsed_s = self._elapsed_timer.elapsed() / 1000.0
            remaining_s = max(0, self._estimated_seconds - elapsed_s)

            # Update progress bar (don't exceed max)
            progress_val = min(
                int(elapsed_s * 10),
                int(self._estimated_seconds * 10),
            )
            self.progress_bar.setValue(progress_val)

            elapsed_str = self._format_time(elapsed_s)
            est_str = self._format_time(self._estimated_seconds)

            if remaining_s > 0:
                remain_str = self._format_time(remaining_s)
                self.timer_label.setText(
                    f"Elapsed: {elapsed_str}  |  "
                    f"Estimated: ~{est_str}  |  "
                    f"Remaining: ~{remain_str}"
                )
            else:
                self.timer_label.setText(
                    f"Elapsed: {elapsed_str}  |  "
                    f"Estimated: ~{est_str}  |  "
                    f"Finishing up..."
                )
                # Switch to indeterminate mode when past estimate
                self.progress_bar.setRange(0, 0)

        @staticmethod
        def _format_time(seconds: float) -> str:
            """Format seconds as M:SS or H:MM:SS."""
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

        def _on_run(self) -> None:
            """Run the full FE analysis in a background thread."""
            try:
                config = self._build_config()
            except Exception as e:
                QMessageBox.critical(self, "Configuration Error", str(e))
                return

            self.run_btn.setEnabled(False)

            self.stop_btn.setEnabled(True)
            self.stop_btn.setVisible(True)
            self.statusBar().showMessage("Running analysis...")
            self.output_text.clear()

            est = self._estimate_time(config)
            self._start_countdown(est)
            self._current_config = config

            self._worker = AnalysisWorker(config)
            self._worker.finished.connect(self._on_analysis_done)
            self._worker.error.connect(self._on_analysis_error)
            self._worker.progress.connect(self._on_progress)
            self._worker.start()

        def _on_analysis_done(self, result) -> None:
            """Handle completed analysis."""
            actual_s = self._elapsed_timer.elapsed() / 1000.0
            self._stop_countdown()

            # Cache actual time for future estimates
            if hasattr(self, '_current_config'):
                cfg = self._current_config
                n_plies = len(cfg.angles) if cfg.angles else 24
                mesh_key = (cfg.nx, cfg.ny, cfg.nz_per_ply, n_plies)
                self._last_run_seconds[mesh_key] = actual_s

            self._result = result
            self.run_btn.setEnabled(True)

            self.stop_btn.setEnabled(False)
            self.stop_btn.setVisible(False)
            self.output_text.setPlainText(result.summary())
            self._update_plots(result)
            self.statusBar().showMessage(
                f"Analysis complete in {actual_s:.1f}s. Strength = "
                f"{result.analytical_strength_MPa:.1f} MPa"
            )

        def _on_analysis_error(self, msg: str) -> None:
            """Handle analysis error."""
            self._stop_countdown()
            self.run_btn.setEnabled(True)

            self.stop_btn.setEnabled(False)
            self.stop_btn.setVisible(False)
            self.output_text.setPlainText(f"ERROR:\n{msg}")
            self.statusBar().showMessage("Analysis failed.")
            QMessageBox.critical(self, "Analysis Error", msg[:500])

        def _on_stop(self) -> None:
            """Stop the running analysis."""
            self._stop_countdown()

            if self._worker is not None and self._worker.isRunning():
                self._worker.terminate()
                self._worker.wait(2000)
                self._worker = None

            self.run_btn.setEnabled(True)

            self.stop_btn.setEnabled(False)
            self.stop_btn.setVisible(False)
            self.statusBar().showMessage("Analysis stopped by user.")
            self.output_text.setPlainText("Analysis was stopped by user.")

        def _on_progress(self, msg: str) -> None:
            """Update status bar with progress message."""
            self.statusBar().showMessage(msg)

        def _on_stress_component_changed(self, *_args) -> None:
            """Re-render when stress component dropdown changes.

            Auto-switches to Global for out-of-plane components.
            """
            combo_idx = self.stress_component_combo.currentIndex()
            _OOP = {2, 3, 4, 8}  # σ₃₃, τ₂₃, τ₁₃, interlaminar
            if combo_idx in _OOP:
                self.stress_coord_combo.blockSignals(True)
                self.stress_coord_combo.setCurrentIndex(1)  # Global
                self.stress_coord_combo.blockSignals(False)

            if hasattr(self, '_result') and self._result is not None:
                self._plot_stress(self._result)

        def _on_morphology_changed(self, name: str) -> None:
            """Show/hide decay floor spinner based on morphology selection."""
            is_graded = name.lower().strip() == "graded"
            self.decay_floor_label.setVisible(is_graded)
            self.decay_floor_spin.setVisible(is_graded)

        def _on_material_changed(self, name: str) -> None:
            """Handle material selection change (reserved for future use)."""
            pass

        def _on_coord_changed(self, *_args) -> None:
            """Re-render when Global/Local coordinate toggle changes."""
            if hasattr(self, '_result') and self._result is not None:
                self._plot_stress(self._result)

        # ----------------------------------------------------------
        # Shared plot helpers
        # ----------------------------------------------------------

        def _extract_plane_elements(self, mesh, plane="xz"):
            """Get element indices at mid-y (xz) or top surface (xy).

            Selects only a single layer of elements — the one closest to the
            midplane (for xz) or top surface (for xy).

            Returns (indices, centers) where centers is (n_elements, 3).
            """
            centers = mesh.nodes[mesh.elements].mean(axis=1)
            if plane == "xz":
                mid_y = (centers[:, 1].min() + centers[:, 1].max()) / 2.0
                # Find the single y-layer closest to mid_y
                unique_y = np.unique(np.round(centers[:, 1], decimals=6))
                closest_y = unique_y[np.argmin(np.abs(unique_y - mid_y))]
                tol = (centers[:, 1].max() - centers[:, 1].min()) / max(mesh.ny, 1) * 0.05
                mask = np.abs(centers[:, 1] - closest_y) < max(tol, 1e-10)
            elif plane == "xy":
                # Find the single z-layer at the top
                unique_z = np.unique(np.round(centers[:, 2], decimals=6))
                top_z = unique_z[-1]
                tol = (centers[:, 2].max() - centers[:, 2].min()) / max(mesh.nz, 1) * 0.05
                mask = np.abs(centers[:, 2] - top_z) < max(tol, 1e-10)
            else:
                raise ValueError(f"Unknown plane: {plane}")
            return np.flatnonzero(mask), centers

        def _get_stress_values(self, field_results, combo_index):
            """Extract per-element stress for the selected component.

            Uses Global or Local coordinates based on the stress_coord_combo.

            Returns (values, label, use_diverging).
            """
            use_global = self.stress_coord_combo.currentIndex() == 1
            coord_tag = "global" if use_global else "local"
            stress_field = (
                field_results.stress_global if use_global
                else field_results.stress_local
            )

            labels_local = [
                "$\\sigma_{11}$", "$\\sigma_{22}$",
                "$\\sigma_{33}$", "$\\tau_{23}$",
                "$\\tau_{13}$", "$\\tau_{12}$",
            ]
            labels_global = [
                "$\\sigma_{xx}$", "$\\sigma_{yy}$",
                "$\\sigma_{zz}$", "$\\tau_{yz}$",
                "$\\tau_{xz}$", "$\\tau_{xy}$",
            ]
            base_labels = labels_global if use_global else labels_local

            if combo_index <= 5:
                values = stress_field[:, :, combo_index].mean(axis=1)
                label = f"{base_labels[combo_index]} (MPa) [{coord_tag}]"
                use_diverging = True
            elif combo_index == 6:
                values = field_results.von_mises.mean(axis=1)
                label = "Von Mises (MPa)"
                use_diverging = False
            elif combo_index == 7:
                values = field_results.max_principal_stress.mean(axis=1)
                label = "Max Principal (MPa)"
                use_diverging = True
            elif combo_index == 8:
                values = stress_field[:, :, 4].mean(axis=1)
                label = f"Interlaminar $\\tau_{{xz}}$ (MPa) [{coord_tag}]" if use_global else f"Interlaminar $\\tau_{{13}}$ (MPa) [{coord_tag}]"
                use_diverging = True
            else:
                values = np.zeros(stress_field.shape[0])
                label = "Stress (MPa)"
                use_diverging = True

            return values, label, use_diverging

        def _smooth_to_nodes(self, mesh, elem_stress, elem_indices, plane="xz"):
            """Average element stresses to nodes for smooth contour rendering.

            For each node shared by elements in the slice, compute the mean
            of contributing element values.

            Returns (node_coords_2d, node_values, triangles) ready for tricontourf.
            """
            from matplotlib.tri import Triangulation

            elems = mesh.elements
            nodes = mesh.nodes

            # Collect unique nodes from slice elements
            slice_conn = elems[elem_indices]  # (n_slice, 8)

            # Pick face nodes for the 2D plane
            if plane == "xz":
                face = [0, 1, 5, 4]  # x-z face of hex
                ci, cj = 0, 2  # x, z
            else:
                face = [4, 5, 6, 7]  # x-y top face
                ci, cj = 0, 1  # x, y

            # Build node → element stress mapping
            face_nodes = slice_conn[:, face]  # (n_slice, 4)
            unique_nids, inverse = np.unique(face_nodes.ravel(), return_inverse=True)
            n_unique = len(unique_nids)

            # Accumulate element values at nodes
            node_sum = np.zeros(n_unique)
            node_count = np.zeros(n_unique)
            for i, eid in enumerate(elem_indices):
                val = elem_stress[eid]
                for j in range(4):
                    nidx = inverse[i * 4 + j]
                    node_sum[nidx] += val
                    node_count[nidx] += 1

            node_count[node_count == 0] = 1
            node_vals = node_sum / node_count

            # Node coordinates in 2D
            coords_2d = nodes[unique_nids][:, [ci, cj]]

            # Build triangles from quad faces (2 triangles per quad)
            tris = []
            for i in range(len(elem_indices)):
                n0 = inverse[i * 4 + 0]
                n1 = inverse[i * 4 + 1]
                n2 = inverse[i * 4 + 2]
                n3 = inverse[i * 4 + 3]
                tris.append([n0, n1, n2])
                tris.append([n0, n2, n3])

            triangulation = Triangulation(
                coords_2d[:, 0], coords_2d[:, 1], triangles=np.array(tris)
            )
            return triangulation, node_vals

        def _get_interior_elements(self, mesh):
            """Return indices of elements not adjacent to BC faces (x_min, x_max).

            Excludes elements within one element width of the loaded faces
            to remove boundary effect artifacts from failure assessment.
            """
            centers = mesh.nodes[mesh.elements].mean(axis=1)
            x_min, x_max = centers[:, 0].min(), centers[:, 0].max()
            elem_width = (x_max - x_min) / max(mesh.nx, 1)
            mask = (
                (centers[:, 0] > x_min + elem_width)
                & (centers[:, 0] < x_max - elem_width)
            )
            indices = np.flatnonzero(mask)
            # Guard: if trimming removed everything, fall back to all elements
            if indices.size == 0:
                return np.arange(mesh.n_elements)
            return indices

        def _element_quads(self, mesh, elem_indices, plane="xz"):
            """Build quad vertices for PolyCollection.

            Returns list of (4,2) arrays.
            """
            nodes = mesh.nodes
            elems = mesh.elements
            verts = []
            if plane == "xz":
                for eid in elem_indices:
                    conn = elems[eid]
                    corners = nodes[conn][[0, 1, 5, 4]]
                    verts.append(corners[:, [0, 2]])
            elif plane == "xy":
                for eid in elem_indices:
                    conn = elems[eid]
                    corners = nodes[conn][[4, 5, 6, 7]]
                    verts.append(corners[:, [0, 1]])
            return verts

        # ----------------------------------------------------------
        # Plot updates
        # ----------------------------------------------------------

        def _update_plots(self, result) -> None:
            """Update all plot tabs with analysis results."""
            if not HAS_MPL_QT or self.profile_canvas is None:
                return

            self._plot_profile(result)
            self._plot_mesh(result)
            self._plot_stress(result)

        def _plot_profile(self, result) -> None:
            """Draw wrinkle profile and morphology comparison on Profile tab."""
            fig = self.profile_canvas.figure
            fig.clear()

            try:
                config = result.config

                if result.wrinkle_config is not None and result.wrinkle_config.n_wrinkles() >= 2:
                    # Dual wrinkle profiles with phase offset
                    ax1 = fig.add_subplot(121)
                    wc = result.wrinkle_config
                    w_upper = wc.wrinkles[0]
                    w_lower = wc.wrinkles[1]
                    p_upper = w_upper.profile
                    p_lower = w_lower.profile

                    # Phase offset → geometric shift for lower wrinkle
                    delta_x_upper = w_upper.phase_offset * p_upper.wavelength / (2.0 * np.pi)
                    delta_x_lower = w_lower.phase_offset * p_lower.wavelength / (2.0 * np.pi)

                    # Use full domain length for x range
                    Lx = config.domain_length
                    x = np.linspace(0, Lx, 500)

                    # Evaluate with phase-offset shifts
                    z_upper = p_upper.displacement(x - delta_x_upper)
                    z_lower = p_lower.displacement(x - delta_x_lower)

                    ax1.plot(x, z_upper, color="#1f77b4", linewidth=1.5, label="Upper wrinkle")
                    ax1.plot(x, z_lower, color="#d62728", linewidth=1.5, label="Lower wrinkle")
                    ax1.fill_between(x, z_lower, z_upper, alpha=0.15, color="#2ca02c",
                                     label="Interface gap")
                    ax1.axhline(0, color="0.6", linewidth=0.5, zorder=0)
                    ax1.set_xlabel("x (mm)")
                    ax1.set_ylabel("z (mm)")
                    ax1.set_title(f"Dual-Wrinkle Profiles — {config.morphology}")
                    ax1.legend(fontsize=7, loc="upper right")

                    # Fiber angle distribution for both wrinkles
                    ax2 = fig.add_subplot(122)
                    slope_upper = p_upper.slope(x - delta_x_upper)
                    slope_lower = p_lower.slope(x - delta_x_lower)
                    angle_upper = np.degrees(np.arctan(np.abs(slope_upper)))
                    angle_lower = np.degrees(np.arctan(np.abs(slope_lower)))

                    ax2.plot(x, angle_upper, color="#1f77b4", linewidth=1.5,
                             label="Upper")
                    ax2.plot(x, angle_lower, color="#d62728", linewidth=1.5,
                             linestyle="--", label="Lower")
                    ax2.axhline(0, color="0.6", linewidth=0.5, zorder=0)
                    ax2.set_xlabel("x (mm)")
                    ax2.set_ylabel("Fiber angle (deg)")
                    ax2.set_title("Fiber Misalignment Angle")
                    ax2.legend(fontsize=7)

                else:
                    # Analytical summary plot
                    ax1 = fig.add_subplot(121)
                    labels = ["Max angle", "Eff. angle"]
                    vals = [np.degrees(result.max_angle_rad),
                            np.degrees(result.effective_angle_rad)]
                    colors = ["#1f77b4", "#ff7f0e"]
                    ax1.bar(labels, vals, color=colors)
                    ax1.set_ylabel("Angle (deg)")
                    ax1.set_title("Wrinkle Angles")

                    ax2 = fig.add_subplot(122)
                    labels2 = ["Knockdown", "Damage"]
                    vals2 = [result.analytical_knockdown, result.damage_index]
                    colors2 = ["#2ca02c", "#d62728"]
                    ax2.bar(labels2, vals2, color=colors2)
                    ax2.set_ylim(0, 1.1)
                    ax2.set_ylabel("Factor / Index")
                    ax2.set_title("Knockdown & Damage")

                fig.tight_layout()
            except Exception as e:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Profile plot error:\n{e}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="red")
                ax.set_axis_off()

            self.profile_canvas.draw()

        def _plot_mesh(self, result) -> None:
            """Draw 2D FE mesh cross-section on Mesh tab."""
            fig = self.mesh_canvas.figure
            fig.clear()

            try:
                if result.mesh is not None and result.mesh.n_elements > 0:
                    mesh = result.mesh
                    ax = fig.add_subplot(111)

                    xz_indices, _centers = self._extract_plane_elements(mesh, "xz")

                    if xz_indices.size > 0:
                        from matplotlib.collections import PolyCollection
                        import matplotlib.cm as cm

                        verts = self._element_quads(mesh, xz_indices, "xz")
                        ply_colors = mesh.ply_ids[xz_indices].astype(float)
                        n_plies = int(ply_colors.max()) + 1
                        cmap = cm.get_cmap("tab20", n_plies)

                        pc = PolyCollection(
                            verts, array=ply_colors, cmap=cmap,
                            edgecolors="black", linewidths=0.3,
                        )
                        pc.set_clim(-0.5, n_plies - 0.5)
                        ax.add_collection(pc)
                        ax.autoscale_view()
                        ax.set_aspect("equal")

                        cb = fig.colorbar(pc, ax=ax, label="Ply ID", shrink=0.8)
                        cb.set_ticks(range(0, n_plies, max(1, n_plies // 10)))
                    else:
                        ax.text(
                            0.5, 0.5, "No elements found in mid-y plane.",
                            transform=ax.transAxes, ha="center",
                            fontsize=12, color="0.5",
                        )

                    ax.set_xlabel("x (mm)")
                    ax.set_ylabel("z (mm)")
                    morph_label = result.config.morphology if result.config else ""
                    ax.set_title(
                        f"FE Mesh — {morph_label}  |  "
                        f"{mesh.n_nodes:,} nodes, {mesh.n_elements:,} elem, "
                        f"{mesh.n_nodes * 3:,} DOF"
                    )


                else:
                    ax = fig.add_subplot(111)
                    ax.text(
                        0.5, 0.5,
                        "No mesh data.\nRun full analysis (not Analytical Only)\nto generate mesh.",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=11, color="0.5",
                    )
                    ax.set_axis_off()

                fig.tight_layout()
            except Exception as e:
                ax = fig.add_subplot(111)
                ax.text(
                    0.5, 0.5, f"Mesh plot error:\n{e}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="red",
                )
                ax.set_axis_off()

            self.mesh_canvas.draw()

        def _plot_stress(self, result) -> None:
            """Draw stress contours + retention bars on Stress tab."""
            fig = self.stress_canvas.figure
            fig.clear()

            try:
                import matplotlib.gridspec as gridspec

                if result.field_results is not None:
                    # Layout: stress contour on top, retention bar on bottom
                    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.45)

                    fr = result.field_results
                    mesh = result.mesh
                    combo_idx = self.stress_component_combo.currentIndex()

                    stress_vals, label, use_diverging = self._get_stress_values(
                        fr, combo_idx
                    )

                    from matplotlib.collections import PolyCollection
                    from matplotlib.colors import TwoSlopeNorm, Normalize

                    ax = fig.add_subplot(gs[0])
                    xz_indices, _centers = self._extract_plane_elements(mesh, "xz")

                    if xz_indices.size > 0:
                        vals_xz = stress_vals[xz_indices]

                        if use_diverging:
                            vmax = max(abs(vals_xz.min()), abs(vals_xz.max()), 1e-10)
                            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
                            cmap = "RdBu_r"
                        else:
                            norm = Normalize(
                                vmin=max(vals_xz.min(), 0),
                                vmax=max(vals_xz.max(), 1e-10),
                            )
                            cmap = "viridis"

                        # Smooth nodal-averaged contour
                        tri, node_vals = self._smooth_to_nodes(
                            mesh, stress_vals, xz_indices, "xz"
                        )
                        n_levels = 32
                        if use_diverging:
                            levels = np.linspace(-vmax, vmax, n_levels)
                        else:
                            levels = np.linspace(
                                norm.vmin, norm.vmax, n_levels
                            )
                        tcf = ax.tricontourf(
                            tri, node_vals, levels=levels,
                            cmap=cmap, norm=norm, extend="both",
                        )
                        fig.colorbar(tcf, ax=ax, label=label, shrink=0.8)

                        # Optional mesh wireframe overlay
                        if self.show_mesh_lines_check.isChecked():
                            verts_xz = self._element_quads(
                                mesh, xz_indices, "xz"
                            )
                            pc_wire = PolyCollection(
                                verts_xz, facecolors="none",
                                edgecolors="black", linewidths=0.3,
                            )
                            ax.add_collection(pc_wire)

                        ax.autoscale_view()
                        ax.set_aspect("equal")
                    else:
                        ax.text(
                            0.5, 0.5, "No elements in x-z slice",
                            transform=ax.transAxes, ha="center",
                        )

                    ax.set_xlabel("x (mm)")
                    ax.set_ylabel("z (mm)")

                    # Out-of-plane components need fine z-mesh
                    _OOP_INDICES = {2, 3, 4, 8}  # σ₃₃, τ₂₃, τ₁₃, interlaminar
                    nz_per_ply = getattr(result.config, "nz_per_ply", 1)
                    if combo_idx in _OOP_INDICES and nz_per_ply < 3:
                        ax.text(
                            0.5, 0.02,
                            f"\u26a0 Out-of-plane stress with nz/ply = {nz_per_ply}. "
                            f"Set nz/ply \u2265 3 for reliable through-thickness results.",
                            transform=ax.transAxes, ha="center", va="bottom",
                            fontsize=9, color="#b71c1c",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="#ffcdd2", edgecolor="#b71c1c",
                                alpha=0.9,
                            ),
                        )

                    ax.set_title(
                        f"{self.stress_component_combo.currentText()}  \u2014  "
                        f"x-z Cross Section (mid-y)"
                    )

                    # --- Bottom panel: retention bars ---
                    ax_bar = fig.add_subplot(gs[1])
                    self._plot_mechanism_bar(ax_bar, result)

                    # --- Text report (failure details) ---
                    if result.failure_indices is not None and len(result.failure_indices) > 0:
                        interior = self._get_interior_elements(mesh)
                        n_trimmed = mesh.n_elements - len(interior)
                        criteria = list(result.failure_indices.keys())
                        max_fi = []
                        for name in criteria:
                            fi = result.failure_indices[name]
                            fi_int = fi[interior].mean(axis=1)
                            finite = fi_int[np.isfinite(fi_int)]
                            max_fi.append(float(finite.max()) if finite.size > 0 else 0.0)
                        self._append_failure_report(result, criteria, max_fi, interior, n_trimmed)

                else:
                    # Analytical-only fallback: just show retention bars
                    ax = fig.add_subplot(111)
                    self._plot_mechanism_bar(ax, result)

                fig.tight_layout()
            except Exception as e:
                ax = fig.add_subplot(111)
                ax.text(
                    0.5, 0.5, f"Stress plot error:\n{e}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="red",
                )
                ax.set_axis_off()

            self.stress_canvas.draw()

        def _plot_mechanism_bar(self, ax, result) -> None:
            """Draw strength and modulus retention with mechanism annotation."""
            import numpy as np
            loading = result.config.loading
            kd = result.analytical_knockdown
            mod_ret = result.modulus_retention if result.modulus_retention is not None else 1.0

            # Two bars: strength retention and modulus retention
            labels = ["Strength\nRetention", "Modulus\nRetention"]
            values = [kd, mod_ret]

            # Color by severity
            def bar_color(v):
                if v < 0.3:
                    return "#d62728"  # red
                elif v < 0.6:
                    return "#ff7f0e"  # orange
                else:
                    return "#2ca02c"  # green

            colors = [bar_color(v) for v in values]

            x_pos = np.arange(len(labels))
            bars = ax.bar(x_pos, values, color=colors, width=0.45,
                          edgecolor="black", linewidth=1.0)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
            ax.set_ylim(0, 1.25)
            ax.set_ylabel("Retention", fontsize=9)
            ax.axhline(1.0, color="0.7", linestyle=":", linewidth=0.8)

            # Value labels on bars
            for bar, val in zip(bars, values):
                pct = val * 100
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        "%.1f%%" % pct, ha="center", va="bottom", fontsize=9,
                        fontweight="bold")

            # Mechanism annotation
            mod_note = "Modulus: \u27e8\u03c3\u2093\u2093\u27e9_wrinkled / \u27e8\u03c3\u2093\u2093\u27e9_pristine from FE"
            if loading == "tension" and result.tension_mechanisms is not None:
                m = result.tension_mechanisms
                ctrl = m["mode"]
                # FE initiation (LaRC05 retention) for tension
                fe_ret = None
                if result.retention_factors:
                    fe_ret = result.retention_factors.get("larc05", None)
                init_line = ""
                if fe_ret is not None:
                    init_line = "\nFE Initiation (LaRC05): %.1f%% — conservative lower bound" % (fe_ret * 100)
                note = (
                    "Ultimate: min(cos\u00b2\u03b8=%.2f, matrix=%.2f, "
                    "\u03c3\u2083\u2083=%.2f)  controls: %s%s\n%s"
                ) % (m["kd_fiber"], m["kd_matrix"], m["kd_oop"], ctrl, init_line, mod_note)
            else:
                gamma_Y_eff = getattr(result, "gamma_Y_eff", result.config.material.gamma_Y)
                theta_eff = result.effective_angle_rad
                import math
                theta_deg = math.degrees(theta_eff)
                note = (
                    "Strength: 1/(1+\u03b8/\u03b3_Y) = 1/(1+%.1f\u00b0/%.4f) = %.3f\n%s"
                ) % (theta_deg, gamma_Y_eff, kd, mod_note)

            ax.set_title(note, fontsize=7, style="italic", pad=8)

        def _append_failure_report(self, result, criteria, max_fi, interior, n_trimmed):
            """Append critical element failure detail to the output text area."""
            mesh = result.mesh
            fr = result.field_results
            laminate = result.laminate

            # Find the overall critical criterion and element
            overall_max_fi = 0.0
            crit_criterion = ""
            crit_elem_idx = 0

            for name in criteria:
                fi = result.failure_indices[name]  # (n_elem, n_gauss)
                fi_int = fi[interior].mean(axis=1)
                finite = fi_int[np.isfinite(fi_int)]
                if finite.size == 0:
                    continue
                local_max = float(finite.max())
                if local_max > overall_max_fi:
                    overall_max_fi = local_max
                    crit_criterion = name
                    local_idx = int(np.argmax(fi_int))
                    crit_elem_idx = interior[local_idx]

            if overall_max_fi <= 0:
                return

            # Element info
            ply_idx = int(mesh.ply_ids[crit_elem_idx])
            ply_angle = float(mesh.ply_angles[crit_elem_idx])
            mat = laminate.plies[ply_idx].material

            # Stress state at critical element (avg over Gauss pts)
            stress = fr.stress_local[crit_elem_idx].mean(axis=0)  # (6,)

            # Build allowable comparison table
            # For each component, pick allowable based on sign
            components = [
                ("\u03c3\u2081\u2081 (fiber)",      stress[0], mat.Xt if stress[0] >= 0 else mat.Xc,
                 "Xt" if stress[0] >= 0 else "Xc"),
                ("\u03c3\u2082\u2082 (transverse)",  stress[1], mat.Yt if stress[1] >= 0 else mat.Yc,
                 "Yt" if stress[1] >= 0 else "Yc"),
                ("\u03c3\u2083\u2083 (thickness)",   stress[2], mat.Zt if stress[2] >= 0 else mat.Zc,
                 "Zt" if stress[2] >= 0 else "Zc"),
                ("\u03c4\u2082\u2083",               stress[3], mat.S23, "S23"),
                ("\u03c4\u2081\u2083",               stress[4], mat.S13, "S13"),
                ("\u03c4\u2081\u2082",               stress[5], mat.S12, "S12"),
            ]

            # Status
            status = "FAILED" if overall_max_fi >= 1.0 else "SAFE"

            # FE knockdown
            loading = result.config.loading
            ref_strength_val = mat.Xt if loading == "tension" else mat.Xc
            fe_knockdown = 1.0 / overall_max_fi if overall_max_fi > 0 else float("inf")
            fe_strength = ref_strength_val * fe_knockdown

            # Analytical comparison
            analytical_kd = result.analytical_knockdown
            analytical_strength = result.analytical_strength_MPa

            # Format report
            lines = [
                "",
                "=" * 60,
                "  FAILURE ASSESSMENT \u2014 CRITICAL ELEMENT",
                "=" * 60,
                "",
                f"  Criterion:     {crit_criterion}",
                f"  Max FI:        {overall_max_fi:.3f}  \u2192  {status}",
                f"  Location:      Element {crit_elem_idx}, Ply {ply_idx} ({ply_angle:.0f}\u00b0)",
                f"  Excluded:      {n_trimmed} boundary elements trimmed",
                "",
                "  Stress State at Critical Point vs Material Allowables:",
                "",
                f"  {'Component':<22s} {'Stress (MPa)':>13s}  {'Allowable (MPa)':>16s}  {'Ratio':>7s}",
                "  " + "\u2500" * 62,
            ]

            for comp_name, val, allow, allow_name in components:
                ratio = abs(val) / allow if allow > 0 else float("inf")
                lines.append(
                    f"  {comp_name:<22s} {val:>+13.1f}  "
                    f"{allow_name + ' = ':>6s}{allow:>8.1f}  {ratio:>7.3f}"
                )

            lines += [
                "",
                f"  FE Knockdown Estimate ({loading}):",
                f"    Reference:  {'Xt' if loading == 'tension' else 'Xc'} = {ref_strength_val:.1f} MPa",
                f"    Applied stress at FI=1:  {ref_strength_val:.1f} / {overall_max_fi:.3f} = {fe_strength:.1f} MPa",
                f"    FE knockdown factor:     {fe_knockdown:.3f}",
            ]

            if loading == "tension" and result.tension_mechanisms is not None:
                m = result.tension_mechanisms
                ctrl = m["mode"]
                arrow = "  \u2190 controls"
                mark_f = arrow if ctrl.startswith("fib") else ""
                mark_m = arrow if ctrl.startswith("mat") else ""
                mark_o = arrow if ctrl.startswith("OOP") else ""
                lines += [
                    "",
                    "  Analytical Prediction (Three-Mechanism Tension Model):",
                    "    Fiber (cos\u00b2\u03b8):        %.3f%s" % (m["kd_fiber"], mark_f),
                    "    Matrix (Hashin):       %.3f%s" % (m["kd_matrix"], mark_m),
                    "    OOP \u03c3\u2083\u2083 (curved beam): %.3f%s" % (m["kd_oop"], mark_o),
                    "    " + "\u2500" * 25,
                    "    0\u00b0 ply KD:            %.3f" % m["kd_0"],
                    "    CLT fraction f\u2080:      %.3f" % m["f_0"],
                    "    Laminate KD:           %.3f" % analytical_kd,
                    "    Strength:              %.1f MPa" % analytical_strength,
                ]
            else:
                gamma_Y = mat.gamma_Y
                theta_eff = result.effective_angle_rad
                lines += [
                    "",
                    "  Analytical Prediction (Budiansky-Fleck Compression):",
                    "    KD = 1/(1+\u03b8_eff/\u03b3_Y) = 1/(1+%.3f/%.3f) = %.3f" % (
                        theta_eff, gamma_Y, analytical_kd),
                    "    Strength:    %.1f MPa" % analytical_strength,
                ]

            # Modulus retention
            if result.modulus_retention is not None:
                lines += [
                    "",
                    "  Modulus Retention (from FE):",
                    f"    E_wrinkled / E_pristine = {result.modulus_retention:.3f}"
                    f"  ({result.modulus_retention*100:.1f}%)",
                ]

            # FE initiation — only for tension (compression failures are catastrophic)
            if result.retention_factors is not None and loading == "tension":
                lines.append("")
                lines.append("  FE Damage Initiation (conservative lower bound):")
                for cname in criteria:
                    ret = result.retention_factors.get(cname, 1.0)
                    lines.append(f"    {cname:<18s}  {ret*100:.1f}%")
                lines.append("    (Tension: delamination initiates before fiber fracture)")

            # Mode breakdown
            if result.failure_modes is not None:
                crit_name = criteria[0]
                modes_arr = result.failure_modes[crit_name]
                interior_modes = modes_arr[interior, 0]
                from collections import Counter
                mode_counts = Counter(interior_modes)
                total = len(interior_modes)
                lines.append("")
                lines.append("  Failure Mode Distribution (interior elements):")
                for mode_name in ["fiber_kinking", "fiber_tension",
                                   "matrix_tension", "matrix_compression"]:
                    count = mode_counts.get(mode_name, 0)
                    pct = count / total * 100 if total > 0 else 0
                    bar = "\u2588" * int(pct / 2)
                    lines.append(f"    {mode_name:<22s}  {count:>5d}  ({pct:5.1f}%)  {bar}")

            # Sub-criterion detail at critical element
            try:
                crit_name = criteria[0]
                fi_field = result.failure_indices[crit_name]
                fi_int = fi_field[interior].mean(axis=1)
                if fi_int.size > 0:
                    local_idx = int(np.argmax(fi_int))
                    crit_elem = interior[local_idx]
                    from wrinklefe.failure.larc05 import LaRC05Criterion
                    larc = LaRC05Criterion()
                    stress_crit = fr.stress_local[crit_elem].mean(axis=0)
                    phi_0_crit = float(mesh.element_fiber_angles_array()[crit_elem])
                    ctx = {"misalignment_angle": phi_0_crit}
                    detail_result = larc.evaluate(stress_crit, mat, ctx)
                    d = detail_result.detail
                    if d:
                        fi_f = d.get("fi_fiber", 0.0)
                        fi_m = d.get("fi_matrix", 0.0)
                        gov_f = " \u2190 GOVERNING" if fi_f >= fi_m else ""
                        gov_m = " \u2190 GOVERNING" if fi_m > fi_f else ""
                        lines.append("")
                        lines.append("  LaRC05 Sub-Criterion Decomposition (critical element):")
                        lines.append(
                            f"    Fiber ({d.get('mode_fiber','')}):"
                            f"  FI = {fi_f:.4f}{gov_f}"
                        )
                        lines.append(
                            f"    Matrix ({d.get('mode_matrix','')}):"
                            f"  FI = {fi_m:.4f}{gov_m}"
                        )
                        phi_0_deg = np.degrees(d.get("phi_0", 0))
                        phi_c_deg = np.degrees(d.get("phi_c", 0))
                        lines.append(
                            f"    Misalignment:  \u03c6\u2080 = {phi_0_deg:.2f}\u00b0"
                            f"  \u03c6_c = {phi_c_deg:.2f}\u00b0"
                            f"  total = {phi_0_deg + phi_c_deg:.2f}\u00b0"
                        )
            except Exception:
                pass  # sub-criterion detail is optional

            lines.append("=" * 60)

            # Append to output text
            current = self.output_text.toPlainText()
            self.output_text.setPlainText(current + "\n".join(lines))

        def _on_new(self) -> None:
            """Reset to defaults."""
            self.amplitude_spin.setValue(0.366)
            self.wavelength_spin.setValue(16.0)
            self.width_spin.setValue(12.0)
            self.morphology_combo.setCurrentIndex(0)
            self.loading_combo.setCurrentIndex(0)
            self.nx_spin.setValue(20)
            self.ny_spin.setValue(6)
            self.output_text.clear()
            self._result = None
            self.statusBar().showMessage("Ready")

        def _on_export(self) -> None:
            """Export results to JSON."""
            if self._result is None:
                QMessageBox.information(
                    self, "No Results",
                    "Run an analysis first before exporting."
                )
                return

            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "wrinklefe_results.json",
                "JSON Files (*.json);;All Files (*)"
            )
            if filepath:
                try:
                    from wrinklefe.io.export import export_results_json
                    export_results_json(self._result, filepath)
                    self.statusBar().showMessage(f"Results exported to {filepath}")
                except ImportError:
                    QMessageBox.warning(
                        self, "Export Not Available",
                        "Export module not found. Results shown in text below."
                    )
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", str(e))

        def _on_about(self) -> None:
            """Show about dialog."""
            QMessageBox.about(
                self, "About WrinkleFE",
                "<h3>WrinkleFE v0.1.0</h3>"
                "<p>Finite element analysis of wrinkled composite laminates.</p>"
                "<p>Combines 3D cross-laminated plate theory with advanced "
                "composite failure criteria for modeling fiber waviness "
                "defects.</p>"
                "<p><b>References:</b></p>"
                "<ul>"
                "<li>Elhajjar (2025) - Scientific Reports 15:25977</li>"
                "<li>Jin et al. (2026) - Thin-Walled Structures 219:114237</li>"
                "<li>Budiansky & Fleck (1993) - J. Mech. Phys. Solids</li>"
                "</ul>"
            )


# ======================================================================
# Launch function
# ======================================================================

def launch() -> None:
    """Launch the WrinkleFE GUI application.

    Raises
    ------
    ImportError
        If PyQt6 is not installed.
    """
    _check_pyqt6()

    app = QApplication.instance()
    standalone = app is None
    if standalone:
        app = QApplication(sys.argv)

    window = WrinkleFEMainWindow()
    window.show()

    if standalone:
        sys.exit(app.exec())
