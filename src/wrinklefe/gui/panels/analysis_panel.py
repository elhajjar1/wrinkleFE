"""Analysis controls panel with run/stop buttons, progress bar, and timer."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar, QLabel,
)
from PyQt6.QtCore import pyqtSignal, QTimer, QElapsedTimer


class AnalysisPanel(QGroupBox):
    """Panel for analysis controls: run/stop, progress bar, countdown timer.

    Signals
    -------
    run_requested
        Emitted when the user clicks *Run Analysis*.
    stop_requested
        Emitted when the user clicks *Stop*.
    """

    run_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__("Analysis", parent)
        layout = QVBoxLayout(self)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Timer label
        self.timer_label = QLabel("")
        self.timer_label.setStyleSheet("QLabel { color: #555; font-size: 11px; }")
        self.timer_label.setVisible(False)
        layout.addWidget(self.timer_label)

        # Countdown timer internals
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._tick_countdown)
        self._elapsed_timer = QElapsedTimer()
        self._estimated_seconds = 0
        self._last_run_seconds: dict = {}  # cache: mesh_key -> actual seconds

        # Buttons
        btn_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self.run_btn.clicked.connect(self.run_requested)
        btn_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #F44336; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self.stop_btn.clicked.connect(self.stop_requested)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

    # -- countdown helpers --

    def estimate_time(self, config) -> float:
        """Estimate analysis time in seconds based on problem size."""
        n_plies = len(config.angles) if config.angles else 24
        n_elem = config.nx * config.ny * config.nz_per_ply * n_plies
        n_dof = n_elem * 3 * 2
        t = 0.5 + n_elem * 0.002 + (n_dof / 1000.0) ** 1.3 * 0.1

        mesh_key = (config.nx, config.ny, config.nz_per_ply, n_plies)
        if mesh_key in self._last_run_seconds:
            actual = self._last_run_seconds[mesh_key]
            t = 0.3 * t + 0.7 * actual

        return max(1.0, t)

    def cache_actual_time(self, config, actual_seconds: float) -> None:
        """Store actual elapsed time for future estimates."""
        n_plies = len(config.angles) if config.angles else 24
        mesh_key = (config.nx, config.ny, config.nz_per_ply, n_plies)
        self._last_run_seconds[mesh_key] = actual_seconds

    def start_countdown(self, estimated_seconds: float) -> None:
        """Start the countdown timer display."""
        self._estimated_seconds = estimated_seconds
        self._elapsed_timer.start()
        self.progress_bar.setRange(0, int(estimated_seconds * 10))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.timer_label.setVisible(True)
        self._update_timer_display()
        self._countdown_timer.start()

    def stop_countdown(self) -> None:
        """Stop the countdown timer."""
        self._countdown_timer.stop()
        self.progress_bar.setVisible(False)
        self.timer_label.setVisible(False)

    def elapsed_seconds(self) -> float:
        """Return seconds since countdown started."""
        return self._elapsed_timer.elapsed() / 1000.0

    def set_running(self, running: bool) -> None:
        """Toggle button states for running/idle."""
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.stop_btn.setVisible(running)

    # -- private --

    def _tick_countdown(self) -> None:
        self._update_timer_display()

    def _update_timer_display(self) -> None:
        elapsed_s = self._elapsed_timer.elapsed() / 1000.0
        remaining_s = max(0, self._estimated_seconds - elapsed_s)

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
