"""GUI panels: material editor, laminate builder, wrinkle configurator, etc."""

from wrinklefe.gui.panels.material_panel import MaterialPanel
from wrinklefe.gui.panels.wrinkle_panel import WrinklePanel
from wrinklefe.gui.panels.mesh_panel import MeshPanel
from wrinklefe.gui.panels.analysis_panel import AnalysisPanel
from wrinklefe.gui.panels.sweep_panel import SweepPanel

__all__ = [
    "MaterialPanel",
    "WrinklePanel",
    "MeshPanel",
    "AnalysisPanel",
    "SweepPanel",
]
