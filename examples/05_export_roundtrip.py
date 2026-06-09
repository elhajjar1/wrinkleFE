"""Export round-trip: JSON results, Abaqus .inp, and legacy VTK.

Runs a small FE analysis and writes every export format the package
supports natively (no extra dependencies required):

- ``05_results.json`` — full analysis results for archiving;
- ``05_mesh.inp``     — mesh + ply sets for commercial FE solvers;
- ``05_fields.vtk``   — mesh + displacement/stress fields; open in
  ParaView (File > Open, then Apply) to inspect contours.

Expected runtime: ~10 s (one small FE solve).
Expected output:  the three files above plus a printed size report.
"""

import json
from pathlib import Path

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.io.export import (
    export_abaqus_inp,
    export_results_json,
    export_vtk,
)

config = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="stack", loading="compression",
    nx=12, ny=2, nz_per_ply=1,  # coarse mesh keeps this fast
)
result = WrinkleAnalysis(config).run()

export_results_json(result, "05_results.json")
export_abaqus_inp(result.mesh, result.laminate, "05_mesh.inp")
export_vtk(result.mesh, result.field_results, "05_fields.vtk")

for name in ("05_results.json", "05_mesh.inp", "05_fields.vtk"):
    print(f"Saved: {name} ({Path(name).stat().st_size:,} bytes)")

# Round-trip check: the JSON is plain data, re-loadable anywhere.
data = json.loads(Path("05_results.json").read_text())
kd = data["analytical_predictions"]["analytical_knockdown"]
print(f"JSON knockdown round-trip: {kd:.4f}")
