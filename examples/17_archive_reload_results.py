"""Archive a full result and reload it later — no re-solving.

The report exports (``export_results_json`` and friends) reduce
per-Gauss-point fields to ``{min, max, mean, p95, n}`` so the document
stays a few KB. That is the right trade for a report and the wrong one
for post-processing: once the Python session ends, the expensive part of
an FE or CZM run — stress and strain per Gauss point, failure indices
and modes, cohesive damage and tractions — is gone.

``save_results`` / ``load_results`` are the archive tier underneath:
one compressed, pickle-free ``.npz`` that reloads into an
``AnalysisResults`` the existing ``viz/`` functions accept unchanged.

This script solves once, archives, reloads, and shows that

1. every array comes back bit-for-bit and every scalar exactly;
2. a ``viz/`` plot drawn from the reloaded result is byte-identical;
3. the object graph is preserved (``field_results.mesh is results.mesh``)
   rather than duplicated, so plots that rely on it still work.

Expected runtime: ~15 s (one small FE solve; the reload is milliseconds).
Expected output:  ``17_run.wfr`` plus a printed comparison report and
                  ``17_reloaded_displacement.png``.
"""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.io.archive import (
    ARCHIVE_FORMAT_VERSION,
    load_results,
    save_results,
)
from wrinklefe.viz.plots_3d import plot_displacement_3d

ARCHIVE = Path("17_run.wfr")

config = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="stack", loading="compression",
    nx=12, ny=2, nz_per_ply=1,  # coarse mesh keeps this fast
)
result = WrinkleAnalysis(config).run()


def digest(a: np.ndarray) -> str:
    """SHA-256 of an array's bytes — bit identity, not just closeness."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def render(res, path: Path) -> None:
    """Draw the same figure the same way, to a file, from any result."""
    ax = plot_displacement_3d(res.field_results)
    ax.figure.savefig(path, dpi=100)
    plt.close(ax.figure)


# -- Save --------------------------------------------------------------
save_results(result, ARCHIVE)
print(f"Archive format {ARCHIVE_FORMAT_VERSION}: {ARCHIVE} "
      f"({ARCHIVE.stat().st_size / 1024:.1f} KB)")

# -- Reload (here immediately; in practice, a later session) ------------
restored = load_results(ARCHIVE)
print(f"Reloaded analytical knockdown: {restored.analytical_knockdown:.6f} "
      f"(original {result.analytical_knockdown:.6f})")

# -- 1. Bit-for-bit arrays ---------------------------------------------
fields, restored_fields = result.field_results, restored.field_results
checked = []
for name in ("displacement", "stress_global", "stress_local",
             "strain_global", "strain_local"):
    original = getattr(fields, name, None)
    if original is None:
        continue
    assert digest(original) == digest(getattr(restored_fields, name)), name
    checked.append(name)
print(f"Field arrays bit-identical ({len(checked)}): {', '.join(checked)}")

assert digest(result.mesh.nodes) == digest(restored.mesh.nodes)
assert digest(result.mesh.elements) == digest(restored.mesh.elements)
print(f"Mesh identical: {restored.mesh.nodes.shape[0]} nodes, "
      f"{restored.mesh.elements.shape[0]} elements")

for criterion, values in (result.failure_indices or {}).items():
    assert digest(values) == digest(restored.failure_indices[criterion])
print(f"Failure-index fields identical: "
      f"{', '.join(sorted(result.failure_indices or {}))}")

# -- 2. A viz/ plot renders identically --------------------------------
# The first matplotlib render in a process warms font and cache state up,
# so draw one throwaway figure before comparing bytes.
warmup = Path("17_warmup.png")
render(result, warmup)
warmup.unlink()

original_png = Path("17_original_displacement.png")
reloaded_png = Path("17_reloaded_displacement.png")
render(result, original_png)
render(restored, reloaded_png)
identical = original_png.read_bytes() == reloaded_png.read_bytes()
print(f"plot_displacement_3d byte-identical from reloaded result: {identical}")
original_png.unlink()

# -- 3. Object graph, not copies ---------------------------------------
print("field_results.mesh is results.mesh: "
      f"{restored.field_results.mesh is restored.mesh}")
print("mesh.laminate is results.laminate:  "
      f"{restored.mesh.laminate is restored.laminate}")

# -- The config comes back too -----------------------------------------
print(f"Reloaded config: morphology={restored.config.morphology}, "
      f"amplitude={restored.config.amplitude} mm, nx={restored.config.nx}")
print(f"Saved: {reloaded_png} ({reloaded_png.stat().st_size:,} bytes)")

# The same archive comes out of the command line with:
#     wrinklefe analyze --amplitude 0.366 --wavelength 16 --fe \
#         --save-results 17_run.wfr
