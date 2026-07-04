# WrinkleFE examples

Copy-paste-runnable starting points for the common workflows. Each
script states its expected runtime and output in its header comment and
runs headless (`MPLBACKEND=Agg`); CI executes every script on each push
so they cannot rot.

| Script | Workflow | Runtime |
|--------|----------|---------|
| [`01_basic_knockdown.py`](01_basic_knockdown.py) | Pristine vs wrinkled strength, analytical + FE, wrinkle-profile figure | ~10 s |
| [`02_parametric_sweep.py`](02_parametric_sweep.py) | Amplitude sweep with knockdown-curve plot (`WrinkleAnalysis.parametric_sweep`) | ~1 s |
| [`03_morphology_comparison.py`](03_morphology_comparison.py) | The 5 named morphologies on one laminate (`compare_morphologies`) | ~1 s |
| [`04_czm_delamination.py`](04_czm_delamination.py) | Cohesive-zone delamination run (`enable_czm`) with damage summary | ~10 s |
| [`05_export_roundtrip.py`](05_export_roundtrip.py) | JSON / Abaqus `.inp` / VTK export; open the `.vtk` in ParaView | ~10 s |
| [`06_custom_material.py`](06_custom_material.py) | Defining an `OrthotropicMaterial` not in the preset library | ~1 s |
| [`07_mesh_convergence.py`](07_mesh_convergence.py) | Mesh-convergence study (`mesh_convergence_study` / `wrinklefe converge`) | ~90 s |
| [`08_multi_wrinkle_czm_linkup.py`](08_multi_wrinkle_czm_linkup.py) | Crest-to-crest delamination link-up between adjacent wrinkles (`wrinkles` + `enable_czm`) | ~60 s |

Run any of them from this directory with the package installed
(`pip install -e ..`):

```bash
python 01_basic_knockdown.py
```
