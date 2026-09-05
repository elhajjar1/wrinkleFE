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
| [`08_multi_wrinkle_czm_linkup.py`](08_multi_wrinkle_czm_linkup.py) | Crest-to-crest delamination link-up between adjacent wrinkles (`wrinkles` + `enable_czm`) | ~5 s |
| [`09_acceptance_limit.py`](09_acceptance_limit.py) | Maximum acceptable wrinkle amplitude for a target knockdown (`find_critical_value` / `wrinklefe critical`) | ~5 s |
| [`10_vf_gradient_compaction.py`](10_vf_gradient_compaction.py) | Compaction Vf / ply-thickness gradient between two caul plates (`enable_vf_gradient`) vs the binary surface pockets | ~30 s |
| [`11_penetration_gate.py`](11_penetration_gate.py) | UD (θ, D/T, z) penetration-gate knockdown vs the angle-only model, gated and ungated (`penetration_gate` / `--gate`) | ~1 s |
| [`12_progressive_damage.py`](12_progressive_damage.py) | Load-stepping ply-discount FE to ultimate strength (`enable_progressive_damage` / `--progressive`), with the load-increment table | ~15 s |
| [`13_crest_resin_pocket.py`](13_crest_resin_pocket.py) | Machined crest resin lens on/off, graded vs binary, and a lens-height sweep (`enable_resin_pocket`) | ~20 s |
| [`14_stochastic_knockdown.py`](14_stochastic_knockdown.py) | Measurement uncertainty propagated to percentile knockdowns (`probabilistic_analysis` / `wrinklefe stochastic`) | ~15 s |
| [`15_cure_residual_stress.py`](15_cure_residual_stress.py) | Cure cool-down residual stress on both paths (`delta_T` / `--delta-T`), and why its effect on failure is signed | ~15 s |
| [`transverse_wrinkle_knockdown.py`](transverse_wrinkle_knockdown.py) | Localized (through-width) vs uniform wrinkle knockdown (`transverse_mode`) | ~20 s |

Run any of them from this directory with the package installed
(`pip install -e ..`):

```bash
python 01_basic_knockdown.py
```
