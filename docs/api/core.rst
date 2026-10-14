wrinklefe.core
==============

Materials
---------

.. automodule:: wrinklefe.core.material
   :members: OrthotropicMaterial, MaterialLibrary
   :show-inheritance:

Constituent micromechanics
--------------------------

.. automodule:: wrinklefe.core.micromechanics
   :members: FiberProperties, MatrixProperties, halpin_tsai,
             e1_rule_of_mixtures, nu12_rule_of_mixtures, e2_halpin_tsai,
             g12_halpin_tsai, nu23_rule_of_mixtures, g23_transverse_isotropy,
             alpha1_schapery, alpha2_schapery
   :show-inheritance:

Laminate and layup
------------------

.. automodule:: wrinklefe.core.laminate
   :members: Laminate, LoadState
   :show-inheritance:

.. automodule:: wrinklefe.core.layup
   :members:
   :show-inheritance:

Wrinkle geometry and morphology
-------------------------------

.. automodule:: wrinklefe.core.wrinkle
   :members:
   :show-inheritance:

.. automodule:: wrinklefe.core.morphology
   :members: WrinklePlacement, WrinkleConfiguration
   :show-inheritance:

Fitting a profile to measured data
----------------------------------

.. automodule:: wrinklefe.core.fit
   :members:
   :show-inheritance:

Mesh
----

.. automodule:: wrinklefe.core.mesh
   :members: WrinkleMesh, MeshData
   :show-inheritance:

Resin pocket
------------

.. automodule:: wrinklefe.core.resin_pocket
   :members: ResinPocketSpec, compute_resin_mask, compute_resin_blend,
             SurfacePocketSpec, compute_surface_resin_blend,
             element_heights, element_height_ratio
   :show-inheritance:

Compaction and fibre-volume-fraction gradient
---------------------------------------------

.. automodule:: wrinklefe.core.compaction
   :members: VfGradientSpec, compute_vf_field, build_vf_materials,
             scale_material_to_vf, resolve_constituents
   :show-inheritance:

Penetration gate
----------------

.. automodule:: wrinklefe.core.penetration_gate
   :members: GateParameters, penetration_gate_kd, angle_floor, position_factor,
             predict_from_geometry, calibrate_gate
   :show-inheritance:

Manufacturing process and validated scope
-----------------------------------------

.. automodule:: wrinklefe.core.process
   :members: ManufacturingProcess, parse_process, ProcessEvidence,
             VALIDATION_EVIDENCE, VALIDATED_PROCESSES, process_in_scope,
             process_scope_caveat, gate_scope_caveat
   :show-inheritance:

Validation margins
------------------

.. automodule:: wrinklefe.core.margins
   :members: COVERAGE, MIN_CASES, SIGMA_FLOOR, N_COMPONENTS,
             UD_DOMINATED_FRACTION, MODEL_PATHS, MARGIN_DEFINITION,
             MarginRegime, ValidationMargin, MixtureFit, MarginResidual,
             zero_ply_fraction, laminate_class, analytical_path, regime_for,
             fit_gaussian_mixture, fit_margin, build_margin_table,
             load_margin_table, lookup_margin, margin_for_results
   :show-inheritance:

Coordinate transforms
---------------------

.. automodule:: wrinklefe.core.transforms
   :members:
   :show-inheritance:
