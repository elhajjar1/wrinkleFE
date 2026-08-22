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

Mesh
----

.. automodule:: wrinklefe.core.mesh
   :members: WrinkleMesh, MeshData
   :show-inheritance:

Resin pocket
------------

.. automodule:: wrinklefe.core.resin_pocket
   :members: ResinPocketSpec, compute_resin_mask, compute_resin_blend
   :show-inheritance:

Penetration gate
----------------

.. automodule:: wrinklefe.core.penetration_gate
   :members: GateParameters, penetration_gate_kd, angle_floor, position_factor,
             predict_from_geometry, calibrate_gate
   :show-inheritance:

Coordinate transforms
---------------------

.. automodule:: wrinklefe.core.transforms
   :members:
   :show-inheritance:
