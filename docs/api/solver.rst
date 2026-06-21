wrinklefe.solver
================

The finite-element solver layer: global assembly, boundary conditions,
the linear static solve and its field-results container, the nonlinear
(Newton-Raphson / arc-length) solvers, and the wrinkle-defect routes
(progressive damage and linearized buckling).

Static solve
------------

.. automodule:: wrinklefe.solver.static
   :members: StaticSolver
   :show-inheritance:

Assembly
--------

.. automodule:: wrinklefe.solver.assembler
   :members: GlobalAssembler
   :show-inheritance:

Boundary conditions
-------------------

Boundary-condition handling and CLT-to-3D load mapping:
:class:`~wrinklefe.solver.boundary.BoundaryCondition` describes a single
BC and :class:`~wrinklefe.solver.boundary.BoundaryHandler` resolves and
applies them to the global system ``K u = F``.

.. autoclass:: wrinklefe.solver.boundary.BoundaryCondition
   :members:
   :show-inheritance:

.. autoclass:: wrinklefe.solver.boundary.BoundaryHandler
   :members:
   :show-inheritance:

Field results
-------------

.. automodule:: wrinklefe.solver.results
   :members: FieldResults
   :show-inheritance:

Progressive damage
------------------

.. automodule:: wrinklefe.solver.progressive_damage
   :members: ProgressiveDamageSolver, ProgressiveDamageResult
   :show-inheritance:

Linear buckling
---------------

.. automodule:: wrinklefe.solver.buckling
   :members: LinearBucklingSolver, BucklingResult, microbuckling_knockdown
   :show-inheritance:

Nonlinear solvers
-----------------

.. automodule:: wrinklefe.solver.nonlinear
   :members: NewtonRaphsonSolver
   :show-inheritance:

.. automodule:: wrinklefe.solver.arclength
   :members: ArcLengthSolver
   :show-inheritance:
