"""Core computation modules: materials, laminates, wrinkle geometry, mesh generation."""

from wrinklefe.core.laminate import Laminate, LoadState, Ply
from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial
from wrinklefe.core.mesh import MeshData, WrinkleMesh
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.transforms import (
    reduced_stiffness_matrix,
    rotate_stiffness_3d,
    rotation_matrix_3d,
    strain_transformation_3d,
    stress_transformation_3d,
    transform_reduced_stiffness,
)
from wrinklefe.core.wrinkle import (
    GaussianBump,
    GaussianSinusoidal,
    PureSinusoidal,
    RectangularSinusoidal,
    TriangularSinusoidal,
    WrinkleProfile,
    WrinkleSurface3D,
)

__all__ = [
    "OrthotropicMaterial",
    "MaterialLibrary",
    "rotation_matrix_3d",
    "stress_transformation_3d",
    "strain_transformation_3d",
    "rotate_stiffness_3d",
    "reduced_stiffness_matrix",
    "transform_reduced_stiffness",
    "Ply",
    "Laminate",
    "LoadState",
    "WrinkleProfile",
    "GaussianSinusoidal",
    "RectangularSinusoidal",
    "TriangularSinusoidal",
    "PureSinusoidal",
    "GaussianBump",
    "WrinkleSurface3D",
    "WrinkleConfiguration",
    "WrinkleMesh",
    "MeshData",
]
