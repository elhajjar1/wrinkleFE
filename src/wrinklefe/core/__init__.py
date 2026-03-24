"""Core computation modules: materials, laminates, wrinkle geometry, mesh generation."""

from wrinklefe.core.material import OrthotropicMaterial, MaterialLibrary
from wrinklefe.core.transforms import (
    rotation_matrix_3d,
    stress_transformation_3d,
    strain_transformation_3d,
    rotate_stiffness_3d,
    reduced_stiffness_matrix,
    transform_reduced_stiffness,
)
from wrinklefe.core.laminate import Ply, Laminate, LoadState
from wrinklefe.core.wrinkle import (
    WrinkleProfile,
    GaussianSinusoidal,
    RectangularSinusoidal,
    TriangularSinusoidal,
    PureSinusoidal,
    GaussianBump,
    WrinkleSurface3D,
)
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.mesh import WrinkleMesh, MeshData

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
