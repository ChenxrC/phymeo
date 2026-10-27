"""Fixed, independently reproducible cloth/dynamic-sphere benchmark scene."""

from dataclasses import dataclass

import numpy as np

from ..core.coupled import CoupledInputs, CoupledState
from ..core.rigid import RigidBodyModel, RigidBodyState
from ..core.validation import finite_real
from ..geometry.sphere import SphereShapes
from .cloth import ClothPatch, cloth_grid


@dataclass(frozen=True, eq=False)
class RigidClothScene:
    cloth: ClothPatch
    rigid_model: RigidBodyModel
    initial_state: CoupledState
    inputs: CoupledInputs
    spheres: SphereShapes
    pinned: np.ndarray
    config: dict

    @property
    def particle_model(self):
        return self.cloth.model

    @property
    def constraints(self):
        return self.cloth.constraints


def rigid_cloth_scene(dtype=np.float64, *, sphere_mass=0.05):
    """Seven-by-seven pinned stretch/shear patch and one solid sphere, in SI."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("scene dtype must be float32 or float64")
    mass = finite_real(sphere_mass, "sphere_mass")
    if mass <= 0:
        raise ValueError("sphere_mass must be positive")
    pins = np.array([0, 6, 42, 48], dtype=np.int64)
    pins.flags.writeable = False
    patch = cloth_grid(
        7,
        7,
        spacing=0.1,
        density=0.4,
        structural_compliance=1e-4,
        shear_compliance=1e-4,
        pinned=pins,
        origin=(-0.3, 0.6, -0.3),
        dtype=dtype,
    )
    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        inverse_mass = np.array([1 / mass], dtype=dtype)
        inertia = np.full((1, 3), 2 * mass * 0.18**2 / 5, dtype=dtype)
    model = RigidBodyModel(inverse_mass, inertia)
    state = RigidBodyState(
        np.array([[0.02, 0.30, 0]], dtype=dtype),
        np.array([[0, 0, 0, 1]], dtype=dtype),
        np.array([[0, 2, 0]], dtype=dtype),
        np.zeros((1, 3), dtype=dtype),
    )
    config = {
        "scene": "pinned_cloth_dynamic_sphere_v1",
        "dtype": dtype.name,
        "rows": 7,
        "columns": 7,
        "spacing_m": 0.1,
        "density_kg_m2": 0.4,
        "structural_compliance_m_N": 1e-4,
        "shear_compliance_m_N": 1e-4,
        "cloth_origin_m": [-0.3, 0.6, -0.3],
        "pinned": pins.tolist(),
        "sphere_mass_kg": mass,
        "sphere_radius_m": 0.18,
        "sphere_position_m": [0.02, 0.30, 0],
        "sphere_velocity_mps": [0, 2, 0],
        "gravity_mps2": [0, -9.81, 0],
        "contact": "discrete_vertices_frictionless_e0",
    }
    return RigidClothScene(
        patch,
        model,
        CoupledState(patch.initial_state, state),
        CoupledInputs.zeros(49, 1, dtype=dtype),
        SphereShapes(np.array([0.18], dtype=dtype)),
        pins,
        config,
    )
