"""Same-instant site pose and twist observations for a fixed-base robot."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.robot import RobotModel, RobotState
from ..dynamics.robot_kinematics import _point_jacobians_at, forward_kinematics
from ..dynamics.rotation import quaternion_matrix


def _local_vector(value, length, name):
    raw = np.asarray(value)
    if raw.shape != (length,) or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real vector with {length} components")
    vector = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite")
    return tuple(map(float, vector))


@dataclass(frozen=True)
class RobotSite:
    """A link-local point in metres and an active site-to-link XYZW rotation.

    A None link index binds the site directly to the world. For USD links with a
    displaced inbound anchor, convert the authored point with authored_to_model_points.
    """

    link_index: int | None
    local_position_m: tuple[float, ...] = (0.0, 0.0, 0.0)
    local_orientation_xyzw: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self):
        index = self.link_index
        if index is not None:
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) or index < 0:
                raise ValueError("site link index must be None or a nonnegative integer")
            object.__setattr__(self, "link_index", int(index))
        position = _local_vector(self.local_position_m, 3, "site local position")
        rotation = _local_vector(self.local_orientation_xyzw, 4, "site local orientation")
        if abs(np.hypot.reduce(rotation) - 1) > 1e-10:
            raise ValueError("site local orientation must be a unit XYZW quaternion")
        object.__setattr__(self, "local_position_m", position)
        object.__setattr__(self, "local_orientation_xyzw", rotation)


@dataclass(frozen=True, eq=False)
class RelativeSiteMeasurement:
    """Target pose and twist expressed in the moving reference site.

    rotation maps target-site axes into reference coordinates. The linear velocity
    is the time derivative of position_m. kernel_stats counts the optional TYX FK
    and Jacobian calls; the final point projection and frame composition use HOST.
    """

    position_m: np.ndarray
    rotation: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    kernel_stats: Mapping[str, object]

    def __post_init__(self):
        for name, shape in (
            ("position_m", (3,)),
            ("rotation", (3, 3)),
            ("linear_velocity_m_s", (3,)),
            ("angular_velocity_rad_s", (3,)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise SolverFailure(f"invalid relative site measurement: {name}")
            array = np.frombuffer(value.tobytes(), dtype=np.float64).reshape(shape)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "kernel_stats", MappingProxyType(dict(self.kernel_stats)))


def measure_relative_site(model, state, reference, target, *, backend="host", cache_root=None):
    """Measure two sites on one supplied q/v snapshot, with no independent sensor clock.

    HOST uses FP64 tree kinematics. The optional interpret, cpu_cpp and cuda modes use
    FP32 TYX FK and both point Jacobians, then compose the readbacks on HOST. A pair
    attached entirely to the world needs no robot kernel on any backend.
    """
    if not isinstance(model, RobotModel) or not isinstance(state, RobotState):
        raise TypeError("site sensing requires a fixed-base RobotModel and RobotState")
    if backend not in ("host", "interpret", "cpu_cpp", "cuda"):
        raise ValueError(f"unsupported robot site backend {backend!r}")
    sites = (reference, target)
    for site in sites:
        if not isinstance(site, RobotSite):
            raise TypeError("reference and target must be RobotSite values")
        if site.link_index is not None and site.link_index >= model.dof_count:
            raise ValueError("site link index outside robot")
    sample = RobotState(state.q, state.v)
    if sample.q.shape != (model.dof_count,):
        raise ValueError("sensor state must match robot DOF count")

    positions = np.array([site.local_position_m for site in sites])
    rotations = quaternion_matrix(np.array([site.local_orientation_xyzw for site in sites]))
    velocities = np.zeros((2, 3))
    spins = np.zeros((2, 3))
    rows = [i for i, site in enumerate(sites) if site.link_index is not None]
    stats = {}
    try:
        with np.errstate(over="raise", invalid="raise"):
            if rows:
                links = np.array([sites[i].link_index for i in rows])
                local = positions[rows]
                if backend == "host":
                    geometry = forward_kinematics(model, sample.q)
                    linear, angular = _point_jacobians_at(model, geometry, links, local)
                else:
                    from ..backends.tyx_robot_math import TyxRobotMath

                    with TyxRobotMath(model, backend, cache_root=cache_root) as kernels:
                        geometry = kernels.geometry(sample.q)
                        linear, angular = kernels.jacobians(sample.q, links, local)
                        stats = kernels.stats()
                frames = geometry.rotation[links]
                positions[rows] = geometry.position[links] + np.einsum("nij,nj->ni", frames, local)
                rotations[rows] = frames @ rotations[rows]
                velocities[rows] = linear @ sample.v
                spins[rows] = angular @ sample.v
            ref = rotations[0].T
            offset = positions[1] - positions[0]
            # The axes used to express offset rotate with the reference site.
            motion = velocities[1] - velocities[0] - np.cross(spins[0], offset)
            return RelativeSiteMeasurement(
                ref @ offset, ref @ rotations[1], ref @ motion, ref @ (spins[1] - spins[0]), stats
            )
    except FloatingPointError as error:
        raise SolverFailure(f"relative site measurement is not representable: {error}") from error
