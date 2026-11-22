"""Current-state HOST surface generation feeding one actual TYX physical step.

Meshes are body-local, full pressure assets. This bounded bridge is not CCD or a
device-resident collision pipeline. It never advances coordinates on the host.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..geometry.area_pressure import AreaPressureIdentity, AreaPressureStamp
from ..geometry.area_pressure_intersection import (
    RigidPressureMesh,
    TetraPressureMesh,
    _AreaPressureGeometryInput,
    _build_geometry_inputs,
    _intersect_checked,
)


@dataclass(frozen=True, eq=False)
class BodyAreaPressureAsset:
    identity: AreaPressureIdentity
    pressure_local: TetraPressureMesh
    rigid_local: RigidPressureMesh

    def __post_init__(self):
        if not isinstance(self.identity, AreaPressureIdentity):
            raise ValueError("expected area pressure identity")
        if not isinstance(self.pressure_local, TetraPressureMesh) or not isinstance(
            self.rigid_local, RigidPressureMesh
        ):
            raise ValueError("expected body-local pressure and rigid meshes")
        a, b = self.pressure_local, self.rigid_local
        if a.pressure_domain != "full":
            raise ValueError("scene contact requires a full zero-boundary pressure asset")
        object.__setattr__(
            self,
            "identity",
            AreaPressureIdentity(
                self.identity.compliant_body,
                self.identity.rigid_body,
                self.identity.compliant_material,
                self.identity.rigid_material,
            ),
        )
        object.__setattr__(
            self,
            "pressure_local",
            TetraPressureMesh(
                a.node_ids,
                a.world_vertices,
                a.cells,
                a.nodal_pressure,
                a.cell_ids,
                pressure_domain="full",
            ),
        )
        object.__setattr__(
            self,
            "rigid_local",
            RigidPressureMesh(
                b.node_ids,
                b.world_vertices,
                b.faces,
                b.face_ids,
            ),
        )


def _current_body_poses(runtime):
    """Read fresh TYX FK of the accepted bank, never last_step entry geometry."""
    device, plan = runtime._device, runtime._plan
    old = device.active
    with device.tx.use_runtime(device.runtime):
        args = [device.nworld, device.nbody, plan.nv]
        if plan.kind == 3:
            args.append(len(plan.joint_ids))
        device.geometry[old](
            *args,
            device.parents,
            device.ranges,
            device.body_data,
            device.joints,
            device.joint_data,
            device.vectors,
            device.q[old],
            device.v[old],
            device.motion,
            device.rotation,
            device.axes,
            device.anchors,
            device.work,
            device.work_rotation,
            device.status,
            mode=device.mode,
        )
        device.counts["completed_kernels"] += 1
        device.runtime.sync()
        device.counts["synchronizations"] += 1
        status = device._get(device.status)
        q, v = device._get(device.q[old]), device._get(device.v[old])
        motion, rotation = device._get(device.motion), device._get(device.rotation)
        clock = device._get(device.time[old])
    if np.any(status):
        raise RuntimeError("current TYX body geometry rejected")
    poses = {}
    for index, key in enumerate(runtime._world_keys):
        state = runtime._data[key]
        if (
            not np.array_equal(q[index, : plan.nq], state.qpos)
            or not np.array_equal(v[index, : plan.nv], state.qvel)
            or float(clock[index, 0]) - float(clock[index, 1]) != state.time
        ):
            raise RuntimeError("current device bank differs from accepted state")
        poses[key] = (motion[index, :, 0, :], rotation[index])
    return poses


def _checked_poses(poses, runtime):
    if not isinstance(poses, dict) or set(poses) != set(runtime._world_keys):
        raise RuntimeError("body poses must cover the complete runtime batch")
    result = {}
    nb = runtime._model.nbody
    for key in runtime._world_keys:
        position, rotation = poses[key]
        p, r = np.asarray(position, dtype=np.float64), np.asarray(rotation, dtype=np.float64)
        if p.shape != (nb, 3) or r.shape != (nb, 3, 3):
            raise RuntimeError("invalid body pose shape")
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            raise RuntimeError("nonfinite body pose")
        if not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), rtol=0, atol=3e-5):
            raise RuntimeError("body rotations are not orthonormal")
        if np.any(np.linalg.det(r) <= 0):
            raise RuntimeError("body rotation reverses orientation")
        result[key] = (p.copy(), r.copy())
    return result


def _check_intersector(intersector, mode):
    if intersector is not None:
        from .tyx_area_pressure_intersection import TyxAreaPressureIntersector

        if not isinstance(intersector, TyxAreaPressureIntersector):
            raise TypeError("expected TyxAreaPressureIntersector")
        if intersector._resources.state != "OPEN" or intersector.mode != mode:
            raise ValueError("intersector must be open and match the physics mode")


def _checked_publication_frame(value):
    if type(value) is not str or value not in ("world", "rigid-origin"):
        raise ValueError("publication_frame must be 'world' or 'rigid-origin'")
    return value


def _geometry_requests(runtime, assets_by_world, publication_frame="world"):
    """Collect owned current meshes before any intersection or dynamics work."""
    if set(assets_by_world) != set(runtime._world_keys):
        raise ValueError("assets must cover every world in the current bucket")
    poses = _checked_poses(_current_body_poses(runtime), runtime)
    requests_by_world = {}
    for key in runtime._world_keys:
        position, rotation = poses[key]
        stamp = AreaPressureStamp(
            str(key),
            runtime._ticks[key],
            runtime._data[key].time,
            runtime.area_pressure_configuration_id(key),
        )
        requests = []
        for asset, body_ids in zip(assets_by_world[key], runtime._plan.area_body_ids, strict=True):
            ba, bb = map(int, body_ids)
            a, b = asset.pressure_local, asset.rigid_local
            pressure = dict(
                node_ids=a.node_ids,
                world_vertices=a.world_vertices @ rotation[ba].T + position[ba],
                cells=a.cells,
                nodal_pressure=a.nodal_pressure,
                cell_ids=a.cell_ids,
                pressure_domain="full",
            )
            rigid = dict(
                node_ids=b.node_ids,
                world_vertices=b.world_vertices @ rotation[bb].T + position[bb],
                faces=b.faces,
                face_ids=b.face_ids,
            )
            origin = position[bb] if publication_frame == "rigid-origin" else None
            requests.append(
                _AreaPressureGeometryInput(
                    pressure, rigid, asset.identity, stamp, origin_world=origin
                )
            )
        requests_by_world[key] = tuple(requests)
    return requests_by_world


def _intersect_requests(requests_by_world, intersector=None):
    requests = tuple(request for group in requests_by_world.values() for request in group)
    if intersector is None:
        checked = tuple(_build_geometry_inputs(requests))
        results = []
        for request, (pressure, rigid, identity, stamp) in zip(requests, checked, strict=True):
            frame = {} if request.origin_world is None else {"origin_world": request.origin_world}
            results.append(
                _intersect_checked(pressure, rigid, identity=identity, stamp=stamp, **frame)
            )
        results = tuple(results)
    else:
        results = intersector._intersect_geometry_inputs(requests)
    generated, surfaces, offset = {}, {}, 0
    for key, group in requests_by_world.items():
        intersections = results[offset : offset + len(group)]
        offset += len(group)
        generated[key] = intersections
        surfaces[key] = tuple(item.surface for item in intersections)
    return MappingProxyType(generated), surfaces


def _prepare_geometry(runtime, assets_by_world, intersector=None, publication_frame="world"):
    """Prepare all current input geometry without advancing accepted state."""
    return _intersect_requests(
        _geometry_requests(runtime, assets_by_world, publication_frame), intersector
    )


class MuJoCoTyxAreaPressureScene:
    """Immutable local assets, fresh geometry, existing atomic dynamics owner."""

    def __init__(self, runtime, assets, *, intersector=None, publication_frame="world"):
        self._publication_frame = _checked_publication_frame(publication_frame)
        if not isinstance(assets, tuple) or not 1 <= len(assets) <= 16:
            raise ValueError("assets must contain 1..16 declared pairs")
        owned = []
        for asset in assets:
            if not isinstance(asset, BodyAreaPressureAsset):
                raise ValueError("expected BodyAreaPressureAsset")
            owned.append(
                BodyAreaPressureAsset(asset.identity, asset.pressure_local, asset.rigid_local)
            )
        with runtime._lock:
            runtime._check_open()
            _check_intersector(intersector, getattr(runtime, "_mode", None))
            self._intersector = intersector
            names = tuple((a.identity.compliant_body, a.identity.rigid_body) for a in owned)
            if names != getattr(runtime._plan, "area_pair_names", None):
                raise ValueError("assets must match all declared area pairs in order")
            if runtime._area_friction_laws and tuple(a.identity for a in owned) != tuple(
                law.identity for law in runtime._area_friction_laws
            ):
                raise ValueError("area friction requires complete body/material identities")
            self.runtime = runtime
            self._assets = tuple(owned)
            self._empty = MappingProxyType({})
            self._last_geometry = self._empty
            self._geometry_revision = None

    @property
    def publication_frame(self):
        """Origin for publishing clipped triangles. Solver points stay in world coordinates."""
        return self._publication_frame

    @property
    def assets(self):
        return self._assets

    @property
    def last_geometry(self):
        """Last accepted input geometry; external owner revisions invalidate it."""
        with self.runtime._lock:
            if self.runtime.checkpoint_revision != self._geometry_revision:
                return self._empty
            return self._last_geometry

    def step(self, h, *, expected_revision):
        runtime = self.runtime
        with runtime._lock:
            runtime._check_open()
            if (
                type(expected_revision) is not int
                or expected_revision != runtime.checkpoint_revision
            ):
                raise ValueError("stale area scene revision")
            if not np.isfinite(h) or float(h) != runtime._timestep:
                raise ValueError("scene timestep must equal the compiled timestep")
            publication, surfaces = _prepare_geometry(
                runtime,
                {key: self._assets for key in runtime._world_keys},
                self._intersector,
                self.publication_frame,
            )
            # Prepare all metadata before physics; only two non-failing pointer
            # assignments remain after the runtime's physical commit.
            accepted = runtime.step_area_pressure(h, surfaces, expected_revision=expected_revision)
            self._last_geometry = publication
            self._geometry_revision = runtime._checkpoint_revision
            return accepted

    def observe(self, keys=None):
        """Historical accepted loads and endpoint state; no geometry recomputation."""
        return self.runtime.area_pressure_snapshot(keys)

    def checkpoint(self):
        from .mujoco_area_pressure_state import MuJoCoTyxAreaSceneCheckpoint

        with self.runtime._lock:
            return MuJoCoTyxAreaSceneCheckpoint.make(self._assets, self.runtime.checkpoint())

    def restore(self, checkpoint, *, expected_revision):
        from .mujoco_area_pressure_state import MuJoCoTyxAreaSceneCheckpoint, asset_digest

        with self.runtime._lock:
            self.runtime._check_open()
            if type(checkpoint) is not MuJoCoTyxAreaSceneCheckpoint:
                raise ValueError("expected area scene checkpoint")
            if checkpoint.asset_sha256 != asset_digest(self._assets):
                raise ValueError("area scene asset binding differs")
            self.runtime.restore(checkpoint.runtime_checkpoint, expected_revision=expected_revision)
            self._last_geometry = self._empty
            self._geometry_revision = None

    def reset(self, keys=None):
        with self.runtime._lock:
            before = self.runtime._checkpoint_revision
            self.runtime.reset(keys)
            if self.runtime._checkpoint_revision != before:
                self._last_geometry = self._empty
                self._geometry_revision = None

    def clone(self, world_keys=None):
        with self.runtime._lock:
            cloned = self.runtime.clone(world_keys)
            try:
                return type(self)(
                    cloned,
                    self._assets,
                    intersector=self._intersector,
                    publication_frame=self.publication_frame,
                )
            except BaseException as error:
                try:
                    cloned.close()
                except BaseException as cleanup:
                    self.runtime._checkpoint_retired.append(cloned)
                    raise BaseExceptionGroup(
                        "area scene clone failed and cleanup needs retry on source.close()",
                        [error, cleanup],
                    ) from None
                raise
