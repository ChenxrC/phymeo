"""Prepared device ray queries; geometry is public, pose storage is borrowed."""

import numpy as np
import tyx as tx

from ..geometry.ray_surface import RaySurface


def _box_mesh(size):
    vertices = tuple(
        (x * size[0], y * size[1], z * size[2]) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)
    )
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    faces = tuple(t for a, b, c, d in quads for t in ((a, b, c), (a, c, d)))
    return vertices, faces


def _patches(surface, index):
    kind, size = surface.kind, surface.size
    data = np.zeros(15, np.float32)
    if kind == "plane":
        data[5] = 1
        return [(3, index, data)]
    if kind == "sphere":
        data[12] = size[0]
        return [(0, index, data)]
    if kind == "capsule":
        data[12], data[13] = size[0], 2 * size[1]
        data[2], data[5] = -size[1], -1
        lower = data.copy()
        data[2], data[5] = size[1], 1
        upper = data.copy()
        data[2] = -size[1]
        return [(1, index, lower), (1, index, upper), (2, index, data)]
    if kind in ("cylinder", "ellipsoid"):
        data[3:12] = np.eye(3, dtype=np.float32).reshape(-1)
        data[12 : 12 + len(size)] = size
        return [(5 if kind == "cylinder" else 7, index, data)]
    return []


class PreparedRayQuery:
    """Own static local geometry/BVHs and scratch; borrow two endpoint pose banks.

    Outputs are packed selected-row/ray order: labels index `surfaces`, hits are
    [distance_m, world_normal_xyz]. Misses are +inf/-1/zero. Equal distances keep
    the first surface. Meshes use exact triangles; primitives use analytic roots.
    """

    contract_version = 1

    def __init__(self, storage, surfaces, bodies, max_rays, nworld, bank):
        storage.require_valid()
        surfaces, bodies = tuple(surfaces), tuple(bodies)
        if (
            len(surfaces) != len(bodies)
            or any(not isinstance(s, RaySurface) for s in surfaces)
            or len({s.name for s in surfaces}) != len(surfaces)
        ):
            raise ValueError("ray scene requires unique named RaySurface values and body bindings")
        fields = storage.fields
        if "body_position" not in fields or "body_rotation_wxyz" not in fields:
            raise ValueError("prepared rays require endpoint body position/rotation storage")
        shape = fields["body_position"].shape
        if (
            len(shape) != 3
            or shape[0] != nworld
            or shape[2] != 3
            or fields["body_rotation_wxyz"].shape != (*shape[:2], 4)
            or any(type(i) is not int or not 0 <= i < shape[1] for i in bodies)
        ):
            raise ValueError("ray body bindings differ from pose storage")
        if type(max_rays) is not int or not 1 <= max_rays <= 2**20:
            raise ValueError("max_rays must be a positive integer up to one megapixel")
        if nworld * max_rays > 2**31 - 1:
            raise ValueError("prepared ray capacity exceeds int32 indexing")
        self.storage, self.surfaces, self.nworld = storage, surfaces, nworld
        self.max_rays, self.bank, self.closed = max_rays, bank, False
        self.resources, self.meshes = [], []
        self.query_count = self.dispatched_rays = 0
        self.bodies = self._array(np.asarray(bodies, np.int32), tx.i32)
        self.local_pose = self._array(
            np.asarray([(*s.position, *s.rotation_xyzw) for s in surfaces], np.float32).reshape(
                -1, 7
            ),
            tx.f32,
        )
        patches, ranges, handles, shared_meshes = [], [], [], {}
        for index, surface in enumerate(surfaces):
            begin = len(patches)
            patches.extend(_patches(surface, index))
            ranges.append((begin, len(patches)))
            handle = 0
            if surface.kind in ("box", "mesh"):
                vertices, faces = (
                    _box_mesh(surface.size)
                    if surface.kind == "box"
                    else (surface.vertices, surface.triangles)
                )
                key = (vertices, faces)
                mesh = shared_meshes.get(key)
                if mesh is None:
                    points = self._array(np.asarray(vertices, np.float32), tx.vec3f)
                    indices = self._array(np.asarray(faces, np.int32).reshape(-1), tx.i32)
                    mesh = object.__new__(tx.Mesh)
                    self.meshes.append(mesh)  # retain before construction launches work
                    mesh.__init__(points, indices)
                    shared_meshes[key] = mesh
                handle = int(mesh.id)
            handles.append(handle)
        self.ranges = self._array(np.asarray(ranges, np.int32).reshape(-1, 2), tx.i32)
        self.handles = self._array(np.asarray(handles, np.uint64), tx.u64)
        self.tags = self._array(
            np.asarray([(k, i) for k, i, _ in patches], np.int32).reshape(-1, 2), tx.i32
        )
        self.geom = self._array(
            np.asarray([d for _, _, d in patches], np.float32).reshape(-1, 15), tx.f32
        )
        self.labels = self._empty((nworld * max_rays,), tx.i32)
        self.hits = self._empty((nworld * max_rays, 4), tx.f32)
        self.errors = self._array(np.zeros(1, np.int32), tx.i32)
        self.poses = tuple(
            (storage.borrow("body_position", i), storage.borrow("body_rotation_wxyz", i))
            for i in (0, 1)
        )
        self.native_error = storage.borrow("execution_error", 0)

    def _empty(self, shape, dtype):
        array = tx.empty(shape, dtype, device=self.storage.device)
        self.resources.append(array)
        return array

    def _array(self, value, dtype):
        shape = (len(value),) if dtype == tx.vec3f else value.shape
        array = self._empty(shape, dtype)
        array.assign(value, stream=self.storage.stream)
        return array

    def sample(self, selected, count, gate, origins, directions, *, rows, near=0.0, far=1.0e6):
        """Enqueue unit-direction rays for selected worlds, without copying or waiting.

        Caller prevalidates unique IDs/count==rows in stream order, retains input
        arrays, prevents overlap with plan resources and concurrent mutation/close.
        Read errors after completion before publishing hits. Inputs have shape
        [capacity >= rows, rays_per_world, 3] and world-local physical coordinates (no display
        world offsets). Every ray is computed only for its selected world.
        """
        self.storage.require_valid()
        if self.closed:
            raise RuntimeError("prepared ray query is closed")
        if type(rows) is not int or not 0 <= rows <= self.nworld:
            raise ValueError("ray rows exceed prepared capacity")
        if not np.isfinite([near, far]).all() or not 0 <= near < far <= np.finfo(np.float32).max:
            raise ValueError("ray interval requires finite 0 <= near < far")
        shape = tuple(origins.shape)
        if (
            len(shape) != 3
            or not rows <= shape[0] <= self.nworld
            or shape[2] != 3
            or not 1 <= shape[1] <= self.max_rays
            or tuple(directions.shape) != shape
        ):
            raise ValueError("ray origins/directions require prepared [rows,rays,3] shape")
        rays = shape[1]
        for array, dtype, strides, minimum in (
            (selected, tx.i32, (4,), rows),
            (count, tx.i32, (4,), 1),
            (gate, tx.i32, (4,), 1),
            (origins, tx.f32, (rays * 12, 12, 4), rows * rays * 3),
            (directions, tx.f32, (rays * 12, 12, 4), rows * rays * 3),
        ):
            if (
                array.dtype != dtype
                or str(array.device) != self.storage.device
                or tuple(array.strides) != strides
                or array.size < minimum
            ):
                raise ValueError("ray inputs require same-device contiguous typed arrays")
        from ._prepared_ray_kernels import query
        from .tyx_raycast import _clip

        stream = self.storage.stream
        near, far = _clip(float(near), float(far), 1.0)
        self.errors.zero_(stream=stream)
        if rows:
            query(
                selected,
                count,
                gate,
                self.native_error,
                self.bodies,
                self.local_pose,
                self.ranges,
                self.handles,
                self.tags,
                self.geom,
                *self.poses[self.bank()],
                origins,
                directions,
                self.labels,
                self.hits,
                self.errors,
                rows,
                rays,
                float(near),
                float(far),
                mode="cuda",
                stream=stream,
            )
        self.query_count += 1
        self.dispatched_rays += rows * rays

    def costs(self):
        return dict(
            queries=self.query_count,
            dispatched_rays=self.dispatched_rays,
            owned_array_bytes=sum(a.storage.nbytes for a in self.resources),
            shared_mesh_count=len(self.meshes),
            mesh_internal_bytes="TYX Mesh/BVH owned, additional",
            scene_rebuilds=1,
            pose_upload_bytes=0,
            host_snapshot_bytes=0,
        )

    def close(self):
        if self.resources or self.meshes:
            self.storage.stream.synchronize()
        self.resources.clear()
        self.meshes.clear()
        self.poses = ()
        self.bodies = self.local_pose = self.ranges = self.handles = self.tags = self.geom = None
        self.labels = self.hits = self.errors = None
        self.closed = True
