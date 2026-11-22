"""Opt-in resident Vulkan ABI: indexed assets once, rigid matrices per capture."""
import ctypes as ct
import math

from . import BackendUnavailableError, UnsupportedRequestError
from ._native_device import NativeDevice, Parameters, validate_parameters


class ResidentVertex(ct.Structure):
    _fields_ = [('position', ct.c_float * 4), ('normal', ct.c_float * 4)]


class ResidentMesh(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in
                ('first_vertex', 'vertex_count', 'first_index', 'index_count')]


class ResidentInstance(ct.Structure):
    _fields_ = [('mesh_index', ct.c_uint32), ('instance_id', ct.c_uint32),
                ('reserved', ct.c_uint32 * 2), ('color_roughness', ct.c_float * 4),
                ('material', ct.c_float * 4)]


class ResidentInfo(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        'resident_abi_version', 'max_vertices', 'max_indices', 'max_meshes',
        'max_instances', 'max_bvh_nodes', 'max_instanced_triangles',
        'vertex_count', 'index_count', 'mesh_count', 'instance_count', 'bvh_nodes',
        'instanced_triangles', 'loaded', 'poisoned', 'reserved')] + [
        (name, ct.c_uint64) for name in ('upload_calls', 'geometry_upload_bytes',
        'transform_upload_bytes', 'parameter_upload_bytes', 'captures',
        'shadow_node_budget', 'target_rebuilds', 'host_transform_input_bytes')]


# Host allocation bounds match the resident ABI's absolute ceiling. Device
# storage-buffer limits can lower them; never increase them based on input size.
_LIMITS = dict(max_vertices=1048576, max_indices=1572864, max_meshes=1024,
               max_instances=1024, max_instanced_triangles=1048576)


def _limits(info=None):
    info = info or {}
    return {key: min(value, info.get('resident_' + key, value))
            for key, value in _LIMITS.items()}


def pack_scene(scene, decoded, info=None):
    """One immutable asset table per session; instances share mesh-local indices."""
    limits = _limits(info)
    counts = dict(max_vertices=sum(len(decoded[m.id].positions) for m in scene.meshes),
                  max_indices=sum(3 * len(decoded[m.id].triangles) for m in scene.meshes),
                  max_meshes=len(scene.meshes), max_instances=len(scene.instances),
                  max_instanced_triangles=sum(len(decoded[i.mesh_id].triangles)
                                             for i in scene.instances))
    for key, count in counts.items():
        if count > limits[key]:
            raise UnsupportedRequestError(f'resident {key} capacity exceeded: {count} > {limits[key]}')
    vertices = (ResidentVertex * counts['max_vertices'])()
    indices = (ct.c_uint32 * counts['max_indices'])()
    meshes = (ResidentMesh * len(scene.meshes))()
    instances = (ResidentInstance * len(scene.instances))()
    vertex_offset = index_offset = 0
    mesh_slots = {}
    for slot, asset in enumerate(scene.meshes):
        mesh = decoded[asset.id]
        mesh_slots[asset.id] = slot
        if len(mesh.normals) != len(mesh.positions):
            raise ValueError('resident mesh normals do not cover vertices')
        meshes[slot] = ResidentMesh(vertex_offset, len(mesh.positions), index_offset,
                                    3 * len(mesh.triangles))
        for index, (position, normal) in enumerate(zip(mesh.positions, mesh.normals, strict=True)):
            length = math.hypot(*normal)
            if not math.isfinite(length) or length == 0:
                raise ValueError('resident mesh normal is not finite and nonzero')
            vertex = vertices[vertex_offset + index]
            vertex.position[:] = (*position, 1.)
            vertex.normal[:] = (*(value / length for value in normal), 0.)
            if not all(math.isfinite(x) for x in (*vertex.position, *vertex.normal)):
                raise ValueError('resident vertex FP32 conversion is nonfinite')
        for triangle in mesh.triangles:
            if len(triangle) != 3 or any(type(i) is not int or not 0 <= i < len(mesh.positions) for i in triangle):
                raise ValueError('resident triangle index outside its mesh')
            indices[index_offset:index_offset+3] = triangle
            index_offset += 3
        vertex_offset += len(mesh.positions)
    materials = {material.id: material for material in scene.materials}
    for slot, instance in enumerate(scene.instances):
        target = instances[slot]
        material = materials[instance.material_id]
        target.mesh_index, target.instance_id = mesh_slots[instance.mesh_id], instance.id
        target.color_roughness[:] = (*material.base_color, material.roughness)
        target.material[:] = (material.metallic, 0., 0., 0.)
    return vertices, indices, meshes, instances


class ResidentDevice(NativeDevice):
    _shader_prefix = 'resident'

    def __init__(self, *args):
        self._resident_info = {}
        self._uploaded = False
        self._instance_count = 0
        super().__init__(*args)

    def _bind(self):
        super()._bind()
        if tuple(map(ct.sizeof, (ResidentVertex, ResidentMesh, ResidentInstance, ResidentInfo))) != (32,16,48,128):
            raise BackendUnavailableError('resident C ABI structure layout mismatch')
        lib = self._lib
        lib.pr_resident_abi_version.argtypes = []
        lib.pr_resident_abi_version.restype = ct.c_uint32
        lib.pr_resident_create.argtypes = lib.pr_create.argtypes
        lib.pr_resident_create.restype = ct.c_void_p
        lib.pr_resident_upload.argtypes = [ct.c_void_p, ct.POINTER(ResidentVertex), ct.c_uint32,
            ct.POINTER(ct.c_uint32), ct.c_uint32, ct.POINTER(ResidentMesh), ct.c_uint32,
            ct.POINTER(ResidentInstance), ct.c_uint32]
        lib.pr_resident_upload.restype = ct.c_int
        lib.pr_resident_render.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_uint32,
            ct.POINTER(Parameters), ct.c_uint32, ct.c_uint32, ct.POINTER(ct.c_uint8), ct.c_size_t]
        lib.pr_resident_render.restype = ct.c_int
        lib.pr_resident_info.argtypes = [ct.c_void_p, ct.POINTER(ResidentInfo)]
        lib.pr_resident_info.restype = ct.c_int

    def _create(self, *args):
        if self._lib.pr_resident_abi_version() != 1:
            raise BackendUnavailableError('unsupported resident Vulkan C ABI version')
        return self._lib.pr_resident_create(*args)

    def info(self):
        result = super().info()
        if self._handle is not None:
            value = ResidentInfo()
            if self._lib.pr_resident_info(self._handle, ct.byref(value)):
                raise RuntimeError(self._error())
            self._resident_info = {(name if name == 'resident_abi_version' else 'resident_' + name): int(getattr(value, name))
                                   for name, _ in ResidentInfo._fields_}
        result.update(self._resident_info)
        return result

    def upload(self, vertices, indices, meshes, instances):
        if self._handle is None:
            raise RuntimeError('native device is closed')
        if self._uploaded:
            raise RuntimeError('resident static geometry has already been uploaded')
        if self._lib.pr_resident_upload(self._handle, vertices, len(vertices), indices,
                len(indices), meshes, len(meshes), instances, len(instances)):
            raise RuntimeError(self._error())
        self._uploaded = True
        self._instance_count = len(instances)
        self.info()

    def render(self, matrices, parameters, width, height):
        if self._handle is None or not self._uploaded:
            raise RuntimeError('resident device is closed or has no uploaded scene')
        limits = self._last_info
        if (type(width) is not int or type(height) is not int or
            not 0 < width <= min(limits['max_width'], 2**32-1) or
            not 0 < height <= min(limits['max_height'], 2**32-1) or width*height > 4096*4096):
            raise ValueError('native capture extent or pixel budget exceeded')
        if len(matrices) != 16*self._instance_count or not all(math.isfinite(x) for x in matrices):
            raise ValueError('resident instance matrices are incomplete or nonfinite FP32')
        validate_parameters(parameters)
        if parameters.clipping[0] != width or parameters.clipping[1] != height:
            raise ValueError('native clipping extent mismatch')
        if parameters.counts[3] != self._instance_count:
            raise ValueError('resident parameter instance count mismatch')
        output = (ct.c_uint8 * (width*height*13))()
        if self._lib.pr_resident_render(self._handle, matrices, self._instance_count,
                ct.byref(parameters), width, height, output, ct.sizeof(output)):
            raise RuntimeError(self._error())
        return bytes(output)

    def close(self):
        if self._handle is None:
            return
        # Preserve resident counters at the last live point, without allowing a
        # failed diagnostic read to prevent the shared real Vulkan destruction.
        diagnostic_error = None
        try:
            self.info()
        except BaseException as error:
            diagnostic_error = error
        try:
            super().close()
        except BaseException as cleanup:
            if diagnostic_error is not None:
                raise BaseExceptionGroup('resident diagnostics and cleanup failed',
                                         [diagnostic_error, cleanup])
            raise
        if diagnostic_error is not None:
            raise diagnostic_error
