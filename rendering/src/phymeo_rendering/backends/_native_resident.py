"""Opt-in resident Vulkan ABI: indexed assets once, rigid matrices per capture."""
import ctypes as ct
import math
import hashlib

from . import BackendUnavailableError, UnsupportedRequestError
from ._native_studio import NativeStudio
from ._native_texture import ResidentTexture, ResidentUV, ResidentTextureInfo, TextureUpload
from ._native_environment import ResidentEnvironment, ResidentEnvironmentInfo, EnvironmentUpload
from ._native_device import NativeDevice, Parameters, _Info, validate_parameters
from ._native_pixels import _image_planes
from ..outputs import Channel


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


class ResidentTelemetry(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in
        ('telemetry_abi_version', 'device_luid_valid', 'device_node_mask', 'reserved')] + [
        ('device_luid', ct.c_uint8 * 8), ('reserved_identity', ct.c_uint8 * 8)] + [
        (name, ct.c_uint64) for name in ('device_local_bytes_current', 'device_local_bytes_peak',
        'device_local_bytes_allocated', 'host_visible_bytes_current', 'host_visible_bytes_peak',
        'host_visible_bytes_allocated', 'readback_bytes', 'fence_wait_wall_ns', 'fence_wait_calls',
        'submissions', 'allocation_count', 'free_count')]


class ResidentProfile(ct.Structure):
    _fields_ = [(name,ct.c_uint32) for name in
        ('profile_abi_version','timestamp_supported','timestamp_valid_bits','tlas_node_count')] + [
        ('timestamp_period_ns',ct.c_double)] + [(name,ct.c_uint64) for name in
        ('captures','host_prepare_ns','host_upload_ns','host_record_ns','host_submit_ns','host_wait_ns',
         'host_readback_ns','gpu_render_ns','gpu_copy_ns','gpu_total_ns','tlas_upload_bytes',
         'timestamp_samples','reserved')]


class ResidentReadbackInfo(ct.Structure):
    _fields_ = [(name,ct.c_uint32) for name in
        ('readback_abi_version','memory_type_index','memory_property_flags','persistent_mapped','last_channel_mask')] + [
        ('reserved',ct.c_uint32*3)] + [(name,ct.c_uint64) for name in
        ('capacity_bytes','copied_bytes','map_calls','unmap_calls')]


class ResidentDevice(NativeDevice):
    _shader_prefix = 'resident'

    def __init__(self, *args, area_light_grid=2, secondary_area_light_grid=None, normal_maps=False, reflection=False, indirect=False, progressive=False, spatial=False, settling=False, shared_cuda_device=None):
        if type(area_light_grid) is not int or area_light_grid not in (2, 4, 8):
            raise ValueError('area_light_grid must be 2, 4 or 8')
        if type(normal_maps) is not bool:
            raise TypeError('normal_maps must be bool')
        if type(reflection) is not bool: raise TypeError('reflection must be bool')
        if type(indirect) is not bool:raise TypeError('indirect must be bool')
        self._indirect_requested=indirect
        if type(progressive) is not bool: raise TypeError("progressive must be bool")
        self._progressive_requested=progressive
        if type(spatial) is not bool:
            raise TypeError('spatial must be bool')
        self._spatial_requested = spatial
        if type(settling) is not bool:
            raise TypeError('settling must be bool')
        self._settling_requested = settling
        self._reflection_requested=reflection
        self._normal_maps_requested = normal_maps
        self._area_light_grid = area_light_grid
        from ._native_secondary import secondary_grid
        self._secondary_area_light_grid = secondary_grid(area_light_grid, secondary_area_light_grid, resident=True,
            shadow_mode='ray_query' if self._shader_prefix=='resident_rayquery' else 'software')
        self._shared_cuda_device = shared_cuda_device
        self._shared_exchange = None
        self._shared_identity = None
        if shared_cuda_device is not None:
            if type(shared_cuda_device) is not int or shared_cuda_device < 0:
                raise ValueError('shared_cuda_device must be a nonnegative CUDA ordinal')
            if (self._shader_prefix != 'resident' or area_light_grid != 2 or
                    any((normal_maps, reflection, indirect, progressive, spatial, settling))):
                raise UnsupportedRequestError('shared matrices require the standard resident software raster shaders')
            from ._native_shared_matrices import cuda_identity
            self._shared_identity = cuda_identity(shared_cuda_device)
        self._secondary_budget_marker = None
        self._resident_info = {}
        self._telemetry = None
        self._profile = None
        self._profile_function = None
        self._telemetry_function = None
        self._resident_destroy = None
        self._studio_function = None
        self._has_captured = False
        self._render_planes_function = None
        self._readback_function = None
        self._readback = None
        self._output_buffer = None
        self._output_capacity = 0
        self._texture_function = None
        self._surface_maps_function = None
        self._texture_info_function = None
        self._texture_info = None
        self._texture_abi_version = None
        self._textures_set = False
        self._environment_function = None
        self._environment_info_function = None
        self._environment_info = None
        self._environment_abi_version = None
        self._environment_set = False
        self._uploaded = False
        self._instance_count = 0
        super().__init__(*args)

    def _shader_filename(self, stage):
        prefix=self._shader_prefix+('_normal' if getattr(self,'_normal_maps_requested',False) else '')
        if stage=='frag' and getattr(self,'_reflection_requested',False): prefix+='_reflection'
        if stage=='frag' and getattr(self,'_indirect_requested',False):prefix+='_indirect'
        if stage=='frag' and getattr(self,'_progressive_requested',False):prefix+='_progressive'
        grid=f'_area{self._area_light_grid}' if stage=='frag' and self._area_light_grid!=2 else ''
        secondary = '_secondary2' if stage=='frag' and getattr(self, '_secondary_area_light_grid', self._area_light_grid)!=self._area_light_grid else ''
        return f'{prefix}{grid}{secondary}.{stage}.spv'

    def _validate_shader(self, stage, data):
        if stage=='frag' and self._secondary_area_light_grid!=self._area_light_grid:
            from ._native_secondary import validate_secondary_shader
            self._secondary_budget_marker = validate_secondary_shader(data)

    def _spatial_guide_filename(self):
        prefix = 'resident_rayquery' + ('_normal' if self._normal_maps_requested else '')
        grid = f'_area{self._area_light_grid}' if self._area_light_grid != 2 else ''
        return f'{prefix}{grid}.frag.spv'

    def _bind(self):
        super()._bind()
        if tuple(map(ct.sizeof, (ResidentVertex, ResidentMesh, ResidentInstance, ResidentInfo))) != (32,16,48,128):
            raise BackendUnavailableError('resident C ABI structure layout mismatch')
        lib = self._lib
        if self._shared_identity is not None:
            from ._native_shared_matrices import bind
            bind(lib)
        if self._settling_requested:
            version = getattr(lib, 'pr_resident_settling_abi_version', None)
            if version is None:
                raise BackendUnavailableError('native settling ABI unavailable; no fallback')
            version.argtypes = []
            version.restype = ct.c_uint32
            if version() != 1:
                raise BackendUnavailableError('unsupported native settling ABI')
        if self._spatial_requested:
            version = getattr(lib, 'pr_resident_spatial_abi_version', None)
            if version is None:
                raise BackendUnavailableError('native spatial ABI unavailable; no fallback')
            version.argtypes = []
            version.restype = ct.c_uint32
            if version() != 1:
                raise BackendUnavailableError('unsupported native spatial ABI')
        if self._progressive_requested:
            version = getattr(lib, 'pr_resident_progressive_abi_version', None)
            if version is None:
                raise BackendUnavailableError('native progressive ABI unavailable')
            version.argtypes = []
            version.restype = ct.c_uint32
            if version() != 1:
                raise BackendUnavailableError('unsupported native progressive ABI')
        if self._indirect_requested:
            version=getattr(lib,'pr_resident_indirect_abi_version',None)
            if version is None:raise BackendUnavailableError('native indirect ABI unavailable')
            version.argtypes=[];version.restype=ct.c_uint32
            if version()!=1:raise BackendUnavailableError('unsupported native indirect ABI')
        if self._reflection_requested:
            version=getattr(lib,'pr_resident_reflection_abi_version',None)
            if version is None: raise BackendUnavailableError('native reflection ABI is unavailable')
            version.argtypes=[];version.restype=ct.c_uint32
            if version()!=1: raise BackendUnavailableError('native reflection ABI is unsupported')
        if self._normal_maps_requested:
            version=getattr(lib,'pr_resident_normal_maps_abi_version',None)
            if version is None:
                raise BackendUnavailableError('native resident normal-map ABI is unavailable')
            version.argtypes=[]; version.restype=ct.c_uint32
            if version()!=1:
                raise BackendUnavailableError('native resident normal-map ABI is unsupported')
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
        if tuple(map(ct.sizeof,(ResidentEnvironment,ResidentEnvironmentInfo))) != (160,128):
            raise BackendUnavailableError('resident environment structure layout mismatch')
        environment_version=getattr(lib,'pr_resident_environment_abi_version',None)
        if environment_version is not None:
            environment_version.argtypes=[]; environment_version.restype=ct.c_uint32
            self._environment_abi_version=int(environment_version())
        if self._environment_abi_version==1:
            self._environment_function=getattr(lib,'pr_resident_set_environment',None)
            self._environment_info_function=getattr(lib,'pr_resident_environment_info',None)
        if self._environment_function is not None:
            self._environment_function.argtypes=[ct.c_void_p,ct.POINTER(ResidentEnvironment),
                ct.POINTER(ct.c_float),ct.c_size_t,ct.POINTER(ct.c_uint32),ct.c_size_t,
                ct.POINTER(ct.c_uint32),ct.c_size_t]
            self._environment_function.restype=ct.c_int
        if self._environment_info_function is not None:
            self._environment_info_function.argtypes=[ct.c_void_p,ct.POINTER(ResidentEnvironmentInfo)]
            self._environment_info_function.restype=ct.c_int
        if tuple(map(ct.sizeof,(ResidentTexture,ResidentUV,ResidentTextureInfo))) != (16,8,128):
            raise BackendUnavailableError('resident texture structure layout mismatch')
        texture_version=getattr(lib,'pr_resident_texture_abi_version',None)
        if texture_version is not None:
            texture_version.argtypes=[]; texture_version.restype=ct.c_uint32
            self._texture_abi_version=int(texture_version())
        if self._texture_abi_version==1:
            self._texture_function=getattr(lib,'pr_resident_set_textures',None)
            self._texture_info_function=getattr(lib,'pr_resident_texture_info',None)
        if self._texture_function is not None:
            self._texture_function.argtypes=[ct.c_void_p,ct.POINTER(ResidentTexture),ct.c_uint32,
                ct.POINTER(ct.c_uint8),ct.c_size_t,ct.POINTER(ResidentUV),ct.c_uint32,
                ct.POINTER(ct.c_uint32),ct.c_uint32]
            self._texture_function.restype=ct.c_int
            version=getattr(lib,'pr_resident_surface_maps_abi_version',None)
            if version is not None:
                version.argtypes=[]; version.restype=ct.c_uint32
                if version()==1:
                    self._surface_maps_function=getattr(lib,'pr_resident_set_surface_maps',None)
            if self._surface_maps_function is not None:
                self._surface_maps_function.argtypes=self._texture_function.argtypes
                self._surface_maps_function.restype=ct.c_int
        if self._texture_info_function is not None:
            self._texture_info_function.argtypes=[ct.c_void_p,ct.POINTER(ResidentTextureInfo)]
            self._texture_info_function.restype=ct.c_int
        if ct.sizeof(ResidentTelemetry) != 128:
            raise BackendUnavailableError('resident telemetry structure layout mismatch')
        # Optional, independently versioned extension. Its absence does not
        # change the resident raster capability of an older resident ABI1 DLL.
        self._telemetry_function = getattr(lib, 'pr_resident_telemetry', None)
        self._resident_destroy = getattr(lib, 'pr_resident_destroy', None)
        self._studio_function = getattr(lib, 'pr_resident_set_studio', None)
        self._profile_function = getattr(lib, 'pr_resident_profile', None)
        self._render_planes_function = getattr(lib, 'pr_resident_render_planes', None)
        self._readback_function = getattr(lib, 'pr_resident_readback_info', None)
        if ct.sizeof(ResidentReadbackInfo) != 64:
            raise BackendUnavailableError('resident readback structure layout mismatch')
        if self._render_planes_function is not None:
            self._render_planes_function.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_uint32,
                ct.POINTER(Parameters), ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.POINTER(ct.c_uint8), ct.c_size_t]
            self._render_planes_function.restype = ct.c_int
        if self._readback_function is not None:
            self._readback_function.argtypes = [ct.c_void_p, ct.POINTER(ResidentReadbackInfo)]
            self._readback_function.restype = ct.c_int
        if ct.sizeof(ResidentProfile) != 128:
            raise BackendUnavailableError('resident profile structure layout mismatch')
        if self._profile_function is not None:
            self._profile_function.argtypes = [ct.c_void_p, ct.POINTER(ResidentProfile)]
            self._profile_function.restype = ct.c_int
        if self._studio_function is not None:
            self._studio_function.argtypes = [ct.c_void_p, ct.POINTER(NativeStudio)]
            self._studio_function.restype = ct.c_int
        if self._telemetry_function is not None:
            self._telemetry_function.argtypes = [ct.c_void_p, ct.POINTER(ResidentTelemetry)]
            self._telemetry_function.restype = ct.c_int
        if self._resident_destroy is not None:
            self._resident_destroy.argtypes = [ct.c_void_p, ct.POINTER(_Info), ct.POINTER(ResidentTelemetry)]
            self._resident_destroy.restype = ct.c_int

    def _create(self, *args):
        if self._lib.pr_resident_abi_version() != 1:
            raise BackendUnavailableError('unsupported resident Vulkan C ABI version')
        if self._shared_identity is not None:
            uuid = (ct.c_uint8*16).from_buffer_copy(self._shared_identity[2])
            return self._lib.pr_resident_create_shared(*args, uuid)
        return self._lib.pr_resident_create(*args)

    def _remember_telemetry(self, value, *, final):
        if value.telemetry_abi_version != 1:
            raise RuntimeError('unsupported resident telemetry ABI')
        self._telemetry = {name: int(getattr(value, name)) for name, _ in ResidentTelemetry._fields_
                           if name not in ('device_luid', 'reserved_identity', 'reserved')}
        self._telemetry['device_luid_hex'] = bytes(value.device_luid).hex() if value.device_luid_valid else None
        self._telemetry['final_after_destroy'] = final
        self._telemetry['allocation_scope'] = 'this context explicit VkDeviceMemory allocations; not driver total VRAM'
        self._telemetry['wait_scope'] = 'host wall time blocked in fence waits, including upload; not GPU kernel time'

    def _remember_profile(self, value):
        if value.profile_abi_version != 1:
            raise RuntimeError('unsupported resident profile ABI')
        self._profile = {name: (float(getattr(value,name)) if name=='timestamp_period_ns' else int(getattr(value,name)))
                         for name,_ in ResidentProfile._fields_ if name!='reserved'}
        self._profile['scope'] = 'cumulative successful captures including warmup; excludes initial static geometry upload'
        self._profile['gpu_scope'] = 'Vulkan timestamps: raster plus fragment work, then attachment copies; no per-light kernel claim'
        self._profile['snapshot'] = 'last live query, retained after close; final ownership is checked by telemetry'

    def _remember_environment_info(self, value):
        if value.environment_abi_version != 1:
            raise RuntimeError('unsupported resident environment ABI')
        compatible = int(value.reserved[0])
        active = int(value.reserved[1])
        if compatible not in (0, 1) or active not in (0, 1):
            raise RuntimeError('unsupported resident environment importance flags')
        if active and not (compatible and value.loaded and value.supported and value.shader_compatible):
            raise RuntimeError('inconsistent resident environment importance activation')
        result = {
            name: int(getattr(value, name))
            for name, _ in ResidentEnvironmentInfo._fields_
            if name not in ('reserved', 'reserved64')
        }
        result.update(
            importance_shader_compatible=bool(compatible),
            importance_active=bool(active),
            importance_scope=(
                'native configuration flags: named progressive cubemap-CDF shader '
                'compatibility (SpecId72), and successfully uploaded nonconstant positive-mass map; '
                'not a ray, sample, visibility or capture counter; legacy reserved zeros remain valid'
            ),
            importance_mass_scope=(
                'private upload alpha: footprint-filtered max RGB times exact cube-texel solid angle; '
                'L0 inclusive FP32 cubemap CDF, L1 strict inverse-directory indices; '
                'adjacent stored CDF differences define the proposal; '
                'original public RGBA and every RGB mip unchanged'
                if compatible else
                'no compatible cubemap CDF producer receipt; private-alpha representation unclaimed'
            ),
            capture_scope=(
                'successful RGB captures with an uploaded directional environment; '
                'sky draw count is not foreground pixel count'
            ),
            snapshot='last live query; final GPU allocations are reported by teardown telemetry',
        )
        # A corrupt optional receipt must not replace the last valid snapshot.
        self._environment_info = result
    def _remember_texture_info(self, value):
        if value.texture_abi_version!=1:
            raise RuntimeError('unsupported resident texture ABI')
        self._texture_info={name:int(getattr(value,name)) for name,_ in ResidentTextureInfo._fields_
                            if name not in ('reserved','reserved64')}
        self._texture_info['capture_scope']='successful RGB captures with bound texture instances, not visible textured pixel count'
        self._texture_info['upload_scope']='cumulative actual texture image and metadata GPU writes, including default white sentinel; excluded from geometry_upload_bytes'
        self._texture_info['snapshot']='last live query, not final allocation state; teardown verified by allocation telemetry'
    def _remember_readback(self, value):
        if value.readback_abi_version != 1:
            raise RuntimeError('unsupported resident readback ABI')
        self._readback={name:int(getattr(value,name)) for name,_ in ResidentReadbackInfo._fields_ if name!='reserved'}
        flags=value.memory_property_flags
        self._readback.update(host_visible=bool(flags&2),host_coherent=bool(flags&4),host_cached=bool(flags&8),
            device_local=bool(flags&1),snapshot='last live query, not final mapping state; teardown verified by allocation telemetry',
            capacity_scope='logical Vulkan readback buffer capacity, not total VkDeviceMemory allocation size')

    def info(self):
        result = super().info()
        if self._handle is not None:
            value = ResidentInfo()
            if self._lib.pr_resident_info(self._handle, ct.byref(value)):
                raise RuntimeError(self._error())
            self._resident_info = {(name if name == 'resident_abi_version' else 'resident_' + name): int(getattr(value, name))
                                   for name, _ in ResidentInfo._fields_}
            if self._telemetry_function is not None:
                telemetry = ResidentTelemetry()
                if self._telemetry_function(self._handle, ct.byref(telemetry)):
                    raise RuntimeError(self._error())
                self._remember_telemetry(telemetry, final=False)
            if self._profile_function is not None:
                profile = ResidentProfile()
                if self._profile_function(self._handle, ct.byref(profile)):
                    raise RuntimeError(self._error())
                self._remember_profile(profile)
            if self._readback_function is not None:
                readback = ResidentReadbackInfo()
                if self._readback_function(self._handle, ct.byref(readback)):
                    raise RuntimeError(self._error())
                self._remember_readback(readback)
            if self._texture_info_function is not None:
                texture=ResidentTextureInfo()
                if self._texture_info_function(self._handle,ct.byref(texture)):
                    raise RuntimeError(self._error())
                self._remember_texture_info(texture)
            if self._environment_info_function is not None:
                environment=ResidentEnvironmentInfo()
                if self._environment_info_function(self._handle,ct.byref(environment)):
                    raise RuntimeError(self._error())
                self._remember_environment_info(environment)
        result.update(self._resident_info)
        exchange = getattr(self, '_shared_exchange', None)
        result['resident_shared_matrices'] = exchange.diagnostics() if exchange is not None else None
        result['resident_secondary_light_budget'] = dict(self._secondary_budget_marker) if self._secondary_budget_marker is not None else None
        result['resident_render_planes_available'] = self._render_planes_function is not None
        result['resident_environment_available'] = self._environment_function is not None and (self._environment_info is None or bool(self._environment_info['supported'] and self._environment_info['shader_compatible']))
        result['resident_environment_abi_version'] = self._environment_abi_version
        result['resident_environment_info'] = dict(self._environment_info) if self._environment_info is not None else None
        result['resident_texture_available'] = self._texture_function is not None
        result['resident_linear_roughness_available'] = self._surface_maps_function is not None
        result['resident_texture_abi_version'] = self._texture_abi_version
        result['resident_texture_info'] = dict(self._texture_info) if self._texture_info is not None else None
        if getattr(self,'_normal_info_function',None) is not None and self._handle:
            from ._native_normal import ResidentNormalInfo, normal_info
            value=ResidentNormalInfo()
            if self._normal_info_function(self._handle,ct.byref(value)): raise RuntimeError(self._error())
            self._normal_info=normal_info(value)
        result['resident_normal_info']=dict(self._normal_info) if getattr(self,'_normal_info',None) is not None else None
        if getattr(self,'_reflection_info_function',None) is not None and self._handle:
            from ._native_reflection import ResidentReflectionInfo,reflection_info
            value=ResidentReflectionInfo()
            if self._reflection_info_function(self._handle,ct.byref(value)): raise RuntimeError(self._error())
            self._reflection_info=reflection_info(value)
        result['resident_reflection_info']=dict(self._reflection_info) if getattr(self,'_reflection_info',None) is not None else None
        if getattr(self,'_indirect_info_function',None) is not None and self._handle:
            from ._native_indirect import ResidentIndirectInfo,indirect_info
            value=ResidentIndirectInfo()
            if self._indirect_info_function(self._handle,ct.byref(value)):raise RuntimeError(self._error())
            self._indirect_info=indirect_info(value)
        from ._native_progressive import query_progressive
        result['resident_progressive_info'] = query_progressive(self)
        from ._native_spatial import query_spatial
        result['resident_spatial_info'] = query_spatial(self)
        from ._native_spatial import query_settling
        result['resident_settling_info'] = query_settling(self)
        result['resident_indirect_info']=dict(self._indirect_info) if getattr(self,'_indirect_info',None) is not None else None
        result['resident_readback'] = dict(self._readback) if self._readback is not None else None
        result['resident_profile'] = dict(self._profile) if self._profile is not None else None
        result['resident_telemetry'] = dict(self._telemetry) if self._telemetry is not None else None
        result['resident_studio_available'] = self._studio_function is not None
        query=getattr(self,'_msaa_info_function',None)
        if query is not None and self._handle is not None:
            from ._native_msaa import ResidentMsaaInfo, msaa_info
            value=ResidentMsaaInfo()
            if query(self._handle,ct.byref(value)):
                raise RuntimeError(self._error())
            self._msaa_info=msaa_info(value)
        result['resident_msaa']=dict(self._msaa_info) if getattr(self,'_msaa_info',None) is not None else None
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

    def set_environment(self, upload):
        if self._handle is None or not self._uploaded:
            raise RuntimeError('resident environment requires an open uploaded scene')
        if self._has_captured or self._environment_set:
            raise RuntimeError('resident environment may be set once before the first capture')
        if self._environment_function is None:
            raise BackendUnavailableError('native resident directional environment ABI1 export is unavailable')
        if not isinstance(upload,EnvironmentUpload):
            raise TypeError('environment upload must be EnvironmentUpload')
        shaders=[]; hashes={}
        for stage in ('vert','frag'):
            suffix = '_progressive' if stage == 'frag' and self._progressive_requested else ''
            path=self.shader_directory/f'resident_sky{suffix}.{stage}.spv'
            try:
                with path.open('rb') as stream: raw=stream.read(4*1024*1024+1)
            except OSError as error:
                raise BackendUnavailableError(f'native environment SPIR-V unavailable: {path}') from error
            if len(raw)<20 or len(raw)>4*1024*1024 or len(raw)%4 or raw[:4]!=b'\x03\x02\x23\x07':
                raise BackendUnavailableError(f'invalid native environment SPIR-V: {path}')
            shaders.append((ct.c_uint32*(len(raw)//4)).from_buffer_copy(raw))
            hashes['environment_sky_'+stage]=hashlib.sha256(raw).hexdigest()
        if self._environment_function(self._handle,ct.byref(upload.config),upload.rgba,ct.sizeof(upload.rgba),
                shaders[0],ct.sizeof(shaders[0]),shaders[1],ct.sizeof(shaders[1])):
            raise RuntimeError(self._error())
        self._environment_set=True
        self.shader_sha256.update(hashes)
    def set_textures(self, upload):
        if self._handle is None or not self._uploaded:
            raise RuntimeError('resident textures require an open uploaded scene')
        if self._has_captured or self._textures_set:
            raise RuntimeError('resident textures may be set once before the first capture')
        if self._texture_function is None:
            raise BackendUnavailableError('native resident texture ABI1 export is unavailable')
        if not isinstance(upload,TextureUpload):
            raise TypeError('texture upload must be TextureUpload')
        setter=self._surface_maps_function if upload.uses_roughness else self._texture_function
        if setter is None:
            raise BackendUnavailableError('native resident linear roughness requires surface-maps ABI1')
        if setter(self._handle,upload.records,len(upload.records),
                upload.rgba,len(upload.rgba),upload.uv,len(upload.uv),upload.slots,len(upload.slots)):
            raise RuntimeError(self._error())
        self._textures_set=True

    def set_studio(self, value):
        if self._handle is None:
            raise RuntimeError('resident device is closed')
        if self._has_captured:
            raise RuntimeError('resident studio must be set before the first capture')
        if self._studio_function is None:
            raise BackendUnavailableError('native resident studio export is unavailable')
        if not isinstance(value, NativeStudio):
            raise TypeError('studio upload must be NativeStudio')
        if self._studio_function(self._handle, ct.byref(value)):
            raise RuntimeError(self._error())

    def render(self, matrices, parameters, width, height):
        """Original ABI1 full-attachment read, including older DLL support."""
        return self._capture_bytes(matrices,parameters,width,height,None)

    def render_planes(self, matrices, parameters, width, height, channel_mask):
        """Explicit optional compact ABI; never pretend an old DLL supports it."""
        self._require_compact_mask(parameters, channel_mask)
        return self._capture_bytes(matrices,parameters,width,height,channel_mask)

    def render_image_planes(self, matrices, parameters, width, height, channels):
        """Finish the capture and own each requested plane before returning."""
        mask = sum(1 << bit for bit, channel in enumerate(Channel) if channel in channels)
        self._require_compact_mask(parameters, mask)
        return self._capture_bytes(matrices,parameters,width,height,mask,image_channels=channels)

    def open_shared_matrices(self, *, cuda_device):
        """Import the resident slot once after upload, retaining it until device.close()."""
        if self._handle is None or not self._uploaded:
            raise RuntimeError('shared matrices require an open uploaded resident scene')
        if (type(cuda_device) is not int or self._shared_cuda_device is None or
                cuda_device != self._shared_cuda_device):
            raise ValueError('CUDA ordinal must match the shared resident device configuration')
        if self._last_info['device_uuid'] != self._shared_identity[2].hex():
            raise BackendUnavailableError('resident Vulkan UUID differs from the CUDA device UUID')
        if self._shared_exchange is None:
            from ._native_shared_matrices import SharedMatrices
            self._shared_exchange = SharedMatrices(self)
        return self._shared_exchange.open()

    def _require_compact_mask(self, parameters, channel_mask):
        if type(channel_mask) is not int:
            raise TypeError('resident channel mask must be int, not bool')
        if not 0 < channel_mask <= 15:
            raise ValueError('resident channel mask must be nonzero bits 0..3')
        if self._render_planes_function is None:
            raise BackendUnavailableError('native compact readback export is unavailable')
        if parameters.counts[2] != bool(channel_mask&1):
            raise ValueError('resident RGB uniform disagrees with channel mask')

    def _capture_bytes(self, matrices, parameters, width, height, channel_mask, *, image_channels=None):
        if getattr(self, '_shared_cuda_device', None) is not None:
            if self._shared_exchange is None:
                raise RuntimeError('open the CUDA matrix exchange before rendering a shared capture')
            with self._shared_exchange.capturing(matrices):
                return self._read_capture(matrices, parameters, width, height, channel_mask, image_channels, shared=True)
        return self._read_capture(matrices, parameters, width, height, channel_mask, image_channels, shared=False)

    def _read_capture(self, matrices, parameters, width, height, channel_mask, image_channels, *, shared):
        if self._handle is None or not self._uploaded:
            raise RuntimeError('resident device is closed or has no uploaded scene')
        limits=self._last_info
        if (type(width) is not int or type(height) is not int or
            not 0 < width <= min(limits['max_width'],2**32-1) or
            not 0 < height <= min(limits['max_height'],2**32-1) or width*height > 4096*4096):
            raise ValueError('native capture extent or pixel budget exceeded')
        if not shared and (len(matrices)!=16*self._instance_count or not all(math.isfinite(x) for x in matrices)):
            raise ValueError('resident instance matrices are incomplete or nonfinite FP32')
        validate_parameters(parameters)
        if parameters.clipping[0]!=width or parameters.clipping[1]!=height:
            raise ValueError('native clipping extent mismatch')
        if parameters.counts[3]!=self._instance_count:
            raise ValueError('resident parameter instance count mismatch')
        mask=15 if channel_mask is None else channel_mask
        size=width*height*sum(width for bit,width in enumerate((4,4,4,1)) if mask&(1<<bit))
        if self._output_capacity<size:
            self._output_buffer=(ct.c_uint8*size)()
            self._output_capacity=size
        output=self._output_buffer
        if shared:
            result=self._lib.pr_resident_render_shared_planes(self._handle,ct.byref(parameters),
                width,height,mask,output,size)
        elif channel_mask is None:
            result=self._lib.pr_resident_render(self._handle,matrices,self._instance_count,
                ct.byref(parameters),width,height,output,size)
        else:
            result=self._render_planes_function(self._handle,matrices,self._instance_count,
                ct.byref(parameters),width,height,channel_mask,output,size)
        if result: raise RuntimeError(self._error())
        self._has_captured=True
        if image_channels is not None:
            # The owning session keeps its lock until publication. No view of
            # this reusable ctypes array is allowed to escape the decoder.
            view = memoryview(output).cast("B")[:size]
            return _image_planes(view, width*height, image_channels, mask)
        # This is the single ownership copy from reusable private storage.
        # Only this call's compact length is copied, never the old capacity.
        return ct.string_at(output,size)
    def close(self):
        self._output_buffer = None
        self._output_capacity = 0
        if self._handle is None:
            return
        if getattr(self, '_shared_exchange', None) is not None:
            # The Vulkan slot outlives CUDA imported objects. A timeout leaves
            # both sides owned here so teardown can be retried.
            self._shared_exchange.close()
        # Preserve resident counters at the last live point, without allowing a
        # failed diagnostic read to prevent the shared real Vulkan destruction.
        diagnostic_error = None
        try:
            self.info()
        except BaseException as error:
            diagnostic_error = error
        try:
            if self._resident_destroy is None:
                super().close()
            else:
                final, telemetry = _Info(), ResidentTelemetry()
                result = self._resident_destroy(self._handle, ct.byref(final), ct.byref(telemetry))
                if result == -2:
                    raise TimeoutError(self._error())
                self._handle = None
                self._remember(final)
                self._remember_telemetry(telemetry, final=True)
                if result:
                    raise RuntimeError(self._error())
        except BaseException as cleanup:
            if diagnostic_error is not None:
                raise BaseExceptionGroup('resident diagnostics and cleanup failed',
                                         [diagnostic_error, cleanup])
            raise
        if diagnostic_error is not None:
            raise diagnostic_error
