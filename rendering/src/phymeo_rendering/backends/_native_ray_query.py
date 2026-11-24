"""Explicit Vulkan ray-query shadow context; never substitutes software traversal."""
import ctypes as ct
import hashlib

from . import BackendUnavailableError
from ._native_resident import ResidentDevice


class ResidentRayQueryInfo(ct.Structure):
    _fields_=[(name,ct.c_uint32) for name in (
        'ray_query_abi_version','enabled','blas_count','tlas_instance_count','primitive_count',
        'loaded','poisoned','reserved','max_meshes','max_instances','max_triangles',
        'build_flags','update_flags','trace_flags')]+[('reserved2',ct.c_uint32*2)]+[
        (name,ct.c_uint64) for name in ('max_acceleration_bytes','acceleration_buffer_bytes',
        'scratch_buffer_bytes','blas_builds','tlas_builds','tlas_updates',
        'instance_upload_bytes','shadow_captures')]


class RayQueryResidentDevice(ResidentDevice):
    """Same resident assets and captures, separately selected hardware visibility."""
    _shader_prefix='resident_rayquery'

    def __init__(self,*args,**kwargs):
        self._ray_query_info={}
        self._ray_query_info_function=None
        self._ray_query_create=None
        self._shared_ray_query_create=None
        super().__init__(*args,**kwargs)

    def _bind(self):
        super()._bind()
        if ct.sizeof(ResidentRayQueryInfo)!=128:
            raise BackendUnavailableError('resident ray query structure layout mismatch')
        lib=self._lib
        version=getattr(lib,'pr_resident_ray_query_abi_version',None)
        query=getattr(lib,'pr_resident_ray_query_info',None)
        if version is None or query is None:
            raise BackendUnavailableError('native hardware ray query ABI exports are unavailable')
        version.argtypes=[]; version.restype=ct.c_uint32
        if version()!=1:
            raise BackendUnavailableError('unsupported native hardware ray query ABI version')
        query.argtypes=[ct.c_void_p,ct.POINTER(ResidentRayQueryInfo)]
        query.restype=ct.c_int
        self._ray_query_info_function=query
        if getattr(self, '_shared_identity', None) is not None:
            from ._native_shared_matrices import require_shared_modes
            require_shared_modes(self)
            shared_create = getattr(lib, 'pr_resident_create_shared_ray_query', None)
            if shared_create is None:
                raise BackendUnavailableError('native shared hardware ray-query creation export is unavailable')
            shared_create.argtypes = [*lib.pr_create.argtypes, ct.POINTER(ct.c_uint8),
                                      ct.POINTER(ct.c_uint32), ct.c_size_t]
            shared_create.restype = ct.c_void_p
            self._shared_ray_query_create = shared_create
        else:
            create = getattr(lib, 'pr_resident_create_ray_query', None)
            if create is None:
                raise BackendUnavailableError('native hardware ray query creation export is unavailable')
            create.argtypes = lib.pr_resident_create.argtypes
            create.restype = ct.c_void_p
            self._ray_query_create = create

    def _create(self,*args):
        if self._lib.pr_resident_abi_version()!=1:
            raise BackendUnavailableError('unsupported resident Vulkan C ABI version')
        if getattr(self, '_shared_identity', None) is not None:
            path = self.shader_directory / 'resident_shared_instances.comp.spv'
            try:
                with path.open('rb') as stream:
                    raw = stream.read(4*1024*1024+1)
            except OSError as error:
                raise BackendUnavailableError(f'shared TLAS instance compute shader unavailable: {path}') from error
            if len(raw) < 20 or len(raw) > 4*1024*1024 or len(raw)%4 or raw[:4] != b'\x03\x02\x23\x07':
                raise BackendUnavailableError(f'invalid shared TLAS instance compute SPIR-V: {path}')
            # The native constructor checks the compute entry point, the two
            # storage bindings and the named instance packing ABI (SpecId 75).
            shader = (ct.c_uint32*(len(raw)//4)).from_buffer_copy(raw)
            uuid = (ct.c_uint8*16).from_buffer_copy(self._shared_identity[2])
            self.shader_sha256['shared_instances_compute'] = hashlib.sha256(raw).hexdigest()
            return self._shared_ray_query_create(*args, uuid, shader, len(raw))
        return self._ray_query_create(*args)

    def _remember_ray_query_info(self,value):
        if value.ray_query_abi_version!=1:
            raise RuntimeError('unsupported resident ray query ABI')
        if value.enabled!=1:
            raise RuntimeError('requested hardware ray query context is not enabled')
        self._ray_query_info={name:int(getattr(value,name)) for name,_ in ResidentRayQueryInfo._fields_
                              if name not in ('reserved','reserved2')}
        self._ray_query_info.update(
            capture_scope='successful RGB-requested captures, not ray count or visible shadow pixel count',
            buffer_scope='logical AS and scratch buffer bytes; actual VkDeviceMemory allocation telemetry is separate',
            update_scope='each capture builds or updates TLAS, including two cameras at the same frame key',
            upload_scope='additional VkAccelerationStructureInstanceKHR records, 64 bytes per instance per capture; not the ordinary forward/inverse matrix buffer',
            snapshot='last live query, retained after close; final allocation teardown is reported by resident telemetry')
        if getattr(self, '_shared_identity', None) is not None:
            self._ray_query_info.update(
                update_scope='one GPU TLAS build or update per acquired matrix publication, shared across its camera draws',
                upload_scope='instance_upload_bytes counts host TLAS-instance uploads; shared instance records are generated on the GPU',
                instance_source='Vulkan compute over the shared device matrix buffer')

    def info(self):
        result=super().info()
        if self._handle is not None:
            value=ResidentRayQueryInfo()
            if self._ray_query_info_function(self._handle,ct.byref(value)):
                raise RuntimeError(self._error())
            self._remember_ray_query_info(value)
        if getattr(self, '_shared_identity', None) is not None and result.get('resident_profile') is not None:
            result['resident_profile']['gpu_scope'] = 'capture timestamps cover raster, fragment shadows and attachment copies; shared TLAS work is part of pose publication'
        elif result.get('resident_profile') is not None:
            result['resident_profile']['gpu_scope'] = 'Vulkan timestamps: TLAS build/update plus raster and fragment shadows, then attachment copies; not fragment-only time'
        if getattr(self,'_reflection_requested',False) and result.get('resident_profile') is not None:
            result['resident_profile']['gpu_scope']='Vulkan timestamps: TLAS build/update + raster, shadow queries, single-bounce reflection and terminal lighting; then attachment copies'
        if getattr(self,'_indirect_requested',False) and result.get('resident_profile') is not None:
            result['resident_profile']['gpu_scope']='Vulkan timestamps: TLAS update/build + raster, shadows, 16-sample diffuse indirect and optional specular reflection with terminal lighting; then attachment copies'
        if getattr(self,'_progressive_requested',False) and result.get('resident_profile') is not None:
            result['resident_profile']['gpu_scope']='render timestamp interval: TLAS + raster + bounded one/two-bounce paths + GPU scene-linear mean/display compute + 4-byte status copy; copy interval: requested planes; explicit mean readback is separate'
        if getattr(self, '_spatial_requested', False) and result.get('resident_profile') is not None:
            result['resident_profile']['gpu_scope'] += '; guided first/reset RGB adds an extra GPU direct/environment raster pass with shadow tests and 25-tap display compute; no guide draw on later batches; explicit guide readback is separate'
        if getattr(self, '_settling_requested', False) and result.get('resident_profile') is not None:
            result['resident_profile']['gpu_scope'] += '; settling batches n<16 add display-only mean filtering after an explicit compute barrier; one retained guide draw per RGB state'
        result['resident_ray_query_abi_version']=1
        result['resident_ray_query_info']=dict(self._ray_query_info)
        return result
