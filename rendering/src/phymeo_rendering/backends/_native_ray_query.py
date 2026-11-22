"""Explicit Vulkan ray-query shadow context; never substitutes software traversal."""
import ctypes as ct

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
        super().__init__(*args,**kwargs)

    def _bind(self):
        super()._bind()
        if ct.sizeof(ResidentRayQueryInfo)!=128:
            raise BackendUnavailableError('resident ray query structure layout mismatch')
        lib=self._lib
        version=getattr(lib,'pr_resident_ray_query_abi_version',None)
        create=getattr(lib,'pr_resident_create_ray_query',None)
        query=getattr(lib,'pr_resident_ray_query_info',None)
        if version is None or create is None or query is None:
            raise BackendUnavailableError('native hardware ray query ABI exports are unavailable')
        version.argtypes=[]; version.restype=ct.c_uint32
        if version()!=1:
            raise BackendUnavailableError('unsupported native hardware ray query ABI version')
        create.argtypes=lib.pr_resident_create.argtypes
        create.restype=ct.c_void_p
        query.argtypes=[ct.c_void_p,ct.POINTER(ResidentRayQueryInfo)]
        query.restype=ct.c_int
        self._ray_query_create=create
        self._ray_query_info_function=query

    def _create(self,*args):
        if self._lib.pr_resident_abi_version()!=1:
            raise BackendUnavailableError('unsupported resident Vulkan C ABI version')
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

    def info(self):
        result=super().info()
        if self._handle is not None:
            value=ResidentRayQueryInfo()
            if self._ray_query_info_function(self._handle,ct.byref(value)):
                raise RuntimeError(self._error())
            self._remember_ray_query_info(value)
        if result.get('resident_profile') is not None:
            result['resident_profile']['gpu_scope'] = 'Vulkan timestamps: TLAS build/update plus raster and fragment shadows, then attachment copies; not fragment-only time'
        result['resident_ray_query_abi_version']=1
        result['resident_ray_query_info']=dict(self._ray_query_info)
        return result