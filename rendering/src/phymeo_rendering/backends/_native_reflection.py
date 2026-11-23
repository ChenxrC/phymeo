"""Explicit ideal, single specular bounce for smooth opaque metal receivers.

Eligibility is the actual uploaded FP32 metallic and authored roughness factors.
A roughness image only multiplies that factor by [0,1]; it cannot make a rough
material eligible. This is NOT a rough-reflection estimator or recursive tracer.
"""
import ctypes as ct
from . import BackendUnavailableError, UnsupportedRequestError

METALLIC_MIN=ct.c_float(.99).value
ROUGHNESS_MAX=ct.c_float(.10).value

class ResidentReflectionInfo(ct.Structure):
    _fields_=[(name,ct.c_uint32) for name in ('reflection_abi_version','supported',
        'shader_compatible','enabled','max_bounces','receiver_count','set_calls','reserved')]+[
        ('metallic_min',ct.c_float),('roughness_max',ct.c_float),('reserved_f',ct.c_float*2),
        ('captures',ct.c_uint64),('rgb_captures',ct.c_uint64)]


def reflection_info(value):
    if value.reflection_abi_version!=1: raise RuntimeError('unsupported reflection info ABI')
    result={name:getattr(value,name) for name,_ in ResidentReflectionInfo._fields_ if name not in ('reserved','reserved_f')}
    result['capture_scope']='successful captures after enable; RGB counter is not actual ray/hit count'
    result['snapshot']='last live query retained after close; allocation teardown uses resident telemetry'
    return result


def reflection_declaration(scene,mode):
    if mode=='environment': return dict(mode=mode,bounces=0,receiver_instances=[])
    materials={m.id:m for m in scene.materials}; receivers=[]
    for instance in scene.instances:
        m=materials[instance.material_id]
        metal=ct.c_float(m.metallic).value; rough=ct.c_float(m.roughness).value
        if metal>=METALLIC_MIN and rough<=ROUGHNESS_MAX:
            receivers.append(dict(instance_id=instance.id,material_id=m.id,
                metallic_fp32=metal,authored_roughness_fp32=rough))

    return dict(mode='single_bounce',bounces=1,receiver_instances=receivers,
        receiver_policy='opaque; uploaded FP32 metallic>=.99 and authored roughness<=.10',
        roughness_policy='image is linear [0,1] multiplier; admission uses authored factor; ideal unblurred direction, no rough-reflection integration',
        terminal_lighting='nearest real TLAS triangle: original directional + studio + environment; no recursive reflection',
        miss_policy='original specular environment; zero if no environment; direct lighting retained',
        normal_policy='shading normal determines reflection; primary world-position derivative face normal controls launch bias/inward rejection; terminal triangle-face normal controls hit bias; original primary shadow bias unchanged',
        geometry_policy='same complete BLAS/TLAS, current matrices, barycentric material/primary UV; no screen-space cache')


def configure_reflection(device,declaration):
    if declaration['mode']=='environment': return declaration
    if ct.sizeof(ResidentReflectionInfo)!=64: raise BackendUnavailableError('reflection ABI layout mismatch')
    try:
        version=device._lib.pr_resident_reflection_abi_version
        setter=device._lib.pr_resident_set_reflection
        query=device._lib.pr_resident_reflection_info
    except AttributeError as error:
        raise BackendUnavailableError('native single-bounce reflection ABI unavailable; no fallback') from error
    if not device._handle or not device._uploaded or device._has_captured:
        raise RuntimeError('reflection requires an uploaded uncaptured session')
    version.argtypes=[]; version.restype=ct.c_uint32
    setter.argtypes=[ct.c_void_p,ct.c_uint32]; setter.restype=ct.c_int
    query.argtypes=[ct.c_void_p,ct.POINTER(ResidentReflectionInfo)]; query.restype=ct.c_int
    if version()!=1: raise BackendUnavailableError('unsupported single-bounce reflection ABI')
    if setter(device._handle,1): raise BackendUnavailableError(device._error())
    value=ResidentReflectionInfo()
    if query(device._handle,ct.byref(value)): raise BackendUnavailableError(device._error())
    if not value.enabled or value.receiver_count!=len(declaration['receiver_instances']):
        raise BackendUnavailableError('reflection receiver admission differs from device')
    device._reflection_info_function=query;device._reflection_info=reflection_info(value)
    return declaration
