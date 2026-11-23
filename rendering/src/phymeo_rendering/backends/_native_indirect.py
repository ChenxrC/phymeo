"""Explicit fixed-budget one-bounce diffuse estimator and optional C ABI."""

import ctypes as ct

from . import BackendUnavailableError


class ResidentIndirectInfo(ct.Structure):
    _fields_ = [
        (name, ct.c_uint32)
        for name in (
            'indirect_abi_version', 'supported', 'shader_compatible', 'enabled',
            'samples_per_pixel', 'seed', 'max_bounces', 'set_calls',
            'receiver_count',
        )
    ] + [
        ('reserved', ct.c_uint32 * 3),
        ('captures', ct.c_uint64),
        ('rgb_captures', ct.c_uint64),
    ]


def indirect_info(value):
    if value.indirect_abi_version != 1:
        raise RuntimeError('unsupported indirect info ABI')
    result = {
        name: getattr(value, name)
        for name, _ in ResidentIndirectInfo._fields_
        if name != 'reserved'
    }
    result['capture_scope'] = (
        'successful captures/RGB captures; samples_per_pixel is a budget, '
        'not an actual ray/hit counter'
    )
    result['receiver_scope'] = (
        'instances with uploaded metallic<1 and positive base-color factor; '
        'image-zero pixels may emit no rays'
    )
    result['snapshot'] = (
        'last live query retained after close; allocation teardown uses resident telemetry'
    )
    return result


def indirect_declaration(mode, seed):
    if mode == 'none':
        return dict(mode=mode, seed=seed, samples_per_pixel=0, bounces=0)
    return dict(
        mode=mode, seed=seed, samples_per_pixel=16, bounces=1,
        estimator=(
            'rho_d * sum(Li)/16; 4x4 stratified cosine hemisphere '
            'with hashed 23-bit open-interval jitter'
        ),
        diffuse_coefficient='rho_d=(1-metallic)*albedo*(1-F0); F0=mix(.04,albedo,metallic)',
        environment_policy='replace the original diffuse IBL/SH term, never add a second copy',
        miss_policy=(
            'original HDR level0 radiance along sample direction, '
            'or uniform environment; zero if absent'
        ),
        terminal_lighting=(
            'nearest current real TLAS triangle, original direct + studio + environment; '
            'no recursive indirect/reflection'
        ),
        seed_scope=(
            'uint32 seed + capture-local integer pixel XY + sample index; '
            'no frame/time/epoch key, no accumulation'
        ),
        geometry_policy=(
            'shading-normal cosine distribution clipped by oriented true face; '
            'inward/degenerate sample=0, denominator stays16'
        ),
        combination=(
            'independent primary specular/diffuse lobes may share terminal shading; '
            'no additional bounce'
        ),
        limitations=(
            '16 noisy samples; no denoiser, temporal history, multiple diffuse '
            'bounces or raw-HDR output attachment'
        ),
    )


def configure_indirect(device, declaration):
    if declaration['mode'] == 'none':
        return declaration
    if ct.sizeof(ResidentIndirectInfo) != 64:
        raise BackendUnavailableError('indirect ABI layout mismatch')
    try:
        version = device._lib.pr_resident_indirect_abi_version
        setter = device._lib.pr_resident_set_indirect
        query = device._lib.pr_resident_indirect_info
    except AttributeError as error:
        raise BackendUnavailableError('native diffuse indirect ABI unavailable; no fallback') from error
    if not device._handle or not device._uploaded or device._has_captured:
        raise RuntimeError('indirect requires an uploaded uncaptured session')
    version.argtypes = []
    version.restype = ct.c_uint32
    setter.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_uint32]
    setter.restype = ct.c_int
    query.argtypes = [ct.c_void_p, ct.POINTER(ResidentIndirectInfo)]
    query.restype = ct.c_int
    if version() != 1:
        raise BackendUnavailableError('unsupported indirect ABI')
    if setter(device._handle, 1, declaration['seed']):
        raise BackendUnavailableError(device._error())
    value = ResidentIndirectInfo()
    if query(device._handle, ct.byref(value)):
        raise BackendUnavailableError(device._error())
    if not value.enabled or value.samples_per_pixel != 16 or value.seed != declaration['seed']:
        raise BackendUnavailableError('device indirect configuration disagrees')
    device._indirect_info_function = query
    device._indirect_info = indirect_info(value)
    return declaration
