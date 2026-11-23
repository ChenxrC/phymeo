"""Bounded GPU scene-linear averaging and explicit static-state history keys."""

import ctypes as ct
import hashlib

from . import BackendUnavailableError, UnsupportedRequestError


class ProgressiveInfo(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        'progressive_abi_version', 'supported', 'shader_compatible', 'enabled',
        'bounces', 'max_batches', 'batches', 'last_batch_index', 'seed',
        'samples_per_batch', 'width', 'height',
    )] + [('reserved', ct.c_uint32 * 4)] + [(name, ct.c_uint64) for name in (
        'rgb_captures', 'reset_calls', 'mean_readbacks', 'mean_readback_bytes', 'status_readback_bytes',
    )] + [('reserved64', ct.c_uint64 * 3)]


class ProgressiveHistory:
    """Prepare is read-only; only a successfully published RGB batch commits."""

    def __init__(self, bounces):
        self.bounces = bounces
        self.key = None
        self.batches = 0
        self.reset_count = 0
        self.last_reset_reasons = ['new_session']
        self.pending = 'new_session'
        self.reset_recorded = False

    def prepare(self, frame, camera, lights=()):
        key = dict(
            world=frame.key.world_id, epoch=frame.key.epoch,
            extent=(camera.width, camera.height),
            camera=(camera.id, camera.world_pose, camera.fx_px, camera.fy_px,
                    camera.cx_px, camera.cy_px, camera.near_m, camera.far_m),
            transforms=tuple(sorted((node.node_id, node.world_pose) for node in frame.nodes)),
            lights=tuple(lights),
        )
        reasons = [self.pending] if self.key is None else [
            name for name in key if key[name] != self.key[name]
        ]
        if not reasons and self.batches >= 4096:
            raise UnsupportedRequestError('progressive history reached 4096 batches; reset explicitly')
        return key, reasons

    def commit(self, key, reasons):
        if reasons:
            self.batches = 0
            self.last_reset_reasons = list(reasons)
            if not self.reset_recorded:
                self.reset_count += 1
        self.key = key
        self.batches += 1
        self.pending = None
        self.reset_recorded = False

    def reset(self, reason="explicit_reset"):
        self.key = None
        self.batches = 0
        self.pending = reason
        self.last_reset_reasons = [self.pending]
        self.reset_count += 1
        self.reset_recorded = True

    def diagnostics(self):
        return dict(
            enabled=True, bounces=self.bounces, batches=self.batches,
            batch_index=self.batches - 1 if self.batches else None,
            primary_samples_per_pixel=16 * self.batches,
            samples_per_batch=16, max_batches=4096,
            reset_count=self.reset_count,
            last_reset_reasons=list(self.last_reset_reasons),
            accumulation_space='scene-linear-before-exposure',
            storage='GPU RGBA32F mean',
            sequence='seed + capture-local pixel + 0-based batch; 16 primary paths, at most one continuation each',
            scene_scope='one immutable scene/session; camera, transforms, effective directional lights, world, epoch and extent are checked',
            non_rgb='does not advance or replace RGB history',
        )


def configure_progressive(device, enabled, bounces):
    if not enabled:
        return None
    if ct.sizeof(ProgressiveInfo) != 128:
        raise BackendUnavailableError('progressive info ABI layout mismatch')
    try:
        version = device._lib.pr_resident_progressive_abi_version
        setter = (device._lib.pr_resident_set_spatial if getattr(device, '_spatial_requested', False)
                  else device._lib.pr_resident_set_progressive)
        query = device._lib.pr_resident_progressive_info
        begin = device._lib.pr_resident_begin_progressive
        reset = device._lib.pr_resident_reset_progressive
        read = device._lib.pr_resident_read_progressive_mean
    except AttributeError as error:
        raise BackendUnavailableError('native progressive ABI unavailable; no fallback') from error
    version.argtypes = []
    version.restype = ct.c_uint32
    if version() != 1:
        raise BackendUnavailableError('unsupported progressive ABI')
    filename = 'resident_spatial.comp.spv' if getattr(device, '_spatial_requested', False) else 'resident_progressive.comp.spv'
    path = device.shader_directory / filename
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise BackendUnavailableError(f'progressive compute shader unavailable: {path}') from error
    if len(raw) < 20 or len(raw) % 4 or raw[:4] != b'\x03\x02\x23\x07':
        raise BackendUnavailableError('invalid progressive compute SPIR-V')
    words = (ct.c_uint32 * (len(raw) // 4)).from_buffer_copy(raw)
    setter.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint32), ct.c_size_t]
    setter.restype = ct.c_int
    query.argtypes = [ct.c_void_p, ct.POINTER(ProgressiveInfo)]
    query.restype = ct.c_int
    begin.argtypes = [ct.c_void_p, ct.c_uint32]
    begin.restype = ct.c_int
    reset.argtypes = [ct.c_void_p]
    reset.restype = ct.c_int
    read.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
    read.restype = ct.c_int
    arguments = [device._handle, bounces, words, len(raw)]
    if getattr(device, '_spatial_requested', False):
        guide_path = device.shader_directory / device._spatial_guide_filename()
        try:
            guide_raw = guide_path.read_bytes()
        except OSError as error:
            raise BackendUnavailableError(f'spatial guide shader unavailable: {guide_path}') from error
        if len(guide_raw) < 20 or len(guide_raw) % 4 or guide_raw[:4] != b'\x03\x02\x23\x07':
            raise BackendUnavailableError('invalid spatial guide SPIR-V')
        guide_words = (ct.c_uint32 * (len(guide_raw) // 4)).from_buffer_copy(guide_raw)
        setter.argtypes += [ct.POINTER(ct.c_uint32), ct.c_size_t]
        arguments += [guide_words, len(guide_raw)]
        device.shader_sha256['spatial_guide_fragment'] = hashlib.sha256(guide_raw).hexdigest()
    if setter(*arguments):
        raise BackendUnavailableError(device._error())
    device._progressive_query = query
    device._progressive_begin = begin
    device._progressive_reset = reset
    device._progressive_read = read
    device.shader_sha256['progressive_compute'] = hashlib.sha256(raw).hexdigest()
    return ProgressiveHistory(bounces)


def query_progressive(device):
    query = getattr(device, '_progressive_query', None)
    if query is not None and device._handle:
        value = ProgressiveInfo()
        if query(device._handle, ct.byref(value)):
            raise RuntimeError(device._error())
        device._progressive_info = {
            name: getattr(value, name) for name, _ in ProgressiveInfo._fields_
            if name not in ('reserved', 'reserved64')
        }
    value = getattr(device, '_progressive_info', None)
    return dict(value) if value is not None else None
