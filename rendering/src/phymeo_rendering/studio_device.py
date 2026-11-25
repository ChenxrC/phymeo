"""Inspect the system Vulkan loader and select an actual Phymeo graphics context."""
from contextlib import ExitStack
import ctypes as ct
import os

from .backends import BackendUnavailableError
from .backends._native_msaa import ResidentMsaaInfo


VULKAN_1_1 = (1 << 22) | (1 << 12)
VULKAN_1_2 = (1 << 22) | (2 << 12)
_RAY_EXTENSIONS = {'VK_KHR_ray_query', 'VK_KHR_acceleration_structure',
                   'VK_KHR_deferred_host_operations'}
_TYPES = {1: 'integrated', 2: 'discrete', 3: 'virtual', 4: 'cpu'}
_VALIDATION = 'VK_LAYER_KHRONOS_validation'


class StudioDeviceError(BackendUnavailableError):
    """A device admission failure with a machine-readable category and evidence."""

    def __init__(self, category, message, report=None):
        super().__init__(message)
        self.category = category
        self.report = dict(report or {}, category=category, error=message)


class _Application(ct.Structure):
    _fields_ = [('sType', ct.c_uint32), ('pNext', ct.c_void_p), ('name', ct.c_char_p),
                ('version', ct.c_uint32), ('engine', ct.c_char_p), ('engineVersion', ct.c_uint32),
                ('apiVersion', ct.c_uint32)]


class _Create(ct.Structure):
    _fields_ = [('sType', ct.c_uint32), ('pNext', ct.c_void_p), ('flags', ct.c_uint32),
                ('app', ct.POINTER(_Application)), ('layerCount', ct.c_uint32), ('layers', ct.c_void_p),
                ('extensionCount', ct.c_uint32), ('extensions', ct.c_void_p)]


class _Layer(ct.Structure):
    _fields_ = [('name', ct.c_char * 256), ('spec', ct.c_uint32), ('impl', ct.c_uint32),
                ('description', ct.c_char * 256)]


class _Extension(ct.Structure):
    _fields_ = [('name', ct.c_char * 256), ('spec', ct.c_uint32)]


class _Properties(ct.Structure):
    # Fixed VkPhysicalDeviceProperties prefix; aligned storage for the remaining
    # pipeline-cache UUID, limits and sparse properties, which native queries itself.
    _fields_ = [('api', ct.c_uint32), ('driver', ct.c_uint32), ('vendor', ct.c_uint32),
                ('device', ct.c_uint32), ('type', ct.c_uint32), ('name', ct.c_char * 256),
                ('tail', ct.c_uint64 * 512)]


class _DeviceID(ct.Structure):
    _fields_ = [('sType', ct.c_uint32), ('pNext', ct.c_void_p), ('uuid', ct.c_ubyte * 16),
                ('driver_uuid', ct.c_ubyte * 16), ('luid', ct.c_ubyte * 8),
                ('node_mask', ct.c_uint32), ('luid_valid', ct.c_uint32)]


class _Properties2(ct.Structure):
    _fields_ = [('sType', ct.c_uint32), ('pNext', ct.c_void_p), ('properties', _Properties)]


class _ImageFormat(ct.Structure):
    _fields_ = [('extent', ct.c_uint32 * 3), ('mips', ct.c_uint32), ('layers', ct.c_uint32),
                ('samples', ct.c_uint32), ('max_resource_bytes', ct.c_uint64)]


def _msaa_formats(query, handle):
    # Match resident_msaa.inc: RGBA16F color/transfer, D32F depth and the
    # sampled single-sample RGBA16F resolve. VkResult can also report OOM.
    counts = []
    for fmt, usage in ((97, 0x11), (126, 0x20), (97, 0x15)):
        out = _ImageFormat()
        status = query(handle, fmt, 1, 0, usage, 0, ct.byref(out))
        if status == -11:  # VK_ERROR_FORMAT_NOT_SUPPORTED
            counts.append(0)
        elif status == 0:
            counts.append(out.samples)
        else:
            return dict(msaa_query_error=f'vkGetPhysicalDeviceImageFormatProperties(format={fmt}, '
                        f'usage={usage}) failed (VkResult {status})')
    return dict(msaa_format_queries_supported=bool(counts[0] and counts[1] and counts[2] & 1))


def _bind(loader, name, result, *arguments):
    try:
        fn = getattr(loader, name)
    except AttributeError as error:
        raise StudioDeviceError('loader', f'Vulkan loader lacks {name}') from error
    fn.argtypes = arguments
    fn.restype = result
    return fn


def _enumerate(fn, arguments, item_type):
    count = ct.c_uint32()
    for _ in range(4):
        status = fn(*arguments, ct.byref(count), None)
        if status != 0 or count.value > 65536:
            break
        if not count.value:
            return []
        items = (item_type * count.value)()
        status = fn(*arguments, ct.byref(count), items)
        if status == 0:
            return list(items[:count.value])
        if status != 5:  # VK_INCOMPLETE: a driver or layer appeared during enumeration.
            break
    raise StudioDeviceError('icd', f'{fn.__name__} failed to enumerate Vulkan resources (VkResult {status})')


def inspect_vulkan():
    """Enumerate the same Vulkan physical indices as the native renderer, without a logical device."""
    name = 'vulkan-1.dll' if os.name == 'nt' else 'libvulkan.so.1'
    try:
        loader = ct.WinDLL(name) if os.name == 'nt' else ct.CDLL(name)
    except OSError as error:
        raise StudioDeviceError('loader', f'Could not load the Vulkan loader {name}: {error}') from error
    u32, ptr, i32 = ct.c_uint32, ct.c_void_p, ct.c_int32
    count = ct.POINTER(u32)
    if not hasattr(loader, 'vkEnumerateInstanceVersion'):
        raise StudioDeviceError('loader', 'The Vulkan loader provides only Vulkan 1.0. Phymeo requires Vulkan 1.1.')
    version = u32()
    result = _bind(loader, 'vkEnumerateInstanceVersion', i32, count)(ct.byref(version))
    if result != 0 or version.value < VULKAN_1_1:
        raise StudioDeviceError('loader', f'Vulkan 1.1 loader unavailable (VkResult {result}, version {version.value})')
    layers = _bind(loader, 'vkEnumerateInstanceLayerProperties', i32, count, ct.POINTER(_Layer))
    layer_names = {row.name.decode('utf8', 'replace') for row in _enumerate(layers, (), _Layer)}
    result = dict(loader=name, loader_api_version=version.value, validation_layer=_VALIDATION in layer_names,
                  adapters=[])
    create = _bind(loader, 'vkCreateInstance', i32, ct.POINTER(_Create), ptr, ct.POINTER(ptr))
    destroy = _bind(loader, 'vkDestroyInstance', None, ptr, ptr)
    devices = _bind(loader, 'vkEnumeratePhysicalDevices', i32, ptr, count, ct.POINTER(ptr))
    properties = _bind(loader, 'vkGetPhysicalDeviceProperties', None, ptr, ct.POINTER(_Properties))
    properties2 = (_bind(loader, 'vkGetPhysicalDeviceProperties2', None, ptr, ct.POINTER(_Properties2))
                   if hasattr(loader, 'vkGetPhysicalDeviceProperties2') else None)
    extensions = _bind(loader, 'vkEnumerateDeviceExtensionProperties', i32, ptr, ct.c_char_p, count,
                       ct.POINTER(_Extension))
    image_formats = _bind(loader, 'vkGetPhysicalDeviceImageFormatProperties', i32, ptr, i32, i32,
                          i32, u32, u32, ct.POINTER(_ImageFormat))
    app = _Application(name=b'phymeo-studio-probe', apiVersion=VULKAN_1_1)
    create_info = _Create(sType=1, app=ct.pointer(app))
    instance = ptr()
    status = create(ct.byref(create_info), None, ct.byref(instance))
    if status != 0:
        raise StudioDeviceError('icd', f'vkCreateInstance(Vulkan 1.1) failed (VkResult {status}). '
                               'Check the installed GPU driver, ICD manifest and its library.', result)
    try:
        for index, handle in enumerate(_enumerate(devices, (instance,), ptr)):
            p = _Properties()
            properties(handle, ct.byref(p))
            uuid = None
            if properties2 is not None and p.api >= VULKAN_1_1:
                identity = _DeviceID(sType=1000071004)  # VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES
                query = _Properties2(sType=1000059001, pNext=ct.cast(ct.pointer(identity), ptr))
                properties2(handle, ct.byref(query))
                if any(identity.uuid):
                    uuid = bytes(identity.uuid).hex()
            names = {row.name.decode('ascii') for row in _enumerate(extensions, (handle, None), _Extension)}
            result['adapters'].append(dict(index=index, name=p.name.decode('utf8', 'replace'),
                type=_TYPES.get(p.type, 'other'), api_version=p.api, driver_version=p.driver,
                vendor_id=p.vendor, device_id=p.device, device_uuid=uuid,
                missing_ray_extensions=sorted(_RAY_EXTENSIONS - names),
                **_msaa_formats(image_formats, handle)))
    finally:
        destroy(instance, None)
    if not result['adapters']:
        raise StudioDeviceError('icd', 'Vulkan enumerated no physical devices. Check the driver ICD and GPU access.', result)
    return result


def resident_capabilities(device, size, *, hdr):
    """Read the native framebuffer, HDR and exact format/usage/sample intersection."""
    info = device.info()
    w, h = size
    if w > info['max_width'] or h > info['max_height'] or w*h > 4096**2:
        raise StudioDeviceError('device', f'The requested camera extent {w}x{h} exceeds native device limits')
    if hdr and not info.get('resident_environment_available'):
        env = info.get('resident_environment_info') or {}
        if env and not env.get('supported'):
            raise StudioDeviceError('device', 'The GPU lacks the resident HDR environment format or independent blending')
        raise StudioDeviceError('native_installation', 'The native library or shaders lack the resident HDR environment ABI')
    result = dict(device_uuid=info['device_uuid'], native_device_name=info.get('device_name'), msaa4=False)
    try:
        version = device._lib.pr_resident_msaa_abi_version
        query = device._lib.pr_resident_msaa_info
        device._lib.pr_resident_set_msaa
    except AttributeError:
        return dict(result, msaa_installation_error='Native resident MSAA ABI exports are missing')
    version.argtypes, version.restype = [], ct.c_uint32
    query.argtypes, query.restype = [ct.c_void_p, ct.POINTER(ResidentMsaaInfo)], ct.c_int
    if ct.sizeof(ResidentMsaaInfo) != 128 or version() != 1:
        return dict(result, msaa_installation_error='Unsupported native resident MSAA ABI')
    caps = ResidentMsaaInfo()
    if query(device._handle, ct.byref(caps)) != 0:
        raise StudioDeviceError('native_runtime', device._error())
    if caps.msaa_abi_version != 1 or not caps.shader_compatible:
        return dict(result, msaa_installation_error='Resident shader or native MSAA ABI is incompatible')
    result.update(msaa4=bool(caps.color_sample_counts & caps.depth_sample_counts & 4 and
                             w <= caps.max_width and h <= caps.max_height),
                  msaa_max_extent=[caps.max_width, caps.max_height],
                  msaa_format_query_empty=not (caps.max_width or caps.max_height or caps.color_sample_counts
                                               or caps.depth_sample_counts))
    return result


# Only explicit feature/capacity checks from the native device constructor are
# recoverable. OOM, device loss, invalid shaders or validation errors propagate.
_RAY_FAILURES = {
    'ray-query shadows require Vulkan 1.2',
    'required ray-query device extension unavailable; no software fallback',
    'required ray-query/accelerationStructure/bufferDeviceAddress feature unavailable',
    'ray-query acceleration descriptor capacity unavailable',
    'ray-query float3 vertex format unavailable',
    'ray-query scratch alignment unavailable',
}
_DEVICE_FAILURES = {
    'Vulkan 1.1 device is required', 'four color attachments are required',
    'uniform capacity unavailable', 'required attachment format unavailable',
    'resident rendering requires eight fragment storage buffers', 'resident uniform capacity unavailable',
    'resident texture sampler capacity unavailable', 'resident sRGB linear texture format unavailable',
    'resident texture guard extent unavailable', 'required graphics/compute queue unavailable',
    'resident Vulkan device UUID differs from CUDA producer UUID',
}


def native_error(error):
    """Keep driver capability misses separate from broken installs and runtime failures."""
    if isinstance(error, StudioDeviceError):
        return error
    message = str(error)
    if message in _RAY_FAILURES:
        category = 'ray_query'
    elif message in _DEVICE_FAILURES:
        category = 'device'
    elif message in ('requested VK_LAYER_KHRONOS_validation is not installed', 'debug-utils messenger unavailable',
                     'vkCreateInstance VkResult=-6', 'vkCreateInstance VkResult=-7'):
        category = 'validation'
    elif message == 'vkCreateInstance VkResult=-9':
        category = 'icd'
    elif 'VkResult=' in message:
        category = 'native_runtime'
    elif any(term in message.lower() for term in ('abi', 'spir-v', 'shader', 'library')):
        category = 'native_installation'
    else:
        category = 'native_runtime'
    return StudioDeviceError(category, message)


def _ray_reason(adapter, loader_version):
    if min(adapter['api_version'], loader_version) < VULKAN_1_2:
        return 'Ray-query shadows require Vulkan 1.2 on the loader and GPU'
    missing = adapter['missing_ray_extensions']
    if missing:
        return 'Missing Vulkan ray-query extensions: ' + ', '.join(missing)
    return None


def select_device(renderer, scene, *, requested_index, requested_shadow, requested_samples):
    """Return one live scene device and the admission report. Close every discarded context."""
    inventory = inspect_vulkan()
    report = dict(inventory, profile=renderer.profile, requested_device_index=requested_index,
                  validation_enabled=renderer.validation, attempts=[])
    if renderer.validation and not inventory['validation_layer']:
        raise StudioDeviceError('validation', f'Requested {_VALIDATION} is not installed', report)
    adapters = list(inventory['adapters'])
    if not adapters:
        raise StudioDeviceError('icd', 'Vulkan enumerated no physical devices', report)
    if requested_index is not None:
        adapters = [a for a in adapters if a['index'] == requested_index]
        if not adapters:
            raise StudioDeviceError('device', f'Vulkan device index {requested_index} is unavailable', report)
    adapters = [a for a in adapters if a['type'] != 'cpu' and a['api_version'] >= VULKAN_1_1]
    if not adapters:
        raise StudioDeviceError('device', 'A non-CPU Vulkan 1.1 GPU is required for studio rendering', report)
    rank = {'discrete': 0, 'integrated': 1, 'virtual': 2, 'other': 3}
    adapters.sort(key=lambda a: (rank.get(a['type'], 4), a['index']))
    modes = (requested_shadow,) if requested_shadow else ('ray_query', 'software')
    hdr = renderer.studio is not None and renderer.studio.environment_map is not None
    ray_reasons = {}
    lacks_required_msaa = set()
    last_failure = None

    def reject(index, mode, category, message):
        nonlocal last_failure
        report['attempts'].append(dict(device_index=index, shadow_mode=mode, category=category, reason=message))
        if category == 'ray_query':
            ray_reasons[index] = message
        last_failure = StudioDeviceError(category, message, report)

    for mode in modes:
        # Retain at most one otherwise usable 1x candidate while looking for
        # a full-quality adapter in the same shadow mode.
        with ExitStack() as pending:
            fallback = None
            for adapter in adapters:
                index = adapter['index']
                if index in lacks_required_msaa:
                    continue
                if adapter.get('msaa_query_error'):
                    raise StudioDeviceError('native_runtime', adapter['msaa_query_error'], report)
                reason = _ray_reason(adapter, inventory['loader_api_version'])
                if mode == 'ray_query' and reason:
                    reject(index, mode, 'ray_query', reason)
                    continue
                with ExitStack() as owned:
                    try:
                        device = renderer._candidate_device(scene, index, mode)
                        owned.callback(device.close)
                        if renderer.resident:
                            caps = resident_capabilities(device, renderer.size, hdr=hdr)
                        else:
                            info = device.info()
                            caps = dict(msaa4=False, device_uuid=info['device_uuid'], native_device_name=info['device_name'])
                    except BackendUnavailableError as error:
                        issue = native_error(error)
                        if issue.category in ('device', 'ray_query'):
                            reject(index, mode, issue.category, str(issue))
                            continue
                        raise StudioDeviceError(issue.category, str(issue), report) from error
                    # The loader inventory and the native graphics context create
                    # separate Vulkan instances. Indices alone cannot bind them.
                    uuid = adapter.get('device_uuid')
                    if uuid:
                        matches = caps['device_uuid'] == uuid
                    else:
                        matches = caps.get('native_device_name') == adapter['name']
                        if sum(a['name'] == adapter['name'] for a in inventory['adapters']) != 1:
                            matches = False
                    if not matches:
                        raise StudioDeviceError('device', f'Vulkan device identity changed at index {index}: '
                                                f'enumerated {adapter["name"]} ({uuid}), opened '
                                                f'{caps.get("native_device_name")} ({caps["device_uuid"]})', report)
                    if requested_samples != 1 and caps.get('msaa_installation_error'):
                        raise StudioDeviceError('native_installation', caps['msaa_installation_error'], report)
                    if caps.get('msaa_format_query_empty') and adapter.get('msaa_format_queries_supported'):
                        raise StudioDeviceError('native_runtime', 'Native MSAA format query failed after the Vulkan '
                                                'loader confirmed the same formats', report)
                    if requested_samples == 4 and not caps['msaa4']:
                        lacks_required_msaa.add(index)
                        reject(index, mode, 'msaa', 'The GPU lacks 4x MSAA for the studio color/depth formats and extent')
                        continue
                    samples = requested_samples or (4 if caps['msaa4'] else 1)
                    degradations = []
                    if mode == 'software' and requested_shadow is None:
                        degradations.append(ray_reasons.get(index, 'Hardware ray-query rendering is unavailable'))
                    if requested_samples is None and samples == 1:
                        degradations.append('4x MSAA unavailable at this extent; using one color sample')
                    selected = dict(report, **caps, device_name=adapter['name'], device_index=index,
                                    shadow_mode=mode, msaa_samples=samples, degradations=degradations,
                                    ray_query=True if mode == 'ray_query' else
                                    (False if reason or index in ray_reasons else None))
                    if requested_samples is None and samples == 1 and len(adapters) > 1:
                        if fallback is None:
                            pending.enter_context(owned.pop_all())
                            fallback = device, selected
                        continue
                    pending.close()
                    owned.pop_all()
                    return device, selected
            if fallback is not None:
                pending.pop_all()
                return fallback
    if last_failure is not None:
        raise last_failure
    raise StudioDeviceError('device', 'No suitable graphics device is available', report)
