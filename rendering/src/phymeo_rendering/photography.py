"""Public photographic camera and resident Vulkan renderer defaults."""
from copy import copy, deepcopy
from math import radians, tan
from numbers import Real
from pathlib import Path
from threading import RLock

from . import studio_device
from .backends.native import NativeBackend
from .cameras import Camera
from .environment import RadianceEnvironment
from .scene import VisualScene
from .studio import StudioLighting


STUDIO_SIZE = (1440, 810)
STUDIO_EXPOSURE = 7.0
STUDIO_DISPLAY = 'pbr-neutral'


def studio_lighting():
    """The manipulation studio's two softboxes and pinned workshop HDR environment."""
    package = Path(__file__).resolve().parent
    resources = package / '_studio_assets'
    if not resources.exists() and package.parent.name == 'src':
        resources = package.parent.parent / 'examples' / 'materials'
    hdr = resources / 'university_workshop_1k.hdr'
    sky = RadianceEnvironment(hdr.as_uri(),
        'a125ada38fc602025326f29b68df09531bb16f899f1973b03d9ac5ac98142548',
        strength=.075, yaw_degrees=35., face_size=128)
    return StudioLighting(center_m=(.19, 0., .10), scale_m=.64,
                          panel_size_scale=.48, panel_strength=.135, environment_map=sky)


def _size(size):
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError('size must contain a width and height in pixels')
    if any(type(n) is not int or n < 1 for n in size):
        raise ValueError('camera dimensions must be positive integers')
    return tuple(size)


class StudioRenderer(NativeBackend):
    """Photographic Vulkan capture with the usual ``open_scene`` / ``capture`` API.

    Cameras default to 1440×810, with hardware ray-query shadows, 4× color
    MSAA and 16 quadrature samples per rectangular emitter. The default
    ``auto`` profile selects a GPU and uses GPU software shadows or 1× MSAA
    when the driver lacks the corresponding feature. Explicit shadow and sample
    settings are requirements. ``strict`` requires ray queries and 4× MSAA.
    ``compatible`` selects GPU software shadows and the best available MSAA.
    ``panel_samples``
    is a count per emitter (4, 16 or 64); ``area_light_grid`` is its side
    length (2, 4 or 8). Consistent explicit settings can be supplied together.
    Other keyword arguments are ordinary NativeBackend options. Constructing
    the renderer or its cameras does not open a GPU session.
    """

    def __init__(self, *, size=STUDIO_SIZE, panel_samples=None, profile='auto',
                 device_index=None, **native_options):
        self.size = _size(size)
        if profile not in ('auto', 'strict', 'compatible'):
            raise ValueError("profile must be 'auto', 'strict' or 'compatible'")
        explicit_shadow = native_options.get('shadow_mode')
        explicit_samples = native_options.get('msaa_samples')
        if profile == 'strict' and (explicit_shadow not in (None, 'ray_query') or explicit_samples not in (None, 4)):
            raise ValueError('The strict studio profile requires shadow_mode=ray_query and msaa_samples=4')
        if profile == 'compatible' and explicit_shadow not in (None, 'software'):
            raise ValueError('The compatible studio profile requires shadow_mode=software')
        if panel_samples is not None:
            if type(panel_samples) is not int:
                raise TypeError('panel_samples must be an integer count per rectangular light')
            grids = {4: 2, 16: 4, 64: 8}
            if panel_samples not in grids:
                raise ValueError('panel_samples must be 4, 16 or 64 per rectangular light')
            grid = grids[panel_samples]
            if native_options.setdefault('area_light_grid', grid) != grid:
                raise ValueError('panel_samples and area_light_grid specify different quadrature')
        options = dict(resident=True, area_light_grid=4, msaa_samples=4,
                       shadow_mode='software' if profile == 'compatible' else 'ray_query',
                       exposure=STUDIO_EXPOSURE, display_transform=STUDIO_DISPLAY)
        options.update(native_options)
        if 'studio' not in options:
            options['studio'] = studio_lighting()
        super().__init__(device_index=0 if device_index is None else device_index, **options)
        self.profile = profile
        self._requested_index = device_index
        self._requested_shadow = explicit_shadow
        if (profile != 'auto' or self.reflection_mode != 'environment' or self.indirect_mode != 'none'
                or self.area_light_grid != self.secondary_area_light_grid):
            self._requested_shadow = self.shadow_mode
        self._requested_samples = 4 if profile == 'strict' else explicit_samples
        self._admission_lock = RLock()
        self._device_report = None

    @classmethod
    def for_device(cls, *, profile='auto', device_index=None, **options):
        """Probe the installed driver and native context now, and return the configured renderer."""
        renderer = cls(profile=profile, device_index=device_index, **options)
        renderer.probe_device()
        return renderer

    def _candidate_device(self, scene, index, shadow_mode):
        # Device construction reads NativeBackend's validated settings. Use a
        # shallow config copy so unsuccessful candidates never alter this renderer.
        config = copy(self)
        config.device_index, config.shadow_mode = index, shadow_mode
        return NativeBackend._scene_device(config, scene)

    def _scene_device(self, scene):
        with self._admission_lock:
            if self._device_report is None:
                device, report = studio_device.select_device(
                    self, scene, requested_index=self._requested_index,
                    requested_shadow=self._requested_shadow, requested_samples=self._requested_samples)
                self.device_index = report['device_index']
                self.shadow_mode = report['shadow_mode']
                self.msaa_samples = report['msaa_samples']
                self._device_report = report
                return device
            device = self._candidate_device(scene, self.device_index, self.shadow_mode)
            try:
                info = device.info()
                uuid = self._device_report['device_uuid']
                if (info['device_uuid'] != uuid or
                        (not uuid.strip('0') and info['device_name'] != self._device_report['device_name'])):
                    raise studio_device.StudioDeviceError('device', 'Vulkan device identity changed after selection',
                                                         self._device_report)
            except BaseException:
                device.close()
                raise
            return device

    def probe_device(self):
        """Return JSON-ready driver, adapter, quality and degradation evidence.

        Before the first scene this opens and closes one selected native
        graphics context (and any rejected candidates). Scene preparation and
        upload share the selection; they still own separate native contexts.
        """
        with self._admission_lock:
            if self._device_report is None:
                device = self._scene_device(VisualScene('studio-device-probe', (), (), ()))
                try:
                    device.close()
                except BaseException:
                    self._device_report = None
                    raise
            return deepcopy(self._device_report)

    def camera(self, id, frame_key, world_pose, *, size=None, vertical_fov_degrees=45., near_m=.01, far_m=20.):
        """Construct a centred, square-pixel camera bound to this exposure."""
        if isinstance(vertical_fov_degrees, bool) or not isinstance(vertical_fov_degrees, Real):
            raise TypeError('vertical_fov_degrees must be a real angle')
        if not 0 < vertical_fov_degrees < 180:
            raise ValueError('vertical_fov_degrees must lie between 0 and 180')
        w, h = self.size if size is None else _size(size)
        focal = .5*h / tan(.5*radians(vertical_fov_degrees))
        return Camera(id, frame_key, world_pose, w, h, focal, focal, w/2, h/2, near_m, far_m)
