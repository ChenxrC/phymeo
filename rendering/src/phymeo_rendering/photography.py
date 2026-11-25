"""Public photographic camera and resident Vulkan renderer defaults."""
from math import radians, tan
from numbers import Real
from pathlib import Path

from .backends.native import NativeBackend
from .cameras import Camera
from .environment import RadianceEnvironment
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
    MSAA and 16 quadrature samples per rectangular emitter. ``panel_samples``
    is a count per emitter (4, 16 or 64); ``area_light_grid`` is its side
    length (2, 4 or 8). Consistent explicit settings can be supplied together.
    Other keyword arguments are ordinary NativeBackend options. Constructing
    the renderer or its cameras does not open a GPU session.
    """

    def __init__(self, *, size=STUDIO_SIZE, panel_samples=None, **native_options):
        self.size = _size(size)
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
                       shadow_mode='ray_query', exposure=STUDIO_EXPOSURE, display_transform=STUDIO_DISPLAY)
        options.update(native_options)
        if 'studio' not in options:
            options['studio'] = studio_lighting()
        super().__init__(**options)

    def camera(self, id, frame_key, world_pose, *, size=None, vertical_fov_degrees=45., near_m=.01, far_m=20.):
        """Construct a centred, square-pixel camera bound to this exposure."""
        if isinstance(vertical_fov_degrees, bool) or not isinstance(vertical_fov_degrees, Real):
            raise TypeError('vertical_fov_degrees must be a real angle')
        if not 0 < vertical_fov_degrees < 180:
            raise ValueError('vertical_fov_degrees must lie between 0 and 180')
        w, h = self.size if size is None else _size(size)
        focal = .5*h / tan(.5*radians(vertical_fov_degrees))
        return Camera(id, frame_key, world_pose, w, h, focal, focal, w/2, h/2, near_m, far_m)
