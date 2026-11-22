"""Opt-in neutral PT studio using public SAPIEN light and cubemap APIs.

This module imports neither SAPIEN nor NumPy. RGBE inputs are linear radiance,
not sRGB images; the public capture applies its one display transform later.
"""
from dataclasses import dataclass
from hashlib import sha256
from math import frexp, hypot, isfinite, pi
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory

from ..environment import RadianceEnvironment, write_sapien_faces
from . import UnsupportedRequestError


_MAX_INSTANCES = 256
_MAX_MESH_SLOTS = 256
_MAX_UPLOADED_TRIANGLES = 200_000
_MAX_UPLOADED_VERTICES = 600_000


def _finite(value, name, *, minimum=None, maximum=1e6):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        valid = isfinite(value) and abs(value) <= maximum
    except OverflowError:
        valid = False
    if not valid or (minimum is not None and value < minimum):
        raise ValueError(f"{name} is outside the finite supported range")


@dataclass(frozen=True, slots=True)
class SapienStudio:
    """World-Z-up, metre-scale neutral studio for the PT reference path.

    Two real rectangular emitters and a linear HDR environment are
    added to the declared scene lights. ``panel_size_scale`` spreads the same
    power over a larger emitter: radiance is divided by its square. This is a
    lighting preset, not a material or camera change. Strengths are relative
    renderer radiance units, not a calibrated photometric-lux claim.
    An explicit environment_map replaces only the uniform environment; its
    declared strength and yaw are applied once by the shared map converter.
    """
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_m: float = 1.0
    panel_size_scale: float = 1.0
    panel_strength: float = 1.0
    environment_radiance: float = 0.125
    environment_map: RadianceEnvironment | None = None

    def __post_init__(self):
        if not isinstance(self.center_m, tuple):
            raise TypeError("center_m must be an immutable xyz tuple")
        if len(self.center_m) != 3:
            raise ValueError("center_m must contain three coordinates")
        for value in self.center_m:
            _finite(value, "center_m")
        _finite(self.scale_m, "scale_m", minimum=0.01, maximum=100.0)
        _finite(self.panel_size_scale, "panel_size_scale", minimum=0.05, maximum=4.0)
        _finite(self.panel_strength, "panel_strength", minimum=0.0, maximum=100.0)
        # The official miss shader clamps environment samples to [0,100].
        _finite(self.environment_radiance, "environment_radiance", minimum=0.0, maximum=100.0)
        if self.environment_map is not None and not isinstance(self.environment_map, RadianceEnvironment):
            raise TypeError("environment_map must be RadianceEnvironment or None")
        if 0 < self.environment_radiance < 2.0**-128:
            raise ValueError("environment_radiance is below the RGBE nonzero range")


def studio_recipe(studio):
    panels = []
    for name, offset, half_size, radiance in (
        ("key", (-1.2, -1.4, 1.8), (0.8, 0.55), 6.0),
        ("fill", (1.4, 0.7, 1.2), (0.65, 0.8), 2.5),
    ):
        position = [float(c) + float(studio.scale_m)*d for c, d in zip(studio.center_m, offset)]
        inward = [-d for d in offset]
        length = hypot(*inward)
        emission = radiance * float(studio.panel_strength) / float(studio.panel_size_scale)**2
        panels.append({
            "name": name,
            "position_m": position,
            "emission_direction_world": [d/length for d in inward],
            "half_size_m": [d*float(studio.scale_m)*float(studio.panel_size_scale) for d in half_size],
            "radiance_linear_rgb": [emission]*3,
            "public_component": "RenderParallelogramLightComponent",
            "visible_geometry": False,
        })
    recipe = {
        "preset": "neutral-area-studio-v1",
        "center_m": list(studio.center_m), "scale_m": float(studio.scale_m),
        "panel_size_scale": float(studio.panel_size_scale),
        "panel_strength": float(studio.panel_strength),
        "environment_radiance": float(studio.environment_radiance),
        "panels": panels,
        "units": "positions and emitter half sizes in metres; colors linear relative radiance",
        "declared_scene_lights": "retained unchanged",
        "soft_shadow_method": "official RT samples emitter surface and traces finite shadow rays",
        "reflection_method": "official diffuse/GGX direct area-light estimator plus traced HDR environment",
        "support": "path_trace RGB only; no raster studio approximation",
    }
    if studio.environment_map is not None:
        environment = studio.environment_map
        recipe.update(
            environment_radiance=None,
            environment_source="directional-radiance-map",
            uniform_environment_radiance_ignored=float(studio.environment_radiance),
            environment_map={"uri": environment.uri, "sha256": environment.sha256,
                             "strength": float(environment.strength),
                             "yaw_degrees": float(environment.yaw_degrees),
                             "face_size": environment.face_size},
        )
    return recipe


def uniform_hdr(radiance):
    """A deterministic 64-square, flat RGBE Radiance file (no gamma encoding).

    The legacy flat payload is legal under FORMAT=32-bit_rle_rgbe and is
    supported by stb_image, SAPIEN's image loader. A uniform cubemap needs no
    orientation-dependent face resampling and has no seams.
    """
    _finite(radiance, "radiance", minimum=0.0, maximum=100.0)
    if 0 < radiance < 2.0**-128:
        raise ValueError("radiance is below the RGBE nonzero range")
    value = float(radiance)
    if value == 0:
        pixel = bytes(4)
    else:
        mantissa, exponent = frexp(value)
        code = int(mantissa * 256.0)
        pixel = bytes((code, code, code, exponent + 128))
    return b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 64 +X 64\n" + pixel*(64*64)


def validate_scene_slots(mesh_slots, instance_slots):
    if mesh_slots > _MAX_MESH_SLOTS:
        raise UnsupportedRequestError("SAPIEN scene exceeds 256 mesh slots")
    if instance_slots > _MAX_INSTANCES:
        raise UnsupportedRequestError("SAPIEN scene exceeds 256 instance slots")


def geometry_budget(counts, *, mesh_slots):
    """Bound actual duplicated uploads, not just distinct mesh filenames.

    The 134,924-triangle Panda visual plus the small authored table/cube fits;
    these explicit reference budgets are not hardware maximum claims.
    """
    validate_scene_slots(mesh_slots, len(counts))
    vertices = sum(pair[0] for pair in counts)
    triangles = sum(pair[1] for pair in counts)
    if vertices > _MAX_UPLOADED_VERTICES:
        raise UnsupportedRequestError("SAPIEN scene exceeds 600000 uploaded vertices")
    if triangles > _MAX_UPLOADED_TRIANGLES:
        raise UnsupportedRequestError("SAPIEN scene exceeds 200000 uploaded triangles")
    return {"mesh_slots": mesh_slots, "instances": len(counts),
            "uploaded_vertices": vertices, "uploaded_triangles": triangles,
            "limits": {"mesh_slots": _MAX_MESH_SLOTS, "instances": _MAX_INSTANCES,
                       "uploaded_vertices": _MAX_UPLOADED_VERTICES,
                       "uploaded_triangles": _MAX_UPLOADED_TRIANGLES},
            "scope": "per-instance uploaded geometry; shared per-mesh decoding guards unchanged"}


class StudioResources:
    """Per-scene resources; no modification of process-global light defaults."""
    def __init__(self, sapien, scene, studio):
        self.directory = None
        self.directory_path = None
        self.cubemap = None
        self.entities = []
        self.closed = False
        self.diagnostics = studio_recipe(studio)
        try:
            self.directory = TemporaryDirectory(prefix="phymeo-sapien-studio-")
            self.directory_path = Path(self.directory.name)
            if studio.environment_map is None:
                source = uniform_hdr(studio.environment_radiance)
                hdr_path = self.directory_path / "neutral-linear.hdr"
                hdr_path.write_bytes(source)
                faces = (hdr_path,) * 6
                environment_info = {
                    "encoding": "linear-radiance-rgbe", "face_count": 6,
                    "face_resolution": [64, 64], "radiance_rgb": [float(studio.environment_radiance)]*3,
                    "source_sha256": sha256(source).hexdigest(),
                    "source_file_bytes": len(source), "source_kind": "procedural uniform neutral HDR",
                    "encoded_radiance_rgb": [source[-4] * 2.0**(source[-1]-136)]*3,
                    "source_quantization": "RGBE shared 8-bit mantissa; default 0.125 is exact",
                    "orientation": "uniform all directions; world-Z-up panel recipe",
                }
            else:
                # This shared conversion owns the world-to-SAPIEN orientation,
                # linear gain and yaw. Do not transform or multiply again.
                faces, converted = write_sapien_faces(studio.environment_map, self.directory_path)
                environment_info = dict(converted)
                environment_info["encoding"] = "linear-radiance-rgbe"
            # Public six-file API. Both paths keep sources alive until the
            # owning scene has been cleared and its RenderSystem released.
            self.cubemap = sapien.render.RenderCubemap(*(str(path) for path in faces))
            scene.render_system.cubemap = self.cubemap
            environment_info.update(
                loader="public RenderCubemap(px,nx,py,ny,pz,nz)",
                source_lifetime="owned temporary file until scene close",
            )
            self.diagnostics["environment"] = environment_info
            if studio.panel_strength > 0:
                for panel in self.diagnostics["panels"]:
                    light = sapien.render.RenderParallelogramLightComponent()
                    light.color = panel["radiance_linear_rgb"]
                    light.set_shape(*panel["half_size_m"], pi/2)
                    entity = sapien.Entity()
                    entity.name = "phymeo-studio-" + panel["name"]
                    entity.add_component(light)
                    # SAPIEN light entities emit along +X; the renderer's GL
                    # conversion is internal, as for its directional lights.
                    rotation = sapien.math.shortest_rotation((1.0, 0.0, 0.0),
                                                            panel["emission_direction_world"])
                    entity.pose = sapien.Pose(panel["position_m"], rotation)
                    scene.add_entity(entity)
                    self.entities.append(entity)
            self.diagnostics["attached_area_lights"] = len(self.entities)
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                exc.add_note(f"studio source cleanup also failed: {cleanup_error}")
            raise

    @property
    def temporary_files_exist(self):
        return self.directory_path is not None and self.directory_path.exists()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.entities.clear()
        self.cubemap = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
