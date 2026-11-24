"""Metre-scaled repeat textures for the static R1 workshop benches and floor."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .finish_details import _store, _texture, _with_uv


RESOLUTION = 512
TILE_METRES = {"benchtop": .80, "floor": 2.56}


def _grain(rng, tile_m, sigma_u_m, sigma_v_m):
    """Periodic, Gaussian-filtered grain with a physical width in each direction."""
    size = RESOLUTION
    frequency_u = np.fft.rfftfreq(size, d=tile_m / size)[None, :]
    frequency_v = np.fft.fftfreq(size, d=tile_m / size)[:, None]
    radial = (sigma_u_m * frequency_u) ** 2 + (sigma_v_m * frequency_v) ** 2
    kernel = np.exp((-2 * np.pi ** 2) * radial)
    kernel[0, 0] = 0
    spectrum = np.fft.rfft2(rng.standard_normal((size, size)))
    result = np.fft.irfft2(spectrum * kernel)
    return result / result.std()


def _surface_images(surface):
    tile_m = TILE_METRES[surface]
    if surface == "benchtop":
        # Pigmented, matte laminate. The sparse directional wear runs along the
        # long Y axis of the tabletop and locally polishes its pebbled finish.
        rng = np.random.default_rng(73418)
        pebble = _grain(rng, tile_m, .0022, .0024)
        lay = _grain(rng, tile_m, .0028, .030)
        wear = np.clip(lay - .95, 0., 2.)
        binder = _grain(rng, tile_m, .028, .045)
        swatch = (148., 151., 150.)
        value = 2.6 * pebble + 1.7 * binder + 2.7 * wear
        roughness = .73 + .019 * pebble - .048 * wear
    else:
        # Ground concrete has a mottled paste and scattered darker aggregate.
        # Keep the aggregate soft at distance for the native single-mip sampler.
        rng = np.random.default_rng(21459)
        paste = _grain(rng, tile_m, .078, .064)
        sand = _grain(rng, tile_m, .0065, .007)
        aggregate = np.clip(sand - .85, 0., 2.2)
        swatch = (133., 139., 138.)
        value = 2.8 * paste + 1.4 * sand - 4.5 * aggregate
        roughness = .83 + .018 * paste - .022 * aggregate

    color = np.rint(np.clip(swatch + value[:, :, None], 0., 255.)).astype("u1")
    data = np.rint(255 * np.clip(roughness, 0., 1.)).astype("u1")
    return color, np.repeat(data[:, :, None], 3, axis=2)


def _planar_uv(mesh, tile_m):
    # append_workshop emits the static node and its instances at identity.
    # These vertices are in world metres; box edges already split by normal.
    points = np.asarray(mesh.positions)
    axes = np.argmax(np.abs(mesh.normals), axis=1)
    faces = axes[np.asarray(mesh.triangles)]
    if np.any(faces != faces[:, :1]):
        raise ValueError("workshop finish needs separate vertices on planar edges")
    uv = np.empty((len(points), 2), dtype="<f4")
    for axis, plane in enumerate(((2, 1), (0, 2), (0, 1))):
        select = axes == axis
        uv[select] = points[select][:, plane] / tile_m
    return uv


def apply_workshop_surfaces(scene, directory):
    """Bind world-projected sRGB color and linear roughness to the two static meshes."""
    from phymeo_rendering.assets import TexturedMaterial, TextureChannel
    from phymeo_rendering.mesh_data import _local_path, load_mesh
    from phymeo_rendering.scene import Pose
    from preview_stable_lighting.shell_normals import _publish

    directory = Path(directory).resolve()
    directory.mkdir(exist_ok=True, parents=True)
    meshes = {item.id: item for item in scene.meshes}
    materials = {item.id: item for item in scene.materials}
    bindings = {}
    for surface, tile_m in TILE_METRES.items():
        mesh_id = "r1-workshop-" + surface
        mat_id = mesh_id + "-material"
        users = [item for item in scene.instances if item.mesh_id == mesh_id or item.material_id == mat_id]
        if len(users) != 1 or (users[0].node_id, users[0].mesh_id, users[0].material_id,
                               users[0].local_pose) != ("r1-workshop", mesh_id, mat_id, Pose()):
            raise ValueError(f"expected one identity-bound static workshop instance for {surface}")
        original = meshes[mesh_id]
        mesh = load_mesh(original)
        raw = _local_path(original.uri).read_bytes()
        if sha256(raw).hexdigest() != mesh.source_sha256:
            raise ValueError(f"workshop geometry changed while mapping {mesh_id}")
        uv = _planar_uv(mesh, tile_m)
        path, mesh_digest = _store(directory, mesh_id + "-uv", _with_uv(raw, uv), "gltf")
        color_pixels, rough_pixels = _surface_images(surface)
        color = _texture(directory, surface + "-color", color_pixels)
        rough = _texture(directory, surface + "-roughness", rough_pixels)
        materials[mat_id] = TexturedMaterial(mat_id, (1., 1., 1.), 0., 1., color, TextureChannel(rough, "r"))
        meshes[mesh_id] = replace(original, uri=path.as_uri())
        bindings[mesh_id] = dict(material=mat_id, source_mesh_sha256=mesh.source_sha256,
                                 uv_mesh_sha256=mesh_digest, tile_metres=tile_m,
                                 color_srgb_sha256=color.sha256, roughness_linear_sha256=rough.sha256)

    receipt = json.dumps(dict(source_scene=scene.key, resolution=[RESOLUTION, RESOLUTION],
                              mapping="world metres, dominant-axis planar UV, bilinear repeat",
                              bindings=bindings), sort_keys=True).encode()
    _publish(directory / "workshop-surfaces.json", receipt)
    stamp = sha256(receipt).hexdigest()[:16]
    return replace(scene, key=f"{scene.key}:workshop-surfaces-{stamp}",
                   meshes=tuple(meshes.values()), materials=tuple(materials.values()))
