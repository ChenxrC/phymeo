"""UV-only finishes on measured, link-local Galaxea wheel and camera geometry."""
import argparse
import base64
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image


TARGETS = ("wheel_motor_link1", "wheel_motor_link2", "wheel_motor_link3", "zed_link")

# sRGB reflectance swatches and linear roughness. The bases and hub are
# coated dielectrics: the underlying metal is covered by a pigmented finish.
WHEEL = (((131, 141, 149), .36), ((46, 49, 51), .85))
CAMERA = (((191, 199, 201), .40), ((54, 67, 74), .26))
CAMERA_RIM = ((90, 101, 107), .34)

# A 0.5 mm texel in the CAD's local XY plane. Both windows lie well away
# from the chart edges, with solid shell texels reserved for non-front islands.
_CAMERA_MIN = np.array([-.064, -.064])
_CAMERA_EXTENT = np.array([.256, .192])
_CAMERA_WIDTH, _CAMERA_HEIGHT = 512, 384
_CAMERA_BLANK = (.5, .91)
_CAMERA_CENTERS = (0., .120)


def _with_uv(raw, uv):
    doc = json.loads(raw)
    primitive, = doc["meshes"][0]["primitives"]
    attrs = primitive["attributes"]
    if "TEXCOORD_0" in attrs:
        raise ValueError("expected the untextured converted R1 mesh")
    n = doc["accessors"][attrs["POSITION"]]["count"]
    uv = np.ascontiguousarray(uv, dtype="<f4")
    if uv.shape != (n, 2) or not np.isfinite(uv).all():
        raise ValueError("expected one finite UV pair for each original vertex")
    buffer, = doc["buffers"]
    prefix, encoded = buffer["uri"].split(",", 1)
    binary = base64.b64decode(encoded, validate=True)
    if len(binary) != buffer["byteLength"]:
        raise ValueError("source mesh buffer extent differs")
    start = (len(binary)+3) & ~3
    binary += bytes(start-len(binary)) + uv.tobytes()
    view = len(doc["bufferViews"])
    index = len(doc["accessors"])
    doc["bufferViews"].append(dict(buffer=0, byteOffset=start, byteLength=uv.nbytes))
    doc["accessors"].append(dict(bufferView=view, componentType=5126, type="VEC2", count=n))
    attrs["TEXCOORD_0"] = index
    buffer.update(byteLength=len(binary), uri=prefix+","+base64.b64encode(binary).decode("ascii"))
    return json.dumps(doc, separators=(",", ":")).encode()


def _store(directory, kind, data, extension):
    from preview_stable_lighting.shell_normals import _publish
    code = sha256(data).hexdigest()
    path = directory / f"{kind}-{code[:18]}.{extension}"
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != code:
        _publish(path, data)
    return path, code


def _texture(directory, kind, rgb):
    from phymeo_rendering.assets import TextureMap
    data = BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype="u1")).save(data, format="PNG")
    path, code = _store(directory, kind, data.getvalue(), "png")
    return TextureMap(path.as_uri(), code)


def _swatches(directory, name, palette):
    # Broad uniform texels around each sample prevent bilinear blending across
    # industrial finishes or a repeat seam. The stored RGB is sRGB, while the
    # roughness image is consumed in a linear channel.
    blend = np.linspace(0,1,64)
    blend = np.clip((blend-.35)/.3,0,1)
    a,b = palette
    base = np.rint((1.-blend[:,None])*a[0] + blend[:,None]*b[0]).astype("u1")
    rough = np.rint(255*((1.-blend)*a[1]+blend*b[1])).astype("u1")
    color = np.tile(base[None,:,:],(4,1,1))
    linear = np.tile(rough[None,:,None],(4,1,3))
    return _texture(directory, name+"-color", color), _texture(directory, name+"-roughness", linear)


def _groups(chunks, *, keep_normal_splits=False):
    """Measure components across chunks, optionally respecting existing normal seams."""
    positions = np.concatenate([np.asarray(mesh.positions, dtype=np.float32) for mesh in chunks])
    triangles = []
    first = 0
    for mesh in chunks:
        triangles.append(first + np.asarray(mesh.triangles, dtype=np.int32))
        first += len(mesh.positions)
    weld_keys = positions
    if keep_normal_splits:
        normals = np.concatenate([np.asarray(mesh.normals, dtype=np.float32) for mesh in chunks])
        weld_keys = np.column_stack((positions, normals))
    unique, lookup = np.unique(weld_keys, axis=0, return_inverse=True)
    vertices = unique[:, :3]
    lookup = lookup.reshape(-1)
    triangles = lookup[np.concatenate(triangles)]
    parent = np.arange(len(vertices))
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b, c in triangles:
        ra = root(a)
        parent[root(b)] = ra
        parent[root(c)] = ra
    _, labels = np.unique([root(i) for i in range(len(vertices))], return_inverse=True)
    face_component = labels[triangles[:, 0]]
    component = labels[lookup]
    rows = []
    for i in range(int(labels.max())+1):
        v = vertices[labels == i]
        radius = np.hypot(v[:, 0], v[:, 2])
        rows.append(dict(group=i, vertices=len(v), faces=int(np.count_nonzero(face_component == i)),
                         low=v.min(axis=0).tolist(), high=v.max(axis=0).tolist(),
                         radius_xz_m=[float(radius.min()), float(radius.max())]))
    return component, rows


def _wheel_weights(groups, rows):
    # The joint axis is local Y. Only the continuous contact ring reaches the
    # rolling radius. Hub covers reach ~65 mm too, so selecting every vertex
    # above a radius would paint part of the hub as rubber.
    radius = max(row["radius_xz_m"][1] for row in rows)
    tire = [row["group"] for row in rows if
            row["radius_xz_m"][1] > .99 * radius and row["radius_xz_m"][0] > .80 * radius and
            all(row["high"][i] - row["low"][i] > 1.9 * radius for i in (0, 2))]
    if len(tire) != 1:
        raise ValueError("could not identify one continuous outer wheel contact ring")
    weight = (groups == tire[0]).astype(np.float32)
    return weight, dict(wheel_axis="Y", rolling_radius_m=radius, tread_group=tire[0],
                        selection="connected annulus at the outer rolling radius in local XZ")


def _camera_uv(position, groups, rows, islands, island_rows):
    # A complete connected UV island has a single chart. The original CAD
    # normal splits already separate the shallow recesses from the back and
    # supports, so no geometry attributes or triangle indices need changing.
    main_component = max(rows, key=lambda row: row["faces"])["group"]
    x, y = position[:, :2].T
    front_ids = [row["group"] for row in island_rows
                 if row["low"][2] >= -.00005 and row["high"][2] <= .00245]
    front = (groups == main_component) & np.isin(islands, front_ids)
    evidence = []
    for center in _CAMERA_CENTERS:
        radius = np.hypot(x - center, y)
        if not np.any(front & (np.abs(radius - .0096) < .0001)):
            raise ValueError(f"missing ZED aperture surface at local X {center}")
        evidence.append(dict(center_xy_m=[center, 0.], measured_rings_m=[.0096, .0117, .0122, .0125]))
    uv = np.tile(_CAMERA_BLANK, (len(position), 1))
    uv[front] = (position[front, :2] - _CAMERA_MIN) / _CAMERA_EXTENT
    return uv, dict(windows=evidence, surface_group=main_component, local_z_range_m=[-.00005, .00245],
                    front_vertices=int(front.sum()), shell_vertices=int((~front).sum()),
                    front_islands=np.unique(islands[front]).tolist(), solid_shell_uv=_CAMERA_BLANK,
                    selection="whole existing normal-split front islands; affine local XY in a 2D atlas")


def _camera_images():
    u = (np.arange(_CAMERA_WIDTH) + .5) / _CAMERA_WIDTH
    v = (np.arange(_CAMERA_HEIGHT) + .5) / _CAMERA_HEIGHT
    x = _CAMERA_MIN[0] + _CAMERA_EXTENT[0] * u[None, :]
    y = _CAMERA_MIN[1] + _CAMERA_EXTENT[1] * v[:, None]
    radius = np.minimum(np.hypot(x - _CAMERA_CENTERS[0], y), np.hypot(x - _CAMERA_CENTERS[1], y))
    # The dark entrance window extends to the 9.6 mm recess. Its graphite
    # surround ends at the 12.5 mm rim. All remaining texels are housing.
    # Shading and highlight position come from the actual PBR normal and light.
    rim = np.clip((.0125 - radius) / .0008, 0., 1.)
    lens = np.clip((.0098 - radius) / .0005, 0., 1.)
    shell, glass = CAMERA
    def mix(channel):
        a = np.asarray(shell[channel])
        b = np.asarray(CAMERA_RIM[channel])
        c = np.asarray(glass[channel])
        outer = a + rim[:, :, None] * (b - a)
        return outer + lens[:, :, None] * (c - outer)
    color = np.rint(mix(0)).astype("u1")
    rough = np.rint(255 * mix(1)).astype("u1")
    return color, np.repeat(rough, 3, axis=2)


def _write_link(scene, link, directory, meshes, materials):
    from phymeo_rendering.assets import TexturedMaterial, TextureChannel
    from phymeo_rendering.mesh_data import load_mesh, _local_path
    links = [item for item in scene.instances if item.node_id == link]
    ids = list(dict.fromkeys(item.mesh_id for item in links))
    mat_ids = {item.material_id for item in links}
    if not links:
        return None
    if len(mat_ids) != 1 or any(item.node_id != link and
            (item.mesh_id in ids or item.material_id in mat_ids) for item in scene.instances):
        raise ValueError(f"R1 detail mesh or material shared across links: {link}")
    originals = [meshes[key] for key in ids]
    chunks = [load_mesh(mesh) for mesh in originals]
    if any(mesh.texcoords is not None for mesh in chunks):
        raise ValueError(f"R1 imported mesh already has UVs: {link}")
    position = np.concatenate([np.asarray(mesh.positions, dtype=np.float32) for mesh in chunks])
    groups, rows = _groups(chunks)
    if link == "zed_link":
        islands, island_rows = _groups(chunks, keep_normal_splits=True)
        uv, evidence = _camera_uv(position, groups, rows, islands, island_rows)
        color_pixels, rough_pixels = _camera_images()
        color = _texture(directory, "stereo-color", color_pixels)
        rough = _texture(directory, "stereo-roughness", rough_pixels)
        palette = dict(shell=CAMERA[0], rim=CAMERA_RIM, window=CAMERA[1])
    else:
        weight, evidence = _wheel_weights(groups, rows)
        if not np.any(weight == 1.) or not np.any(weight == 0.):
            raise ValueError(f"R1 {link} finish failed to distinguish two surfaces")
        evidence.update(tread_vertices=int(weight.sum()), hub_vertices=int((weight == 0.).sum()))
        uv = np.column_stack((.125 + .75 * weight, np.full(len(weight),.5)))
        palette = WHEEL
        color, rough = _swatches(directory, "wheel", palette)
    source_parts = []
    offset = 0
    for mesh, data in zip(originals, chunks):
        count = len(data.positions)
        raw = _local_path(mesh.uri).read_bytes()
        if sha256(raw).hexdigest() != data.source_sha256:
            raise ValueError("R1 mesh changed during UV derivation")
        path, code = _store(directory, mesh.id+"-uv", _with_uv(raw, uv[offset:offset+count]), "gltf")
        source_parts.append(dict(id=mesh.id, source_sha256=data.source_sha256, derived_sha256=code,
                                 vertices=count, triangles=len(data.triangles)))
        meshes[mesh.id] = replace(mesh, uri=path.as_uri())
        offset += count
    mat_id, = mat_ids
    materials[mat_id] = TexturedMaterial(mat_id, (1.,1.,1.), 0., 1., color, TextureChannel(rough,"r"))
    return dict(link=link, components=rows, mapping=evidence, parts=source_parts,
                color_srgb_and_linear_roughness=palette, base_sha256=color.sha256, roughness_sha256=rough.sha256)


def apply_details(scene, cache_dir):
    """Return a scene with UV-only meshes and rigidly bound color/roughness maps."""
    from preview_stable_lighting.shell_normals import _publish
    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    meshes = {mesh.id:mesh for mesh in scene.meshes}
    materials = {material.id:material for material in scene.materials}
    rows = []
    for link in TARGETS:
        row = _write_link(scene, link, directory, meshes, materials)
        if row is not None:
            rows.append(row)
    if not rows:
        return scene
    evidence = json.dumps(dict(units="metres", coordinates="original mesh axes before visual and joint transforms",
                               source_scene=scene.key, links=rows), sort_keys=True, allow_nan=False).encode()
    _publish(directory/"details.json", evidence)
    suffix = sha256(evidence).hexdigest()[:12]
    return replace(scene, key=f"{scene.key}:r1-details-{suffix}", meshes=tuple(meshes.values()),
                   materials=tuple(materials.values()))


def _front_profile(chunks):
    vertices = np.unique(np.concatenate([np.asarray(c.positions) for c in chunks]), axis=0)
    # The ZED's two optical-frame origins are on its local X baseline. A ring
    # is evidence for an aperture only when its two copies have similar depth.
    sample = vertices[vertices[:, 2] >= -.005]
    levels, counts = np.unique(np.round(sample[:, 2], 4), return_counts=True)
    planes = []
    for index in np.argsort(counts)[::-1][:20]:
        layer = sample[np.round(sample[:, 2], 4) == levels[index]]
        planes.append(dict(z=float(levels[index]), count=len(layer),
                           xy_min=layer[:, :2].min(axis=0).tolist(),
                           xy_max=layer[:, :2].max(axis=0).tolist()))
    rings = []
    for x in (0., .12):
        radii = np.round(np.hypot(sample[:, 0] - x, sample[:, 1]), 4)
        values, counts = np.unique(radii[radii < .035], return_counts=True)
        entries = []
        for index in np.argsort(counts)[::-1][:15]:
            layer = sample[radii == values[index]]
            entries.append(dict(radius=float(values[index]), count=len(layer),
                                z_range=[float(layer[:,2].min()), float(layer[:,2].max())]))
        rings.append(dict(center_x=x, rings=entries))
    return dict(planes=planes, radial_profiles=rings)


def inspect_files(paths, front_profile=False, surface_islands=False):
    from phymeo_rendering.assets import Mesh
    from phymeo_rendering.mesh_data import load_mesh
    chunks = [load_mesh(Mesh(p.stem,p.resolve().as_uri())) for p in paths]
    if front_profile:
        result = _front_profile(chunks)
    else:
        _, rows = _groups(chunks, keep_normal_splits=surface_islands)
        result = sorted(rows,key=lambda r:-r["faces"])
    print(json.dumps(result,allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meshes", type=Path, nargs="+")
    parser.add_argument("--front-profile", action="store_true")
    parser.add_argument("--surface-islands", action="store_true")
    args = parser.parse_args()
    inspect_files(args.meshes, args.front_profile, args.surface_islands)
