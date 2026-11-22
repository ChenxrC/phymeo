"""Prepare the pinned classic Panda visual surfaces without simplification.

Standard library only. COLLADA 1.4.1: metre/Z_UP, a flat visual scene, one affine
matrix per node, indexed triangles with POSITION and NORMAL, opaque Phong colors.
All source scene matrices are baked (row-major serialization, column vectors).
Normals use inverse transpose then unit normalization. Unsupported semantics
fail closed. The six source CAD line segments are retained as explicit metadata
and in lossless source archives; the triangle-only renderer cannot draw them.

The resulting embedded glTF uses FLOAT32 as required by shared load_mesh;
the exact source DAE bytes, original Phong values, matrix and rounding errors
are retained. This is a visual resource conversion, never a physical model.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.request
import xml.etree.ElementTree as ET

COMMIT = "30e598aa6fb703cc80a203481e6427f397337b4c"
REPOSITORY = "https://github.com/frankarobotics/franka_ros"
RAW = f"https://raw.githubusercontent.com/frankarobotics/franka_ros/{COMMIT}/"
PINS = {
    "link0": "050fe608136d11913287eaea6906538ddc9112cdbf7337a3181d9ed03f55f619",
    "link1": "b061af1bcfb57f32719a1ffda44290619549a41b33a37a49bfebe781b20039a4",
    "link2": "01ec435e7aee5812eb401489a83bf1717834eb5f21f8f6233e553b2da9be00bb",
    "link3": "ce28221fe968ee86787d594222b24d00aa0f47bb64f53eaacbaf6172e81c29b4",
    "link4": "d3a4029d1396471ad78ea5fd06e440167cb46c3ed7ab57befb2bcdbf1360efbf",
    "link5": "07bab2ced87c546bfbc15ce8249038db0e0e582de1607b4243ebd3669dac1f21",
    "link6": "4475cb620e2e7293eb0f317b5f968f6520f703a4e5d4f7b95f75249e4302fd21",
    "link7": "463227bda9eb9901637107068f59e861fcb81ac89880706d41c26d18c4b46abe",
    "hand": "fe5d445d509e44a9bd107ff78f0b03c49752c98e5e0a8ebafddf2b6cf5a8b380",
    "finger": "caefea762f2d18ca9412cf6c2e64e5007ad03571181d790a8e0c828b1b2035cf",
}
LICENSE_PINS = {
    "LICENSE": "a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9",
    "NOTICE": "d8ded0c268405e7847c281faa15042d381fbf7d308ec91d9941fbd943a6e1053",
}
METADATA_PINS = {
    "franka_hand.xacro": ("franka_description/robots/common/franka_hand.xacro",
                         "61bfc98d331fdcc9f340f028efff8b6ab6fbaf1db123384afb6c52d1c8ca9f17"),
    "utils.xacro": ("franka_description/robots/common/utils.xacro",
                   "f0124cd8ccb470b94be7c8da5e2dacf69671ec22c385013d34c9bed5409eca9b"),
    "package.xml": ("franka_description/package.xml",
                    "88832657e816997861731e2363477a2e02d0e3c95ffd1999c4d87cce16c0b2cc"),
}
NS = "{http://www.collada.org/2005/11/COLLADASchema}"
LIMIT = 8 * 1024 * 1024
IDENTITY = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path):
    with Path(path).open("rb") as stream:
        data = stream.read(LIMIT + 1)
    require(len(data) <= LIMIT, f"file exceeds {LIMIT} bytes: {path}")
    return data


def children(node, allowed):
    require(node is not None, "missing source element")
    require(all(child.tag in {NS + name for name in allowed} for child in node),
            f"unsupported children in {node.tag}")


def one(node, name):
    found = node.findall(NS + name)
    require(len(found) == 1, f"expected one {name}")
    return found[0]


def floats(text, count=None):
    try:
        result = tuple(float(value) for value in (text or "").split())
    except ValueError as exc:
        raise ValueError("invalid source numbers") from exc
    require(all(math.isfinite(value) for value in result), "nonfinite source numbers")
    require(count is None or len(result) == count, "source number count mismatch")
    return result


def integer(value):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid source integer") from exc
    require(result >= 0 and str(result) == value, "invalid nonnegative integer")
    return result


def reference(value):
    require(isinstance(value, str) and value.startswith("#") and len(value) > 1,
            "external or missing reference")
    return value[1:]


def by_id(nodes):
    result = {}
    for node in nodes:
        key = node.get("id")
        require(key and key not in result, "missing or duplicate source ID")
        result[key] = node
    return result


def matrix_data(node):
    children(node, {"matrix", "instance_geometry"})
    m = floats(one(node, "matrix").text, 16)
    require(m[12:] == (0, 0, 0, 1), "projective matrices are unsupported")
    a, b, c, d, e, f, g, h, i = (m[k] for k in (0, 1, 2, 4, 5, 6, 8, 9, 10))
    cof = ((e*i-f*h, f*g-d*i, d*h-e*g),
           (c*h-b*i, a*i-c*g, b*g-a*h),
           (b*f-c*e, c*d-a*f, a*e-b*d))
    det = a*cof[0][0] + b*cof[0][1] + c*cof[0][2]
    require(math.isfinite(det) and det > 0, "singular or reflected matrix unsupported")
    inverse_transpose = tuple(tuple(x / det for x in row) for row in cof)
    return m, inverse_transpose


def transform_point(m, point):
    result = tuple(sum(m[4*r+c]*point[c] for c in range(3)) + m[4*r+3] for r in range(3))
    require(all(math.isfinite(x) for x in result), "unrepresentable transformed point")
    return result


def transform_normal(matrix, normal):
    result = tuple(sum(matrix[r][c]*normal[c] for c in range(3)) for r in range(3))
    length = math.hypot(*result)
    require(math.isfinite(length) and length > 0, "invalid source normal")
    return tuple(x / length for x in result)


def decode_phong(effect):
    children(effect, {"profile_COMMON"})
    profile = one(effect, "profile_COMMON")
    children(profile, {"technique"})
    technique = one(profile, "technique")
    children(technique, {"phong"})
    phong = one(technique, "phong")
    colors = ("emission", "ambient", "diffuse", "specular", "transparent")
    scalars = ("shininess", "transparency", "index_of_refraction")
    children(phong, set(colors + scalars))
    values = {}
    for name in colors:
        channel = one(phong, name)
        children(channel, {"color"})
        values[name] = floats(one(channel, "color").text, 4)
        require(all(0 <= x <= 1 for x in values[name]), "color outside [0,1]")
    for name in scalars:
        channel = one(phong, name)
        children(channel, {"float"})
        values[name] = floats(one(channel, "float").text, 1)[0]
    require(one(phong, "transparent").get("opaque") == "A_ONE", "unsupported opacity mode")
    require(values["diffuse"][3] == 1 and values["transparent"] == (1, 1, 1, 1)
            and values["transparency"] == 1, "only fully opaque source materials supported")
    require(values["shininess"] >= 0, "negative shininess")
    # Current Material has no emission channel: nonzero would be silently lost.
    require(values["emission"][:3] == (0, 0, 0), "emissive source unsupported")
    return values


def source_arrays(mesh):
    result = {}
    for sid, source in by_id(mesh.findall(NS + "source")).items():
        children(source, {"float_array", "technique_common"})
        array = one(source, "float_array")
        values = floats(array.text, integer(array.get("count")))
        accessor = one(one(source, "technique_common"), "accessor")
        require(reference(accessor.get("source")) == array.get("id"), "accessor source mismatch")
        require(accessor.get("stride") == "3" and accessor.get("offset", "0") == "0",
                "only packed xyz arrays supported")
        require(len(values) == 3 * integer(accessor.get("count")), "accessor count mismatch")
        params = accessor.findall(NS + "param")
        require([(p.get("name"), p.get("type")) for p in params] ==
                [("X", "float"), ("Y", "float"), ("Z", "float")], "unsupported source layout")
        result[sid] = tuple(tuple(values[k:k+3]) for k in range(0, len(values), 3))
    return result


def decode_dae(path):
    data = read_bytes(path)
    require(b"<!DOCTYPE" not in data.upper() and b"<!ENTITY" not in data.upper(),
            "DTD and entities unsupported")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError("invalid COLLADA XML") from exc
    require(root.tag == NS + "COLLADA" and root.get("version") == "1.4.1", "COLLADA 1.4.1 required")
    children(root, {"asset", "library_images", "library_effects", "library_materials",
                    "library_geometries", "library_controllers", "library_visual_scenes", "scene"})
    asset = one(root, "asset")
    require(float(one(asset, "unit").get("meter")) == 1
            and one(asset, "up_axis").text == "Z_UP", "metre Z_UP source required")
    for name in ("library_images", "library_controllers"):
        for library in root.findall(NS + name):
            require(len(library) == 0, f"{name} contents unsupported")
    effects = {key: decode_phong(value) for key, value in
               by_id(one(root, "library_effects").findall(NS + "effect")).items()}
    materials = {}
    for key, value in by_id(one(root, "library_materials").findall(NS + "material")).items():
        children(value, {"instance_effect"})
        effect_id = reference(one(value, "instance_effect").get("url"))
        require(effect_id in effects, "unresolved effect")
        materials[key] = effects[effect_id]
    geometries = by_id(one(root, "library_geometries").findall(NS + "geometry"))
    scenes = by_id(one(root, "library_visual_scenes").findall(NS + "visual_scene"))
    active = reference(one(one(root, "scene"), "instance_visual_scene").get("url"))
    require(len(scenes) == 1 and active in scenes, "one active visual scene required")
    scene = scenes[active]
    children(scene, {"node"})
    used = set()
    parts = []
    for node in scene:
        matrix, normal_matrix = matrix_data(node)
        instance = one(node, "instance_geometry")
        gid = reference(instance.get("url"))
        require(gid in geometries and gid not in used, "missing or repeated geometry instance")
        used.add(gid)
        bindings = {}
        for binding in one(one(instance, "bind_material"), "technique_common"):
            require(binding.tag == NS + "instance_material" and len(binding) == 0,
                    "unsupported material binding")
            symbol, target = binding.get("symbol"), reference(binding.get("target"))
            require(symbol and symbol not in bindings and target in materials, "invalid material binding")
            bindings[symbol] = target
        mesh = one(geometries[gid], "mesh")
        children(mesh, {"source", "vertices", "triangles", "lines"})
        arrays = source_arrays(mesh)
        vertices = {}
        for vid, value in by_id(mesh.findall(NS + "vertices")).items():
            inputs = list(value)
            require(len(inputs) == 1 and inputs[0].tag == NS+"input"
                    and inputs[0].get("semantic") == "POSITION", "only POSITION vertices supported")
            sid = reference(inputs[0].get("source"))
            require(sid in arrays, "unresolved positions")
            vertices[vid] = arrays[sid]
        line_segments = []
        for lines in mesh.findall(NS + "lines"):
            children(lines, {"input", "p"})
            inp = one(lines, "input")
            require(inp.get("semantic") == "VERTEX" and inp.get("offset") == "0", "unsupported lines")
            vid = reference(inp.get("source"))
            require(vid in vertices, "unresolved line vertices")
            indices = tuple(integer(x) for x in (one(lines, "p").text or "").split())
            require(len(indices) == 2*integer(lines.get("count")), "line count mismatch")
            require(all(x < len(vertices[vid]) for x in indices), "line index out of range")
            line_segments.extend(tuple(transform_point(matrix, vertices[vid][j]) for j in indices[k:k+2])
                                 for k in range(0, len(indices), 2))
        triangle_nodes = mesh.findall(NS + "triangles")
        require(triangle_nodes, "geometry has no triangle surface")
        for ordinal, triangles in enumerate(triangle_nodes):
            children(triangles, {"input", "p"})
            inputs = triangles.findall(NS + "input")
            require(len(inputs) == 2, "only vertex and normal indices supported")
            lookup = {inp.get("semantic"): inp for inp in inputs}
            require(set(lookup) == {"VERTEX", "NORMAL"}, "unsupported triangle attributes")
            require({x.get("offset") for x in inputs} == {"0", "1"}, "unsupported index offsets")
            vid = reference(lookup["VERTEX"].get("source"))
            nid = reference(lookup["NORMAL"].get("source"))
            require(vid in vertices and nid in arrays, "unresolved triangle inputs")
            positions, normals = vertices[vid], arrays[nid]
            vi, ni = int(lookup["VERTEX"].get("offset")), int(lookup["NORMAL"].get("offset"))
            indices = tuple(integer(x) for x in (one(triangles, "p").text or "").split())
            count = integer(triangles.get("count"))
            require(count > 0 and len(indices) == 6*count, "triangle count mismatch")
            symbol = triangles.get("material")
            require(symbol in bindings, "unbound triangle material")
            target = bindings[symbol]
            output_positions, output_normals, output_indices, source_positions = [], [], [], []
            unique = {}
            for k in range(0, len(indices), 2):
                pair = (indices[k+vi], indices[k+ni])
                require(pair[0] < len(positions) and pair[1] < len(normals), "triangle index out of range")
                if pair not in unique:
                    unique[pair] = len(unique)
                    output_positions.append(transform_point(matrix, positions[pair[0]]))
                    source_positions.append(positions[pair[0]])
                    output_normals.append(transform_normal(normal_matrix, normals[pair[1]]))
                output_indices.append(unique[pair])
            require(len(unique) <= 65536 and count <= 65536, "source part exceeds shared mesh admission")
            parts.append({"geometry": gid, "primitive": ordinal, "node": node.get("id"),
                          "source_matrix_row_major": matrix, "material": target,
                          "phong": materials[target], "positions": tuple(output_positions),
                          "source_positions": tuple(source_positions),
                          "normals": tuple(output_normals),
                          "triangles": tuple(tuple(output_indices[k:k+3]) for k in range(0, len(output_indices), 3)),
                          "non_surface_lines_m": tuple(line_segments) if ordinal == 0 else ()})
    require(used == set(geometries), "unused source geometry would be lost")
    return tuple(parts)


def pbr_material(phong):
    return {"base_color": list(phong["diffuse"][:3]), "metallic": 0.0,
            "roughness": math.sqrt(2.0 / (phong["shininess"] + 2.0))}


def f32(value):
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("geometry exceeds FLOAT32") from exc
    require(math.isfinite(result), "geometry exceeds FLOAT32")
    return result


def _zero_area(points):
    a, b, c = points
    u = tuple(b[k]-a[k] for k in range(3))
    v = tuple(c[k]-a[k] for k in range(3))
    return (u[1]*v[2]-u[2]*v[1] == 0 and u[2]*v[0]-u[0]*v[2] == 0
            and u[0]*v[1]-u[1]*v[0] == 0)


def admitted_part(part):
    """Remove only proven source-zero-area indices, with an exhaustive receipt.

    Positive-area surfaces collapsing in FLOAT32 are rejected, never repaired,
    moved or dropped. Source-zero candidates are checked again with exact
    rational arithmetic on the parsed source coordinates, before any transform.
    Vertices/normals remain intact; retained source ordinals keep original order.
    """
    from fractions import Fraction

    rounded = tuple(tuple(f32(x) for x in p) for p in part["positions"])
    original = part.get("source_positions", part["positions"])
    kept, removed, runs = [], [], []
    raw_zero_count = rounded_zero_count = 0
    run_start = None
    for ordinal, triangle in enumerate(part["triangles"]):
        source_points = tuple(original[i] for i in triangle)
        zero = _zero_area(source_points)
        fp_zero = _zero_area(tuple(rounded[i] for i in triangle))
        raw_zero_count += int(zero)
        rounded_zero_count += int(fp_zero)
        if zero:
            exact = tuple(tuple(Fraction(x) for x in p) for p in source_points)
            require(_zero_area(exact), "source zero-area classification is not exact")
            removed.append({"source_ordinal": ordinal, "vertex_indices": triangle,
                "source_positions_m": source_points,
                "baked_positions_m": tuple(part["positions"][i] for i in triangle),
                "float32_positions_m": tuple(rounded[i] for i in triangle),
                "reason": "exact source zero area; no triangle raster surface coverage",
                "float32_zero_area": fp_zero,
                "position_rounding_max_abs_m": max(abs(rounded[i][k]-part["positions"][i][k])
                                                  for i in triangle for k in range(3))})
            if run_start is not None:
                runs.append((run_start, ordinal))
                run_start = None
        else:
            require(not fp_zero, "positive-area source triangle collapses in FLOAT32")
            kept.append(triangle)
            if run_start is None:
                run_start = ordinal
    if run_start is not None:
        runs.append((run_start, len(part["triangles"])))
    require(kept, "part has no visible triangle surface after exact zero-area classification")
    report = {"source_triangle_count": len(part["triangles"]),
        "source_triangle_runs": tuple(runs), "omitted_zero_area_triangles": tuple(removed),
        "source_zero_area_count": raw_zero_count, "float32_zero_area_count": rounded_zero_count,
        "quantization_surface_loss_count": 0}
    return {**part, "triangles": tuple(kept)}, report


def gltf_bytes(part):
    part, _ = admitted_part(part)
    positions = tuple(tuple(f32(x) for x in p) for p in part["positions"])
    normals = tuple(tuple(f32(x) for x in n) for n in part["normals"])
    pbytes = b"".join(struct.pack("<3f", *p) for p in positions)
    nbytes = b"".join(struct.pack("<3f", *n) for n in normals)
    ibytes = b"".join(struct.pack("<3I", *t) for t in part["triangles"])
    data = pbytes+nbytes+ibytes
    doc = {
        "asset": {"version": "2.0", "generator": "Phymeo pinned Panda visual preparation v1"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4}]}],
        "buffers": [{"byteLength": len(data), "uri": "data:application/octet-stream;base64,"+base64.b64encode(data).decode("ascii")}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pbytes)},
            {"buffer": 0, "byteOffset": len(pbytes), "byteLength": len(nbytes)},
            {"buffer": 0, "byteOffset": len(pbytes)+len(nbytes), "byteLength": len(ibytes)}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "min": [min(p[k] for p in positions) for k in range(3)],
             "max": [max(p[k] for p in positions) for k in range(3)]},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": len(part["triangles"])*3, "type": "SCALAR"}]}
    return (json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8")


def pinned_read(path, expected):
    data = read_bytes(path)
    require(digest(data) == expected, f"source SHA256 mismatch: {path}")
    return data


def download(source_dir):
    source_dir.mkdir(parents=True, exist_ok=True)
    records = [(name+".dae", RAW+"franka_description/meshes/visual/"+name+".dae", sha)
               for name, sha in PINS.items()]
    records += [(name, RAW+name, sha) for name, sha in LICENSE_PINS.items()]
    records += [(name, RAW+path, sha) for name, (path, sha) in METADATA_PINS.items()]
    for name, url, sha in records:
        path = source_dir/name
        if path.exists():
            pinned_read(path, sha)
            continue
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(LIMIT+1)
        require(len(data) <= LIMIT and digest(data) == sha, f"download SHA256 mismatch: {url}")
        with path.open("xb") as stream:
            stream.write(data)


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def gzip_bytes(data):
    # GzipFile specifies OS=255 consistently, unlike gzip.compress on Python 3.11.
    import io
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as stream:
        stream.write(data)
    return buffer.getvalue()


def prepare(source_dir, output):
    require(not (output/"manifest.json").exists() and not (output/"meshes").exists()
            and not (output/"original").exists(), "output already contains prepared assets")
    # Validate every input before publishing any prepared resource.
    raw_sources = {name: pinned_read(source_dir/(name+".dae"), sha) for name, sha in PINS.items()}
    licenses = {name: pinned_read(source_dir/name, sha) for name, sha in LICENSE_PINS.items()}
    metadata = {name: pinned_read(source_dir/name, sha) for name, (_, sha) in METADATA_PINS.items()}
    decoded = {name: decode_dae(source_dir/(name+".dae")) for name in PINS}
    manifest = {
        "schema": "phymeo-panda-visuals-v1", "source_repository": REPOSITORY,
        "source_commit": COMMIT, "unit": "m", "coordinate_frame": "authored URDF link local",
        "source_matrix_convention": "COLLADA row-major serialization, column-vector multiplication",
        "preparation": {
            "surface_simplification": False, "degenerate_indices": "only exact source zero-area indices omitted with ordinal mapping; positive-area FLOAT32 collapse rejected",
            "source_precision": "exact DAE bytes in original/*.dae.gz",
            "position_encoding": "baked float64 arithmetic then glTF FLOAT32",
            "normal_encoding": "inverse transpose of exact source matrix, unit normalize, then FLOAT32",
            "pbr_approximation": "source diffuse RGB as linear; metallic=0; roughness=sqrt(2/(shininess+2))",
            "pbr_limitations": "Phong specular/ambient/IOR retained but not represented by current opaque Material",
            "textures": "none in pinned source", "uv": "none in pinned source",
            "non_surface_lines": "retained in metadata and exact source; triangle backend does not draw lines",
            "right_finger_visual": "panda_rightfinger uses finger.dae at local Rz(pi), per franka_hand.xacro"},
        "sources": [], "licenses": [], "visual_placement_sources": [], "meshes": [], "materials": [], "instances": []}
    part_records = {}
    for name, parts in decoded.items():
        source = raw_sources[name]
        archive = gzip_bytes(source)
        archive_file = "original/"+name+".dae.gz"
        write_new(output/archive_file, archive)
        manifest["sources"].append({"file": name+".dae", "sha256": digest(source), "bytes": len(source),
            "url": RAW+"franka_description/meshes/visual/"+name+".dae",
            "archive_file": archive_file, "archive_sha256": digest(archive),
            "triangles": sum(len(p["triangles"]) for p in parts),
            "non_surface_line_segments": sum(len(p["non_surface_lines_m"]) for p in parts)})
        part_records[name] = []
        for index, raw_part in enumerate(parts):
            part, conversion = admitted_part(raw_part)
            mid = f"panda_visual_{name}_{index:02d}"
            matid = mid+"_material"
            relative = "meshes/"+name+f"-{index:02d}.gltf"
            encoded = gltf_bytes(part)
            write_new(output/relative, encoded)
            p_round = max(abs(f32(x)-x) for p in part["positions"] for x in p)
            n_round = max(abs(f32(x)-x) for n in part["normals"] for x in n)
            manifest["meshes"].append({"id": mid, "file": relative, "sha256": digest(encoded),
                "source_file": name+".dae", "source_geometry": part["geometry"],
                "source_node": part["node"], "source_primitive": part["primitive"],
                "source_matrix_row_major": part["source_matrix_row_major"],
                "vertices": len(part["positions"]), "triangles": len(part["triangles"]), **conversion,
                "position_rounding_max_abs_m": p_round, "normal_rounding_max_abs": n_round,
                "non_surface_lines_m": part["non_surface_lines_m"]})
            manifest["materials"].append({"id": matid, **pbr_material(part["phong"]),
                "source_file": name+".dae", "source_material": part["material"], "source_phong": part["phong"]})
            part_records[name].append((mid, matid))
    for name, data in licenses.items():
        if (output/name).exists():
            pinned_read(output/name, LICENSE_PINS[name])
        else:
            write_new(output/name, data)
        manifest["licenses"].append({"file": name, "sha256": digest(data), "url": RAW+name})
    for name, data in metadata.items():
        file = "original/"+name
        write_new(output/file, data)
        manifest["visual_placement_sources"].append({"file": file, "sha256": digest(data),
            "url": RAW+METADATA_PINS[name][0]})
    links = [(name, "panda_"+name) for name in PINS if name != "finger"]
    links += [("finger", "panda_leftfinger"), ("finger", "panda_rightfinger")]
    for name, node in links:
        for mid, matid in part_records[name]:
            manifest["instances"].append({"id": 100+len(manifest["instances"]), "mesh_id": mid,
                "material_id": matid, "node_id": node, "position_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 1.0, 0.0] if node == "panda_rightfinger" else [0.0, 0.0, 0.0, 1.0]})
    counts = {row["id"]: row["triangles"] for row in manifest["meshes"]}
    manifest["totals"] = {"unique_meshes": len(counts), "instances": len(manifest["instances"]),
        "source_triangles_instanced": sum(row["source_triangle_count"] for row in manifest["meshes"])+sum(len(p["triangles"]) for p in decoded["finger"]),
        "source_zero_area_triangles_instanced": sum(row["source_zero_area_count"] for row in manifest["meshes"]),
        "surface_triangles_instanced": sum(counts[row["mesh_id"]] for row in manifest["instances"]),
        "both_fingers_triangles": 2*sum(len(p["triangles"]) for p in decoded["finger"]),
        "non_surface_line_segments": sum(row["non_surface_line_segments"] for row in manifest["sources"])}
    encoded = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)+"\n").encode("utf-8")
    write_new(output/"manifest.json", encoded)
    return digest(encoded), manifest["totals"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new prepared asset directory")
    parser.add_argument("--download", action="store_true", help="fetch exact pinned missing sources first")
    args = parser.parse_args()
    if args.download:
        download(args.source_dir)
    sha, totals = prepare(args.source_dir, args.output)
    print(json.dumps({"manifest_sha256": sha, "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
