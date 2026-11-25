"""Import the ten original household GLBs into native resident studio geometry."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import struct

import numpy as np
from PIL import Image

from _panda_scene import write_mesh
from galaxea_r1.finish_details import _store
from phymeo_rendering.assets import Mesh, TexturedMaterial, TextureChannel, TextureMap
from phymeo_rendering.scene import Instance


@dataclass(frozen=True)
class HomeAsset:
    key: str
    source: Path
    node_key: str
    bounds: tuple
    meshes: tuple[Mesh, ...]
    materials: tuple[TexturedMaterial, ...]
    instances: tuple[Instance, ...]
    details: dict


def _order(path):
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r'(\d+)', path.name)]


def _read_glb(raw):
    magic, version, length = struct.unpack_from('<4sII', raw)
    if magic != b'glTF' or version != 2 or length != len(raw):
        raise ValueError('expected a complete glTF 2 binary file')
    size, kind = struct.unpack_from('<I4s', raw, 12)
    if kind != b'JSON':
        raise ValueError('GLB JSON chunk missing')
    doc = json.loads(raw[20:20 + size])
    offset = 20 + size
    length, kind = struct.unpack_from('<I4s', raw, offset)
    if kind != b'BIN\0' or offset + 8 + length != len(raw) or doc.get('extensionsRequired'):
        raise ValueError('expected one standard embedded GLB binary buffer')
    return doc, memoryview(raw)[offset + 8:]


def _accessor(doc, buf, i):
    accessor = doc['accessors'][i]
    view = doc['bufferViews'][accessor['bufferView']]
    if view['buffer'] != 0 or 'sparse' in accessor or accessor.get('normalized'):
        raise ValueError('expected uncompressed embedded source attributes')
    dtype = np.dtype({5121: 'u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}[accessor['componentType']])
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3}[accessor['type']]
    count = accessor['count']
    stride = view.get('byteStride', width * dtype.itemsize)
    start = accessor.get('byteOffset', 0)
    if count < 1 or start + (count - 1) * stride + width * dtype.itemsize > view['byteLength']:
        raise ValueError('invalid GLB accessor range')
    return np.ndarray((count, width), dtype=dtype, buffer=buf,
                      offset=view.get('byteOffset', 0) + start, strides=(stride, dtype.itemsize))


def _node_matrix(node):
    if 'matrix' in node:
        return np.asarray(node['matrix'], dtype=float).reshape(4, 4).T
    quaternion = np.array(node.get('rotation', (0, 0, 0, 1)), dtype=float)
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    rotation = np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                         (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                         (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))
    out = np.eye(4)
    out[:3, :3] = rotation @ np.diag(node.get('scale', (1, 1, 1)))
    out[:3, 3] = node.get('translation', (0, 0, 0))
    return out


def _parts(doc, buf, vendor):
    # Vendor scale is applied in glTF Y-up world. Rotate +90 about X for Z-up.
    y_to_z = np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    root = np.eye(4)
    root[:3, :3] = y_to_z @ np.diag(vendor['scale'])
    root = root @ np.array(vendor.get('transform_matrix', np.eye(4)), dtype=float)

    def visit(index, parent):
        node = doc['nodes'][index]
        matrix = parent @ _node_matrix(node)
        if 'mesh' in node:
            linear = matrix[:3, :3]
            for prim in doc['meshes'][node['mesh']]['primitives']:
                if prim.get('mode', 4) != 4:
                    raise ValueError('the studio expects triangle primitives')
                attrs = prim['attributes']
                pos = _accessor(doc, buf, attrs['POSITION']) @ linear.T + matrix[:3, 3]
                norm = _accessor(doc, buf, attrs['NORMAL']) @ np.linalg.inv(linear)
                uv = _accessor(doc, buf, attrs['TEXCOORD_0'])
                tri = _accessor(doc, buf, prim['indices']).reshape(-1, 3)
                if np.linalg.det(linear) < 0:
                    tri = tri[:, (0, 2, 1)]
                yield prim['material'], pos, norm, uv, tri
        for child in node.get('children', ()):
            yield from visit(child, matrix)

    for index in doc['scenes'][doc.get('scene', 0)]['nodes']:
        yield from visit(index, root)


def _image(doc, buf, ref, role, folder):
    if ref.get('texCoord', 0) != 0 or ref.get('extensions') or ref.get('scale', 1) != 1:
        raise ValueError('expected untransformed primary-UV texture, unit normal scale')
    image = doc['images'][doc['textures'][ref['index']]['source']]
    view = doc['bufferViews'][image['bufferView']]
    start = view.get('byteOffset', 0)
    raw = bytes(buf[start:start + view['byteLength']])
    with Image.open(BytesIO(raw)) as src:
        rgb = src.convert('RGB')  # All imported glTF materials use OPAQUE.
        original_size = rgb.size
    size = tuple(max(1, round(n * min(1, 1024 / max(original_size)))) for n in original_size)
    if size != original_size:
        if role == 'color':
            # Average albedo in linear light then re-encode for the sRGB sampler.
            srgb = np.arange(256, dtype='f4') / 255
            linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
            channels = []
            for channel in rgb.split():
                data = linear[np.asarray(channel)]
                down = np.asarray(Image.fromarray(data).resize(size, Image.Resampling.BOX))
                out = np.where(down <= .0031308, 12.92 * down, 1.055 * down ** (1 / 2.4) - .055)
                channels.append(Image.fromarray(np.uint8(np.clip(out * 255 + .5, 0, 255))))
            rgb = Image.merge('RGB', channels)
        else:
            rgb = rgb.resize(size, Image.Resampling.BOX)
            if role == 'normal':
                v = np.asarray(rgb).astype('f4') * (2 / 255) - 1
                v /= np.maximum(np.linalg.norm(v, axis=2, keepdims=True), 1e-8)
                rgb = Image.fromarray(np.uint8(np.clip((v + 1) * 127.5 + .5, 0, 255)))
    packed = BytesIO()
    rgb.save(packed, 'PNG', compress_level=2)
    path, digest = _store(folder, role, packed.getvalue(), 'png')
    details = dict(source_name=image['name'], source_sha256=sha256(raw).hexdigest(),
                   source_size=original_size, derived_sha256=digest, derived_size=rgb.size)
    return TextureMap(path.resolve().as_uri(), digest), rgb, details


def _surface_maps(doc, buf, mat, folder):
    if mat.get('alphaMode', 'OPAQUE') != 'OPAQUE':
        raise ValueError('transparent GLB material needs a dedicated render path')
    pbr = mat['pbrMetallicRoughness']
    color, _, col = _image(doc, buf, pbr['baseColorTexture'], 'color', folder)
    packed, mr, mr_info = _image(doc, buf, pbr['metallicRoughnessTexture'], 'metal-rough', folder)
    normal, _, nrm = _image(doc, buf, mat['normalTexture'], 'normal', folder)
    factors = tuple(pbr.get('baseColorFactor', (1, 1, 1, 1))[:3]), pbr.get('roughnessFactor', 1.)
    metal = (np.asarray(mr.getchannel('B')).astype('f4') / 255) * pbr.get('metallicFactor', 1.)
    return (color, TextureChannel(packed, 'g'), normal, factors, metal,
            dict(name=mat.get('name'), color=col, metallic_roughness=mr_info, normal=nrm))


def _load_one(source, folder, ordinal):
    group = source.parent.parent.name
    raw = source.read_bytes()
    doc, buf = _read_glb(raw)
    metadata, = source.parent.glob('model_data*.json')
    vendor = json.loads(metadata.read_text(encoding='utf-8-sig'))
    parts = list(_parts(doc, buf, vendor))
    low = np.min([part[1].min(axis=0) for part in parts], axis=0)
    high = np.max([part[1].max(axis=0) for part in parts], axis=0)
    origin = (low + high) * .5
    origin[2] = low[2]
    node = 'home-' + group
    meshes, materials, instances = [], [], []
    material_cache = {}
    stats = []
    folder.mkdir(parents=True, exist_ok=True)
    for part_id, (mi, pos, norm, uv, tri) in enumerate(parts):
        if mi not in material_cache:
            material_cache[mi] = _surface_maps(doc, buf, doc['materials'][mi], folder)
        color, rough, normal_map, (color_factor, rough_factor), metal, _ = material_cache[mi]
        # Check the actual stored FP32 positions and UVs in FP64 intermediates,
        # exactly as the strict geometry and derivative-tangent importers do.
        positions = (pos-origin).astype('<f4')
        corners = positions[tri].astype(float)
        face_vectors = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        good = np.any(face_vectors != 0, axis=1)
        dropped = int((~good).sum())
        tri, face_vectors = tri[good], face_vectors[good]
        sizes = np.linalg.norm(norm, axis=1)
        if np.any(sizes == 0):
            summed = np.zeros_like(norm)
            for axis in range(3):
                np.add.at(summed, tri[:, axis], face_vectors)
            norm[sizes == 0] = summed[sizes == 0]
            sizes = np.linalg.norm(norm, axis=1)
        if np.any(sizes[tri] == 0):
            raise ValueError(f'{group}: original shaded vertex has no defined normal')
        norm = norm / np.where(sizes == 0, 1, sizes)[:, None]
        uvs = uv[tri].astype(float)
        u, v = uvs[:, 1]-uvs[:, 0], uvs[:, 2]-uvs[:, 0]
        a, b = u[:, 0]*v[:, 1], u[:, 1]*v[:, 0]
        stable = np.abs(a-b) > (2**-18) * (np.abs(a)+np.abs(b))
        # Native metallic is scalar. Use the source B channel at triangle
        # centroids to group faces by metallicity; preserve source UVs and all
        # original nonzero-area triangles, including singular-UV triangles.
        h, w = metal.shape
        coord = np.floor((uvs.mean(axis=1) % 1.) * [w, h]).astype(int)
        metallic = metal[coord[:, 1], coord[:, 0]]
        slots = np.rint(metallic * 7).astype(int) * 2 + stable
        for slot in np.unique(slots):
            selected = slots == slot
            refs, reindex = np.unique(tri[selected].ravel(), return_inverse=True)
            key = f'{node}-p{part_id}-s{slot}'
            if len(refs) > 65536 or selected.sum() > 65536:
                raise ValueError(f'{key}: primitive exceeds native mesh-file capacity')
            mesh = write_mesh(folder, key, positions[refs], norm[refs], reindex.reshape(-1, 3), texcoords=uv[refs])
            average = float(np.average(metallic[selected], weights=np.linalg.norm(face_vectors[selected], axis=1)))
            mat = TexturedMaterial(key+'-pbr', color_factor, average, rough_factor,
                                   color, rough, normal_map if slot % 2 else None)
            meshes.append(mesh)
            materials.append(mat)
            instances.append(Instance(2000+ordinal*100+len(instances), mesh.id, mat.id, node))
        stats.append(dict(source_triangles=len(good), retained_triangles=len(tri), zero_area_removed=dropped,
                          derivative_normal_triangles=int(stable.sum()),
                          singular_uv_using_original_vertex_normals=int((~stable).sum())))
    bounds = tuple(tuple(float(n) for n in x-origin) for x in (low, high))
    details = dict(group=group, variant=source.parent.name, source=str(source), source_sha256=sha256(raw).hexdigest(),
                   metadata=str(metadata), vendor_scale=vendor['scale'], bounds_m=bounds, nodes=len(doc['nodes']),
                   material_maps=[entry[-1] for entry in material_cache.values()], parts=stats,
                   metallic='source packed B sampled at triangle centroid, up to eight area-weighted material levels',
                   uv_origin='glTF top left', frame='source node transforms then vendor scale then Y-up to Z-up',
                   tangent_frame='native UV derivatives; original normals on singular UV faces')
    return HomeAsset(group, source, node, bounds, tuple(meshes), tuple(materials), tuple(instances), details)


def load_home_assets(source_root: Path, cache: Path) -> tuple[HomeAsset, ...]:
    """First ten numbered asset groups, lowest numbered authored base per group."""
    groups = sorted((d for d in Path(source_root).iterdir() if d.is_dir() and re.match(r'^\d+_', d.name)),
                    key=_order)[:10]
    if len(groups) != 10:
        raise ValueError('expected at least ten numbered household asset groups')
    result = []
    for ordinal, group in enumerate(groups):
        variants = sorted((d for d in group.iterdir() if d.is_dir() and re.fullmatch(r'base\d+', d.name)),
                          key=_order)
        if not variants:
            raise ValueError(f'{group.name} has no numbered base variants')
        source = (variants[0] / (variants[0].name + '.glb')).resolve(strict=True)
        result.append(_load_one(source, Path(cache).resolve()/group.name, ordinal))
    return tuple(result)
