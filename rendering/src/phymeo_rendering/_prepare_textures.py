"""Plan and fit the two resident Vulkan texture arrays against queried device limits."""
from array import array
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from math import sqrt
from numbers import Integral
import json
from pathlib import Path

from ._prepare_cache import publish_bytes
from .assets import TexturedMaterial, TextureMap
from .backends import UnsupportedRequestError
from .scene import VisualScene
from .texture_data import _MAX_FILE_BYTES, load_texture


_LIMITS = ('max_textures', 'max_layers', 'max_width', 'max_height', 'max_array_bytes')


def _limits(values):
    result = {}
    for name in _LIMITS:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer from the device')
        result[name] = int(value)
    return result


def _bindings(scene):
    """Use exactly the color/roughness pair and normal keys used by native packing."""
    bound = {i.material_id for i in scene.instances}
    groups = {'color': {}, 'normal': {}}
    for m in scene.materials:
        if m.id not in bound or not isinstance(m, TexturedMaterial):
            continue
        if m.base_color_map or m.roughness_map:
            groups['color'].setdefault((m.base_color_map, m.roughness_map), []).append(m.id)
        if m.normal_map:
            groups['normal'].setdefault(m.normal_map, []).append(m.id)
    return groups


def _maps(kind, key):
    if kind == 'normal':
        return (key,)
    color, rough = key
    return tuple(image for image in (color, rough.image if rough else None) if image)


def _usage(sizes):
    w = max((s[0] for s in sizes.values()), default=0)
    h = max((s[1] for s in sizes.values()), default=0)
    n = len(sizes)
    return dict(layers=n, array_width=w+2 if n else 0, array_height=h+2 if n else 0,
                source_rgba_bytes=sum(wi*hi*4 for wi, hi in sizes.values()),
                padded_array_bytes=n*(w+2)*(h+2)*4)


def _inside(size, cap_w, cap_h):
    """Largest same-aspect image within the cap, rounding the minor axis down."""
    w, h = size
    if w <= cap_w and h <= cap_h:
        return size
    if cap_w*h <= cap_h*w:
        return cap_w, max(1, cap_w*h//w)
    return max(1, cap_h*w//h), cap_h


def _plan(kind, group, sizes, caps, protected, allow_adapt):
    original, wanted, locked, mismatches = {}, {}, {}, []
    for key, ids in group.items():
        dims = [sizes[s] for s in _maps(kind, key)]
        ws, hs = zip(*dims)
        original[key] = (max(ws), max(hs))
        # Keep the same normalized primary UV for two differently sized images;
        # the smaller extent on each axis avoids synthesizing source detail.
        wanted[key] = (min(ws), min(hs))
        if len(set(dims)) > 1:
            if protected.intersection(ids) or not allow_adapt:
                raise UnsupportedRequestError(f'{kind} paired maps on {ids} require equal dimensions; got {dims}')
            mismatches.extend(ids)
        if protected.intersection(ids):
            locked[key] = original[key]
    before = _usage(original)
    report = dict(limits=caps, before=before, paired_size_mismatches=mismatches)
    if not group:
        return {}, report
    max_w = min(caps['max_width'], max(w for w, _ in wanted.values()))
    max_h = min(caps['max_height'], max(h for _, h in wanted.values()))
    locked_w = max((w for w, _ in locked.values()), default=1)
    locked_h = max((h for _, h in locked.values()), default=1)
    budget = caps['max_array_bytes']
    if max_w < locked_w or max_h < locked_h or 4*(locked_w+2)*(locked_h+2)*len(group) > budget:
        names = sorted(protected.intersection(i for ids in group.values() for i in ids))
        raise UnsupportedRequestError(f'{kind} array cannot fit device limits {caps}; protected materials {names}')

    def bounded(w, h):
        return {key: _inside(size, w, h) for key, size in wanted.items()}

    target = bounded(max_w, max_h)
    if _usage(target)['padded_array_bytes'] > budget:
        # One global array envelope. Only layers touching it need to shrink.
        # Keep any protected axis at least as large as its largest locked layer.
        scale = max(max_w, max_h)
        low, high = 0, scale
        while low < high:
            mid = (low+high+1)//2
            attempt = bounded(max(locked_w, max_w*mid//scale), max(locked_h, max_h*mid//scale))
            if _usage(attempt)['padded_array_bytes'] <= budget:
                low = mid
            else:
                high = mid-1
        target = bounded(max(locked_w, max_w*low//scale), max(locked_h, max_h*low//scale))
    if not allow_adapt and target != original:
        raise UnsupportedRequestError(f'{kind} array exceeds strict texture limits {caps}; requested {before}')
    return target, report


@lru_cache(maxsize=1)
def _srgb_tables():
    linear = [x/255 for x in range(256)]
    decode = [x/12.92 if x <= .04045 else ((x+.055)/1.055)**2.4 for x in linear]
    encode = bytes(round(255*(12.92*v if v <= .0031308 else 1.055*v**(1/2.4)-.055))
                   for v in (x/65535 for x in range(65536)))
    return decode, encode


def _resize(image, role, size):
    from PIL import Image
    rgb = Image.frombytes('RGB', (image.width, image.height), image.rgb8)
    box = Image.Resampling.BOX
    if role == 'roughness':
        return rgb.resize(size, box)
    if role == 'color':
        decode, encode = _srgb_tables()
        channels = []
        for band in rgb.split():
            values = array('f', map(decode.__getitem__, band.tobytes()))
            plane = Image.frombytes('F', rgb.size, values.tobytes()).resize(size, box)
            data = bytes(encode[min(65535, max(0, round(v*65535)))] for v in plane.getdata())
            channels.append(Image.frombytes('L', size, data))
        return Image.merge('RGB', channels)
    # Normal textures are linear UNORM. Average vectors before renormalizing;
    # exactly opposing directions fall back to the geometric tangent-space normal.
    bands = [band.convert('F').resize(size, box) for band in rgb.split()]
    data = bytearray()
    for r, g, b in zip(*(band.getdata() for band in bands)):
        x, y, z = r-127.5, g-127.5, b-127.5
        length = sqrt(x*x+y*y+z*z)
        if length <= 1.e-7:
            data.extend((128, 128, 255))
        else:
            data.extend(round(127.5+127.5*v/length) for v in (x, y, z))
    return Image.frombytes('RGB', size, bytes(data))


def _store(image, directory):
    encoded = BytesIO()
    image.save(encoded, format='PNG')
    raw = encoded.getvalue()
    if len(raw) > _MAX_FILE_BYTES:
        raise UnsupportedRequestError('adapted PNG exceeds the bounded texture loader file size')
    digest = sha256(raw).hexdigest()
    destination = directory / (digest+'.png')
    publish_bytes(destination, raw)
    return TextureMap(destination.resolve().as_uri(), digest)


def fit_textures(scene, *, color_limits, normal_limits, cache, protected_materials=(), allow_adapt=True):
    """Return (scene, JSON-ready report) for the actual resident color and normal arrays.

    Limits are the five queried max_* device fields. Only instanced materials
    consume layers. All bound originals undergo the standard bounded, hash-pinned
    decode before any derivative is written. Protected material IDs keep their
    exact maps. Strict mode rejects any required change. A changed scene gets a
    derived key; an already compatible scene is returned by identity.
    """
    if not isinstance(scene, VisualScene):
        raise TypeError('scene must be a VisualScene')
    if not isinstance(allow_adapt, bool):
        raise TypeError('allow_adapt must be bool')
    protected = {protected_materials} if isinstance(protected_materials, str) else set(protected_materials)
    unknown = protected.difference(m.id for m in scene.materials)
    if unknown:
        raise ValueError(f'unknown protected material IDs: {sorted(unknown)}')
    limits = {'color': _limits(color_limits), 'normal': _limits(normal_limits)}
    groups = _bindings(scene)
    sizes = {}
    for kind, group in groups.items():
        caps = limits[kind]
        if len(group) > min(caps['max_textures'], caps['max_layers']):
            raise UnsupportedRequestError(f'{kind} array needs {len(group)} unique layers; device limits {caps}')
        for key in group:
            for source in _maps(kind, key):
                if source not in sizes:
                    data = load_texture(source)
                    sizes[source] = (data.width, data.height)
                    del data
    report = dict(protected_materials=sorted(protected), reductions=[])
    targets = {}
    for kind, group in groups.items():
        targets[kind], report[kind] = _plan(kind, group, sizes, limits[kind], protected, allow_adapt)

    changes = {}
    def fit(source, role, target, material_id):
        if source is None or sizes[source] == target:
            return source
        key = (source, role, target)
        if key not in changes:
            img = _resize(load_texture(source), role, target)
            result = _store(img, Path(cache))
            sizes[result] = target
            row = dict(role=role, source=source.sha256, result=result.sha256, original=list(sizes[source]),
                       size=list(target), material_ids=[])
            changes[key] = (result, row)
            report['reductions'].append(row)
        result, row = changes[key]
        row['material_ids'].append(material_id)
        return result

    materials = []
    bound = {i.material_id for i in scene.instances}
    for m in scene.materials:
        if m.id in bound and isinstance(m, TexturedMaterial):
            color, rough, normal = m.base_color_map, m.roughness_map, m.normal_map
            if color or rough:
                target = targets['color'][(color, rough)]
                color = fit(color, 'color', target, m.id)
                if rough:
                    rough = replace(rough, image=fit(rough.image, 'roughness', target, m.id))
            if normal:
                normal = fit(normal, 'normal', targets['normal'][normal], m.id)
            if (color, rough, normal) != (m.base_color_map, m.roughness_map, m.normal_map):
                m = replace(m, base_color_map=color, roughness_map=rough, normal_map=normal)
        materials.append(m)
    result = scene
    if changes:
        signature = json.dumps(report['reductions'], sort_keys=True, separators=(',', ':')).encode()
        result = replace(scene, key=f'{scene.key}:textures-{sha256(signature).hexdigest()[:16]}',
                         materials=tuple(materials))
    for kind, group in _bindings(result).items():
        actual = {key: sizes[_maps(kind, key)[0]] for key in group}
        report[kind]['after'] = _usage(actual)
        ids = {material_id for materials in group.values() for material_id in materials}
        report[kind]['bound_instances'] = sum(i.material_id in ids for i in result.instances)
    return result, report
