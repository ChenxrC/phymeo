"""Fit the showcase's two native texture arrays after all room and object maps are known."""
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from math import isqrt

import cv2
import numpy as np
from PIL import Image

from galaxea_r1.finish_details import _texture
from phymeo_rendering.assets import TexturedMaterial
from phymeo_rendering.mesh_data import _local_path


def fit_collection_textures(scene, cache):
    cache.mkdir(parents=True, exist_ok=True)
    materials = [m for m in scene.materials if isinstance(m, TexturedMaterial)]
    pairs = {(m.base_color_map, m.roughness_map) for m in materials if m.base_color_map or m.roughness_map}
    normals = {m.normal_map for m in materials if m.normal_map}
    images = {color for color, _ in pairs if color}
    images.update(channel.image for _, channel in pairs if channel)
    reductions = []

    @lru_cache(None)
    def size(source):
        with Image.open(_local_path(source.uri, suffixes=('.jpg', '.jpeg', '.png'))) as img:
            return img.size

    def limit(sources, count):
        if not count:
            return 2048
        if count > 64:
            raise ValueError(f"{count} unique layers exceed the native texture array")
        w, h = map(max, zip(*(size(s) for s in sources)))
        budget = 128 * 1024 * 1024
        if (w + 2) * (h + 2) * 4 * count <= budget:
            return 2048
        return min(2048, isqrt(budget // (4 * count)) - 2)

    rgb_limit = limit(images, len(pairs))
    normal_limit = limit(normals, len(normals))

    @lru_cache(None)
    def fit(source, role, edge):
        if source is None or max(size(source)) <= edge:
            return source
        raw = _local_path(source.uri, suffixes=('.jpg', '.jpeg', '.png')).read_bytes()
        if sha256(raw).hexdigest() != source.sha256:
            raise ValueError("collection map content differs from its input binding")
        with Image.open(BytesIO(raw)) as img:
            data = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
        w, h = size(source)
        final = tuple(max(1, round(v * edge / max(w, h))) for v in (w, h))
        if role == "color":
            data = np.where(data <= .04045, data / 12.92, ((data + .055) / 1.055) ** 2.4)
        reduced = cv2.resize(data, final, interpolation=cv2.INTER_AREA)
        if role == "color":
            reduced = np.where(reduced <= .0031308, reduced * 12.92, 1.055 * reduced ** (1 / 2.4) - .055)
        elif role == "normal":
            vec = 2 * reduced - 1
            lengths = np.linalg.norm(vec, axis=2, keepdims=True)
            vec = np.where(lengths > 1.e-7, vec / np.maximum(lengths, 1.e-7), (0., 0., 1.))
            reduced = (vec + 1) * .5
        result = _texture(cache, role, np.rint(np.clip(reduced, 0, 1) * 255).astype('u1'))
        reductions.append(dict(role=role, source=source.sha256, result=result.sha256, original=[w, h], size=final))
        return result

    result = []
    for m in scene.materials:
        if isinstance(m, TexturedMaterial):
            rough = m.roughness_map
            if rough:
                rough = replace(rough, image=fit(rough.image, "roughness", rgb_limit))
            m = replace(m, base_color_map=fit(m.base_color_map, "color", rgb_limit),
                        roughness_map=rough, normal_map=fit(m.normal_map, "normal", normal_limit))
        result.append(m)
    info = dict(color_layers=len(pairs), normal_layers=len(normals), color_edge_cap=rgb_limit,
                normal_edge_cap=normal_limit, reductions=reductions)
    return replace(scene, materials=tuple(result)), info
