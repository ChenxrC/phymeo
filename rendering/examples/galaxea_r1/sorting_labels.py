"""One texture atlas for the stationary sorting sign and flush bin labels."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .finish_details import _texture


SIZE = (512, 256)
WALL = (8, 8, 504, 84)
BIN_A = (8, 104, 208, 218)
BIN_B = (232, 104, 432, 218)


def _artwork():
    scale = 2
    image = Image.new("RGB", (SIZE[0] * scale, SIZE[1] * scale), (226, 230, 219))
    draw = ImageDraw.Draw(image)
    ink = (32, 52, 58)
    ivory = (236, 239, 228)

    def rect(box, fill):
        x0, y0, x1, y1 = (n * scale for n in box)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=fill)

    def legend(text, box, height, fill):
        for font_name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
            try:
                font = ImageFont.truetype(font_name, size=height * scale)
                break
            except OSError:
                continue
        else:
            font = ImageFont.load_default(size=height * scale)
        ox, oy, right, bottom = draw.textbbox((0, 0), text, font=font)
        x0, y0, x1, y1 = (n * scale for n in box)
        xy = (round((x0 + x1 - right - ox) / 2), round((y0 + y1 - bottom - oy) / 2))
        draw.text(xy, text, fill=fill, font=font)

    rect(WALL, ink)
    legend("SORTING", WALL, 74, ivory)
    for tile, name in ((BIN_A, "A"), (BIN_B, "B")):
        x, y, right, bottom = tile
        rect(tile, ink)
        rect((x + 4, y + 4, right - 4, bottom - 4), ivory)
        if name == "A":
            rect((x + 18, y + 26, x + 80, y + 88), (211, 104, 44))
        else:
            corners = ((15, 57), (32, 27), (66, 27), (83, 57), (66, 87), (32, 87))
            draw.polygon([((x + u) * scale, (y + v) * scale) for u, v in corners],
                         fill=(53, 125, 167))
        legend(name, (x + 87, y + 5, right - 6, bottom - 5), 110, ink)
    return np.asarray(image.resize(SIZE, Image.Resampling.LANCZOS))


def apply_sorting_labels(scene, directory, layout):
    """Add identification to the surfaced static set. All moving items retain their assets."""
    from _panda_scene import write_mesh
    from phymeo_rendering.assets import TexturedMaterial
    from phymeo_rendering.scene import Instance
    from .workshop_set import STATIC_NODE

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    atlas = _texture(directory, "sorting-identification", _artwork())
    plates = [((-2.788, .02, 1.957), .868, .140, (0, 0, 1), WALL)]
    # The bin wall ends at X + .110 m. Each placard starts at X + .119 m,
    # on the fixed table in front of the bin, 0.7 mm above its laminate.
    for (x, y), tile in zip(layout.bin_centers, (BIN_A, BIN_B), strict=True):
        plates.append(((x + .146, y, layout.support_z - .0075 + .0007),
                       .095, .054, (-1, 0, 0), tile))

    positions, normals, triangles, uvs = [], [], [], []
    right = np.array((0, 1, 0))
    for center, width, height, up, tile in plates:
        up = np.asarray(up)
        first = len(positions)
        for u, v in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)):
            positions.append(center + width * u * right + height * v * up)
            normals.append(np.cross(right, up))
        triangles.extend(((first, first + 1, first + 2), (first, first + 2, first + 3)))
        x0, y0, x1, y1 = tile
        left, top = (x0 + .5) / SIZE[0], (y0 + .5) / SIZE[1]
        last, bottom = (x1 - .5) / SIZE[0], (y1 - .5) / SIZE[1]
        uvs.extend(((left, bottom), (last, bottom), (last, top), (left, top)))

    fixed = write_mesh(directory, "r1-sorting-identification", positions, normals,
                       triangles, texcoords=uvs)
    enamel = TexturedMaterial(fixed.id + "-material", (1., 1., 1.), 0., .62, atlas)
    mesh_bytes = (directory / (fixed.id + ".gltf")).read_bytes()
    stamp = sha256(bytes.fromhex(atlas.sha256) + mesh_bytes).hexdigest()[:16]
    identity = max(item.id for item in scene.instances) + 1
    return replace(scene, key=f"{scene.key}:sorting-labels-{stamp}", meshes=(*scene.meshes, fixed),
                   materials=(*scene.materials, enamel),
                   instances=(*scene.instances, Instance(identity, fixed.id, enamel.id, STATIC_NODE)))
