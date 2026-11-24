"""Station and nest identification plates, and a laser mark on the transfer block."""
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .finish_details import _store, _texture, _with_uv
from .workshop_set import PAYLOAD_NODE, STATIC_NODE


SIZE = (512, 256)
# Top-left origin. Every island has a surround wider than the bilinear filter.
BOARD = (8, 8, 504, 88)
PICK = (8, 105, 109, 175)
PLACE = (120, 105, 221, 175)
PART = (268, 98, 375, 245)
WHITE = (490.5 / SIZE[0], 227.5 / SIZE[1])


def _atlas():
    scale = 3
    image = Image.new("RGB", (SIZE[0] * scale, SIZE[1] * scale), "white")
    draw = ImageDraw.Draw(image)
    dark = (40, 55, 59)
    ivory = (225, 231, 220)
    ochre = (232, 170, 72)

    def fill(rect, color):
        left, top, right, bottom = rect
        draw.rectangle((left*scale, top*scale, right*scale-1, bottom*scale-1), fill=color)

    def legend(label, rect, color, height):
        # Draw in optical bounds, including the offset between the em box and
        # the ink. This keeps the two nest characters on a common baseline.
        left, top, right, bottom = (value * scale for value in rect)
        for family in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
            try:
                font = ImageFont.truetype(family, size=height * scale)
                break
            except OSError:
                continue
        else:
            font = ImageFont.load_default(size=height * scale)
        ink = draw.textbbox((0, 0), label, font=font)
        x = (left + right - ink[2] - ink[0]) / 2
        y = (top + bottom - ink[3] - ink[1]) / 2
        draw.text((round(x), round(y)), label, font=font, fill=color)

    # Porcelain-enamel tool-board plaque; the number has its own ochre field.
    fill(BOARD, dark)
    fill((395, 8, 504, 88), ochre)
    legend("ASSEMBLY", (18, 8, 385, 88), ivory, 65)
    legend("04", (397, 8, 502, 88), dark, 69)
    for tile, name in ((PICK, "A"), (PLACE, "B")):
        x0, y0, x1, y1 = tile
        fill(tile, dark)
        fill((x0+3, y0+3, x1-3, y1-3), ivory)
        fill((x0+3, y0+3, x0+12, y1-3), ochre if name == "A" else (96, 143, 144))
        legend(name, (x0+12, y0+4, x1-3, y1-5), dark, 64)

    # Neutral values modulate the orange finish on the existing +X end face.
    # Reserve pure white outside the engraving so all other faces keep their color.
    legend("P4", (271, 130, 372, 200), (68, 68, 68), 49)
    fill((291, 193, 351, 197), (85, 85, 85))
    # Soften the single-mip sampler's most distant footprint. The physical
    # glyph strokes remain substantially wider than this filter.
    image = image.resize(SIZE, Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(.65))
    return np.asarray(image)


def _bounds(tile):
    left, top, right, bottom = tile
    return ((left+.5)/SIZE[0], (top+.5)/SIZE[1], (right-.5)/SIZE[0], (bottom-.5)/SIZE[1])


def _plates(directory, layout):
    from _panda_scene import write_mesh

    positions, normals, triangles, texcoords = [], [], [], []

    def plate(center, width, height, right, up, tile):
        center = np.asarray(center)
        right, up = np.asarray(right), np.asarray(up)
        normal = np.cross(right, up)
        first = len(positions)
        for u, v in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)):
            positions.append(center + width*u*right + height*v*up)
            normals.append(normal)
        triangles.extend(((first, first+1, first+2), (first, first+2, first+3)))
        l, t, r, b = _bounds(tile)
        texcoords.extend(((l,b), (r,b), (r,t), (l,t)))

    # 2 mm proud of the louver fronts, clear of tools and the window sill.
    plate((-2.788, .02, 1.957), .868, .140, (0,1,0), (0,0,1), BOARD)
    table_z = layout.support_z - .050
    for (x, y), tile in ((layout.pick_xy, PICK), (layout.place_xy, PLACE)):
        # Placards lie 0.7 mm above the table. The near edge of either nest is
        # 79 mm from its center. Leave 12 mm before the label starts at 91 mm.
        plate((x+.116, y, table_z+.0007), .072, .050, (0,1,0), (-1,0,0), tile)
    return write_mesh(directory, "r1-workshop-identification", positions, normals,
                      triangles, texcoords=texcoords)


def _engraved_mesh(original, directory, layout):
    from phymeo_rendering.mesh_data import _local_path, load_mesh

    source = load_mesh(original)
    positions = np.asarray(source.positions)
    normals = np.asarray(source.normals)
    front = (normals[:, 0] > .99999) & np.isclose(positions[:, 0], layout.payload_size[0]/2)
    membership = front[np.asarray(source.triangles)]
    if membership.all(axis=1).sum() != 2 or (membership[:, :1] != membership).any():
        raise ValueError("expected the two flat, normal-split +X triangles of the transfer block")
    uv = np.tile(WHITE, (len(positions), 1))
    y, z = positions[front, 1:].T
    l, t, r, b = _bounds(PART)
    uv[front, 0] = l + (r-l) * (y-y.min()) / np.ptp(y)
    uv[front, 1] = b - (b-t) * (z-z.min()) / np.ptp(z)
    raw = _local_path(original.uri).read_bytes()
    path, _ = _store(directory, "r1-workpiece-engraving", _with_uv(raw, uv), "gltf")
    return replace(original, uri=path.as_uri())


def apply_markings(scene, directory, layout):
    """Bind one atlas to the fixed plates and to the original orange material."""
    from phymeo_rendering.assets import TexturedMaterial
    from phymeo_rendering.scene import Instance

    directory = Path(directory).resolve()
    directory.mkdir(exist_ok=True, parents=True)
    atlas = _texture(directory, "workshop-identification", _atlas())
    fixed = _plates(directory, layout)
    enamel = TexturedMaterial(fixed.id+"-material", (1.,1.,1.), .03, .56, atlas)
    material_instances = [item for item in scene.instances
                          if item.node_id == PAYLOAD_NODE and item.mesh_id == PAYLOAD_NODE+"-part"]
    body, = material_instances
    meshes = {item.id: item for item in scene.meshes}
    materials = {item.id: item for item in scene.materials}
    original = materials[body.material_id]
    meshes[body.mesh_id] = _engraved_mesh(meshes[body.mesh_id], directory, layout)
    materials[body.material_id] = TexturedMaterial(original.id, original.base_color,
                                                   original.metallic, original.roughness, atlas)
    identity = max(item.id for item in scene.instances) + 1
    return replace(scene, key=f"{scene.key}:idplates-{atlas.sha256[:16]}",
                   meshes=(*meshes.values(), fixed), materials=(*materials.values(), enamel),
                   instances=(*scene.instances, Instance(identity, fixed.id, enamel.id, STATIC_NODE)))
