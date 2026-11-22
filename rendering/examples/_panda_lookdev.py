"""Appearance-only dressing for the archived Panda video (metres, linear RGB).

lookdev_scene keeps every input mesh, node attachment and local pose. Panda
manufacturer materials and the red payload remain unchanged. The original small
table gets a matte rubber finish and its existing support a metal finish; neither
is a new physical asset. All other additions are explicitly named lookdev props.

No texture contract exists yet: coherent wood grain is represented by bounded
triangle/material groups, with 130 mm nominal planks and roughly 3 mm grain
spacing on the 1.7 x 1.3 m original display board. This is authored procedural
geometry, not an image filter or a calibrated wood BRDF. Metal is isotropic
metallic/roughness PBR; "satin" does not claim anisotropic brushed reflection.

Added geometry: 12,800 wood triangles plus two 48-triangle metal rims = 12,896.
Decoration instance IDs are 20..45 (below the original Panda IDs 100..158).
The caller must bind replay frames/cameras to the returned, distinct scene.key.
No rendering, simulation, random generator, or retained mutable cache is used.
"""

import math
from dataclasses import replace
from pathlib import Path

from _panda_scene import write_mesh
from phymeo_rendering.assets import Material
from phymeo_rendering.mesh_data import load_mesh
from phymeo_rendering.scene import Instance, VisualScene

WOOD_PALETTE_SIZE = 24
WOOD_PLANKS = 10
WOOD_ROWS_PER_PLANK = 40
WOOD_LENGTH_SEGMENTS = 16
ADDED_TRIANGLES = 12_896
DECORATION_INSTANCE_IDS = tuple(range(20, 46))
_PREFIX = "lookdev-"


def _box_bounds(scene, instance):
    """Read actual FLOAT32 render geometry; do not guess the physical dimensions."""
    mesh = next(mesh for mesh in scene.meshes if mesh.id == instance.mesh_id)
    geometry = load_mesh(mesh)
    low = tuple(min(p[axis] for p in geometry.positions) for axis in range(3))
    high = tuple(max(p[axis] for p in geometry.positions) for axis in range(3))
    corners = set(geometry.positions)
    if (
        len(geometry.triangles) != 12
        or len(corners) != 8
        or any(high[axis] <= low[axis] for axis in range(3))
        or any(p[axis] not in (low[axis], high[axis]) for p in corners for axis in range(3))
    ):
        raise ValueError("lookdev requires the existing axis-aligned box display props")
    return low, high


def _quad(group, points):
    """Append a flat-shaded nondegenerate CCW quad to a material group."""
    positions, normals, triangles = group
    a, b, c, _ = points
    u = tuple(b[k] - a[k] for k in range(3))
    v = tuple(c[k] - a[k] for k in range(3))
    cross = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    length = math.hypot(*cross)
    if not math.isfinite(length) or length == 0:
        raise ValueError("degenerate lookdev surface")
    normal = tuple(component / length for component in cross)
    offset = len(positions)
    positions.extend(points)
    normals.extend((normal,) * 4)
    triangles.extend(((offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3)))


def _wood_groups(low, high):
    """Long continuous grain ribbons; shared boundaries avoid cracks/overlap."""
    groups = [([], [], []) for _ in range(WOOD_PALETTE_SIZE)]
    width = high[0] - low[0]
    plank_width = (high[1] - low[1]) / WOOD_PLANKS
    if width < 0.5 or plank_width < 0.04:
        raise ValueError("lookdev grain requires a metre-scale display workbench")
    # A visual veneer 0.25 mm above the original display board, not coplanar.
    z = high[2] + 0.00025
    seam = 0.0007
    xs = tuple(
        low[0] + width * index / WOOD_LENGTH_SEGMENTS for index in range(WOOD_LENGTH_SEGMENTS + 1)
    )
    for plank in range(WOOD_PLANKS):
        y0 = low[1] + plank * plank_width + seam / 2
        y1 = low[1] + (plank + 1) * plank_width - seam / 2
        phase = plank * 1.731
        knot_x = low[0] + width * (0.22 + 0.56 * ((plank * 7) % 11) / 10)

        def boundary(x, row):
            t = row / WOOD_ROWS_PER_PLANK
            # Both ends of a plank stay fixed; adjacent grain rows evaluate the
            # very same boundary, including their deterministic knot excursion.
            taper = math.sin(math.pi * t)
            slow_wave = 0.0020 * math.sin((x - low[0]) * 4.2 + phase)
            knot = (
                0.0035 * math.exp(-(((x - knot_x) / 0.17) ** 2)) * math.sin(3 * math.pi * t + phase)
            )
            return y0 + (y1 - y0) * t + taper * (slow_wave + knot)

        for row in range(WOOD_ROWS_PER_PLANK):
            for column in range(WOOD_LENGTH_SEGMENTS):
                x0, x1 = xs[column], xs[column + 1]
                y00, y10 = boundary(x0, row), boundary(x1, row)
                y01, y11 = boundary(x0, row + 1), boundary(x1, row + 1)
                if not (y00 < y01 and y10 < y11):
                    raise ValueError("procedural grain ribbons overlap")
                # Most variation follows the grain across a plank; the small
                # long-wave term avoids a repeated barcode without random noise.
                tone = (
                    0.50
                    + 0.22 * math.sin(row * 1.73 + phase)
                    + 0.13 * math.sin(row * 0.41 + phase * 0.7)
                    + 0.06 * math.sin(column * 0.27 + phase)
                )
                shade = min(WOOD_PALETTE_SIZE - 1, max(0, int(tone * WOOD_PALETTE_SIZE)))
                _quad(
                    groups[shade],
                    ((x0, y00, z), (x1, y10, z), (x1, y11, z), (x0, y01, z)),
                )
    return groups


def _rim(low, high):
    """Closed 1 mm bevel around a box, with 0.3 mm side clearance.

    The top lies 1 mm below the source top. No rim triangle projects over the
    source rectangle: it cannot cover the payload or replace its contact plane.
    """
    if high[2] - low[2] <= 0.006:
        raise ValueError("lookdev rim needs a display prop thicker than 6 mm")
    cx, cy = (low[0] + high[0]) / 2, (low[1] + high[1]) / 2
    hx, hy = (high[0] - low[0]) / 2, (high[1] - low[1]) / 2
    z0, z1 = low[2] + 0.001, high[2] - 0.001
    profile = (
        (0.0003, z0),
        (0.0010, z0),
        (0.0020, z0 + 0.001),
        (0.0020, z1 - 0.001),
        (0.0010, z1),
        (0.0003, z1),
    )
    rings = [
        tuple(
            (cx + sx * (hx + inset), cy + sy * (hy + inset), z)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        )
        for inset, z in profile
    ]
    group = ([], [], [])
    for level, ring in enumerate(rings):
        following = rings[(level + 1) % len(rings)]
        for corner in range(4):
            next_corner = (corner + 1) % 4
            _quad(
                group,
                (ring[corner], ring[next_corner], following[next_corner], following[corner]),
            )
    return group


def lookdev_scene(scene: VisualScene, geometry_dir) -> VisualScene:
    """Return a distinct appearance-only scene; preserve source geometry/poses.

    The input must be the ordinary full scene from _panda_scene.build_scene.
    Generated files are written once into geometry_dir alongside ordinary props;
    an existing lookdev file is rejected instead of being silently overwritten.
    The source scene and its objects remain immutable and reusable.
    """
    if not isinstance(scene, VisualScene):
        raise TypeError("scene must be VisualScene")
    props = {instance.id: instance for instance in scene.instances}
    for identifier, node, mesh_id in (
        (1, "cube", "payload"),
        (2, "table", "tabletop"),
        (3, "panda_link0", "floor"),
        (4, "table", "stand"),
    ):
        instance = props.get(identifier)
        if instance is None or instance.node_id != node or instance.mesh_id != mesh_id:
            raise ValueError("lookdev requires the original Panda payload/table/display props")
    if any(identifier in props for identifier in DECORATION_INSTANCE_IDS):
        raise ValueError("lookdev decoration IDs collide with existing instances")
    if any(item.id.startswith(_PREFIX) for item in tuple(scene.meshes) + tuple(scene.materials)):
        raise ValueError("lookdev resource prefix already exists")

    floor = props[3]
    tabletop = props[2]
    floor_bounds = _box_bounds(scene, floor)
    table_bounds = _box_bounds(scene, tabletop)
    groups = _wood_groups(*floor_bounds)
    floor_rim = _rim(*floor_bounds)
    table_rim = _rim(*table_bounds)
    if sum(len(group[2]) for group in groups + [floor_rim, table_rim]) != ADDED_TRIANGLES:
        raise AssertionError("lookdev geometry exceeded its declared fixed budget")

    directory = Path(geometry_dir)
    names = [f"lookdev-wood-{index:02d}" for index, group in enumerate(groups) if group[2]]
    names += ["lookdev-workbench-metal-rim", "lookdev-table-brass-rim"]
    if any((directory / (name + ".gltf")).exists() for name in names):
        raise FileExistsError("lookdev geometry files already exist")
    directory.mkdir(parents=True, exist_ok=True)

    rubber = Material("lookdev-matte-rubber", (0.040, 0.046, 0.051), 0.0, 0.84)
    steel = Material("lookdev-satin-steel", (0.48, 0.54, 0.60), 1.0, 0.28)
    brass = Material("lookdev-satin-brass", (0.62, 0.39, 0.14), 1.0, 0.22)
    extra_materials = [rubber, steel, brass]
    extra_meshes, extra_instances = [], []

    for index, group in enumerate(groups):
        if not group[2]:
            continue
        brightness = 0.70 + 0.60 * index / (WOOD_PALETTE_SIZE - 1)
        material = Material(
            f"lookdev-oak-{index:02d}",
            tuple(value * brightness for value in (0.30, 0.17, 0.075)),
            0.0,
            0.43 + 0.04 * index / (WOOD_PALETTE_SIZE - 1),
        )
        mesh = write_mesh(directory, f"lookdev-wood-{index:02d}", *group)
        extra_meshes.append(mesh)
        extra_materials.append(material)
        extra_instances.append(
            Instance(20 + index, mesh.id, material.id, floor.node_id, floor.local_pose)
        )
    for identifier, name, group, owner, material in (
        (44, "lookdev-workbench-metal-rim", floor_rim, floor, steel),
        (45, "lookdev-table-brass-rim", table_rim, tabletop, brass),
    ):
        mesh = write_mesh(directory, name, *group)
        extra_meshes.append(mesh)
        extra_instances.append(
            Instance(identifier, mesh.id, material.id, owner.node_id, owner.local_pose)
        )

    instances = tuple(
        replace(instance, material_id=rubber.id)
        if instance.id == 2
        else replace(instance, material_id=steel.id)
        if instance.id == 4
        else instance
        for instance in scene.instances
    )
    return VisualScene(
        scene.key + ":lookdev-wood-rubber-brass-v1",
        scene.meshes + tuple(extra_meshes),
        scene.materials + tuple(extra_materials),
        instances + tuple(extra_instances),
        scene.lights,
    )
