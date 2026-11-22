"""Explicit x-ray collision diagnostics, kept separate from beauty rendering."""
from itertools import product
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from phymeo_rendering.scene import Pose, compose_pose


def _origin(element):
    if element is None:
        return Pose()
    xyz = tuple(float(x) for x in element.get("xyz", "0 0 0").split())
    r, p, y = (float(x)/2 for x in element.get("rpy", "0 0 0").split())
    cr, cp, cy, sr, sp, sy = math.cos(r), math.cos(p), math.cos(y), math.sin(r), math.sin(p), math.sin(y)
    return Pose(xyz, (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                      cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy))


def _rotate(quat, vector):
    norm = math.hypot(*quat)
    x, y, z, w = (q/norm for q in quat)
    a, b, c = vector
    cross = (y*c-z*b, z*a-x*c, x*b-y*a)
    twice = (y*cross[2]-z*cross[1], z*cross[0]-x*cross[2], x*cross[1]-y*cross[0])
    return tuple(v+2*w*t+2*u for v, t, u in zip(vector, cross, twice))


def _world(pose, point):
    return tuple(p+v for p, v in zip(pose.position_m, _rotate(pose.orientation_xyzw, point)))


def _positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("collision dimensions must be finite positive")
    return value


def _edges(shape):
    """Authored primitive coordinates; curved lines use 32 segments."""
    if shape.tag == "box":
        size = tuple(_positive(x) for x in shape.attrib["size"].split())
        if len(size) != 3:
            raise ValueError("collision box requires three dimensions")
        points = [tuple(s*d/2 for s, d in zip(signs, size)) for signs in product((-1, 1), repeat=3)]
        for i, a in enumerate(points):
            for j, b in enumerate(points[i+1:], i+1):
                if (i ^ j) in (1, 2, 4):
                    yield a, b
    elif shape.tag in ("sphere", "cylinder"):
        radius = _positive(shape.attrib["radius"])
        angles = [(radius*math.cos(i*math.tau/32), radius*math.sin(i*math.tau/32)) for i in range(32)]
        if shape.tag == "sphere":
            circles = [[(a, b, 0) for a, b in angles],
                       [(a, 0, b) for a, b in angles], [(0, a, b) for a, b in angles]]
        else:
            half = _positive(shape.attrib["length"])/2
            circles = [[(a, b, z) for a, b in angles] for z in (-half, half)]
            for i in range(0, 32, 4):
                yield circles[0][i], circles[1][i]
        for points in circles:
            yield from zip(points, points[1:]+points[:1])
    else:
        raise ValueError(f"unsupported collision geometry {shape.tag!r}; not silently omitted")


def _segment(a, b, camera):
    # +X right, +Y up, -Z forward. Clip depth before perspective division.
    d0, d1 = -a[2], -b[2]
    low, high = 0., 1.
    for origin, delta in ((d0-camera.near_m, d1-d0), (camera.far_m-d0, d0-d1)):
        if delta == 0:
            if origin < 0:
                return None
        elif delta > 0:
            low = max(low, -origin/delta)
        else:
            high = min(high, -origin/delta)
    if low > high:
        return None
    projected = []
    for t in (low, high):
        x, y, z = (p+t*(q-p) for p, q in zip(a, b))
        projected.append((camera.fx_px*x/(-z)+camera.cx_px, camera.cy_px-camera.fy_px*y/(-z)))
    (x, y), (end_x, end_y) = projected
    dx, dy = end_x-x, end_y-y
    low, high = 0., 1.
    for p, q in ((-dx, x), (dx, camera.width-1-x), (-dy, y), (dy, camera.height-1-y)):
        if p == 0:
            if q < 0:
                return None
        elif p < 0:
            low = max(low, q/p)
        else:
            high = min(high, q/p)
    if low > high:
        return None
    return (x+low*dx, y+low*dy, x+high*dx, y+high*dy)


def draw_collision_overlay(image, urdf_path, frame, camera):
    """Copy PIL image, drawing all URDF collision shapes as explicit x-ray lines.

    Missing welded descendants are resolved from fixed joints. A missing moving
    source link is an error. Hidden lines intentionally remain visible; this is
    not an occlusion or collision-test result. Original beauty is never modified.
    """
    from PIL import ImageDraw
    if camera.frame_key != frame.key:
        raise ValueError("collision overlay camera references a foreign frame")
    if image.size != (camera.width, camera.height):
        raise ValueError("collision overlay image dimensions differ from camera")
    root = ET.fromstring(Path(urdf_path).read_bytes())
    poses = {node.node_id: node.world_pose for node in frame.nodes}
    fixed = [j for j in root.findall("joint") if j.get("type") == "fixed"]
    while fixed:
        pending = []
        for joint in fixed:
            parent, child = joint.find("parent").attrib["link"], joint.find("child").attrib["link"]
            if child in poses:
                continue
            if parent not in poses:
                pending.append(joint)
            else:
                poses[child] = compose_pose(poses[parent], _origin(joint.find("origin")))
        if len(pending) == len(fixed):
            break
        fixed = pending
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    rotation = camera.world_pose.orientation_xyzw
    inverse = (-rotation[0], -rotation[1], -rotation[2], rotation[3])
    colors = {"box": (255, 85, 35), "sphere": (30, 220, 255), "cylinder": (245, 210, 30)}
    for link in root.findall("link"):
        collisions = link.findall("collision")
        if not collisions:
            continue
        name = link.attrib["name"]
        if name not in poses:
            raise ValueError(f"collision source link pose unavailable: {name}")
        for collision in collisions:
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError(f"collision {name} requires exactly one geometry")
            shape = geometry[0]
            pose = compose_pose(poses[name], _origin(collision.find("origin")))
            for a, b in _edges(shape):
                ends = [_rotate(inverse, tuple(v-c for v, c in zip(_world(pose, point),
                              camera.world_pose.position_m))) for point in (a, b)]
                clipped = _segment(*ends, camera)
                if clipped is not None:
                    draw.line(clipped, fill=colors[shape.tag], width=1)
    draw.rectangle((0, 0, 230, 18), fill=(0, 0, 0))
    draw.text((4, 3), "X-RAY COLLISION (not beauty)", fill=(255, 255, 255))
    return result
