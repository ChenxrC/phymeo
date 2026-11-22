"""Independent CPU finite-plane/pinhole reference for the public R3 channels.

No backend projection, decoded mesh, shader, or public pose-composition helper
is used. NumPy accelerates only the same analytic equations over pixel centres.
The 2 px exclusion is declared from projected geometry before any capture is
read; all excluded samples remain separately counted and measured.
"""

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath


def _dot(a, b):
    return math.fsum(x*y for x, y in zip(a, b, strict=True))


def _rotation(quaternion):
    # Quaternion conjugation R(v) = (w²-|q|²)v + 2q(q.v) + 2w(q cross v).
    norm = math.sqrt(_dot(quaternion, quaternion))
    x, y, z, w = (v/norm for v in quaternion)
    vector = (x, y, z)
    skew = ((0., -z, y), (z, 0., -x), (-y, x, 0.))
    diagonal = w*w-_dot(vector, vector)
    return tuple(tuple((diagonal if i == j else 0.)+2*vector[i]*vector[j]+2*w*skew[i][j]
                       for j in range(3)) for i in range(3))


def camera_axes(frame, camera_id):
    if camera_id == "fixed":
        return (0., 0., 0.), ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    if camera_id != "wrist":
        raise ValueError("unknown calibration camera")
    wrist = next(node.world_pose for node in frame.nodes if node.node_id == "wrist")
    rotation = _rotation(wrist.orientation_xyzw)
    # Independently specified physical mount: translation (12,0,6) mm, axes unchanged.
    origin = tuple(wrist.position_m[i]+_dot(rotation[i], (0.012, 0., 0.006)) for i in range(3))
    return origin, rotation


def _planes(frame, surfaces):
    nodes = {node.node_id: node.world_pose for node in frame.nodes}
    result = []
    for surface in surfaces:
        pose = nodes[surface.node]
        rotation = _rotation(pose.orientation_xyzw)
        center = tuple(pose.position_m[i]+_dot(rotation[i], surface.local_center) for i in range(3))
        axes = tuple(tuple(rotation[i][j] for i in range(3)) for j in range(3))
        result.append((surface, center, *axes))
    return result


def _project(origin, rotation, point):
    delta = tuple(p-o for p, o in zip(point, origin, strict=True))
    local = tuple(math.fsum(rotation[i][j]*delta[i] for i in range(3)) for j in range(3))
    depth = -local[2]
    if depth <= 0:
        raise ValueError("R3 fixture surface behind its camera")
    return (320.+500.*local[0]/depth, 240.-500.*local[1]/depth, depth)


def ray_hit(frame, camera_id, u, v, surfaces):
    origin, rotation = camera_axes(frame, camera_id)
    ray_local = ((u+0.5-320.)/500., (240.-v-0.5)/500., -1.)
    ray = tuple(_dot(row, ray_local) for row in rotation)
    nearest, identity = math.inf, 0
    for surface, center, horizontal, vertical, normal in _planes(frame, surfaces):
        denominator = _dot(normal, ray)
        if denominator == 0:
            continue
        offset = tuple(o-c for o, c in zip(origin, center, strict=True))
        t = -_dot(normal, offset)/denominator
        if not 0.1 <= t <= 5. or t >= nearest:
            continue
        x = _dot(horizontal, offset)+t*_dot(horizontal, ray)
        y = _dot(vertical, offset)+t*_dot(vertical, ray)
        if abs(x) <= surface.half_size[0] and abs(y) <= surface.half_size[1]:
            nearest, identity = t, surface.identity
    return nearest, identity


def reference_image(frame, camera_id, surfaces):
    import numpy as np
    origin, rotation = camera_axes(frame, camera_id)
    ys, xs = np.mgrid[0:480, 0:640]
    rays = np.stack(((xs+0.5-320.)/500., (240.-ys-0.5)/500., -np.ones_like(xs)), axis=-1)
    rays = rays @ np.asarray(rotation, dtype=np.float64).T
    depth = np.full((480, 640), np.inf, dtype=np.float64)
    identities = np.zeros((480, 640), dtype=np.uint32)
    edge = np.zeros((480, 640), dtype=bool)
    projected = {}
    for surface, center, horizontal, vertical, normal in _planes(frame, surfaces):
        offset = tuple(o-c for o, c in zip(origin, center, strict=True))
        denominator = rays @ np.asarray(normal)
        t = np.full((480, 640), np.inf)
        np.divide(-_dot(normal, offset), denominator, out=t, where=denominator != 0)
        with np.errstate(invalid="ignore"):
            x = _dot(horizontal, offset)+t*(rays @ np.asarray(horizontal))
            y = _dot(vertical, offset)+t*(rays @ np.asarray(vertical))
        hit = ((t >= 0.1) & (t <= 5.) & (t < depth)
               & (np.abs(x) <= surface.half_size[0]) & (np.abs(y) <= surface.half_size[1]))
        depth[hit], identities[hit] = t[hit], surface.identity
        corners = []
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            point = tuple(center[i]+a*surface.half_size[0]*horizontal[i]
                          + b*surface.half_size[1]*vertical[i] for i in range(3))
            corners.append(_project(origin, rotation, point)[:2])
        projected[surface.identity] = {"center": _project(origin, rotation, center), "corners": corners}
        # Exact screen-space segment distance, not an image/ID-based erosion.
        # Invisible outlines may conservatively exclude extra samples; no
        # excluded failure is hidden, and internal triangle diagonals are absent.
        for a, b in zip(corners, corners[1:]+corners[:1], strict=True):
            left, right = max(0, math.floor(min(a[0], b[0])-3)), min(640, math.ceil(max(a[0], b[0])+3))
            top, bottom = max(0, math.floor(min(a[1], b[1])-3)), min(480, math.ceil(max(a[1], b[1])+3))
            if left >= right or top >= bottom:
                continue
            yy, xx = np.mgrid[top:bottom, left:right]
            px, py = xx+0.5-a[0], yy+0.5-a[1]
            dx, dy = b[0]-a[0], b[1]-a[1]
            parameter = np.clip((px*dx+py*dy)/(dx*dx+dy*dy), 0., 1.)
            edge[top:bottom, left:right] |= (px-parameter*dx)**2+(py-parameter*dy)**2 <= 4.
    edge[:2, :] = edge[-2:, :] = True
    edge[:, :2] = edge[:, -2:] = True
    return {"depth": depth, "ids": identities, "edge": edge, "projected": projected}


def audit_capture(capture, case, surfaces):
    import numpy as np
    expected_camera = next(camera for camera in case.cameras if camera.id == capture.camera.id)
    assert capture.frame_key == case.frame.key and capture.time_s == case.frame.time_s, "exposure frame/time"
    assert capture.camera == expected_camera, "exposure calibration/pose"
    assert (expected_camera.width, expected_camera.height, expected_camera.fx_px, expected_camera.fy_px,
            expected_camera.cx_px, expected_camera.cy_px, expected_camera.near_m, expected_camera.far_m
            ) == (640, 480, 500., 500., 320., 240., 0.1, 5.), "R3 intrinsic declaration"
    planes = {plane.channel: plane.data for plane in capture.planes}
    assert set(planes) == set(Channel), "all four public channels required"
    rgb = np.frombuffer(planes[Channel.RGB], dtype=np.uint8).reshape(480, 640, 3)
    depth = np.frombuffer(planes[Channel.FORWARD_DEPTH], dtype="<f4").reshape(480, 640)
    ids = np.frombuffer(planes[Channel.INSTANCE_ID], dtype="<u4").reshape(480, 640)
    valid = np.frombuffer(planes[Channel.VALID], dtype=np.uint8).reshape(480, 640)
    reference = reference_image(case.frame, expected_camera.id, surfaces)
    expected_depth, expected_ids, edges = (reference[k] for k in ("depth", "ids", "edge"))
    inside = (expected_ids != 0) & ~edges
    background = (expected_ids == 0) & ~edges
    assert np.count_nonzero(background) >= 1000, "background denominator"
    assert np.array_equal(valid, (ids != 0).astype(np.uint8)), "VALID/ID agreement"
    assert np.array_equal(np.isfinite(depth), valid != 0), "VALID/depth agreement"
    errors = np.abs(depth[inside].astype(np.float64)-expected_depth[inside])
    identity_mismatch = int(np.count_nonzero(ids[inside] != expected_ids[inside]))
    background_mismatch = int(np.count_nonzero((ids[background] != 0) | (valid[background] != 0)
                                              | ~np.isposinf(depth[background])))
    edge_correct = edges & (expected_ids != 0) & (ids == expected_ids) & np.isfinite(depth)
    edge_errors = np.abs(depth[edge_correct].astype(np.float64)-expected_depth[edge_correct])
    report = {
        "case": case.name, "camera": expected_camera.id, "frame_key": asdict(case.frame.key),
        "time_s": case.frame.time_s, "interior_pixels": int(np.count_nonzero(inside)),
        "interior_max_depth_error_m": float(errors.max()), "interior_id_mismatches": identity_mismatch,
        "background_pixels": int(np.count_nonzero(background)), "background_mismatches": background_mismatch,
        "edge_policy": "screen distance <=2px to any declared quad edge, plus outer 2px image border",
        "edge_pixels": int(np.count_nonzero(edges)),
        "edge_id_mismatches": int(np.count_nonzero(ids[edges] != expected_ids[edges])),
        "edge_expected_foreground_misses": int(np.count_nonzero(edges & (expected_ids != 0) & (ids == 0))),
        "edge_expected_background_hits": int(np.count_nonzero(edges & (expected_ids == 0) & (ids != 0))),
        "edge_matching_id_depth_samples": int(len(edge_errors)),
        "edge_matching_id_max_depth_error_m": float(edge_errors.max()) if len(edge_errors) else None,
        "surfaces": [], "markers": [],
    }
    for surface in surfaces:
        mask = inside & (expected_ids == surface.identity)
        count = int(np.count_nonzero(mask))
        if surface.node.startswith("board-") and surface.marker_channel is None:
            assert count >= 100, f"missing board interior {surface.identity}"
        report["surfaces"].append({"id": surface.identity, "interior_pixels": count,
            "max_depth_error_m": float(np.abs(depth[mask]-expected_depth[mask]).max()) if count else None,
            "id_mismatches": int(np.count_nonzero(ids[mask] != surface.identity))})
        if surface.marker_channel is None:
            continue
        expected_count = int(np.count_nonzero(expected_ids == surface.identity))
        actual_y, actual_x = np.nonzero(ids == surface.identity)
        item = {"id": surface.identity, "channel": surface.marker_channel,
                "expected_pixels": expected_count, "observed_id_pixels": len(actual_x)}
        center = reference["projected"][surface.identity]["center"]
        corners = reference["projected"][surface.identity]["corners"]
        left, right = max(0, math.floor(min(p[0] for p in corners)-3)), min(640, math.ceil(max(p[0] for p in corners)+3))
        top, bottom = max(0, math.floor(min(p[1] for p in corners)-3)), min(480, math.ceil(max(p[1] for p in corners)+3))
        local = rgb[top:bottom, left:right].astype(np.int16)
        channel = surface.marker_channel
        others = [i for i in range(3) if i != channel]
        mask = (local[:, :, channel] >= 40) & (local[:, :, channel] > 2*local[:, :, others].max(axis=2))
        ry, rx = np.nonzero(mask)
        item["observed_rgb_pixels"] = len(rx)
        if expected_count == 0:
            assert len(actual_x) == len(rx) == 0, f"occluded marker {surface.identity} leaked"
            item["visibility"] = "fully occluded"
        else:
            assert expected_count >= 9 and len(actual_x) >= 9 and len(rx) >= 9, f"marker samples {item}"
            seg_point = (float(actual_x.mean()+0.5), float(actual_y.mean()+0.5))
            rgb_point = (float(rx.mean()+left+0.5), float(ry.mean()+top+0.5))
            item.update(visibility="visible", expected_center_px=list(center[:2]),
                        observed_id_center_px=seg_point, observed_rgb_center_px=rgb_point,
                        id_projection_error_px=math.dist(seg_point, center[:2]),
                        rgb_projection_error_px=math.dist(rgb_point, center[:2]))
            assert max(item["id_projection_error_px"], item["rgb_projection_error_px"]) <= 1., f"marker projection {item}"
        report["markers"].append(item)
    assert identity_mismatch == 0, f"interior ID mismatch: {report}"
    assert background_mismatch == 0, f"background mismatch: {report}"
    assert report["interior_max_depth_error_m"] <= 0.002, f"depth >2mm: {report}"
    report["max_marker_error_px"] = max(max(m["id_projection_error_px"], m["rgb_projection_error_px"])
                                       for m in report["markers"] if m["visibility"] == "visible")
    report["status"] = "passed"
    return report


def exercise_calibration(test, factory, label):
    """Run the real public API and retain every raw channel before auditing it."""
    from ._r0_table import write_png
    from ._r3_scene import build_calibration_fixture
    output = Path(os.environ["PHYMEO_R3_OUTPUT"]) / label
    output.mkdir(parents=True, exist_ok=False)
    scene, cases, surfaces = build_calibration_fixture(output / "assets")
    frozen = json.dumps([asdict(scene), [asdict(case) for case in cases]], sort_keys=True)
    report = {"schema": "phymeo-r3-calibration-v1", "backend": label, "cases": [],
              "scope": "640x480; two cameras; authored finite planes; no physics steps",
              "thresholds": {"depth_m": 0.002, "marker_px": 1., "interior_id_mismatches": 0},
              "source_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
              "assets": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (output/"assets").glob("*.gltf")},
              "status": "running"}
    (output/"fixture.json").write_text(json.dumps({"scene": asdict(scene), "cases": [asdict(c) for c in cases],
        "analytic_surfaces": [asdict(s) for s in surfaces]}, indent=2), encoding="utf-8")
    session = factory().open_scene(scene)
    retained, first = [], {}
    try:
        for case in cases:
            for camera in case.cameras:
                request = CaptureRequest(camera, RenderPath.RASTER, tuple(Channel))
                test.assertTrue(session.supports(request))
                capture = session.capture(case.frame, request)
                prefix = f"{case.name}-{camera.id}"
                raw_hashes = {}
                for plane in capture.planes:
                    filename = f"{prefix}-{plane.channel.value}.raw"
                    (output/filename).write_bytes(plane.data)
                    raw_hashes[plane.channel.value] = hashlib.sha256(plane.data).hexdigest()
                    if plane.channel is Channel.RGB:
                        write_png(output/f"{prefix}.png", 640, 480, plane.data)
                (output/f"{prefix}-capture.json").write_text(json.dumps({
                    "frame_key": asdict(capture.frame_key), "time_s": capture.time_s,
                    "camera": asdict(capture.camera), "channel_sha256": raw_hashes}, indent=2), encoding="utf-8")
                measured = audit_capture(capture, case, surfaces)
                measured["channel_sha256"] = raw_hashes
                report["cases"].append(measured)
                retained.append((capture, raw_hashes))
                if case.name == "front":
                    first[camera.id] = capture
                if case.name == "reset":
                    test.assertEqual(capture.planes, first[camera.id].planes)
                    stale_request = CaptureRequest(first[camera.id].camera, RenderPath.RASTER, tuple(Channel))
                    with test.assertRaises(ValueError):
                        session.capture(case.frame, stale_request)
        test.assertEqual(frozen, json.dumps([asdict(scene), [asdict(case) for case in cases]], sort_keys=True))
        for capture, hashes in retained:
            test.assertEqual(hashes, {p.channel.value: hashlib.sha256(p.data).hexdigest() for p in capture.planes})
        report["diagnostics"] = session.diagnostics()
        report["retained_frames_unchanged"] = True
        report["reset_repeated_pixels_and_rejected_stale_exposure"] = True
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            session.close()
            report["closed_diagnostics"] = session.diagnostics()
        finally:
            (output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
