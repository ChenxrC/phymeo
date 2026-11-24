"""Color-coded parts at the fixed, indexed pickup station."""
from dataclasses import asdict
from hashlib import sha256

import numpy as np
from PIL import Image


def classify_station_rgb(rgb):
    """Read the centre of the registered inspection window. Fail closed on ambiguous colors."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("station camera expects an RGB8 image")
    h, w, _ = rgb.shape
    if min(h, w) < 32:
        raise ValueError("station inspection window is too small")
    roi = (3*w//8, 3*h//8, 5*w//8, 5*h//8)
    x0,y0,x1,y1 = roi
    r,g,b = rgb[y0:y1, x0:x1].astype(np.int16).transpose(2,0,1)
    orange = (r-g >= 18) & (r-b >= 35)
    blue = (b-r >= 32) & (g-r >= 8)
    fractions = {"block": float(orange.mean()), "hex": float(blue.mean())}
    chosen = max(fractions, key=fractions.get)
    if fractions[chosen] < .55 or min(fractions.values()) > .12:
        raise ValueError(f"unclassified or mixed workpiece in station window: {fractions}")
    return chosen, dict(roi_xyxy=roi, fraction=fractions)


def station_camera(frame_key, layout):
    """Registered intrinsics and optical pose of the conveyor's fixed RGB reader."""
    from _panda_scene import look_at
    from phymeo_rendering.cameras import Camera
    x,y = layout.pick_xy
    z = layout.support_z
    return Camera("belt-color-reader", frame_key,
                  look_at((x+.29,y+.06,z+.205),(x,y,z+.032),(0.,0.,1.)),
                  320,192,620.,620.,160.,96.,.015,8.)


def read_part(owner, frame, layout, output):
    from preview_compact_rigid_history.state import bind_presentations, frame_digest
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    camera = station_camera(frame.key, layout)
    # Arrivals are sparse in time. Each sensor exposure starts a new observation.
    owner.reset_history()
    source = frame_digest(frame)
    binding, = bind_presentations([frame], source)
    capture = owner.capture(frame, CaptureRequest(camera, RenderPath.RASTER, (Channel.RGB,)),
                            presentation=binding)
    if (capture.frame_key,capture.camera,capture.time_s) != (frame.key,camera,frame.time_s):
        raise ValueError("optical read does not correspond to the stopped conveyor exposure")
    raw, = capture.planes
    rgb = np.frombuffer(raw.data, dtype=np.uint8).reshape(camera.height,camera.width,3)
    filename = f"reader-{frame.key.tick:03}.png"
    Image.fromarray(rgb).save(output/filename)
    kind, measurement = classify_station_rgb(rgb)
    return dict(frame=frame.key.tick, source_pose_sha256=source, image=filename,
                rgb_sha256=sha256(raw.data).hexdigest(), camera=asdict(camera),
                decision=kind, **measurement)
