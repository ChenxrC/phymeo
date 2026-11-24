"""True subpixel camera exposures with a scene-linear pixel-area resolve."""
from dataclasses import replace

import numpy as np


# Offsets are measured relative to a pixel's center in top-left image coordinates.
ROTATED_FOUR = ((-.125, -.375), (.375, -.125), (-.375, .125), (.125, .375))


def scaled_camera(camera, scale, box=None):
    """Scale a pinhole sensor or a crop of it without changing any physical ray."""
    if type(scale) is not int or scale < 1:
        raise ValueError("camera scale must be a positive integer")
    x0, y0, x1, y1 = box if box is not None else (0, 0, camera.width, camera.height)
    if (not all(type(i) is int for i in (x0, y0, x1, y1)) or
            not (0 <= x0 < x1 <= camera.width and 0 <= y0 < y1 <= camera.height)):
        raise ValueError("crop must fit within the source sensor")
    return replace(camera, width=(x1-x0)*scale, height=(y1-y0)*scale,
                   fx_px=camera.fx_px*scale, fy_px=camera.fy_px*scale,
                   cx_px=(camera.cx_px-x0)*scale, cy_px=(camera.cy_px-y0)*scale)


def block_mean(high, scale):
    """Integrate uniform fine pixels in linear space; the area groups never overlap."""
    h, w, channels = high.shape
    if type(scale) is not int or scale <= 0 or h % scale or w % scale or channels != 3:
        raise ValueError("linear source must contain complete RGB pixel-area groups")
    # Accumulate in float64, preserving HDR values above one.
    return high.reshape(h//scale, scale, w//scale, scale, 3).mean(axis=(1,3), dtype=np.float64).astype(np.float32)


def _capture(owner, frame, request, presentation):
    result = owner.capture(frame, request, **({"presentation": presentation} if presentation is not None else {}))
    if result.frame_key != frame.key or result.time_s != frame.time_s or result.camera != request.camera:
        raise RuntimeError("capture does not belong to this frame and camera")
    return result


def linear_exposure(owner, frame, request, *, presentation=None):
    """Read the color owner's actual linear exposure at exactly the requested camera."""
    from phymeo_rendering.outputs import Channel
    camera = request.camera
    _capture(owner, frame, replace(request, channels=(Channel.RGB,)), presentation)
    read = getattr(owner, "read_reconstructed_linear", None)
    packet = read() if read is not None else owner.read_linear_mean()
    if (packet["format"] != "RGBA32F_LE" or (packet["width"], packet["height"]) != (camera.width, camera.height)
            or "scene-linear" not in packet["space"] or len(packet["data"]) != camera.width*camera.height*16):
        raise ValueError("color owner did not return the requested scene-linear RGBA32F exposure")
    result = np.frombuffer(packet["data"], "<f4").reshape(camera.height, camera.width, 4)[..., :3]
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("negative or nonfinite pre-display radiance")
    return result


def film_capture(color_owner, frame, request, *, presentation=None, sensor_owner=None,
                 exposure=8., display_transform="pbr-neutral", samples="grid2"):
    """Return (ordinary full capture, owned scene-linear RGB or None for sensor-only).

    grid2 captures a double-width/double-height physical sensor. Each new
    lattice position is independently rasterized and shaded; its backend's
    own fine-pixel coverage resolver still applies. rotated4 instead evaluates
    four full-resolution off-axis cameras. Both return the original camera.
    A separate sensor owner always uses the unshifted original pixel centers.
    """
    from phymeo_rendering.backends._sapien_pixels import encode_linear_image
    from phymeo_rendering.outputs import Capture, Channel, ImagePlane
    from phymeo_rendering.pipeline import validate_capture_request
    validate_capture_request(request)
    if samples not in ("grid2", "rotated4"):
        raise ValueError("supported film sampling modes are grid2 and rotated4")
    sensors = tuple(channel for channel in request.channels if channel != Channel.RGB)
    wants_color = Channel.RGB in request.channels
    if sensors and wants_color and (sensor_owner is None or sensor_owner is color_owner):
        raise ValueError("center sensor channels require an independent owner")
    planes, hdr = {}, None
    if sensors:
        result = _capture(sensor_owner if sensor_owner is not None else color_owner,
                          frame, replace(request, channels=sensors), presentation)
        planes = {plane.channel: plane for plane in result.planes}
    if wants_color:
        camera = request.camera
        if samples == "grid2":
            high = replace(request, camera=scaled_camera(camera, 2))
            hdr = block_mean(linear_exposure(color_owner, frame, high, presentation=presentation), 2)
        else:
            total = np.zeros((camera.height, camera.width, 3), dtype=np.float64)
            for dx, dy in ROTATED_FOUR:
                offset = replace(camera, cx_px=camera.cx_px-dx, cy_px=camera.cy_px-dy)
                total += linear_exposure(color_owner, frame, replace(request, camera=offset), presentation=presentation)
            hdr = (total*.25).astype(np.float32)
        planes[Channel.RGB] = ImagePlane(Channel.RGB, encode_linear_image(hdr, exposure=exposure, display_transform=display_transform))
    return Capture(frame.key, frame.time_s, request.camera, tuple(planes[ch] for ch in request.channels)), hdr
