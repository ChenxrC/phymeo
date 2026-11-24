"""Display-linear ground truth on contour pixels selected by supersampled geometry."""
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw

from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath


def _linear(rgb):
    x = rgb.astype(np.float64) / 255.
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def reference(session, frame, camera, scale, moving_ids):
    high = replace(camera, width=camera.width * scale, height=camera.height * scale,
                   fx_px=camera.fx_px * scale, fy_px=camera.fy_px * scale,
                   cx_px=camera.cx_px * scale, cy_px=camera.cy_px * scale)
    capture = session.capture(frame, CaptureRequest(high, RenderPath.RASTER,
                                                    (Channel.RGB, Channel.INSTANCE_ID)))
    if (capture.frame_key, capture.camera, capture.time_s) != (frame.key, high, frame.time_s):
        raise RuntimeError("supersample reference has a different exposure")
    planes = {p.channel: p.data for p in capture.planes}
    h, w = camera.height, camera.width
    rgb = np.frombuffer(planes[Channel.RGB], np.uint8).reshape(h, scale, w, scale, 3)
    ids = np.frombuffer(planes[Channel.INSTANCE_ID], '<u4').reshape(h, scale, w, scale)
    linear = _linear(rgb).mean(axis=(1, 3))
    display = np.where(linear <= .0031308, 12.92 * linear,
                       1.055 * np.power(linear, 1. / 2.4) - .055)
    mask = (ids.min(axis=(1, 3)) != ids.max(axis=(1, 3))) & np.isin(ids, moving_ids).any(axis=(1, 3))
    if mask.sum() < 20:
        raise ValueError("too few independently identified subpixel contour pixels")
    return linear, np.rint(255 * display).astype(np.uint8), mask


def contour_error(actual, ref):
    linear, display, mask = ref
    error = np.abs(_linear(actual[mask]) - linear[mask])
    codes = np.abs(actual[mask].astype(float) - display[mask])
    return dict(contour_pixels=int(mask.sum()), linear_mae=float(error.mean()),
                linear_rmse=float(np.sqrt((error**2).mean())), srgb8_mae=float(codes.mean()))


def write_board(path, panels, refs, stage, world):
    """Each row uses one fixed RGB, a matching reference and a separate mounted camera."""
    first = next(iter(refs.values()))
    y, x = np.nonzero(first[2])
    h, w = first[2].shape
    # Shared coordinates across all rows, nearest-neighbour for the inspection crops.
    cx, cy = int((x.min() + x.max()) // 2), int((y.min() + y.max()) // 2)
    side = min(h, w, max(42, int(max(np.ptp(x), np.ptp(y))) + 10))
    left, top = min(max(0, cx-side//2), w-side), min(max(0, cy-side//2), h-side)
    crop = left, top, left + side, top + side
    gap, header = 8, 24
    widths = (w, h, h, w)
    starts = [gap + sum(widths[:i]) + i * gap for i in range(4)]
    board = Image.new('RGB', (sum(widths) + 5 * gap, (h + header + gap) * len(panels) + gap), '#1b252a')
    draw = ImageDraw.Draw(board)
    for row, (mode, (fixed, mounted)) in enumerate(panels.items()):
        source = refs[mode[:2]][1]
        data = (fixed, fixed, source, mounted)
        labels = (f'{mode}  |  fixed  |  world {world}, {stage}', f'{mode}  |  contour detail',
                  'Supersampled reference  |  same crop', f'{mode}  |  mounted camera')
        top_y = gap + row * (h + header + gap)
        for col, (rgb, title) in enumerate(zip(data, labels, strict=True)):
            image = Image.fromarray(rgb)
            if col in (1, 2):
                image = image.crop(crop).resize((h, h), Image.Resampling.NEAREST)
            elif col == 0:
                ImageDraw.Draw(image).rectangle((crop[0], crop[1], crop[2]-1, crop[3]-1), outline='#fbce69')
            draw.text((starts[col], top_y + 3), title, fill='white')
            board.paste(image, (starts[col], top_y + header))
    board.save(path)
    return crop


def write_motion_board(path, mode, images, witnesses, worlds):
    """Same-scale fixed-camera crops track the exact reported shadow witness across time."""
    w, h, detail, gap, header = 320, 240, 180, 8, 23
    widths = (w, w, detail, detail)
    starts = [gap + sum(widths[:i]) + i*gap for i in range(4)]
    board = Image.new('RGB', (sum(widths)+5*gap, len(worlds)*(h+header+gap)+gap), '#1b252a')
    draw = ImageDraw.Draw(board)
    for row, world in enumerate(worlds):
        at = witnesses[f'{mode}/{world}']['witness_xy']
        xx = min(max(0, at[0]-15), w-30)
        yy = min(max(0, at[1]-15), h-30)
        crop = (xx, yy, xx+30, yy+30)
        y = gap + row*(h+header+gap)
        for idx, stage in enumerate(('initial', 'step80', 'initial', 'step80')):
            rgb = Image.fromarray(images[mode, stage, world, 'fixed'][0])
            if idx < 2:
                ImageDraw.Draw(rgb).rectangle((xx, yy, xx+29, yy+29), outline='#fbce69')
                title = f'{mode}  |  world {world}  |  {stage}'
            else:
                rgb = rgb.crop(crop).resize((detail, detail), Image.Resampling.NEAREST)
                title = f'Receiver detail  |  {stage}'
            draw.text((starts[idx], y+3), title, fill='white')
            board.paste(rgb, (starts[idx], y+header))
    board.save(path)
