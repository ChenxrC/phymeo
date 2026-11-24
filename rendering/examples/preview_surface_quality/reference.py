"""Render the registered white-shell crop with four linear subpixel samples.

Uses the full-distance resident Vulkan path, the recorded lighting, and the
same normal-mapped Panda scene as the 720p motion comparison.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

import panda_render_performance as panda
from phymeo_rendering.backends._sapien_pixels import encode_linear_image
from phymeo_rendering.backends.native import NativeBackend
from preview_lowres_filter._helpers import digest, save
from panda_motion_preview import verify_close


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device-index", type=int, default=1)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    scenes, frames, studio, trace = panda.bind_scene(out / "geometry")
    scene = scenes[1]
    frame, request = panda.make_inputs(scene, frames)[60]
    x, y, right, bottom = (457, 110, 795, 355)
    width, height = right - x, bottom - y
    c = request.camera
    camera = replace(c, width=2*width, height=2*height, fx_px=2*c.fx_px,
                     fy_px=2*c.fy_px, cx_px=2*(c.cx_px-x), cy_px=2*(c.cy_px-y))
    backend = NativeBackend(library_path=args.library, shader_directory=args.library.parent,
                            resident=True, shadow_mode="ray_query", reflection_mode="single_bounce",
                            indirect_mode="diffuse_bounce", indirect_bounces=1, indirect_seed=1,
                            progressive=True, spatial_filter="none", area_light_grid=8, msaa_samples=1,
                            studio=studio, exposure=8., display_transform="pbr-neutral",
                            validation=True, device_index=args.device_index)
    session = backend.open_scene(scene)
    try:
        for _ in range(64):
            shot = session.capture(frame, replace(request, camera=camera))
        packet = session.read_linear_mean()
        if (packet["batches"], packet["primary_samples_per_pixel"]) != (64, 1024):
            raise RuntimeError("incomplete reference accumulation")
        if shot.frame_key != frame.key or shot.camera != camera:
            raise RuntimeError("reference camera or state differs")
        samples = np.frombuffer(packet["data"], "<f4").reshape(height, 2, width, 2, 4)
        rgb = samples[..., :3].mean(axis=(1, 3))
        if not np.isfinite(rgb).all():
            raise RuntimeError("nonfinite reference radiance")
        image = Image.frombytes("RGB", (width, height), encode_linear_image(
            rgb, exposure=8., display_transform="pbr-neutral"))
        image.save(out / "white-shell.png")
    finally:
        session.close()
        verify_close(session.diagnostics()["device"])
    save(out / "reference.json", dict(box=[x, y, right, bottom], frame=60, scene=scene.key,
         trace_sha256=digest(trace), library_sha256=digest(args.library), subpixels=4,
         diffuse_samples_per_subpixel=1024, panel_grid=8, indirect_bounces=1,
         rgb_sha256=digest(out / "white-shell.png")))


if __name__ == "__main__":
    main()
