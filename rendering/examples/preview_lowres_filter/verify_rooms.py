"""Exercise the low-resolution filter on an unlit room and a material substitution."""
import argparse
from contextlib import closing
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from preview_compact.run_smoke import cases
from preview_compact_rigid_history.runtime import activate
from preview_compact_rigid_history.state import bind_presentations, frame_digest
from .runtime import backend_class


def capture(prepared, root, device):
    prepared, _ = activate(prepared)
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath

    root.mkdir(parents=True, exist_ok=False)
    outputs = {}
    captures = {}
    for label, scene, frame, camera, studio, exposure in cases(root, False):
        if label not in {"room", "zero-light", "chromatic"}:
            continue
        source = frame_digest({"model": asdict(scene), "studio": asdict(studio), "frame": asdict(frame)})
        binding, = bind_presentations([frame], source)
        owner_type = backend_class(root/label)
        with closing(owner_type(prepared, device_index=device, studio=studio, seed=0,
                                exposure=exposure, reuse_history=True, validation=True).open_scene(scene)) as owner:
            shot = owner.capture(frame, CaptureRequest(camera, RenderPath.RASTER, tuple(Channel)), presentation=binding)
            if shot.frame_key != frame.key or shot.camera != camera or shot.time_s != frame.time_s:
                raise ValueError("published room differs")
            owner.read_current_incoming()
            filtered = owner.read_filtered_guide()
            data = np.frombuffer(filtered["data"], "<f4").reshape(-1, 4)
            traced = (data[:, 3].astype(np.uint32) & (1 << 20)) != 0
            if not traced.any() or not np.isfinite(data).all():
                raise ValueError("room had no traced finite surface")
            captures[label] = data[:, :3].copy()
            rgb, = (p for p in shot.planes if p.channel == Channel.RGB)
            actual = owner.diagnostics()["actual"]
            if actual["primary_proposals"] != 8 * actual["traced_anchor_pixels"] or actual["invalid"]:
                raise ValueError("unexpected ray proposal counts")
            outputs[label] = dict(geometry={p.channel.value:sha256(p.data).hexdigest()
                                            for p in shot.planes if p.channel != Channel.RGB},
                                  rgb_sha256=sha256(rgb.data).hexdigest(), filter_sha256=sha256(filtered["data"]).hexdigest(),
                                  traced_cells=int(traced.sum()), max_original_display_code=max(rgb.data),
                                  filtered_min=float(data[traced,:3].min()), filtered_max=float(data[traced,:3].max()),
                                  proposals=actual["primary_proposals"])
        device_data = owner.diagnostics()["device"]
        if device_data["validation_errors"] or device_data["live_resources"]:
            raise ValueError("device validation or close failed")
    if len(outputs) != 3 or not (outputs["room"]["geometry"] == outputs["chromatic"]["geometry"] == outputs["zero-light"]["geometry"]):
        raise ValueError("geometry changed under light or material control")
    if np.any(captures["zero-light"]) or outputs["zero-light"]["max_original_display_code"]:
        raise ValueError("zero-light control generated radiance")
    delta = np.abs(captures["room"] - captures["chromatic"])
    if not np.any(delta > 1e-4) or outputs["room"]["rgb_sha256"] == outputs["chromatic"]["rgb_sha256"]:
        raise ValueError("filtered guide and presented color must respond to recoloring")
    results = dict(status="complete", cases=outputs, geometry_equal=True,
                   changed_low_cells=int(np.sum(np.any(delta > 1e-4,axis=1))), maximum_color_difference=float(delta.max()))
    (root/"report.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device-index",type=int,default=1)
    args=parser.parse_args()
    capture(args.prepared,args.output,args.device_index)
