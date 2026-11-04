"""Save metric images of an interactive Panda recording without re-running its physics.

Run robot_payload_pick_place.py first. The html file displays the same samples
as the NPZ, quantized solely for visualization. No external viewer is required.
"""

import argparse
import base64
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np

import phymeo as ph
from phymeo.dynamics.rotation import matrix_quaternion


def look_at(eye, target):
    backward = np.asarray(eye) - target
    backward /= np.linalg.norm(backward)
    right = np.cross([0, 0, 1.0], backward)
    right /= np.linalg.norm(right)
    return matrix_quaternion(np.column_stack((right, np.cross(backward, right), backward)))


def encode(frame):
    return base64.b64encode(np.ascontiguousarray(frame).tobytes()).decode("ascii")


def observe(record, output, *, operator=None):
    width, height = 192, 144
    table = np.asarray(record["table_center_m"])
    eye = table + [0.28, -0.48, 0.28]
    target = table + [0, 0, 0.075]
    camera = ph.DepthCamera(width, height, 0.68)
    view = look_at(eye, target)
    model = None
    samples, stamps, panels = [], [], []
    for label, event in record["events"].items():
        parts = event["static_motions"]
        keys = tuple(ph.CollisionKey(*p["key"]) for p in parts)
        geometry = tuple(
            ph.CollisionShape(
                k,
                p["kind"],
                tuple(p["vertex_ids"]),
                radius=p["radius_m"],
                group=p["group"],
                mask=p["mask"],
            )
            for p, k in zip(parts, keys, strict=True)
        )
        if model is None:
            model = ph.CollisionModel(geometry)
            names_by_key = dict(zip(keys, [p["name"] for p in parts], strict=True))
            names = [names_by_key[s.key] for s in model.shapes]
        elif geometry != model.shapes or [p["name"] for p in parts] != names:
            raise ValueError("recording changed camera shape registration")
        poses = {
            k: ph.CollisionMotion(p["start_m"], body_orientation_xyzw=p["body_orientation_xyzw"])
            for k, p in zip(keys, parts, strict=True)
        }
        image = camera.capture(
            model, poses, position_m=eye, orientation_xyzw=view, operator=operator
        )
        samples.append(image)
        stamps.append(event["time_s"])
        mask = image.shape_index >= 0
        zmm = np.zeros((height, width), dtype="<u2")
        zmm[mask] = np.clip(np.rint(image.ray_depth_m[mask] * 1000), 1, 65535).astype(np.uint16)
        rgb_n = np.clip(np.rint((image.world_normals + 1) * 127.5), 0, 255).astype(np.uint8)
        panels.append(
            dict(
                label=label,
                t=event["time_s"],
                load_t=event["load_time_s"],
                table=event["table_force_N"],
                pads=event["bilateral_normal_N"],
                mm=encode(zmm),
                ids=encode(image.shape_index.astype("<i2")),
                normals=encode(rgb_n),
                payload_pixels=int(np.count_nonzero(image.shape_index == names.index("payload"))),
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    files = [output / name for name in ("observations.npz", "observations.html")]
    if any(f.exists() for f in files):
        raise ValueError("choose a new output directory")
    ids = [s.key for s in model.shapes]
    np.savez_compressed(
        files[0],
        time_s=stamps,
        shape_keys=[[i.world, i.body, i.shape] for i in ids],
        names=names,
        eye_m=eye,
        orientation_xyzw=view,
        ray_depth_m=[s.ray_depth_m for s in samples],
        forward_depth_m=[s.forward_depth_m for s in samples],
        world_normals=[s.world_normals for s in samples],
        shape_index=[s.shape_index for s in samples],
    )
    data = json.dumps(
        dict(w=width, h=height, names=names, frames=panels), allow_nan=False, separators=(",", ":")
    ).replace("<", "\\u003c")
    page = """<!doctype html><html lang=en><meta charset=utf-8><title>Robot depth observations</title>
<style>body{background:#111d29;color:#e2eef6;font:17px system-ui;padding:30px;max-width:1150px;margin:auto}h1{font-size:28px}small{color:#a9bfd0}main{display:flex;gap:12px;flex-wrap:wrap;margin:25px 0}figure{margin:0;flex:1;min-width:250px}canvas{width:100%;image-rendering:pixelated;background:#192937}figcaption{font-weight:600}input{width:min(100%,620px)} #details{min-height:3em}</style>
<h1>Metric observations from a Panda payload session</h1><small>All panels use the same accepted pose. Contact force refers to the preceding physical step.</small>
<main><figure><canvas></canvas><figcaption>Ray depth · nearer is brighter</figcaption></figure><figure><canvas></canvas><figcaption>Shape ID · payload in amber</figcaption></figure><figure><canvas></canvas><figcaption>World-space surface normal</figcaption></figure></main>
<label>Recorded event <input type=range value=0 min=0></label><p id=stats></p><p id=details>Point to a pixel for the shape and ray distance.</p>
<script id=reading type=application/json>DATA</script>
<script>
const recording = JSON.parse(document.querySelector('#reading').textContent);
const slider = document.querySelector('input');
const canvases = Array.from(document.querySelectorAll('canvas'));
let labels, depths;
slider.max = recording.frames.length - 1;

function bytes(encoded) {
  return Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
}
function pixelColor(panel, id, distance, normals, index) {
  if (id < 0) return [15,27,39];
  if (panel === 0) {
    const level = Math.max(0, Math.min(1, (940 - distance) / 680));
    return [26 + 207 * level, 50 + 176 * level, 80 + 148 * level];
  }
  if (panel === 2) return normals.subarray(3 * index, 3 * index + 3);
  if (recording.names[id] === 'payload') return [255,170,73];
  if (recording.names[id] === 'table') return [72,108,126];
  return [65 + (id*53)%126, 105 + (id*23)%118, 126 + (id*31)%108];
}
function draw() {
  const frame = recording.frames[Number(slider.value)];
  labels = new DataView(bytes(frame.ids).buffer);
  depths = new DataView(bytes(frame.mm).buffer);
  const normals = bytes(frame.normals);
  const pads = frame.pads.map(x => x.toFixed(3)).join(' / ');
  document.querySelector('#stats').textContent =
    frame.label + ' | t = ' + frame.t.toFixed(4) + ' s | table ' + frame.table.toFixed(3) +
    ' N | fingers ' + pads + ' N | ' + frame.payload_pixels + ' payload pixels';
  canvases.forEach((canvas, panel) => {
    canvas.width = recording.w;
    canvas.height = recording.h;
    const context = canvas.getContext('2d');
    const image = context.createImageData(canvas.width, canvas.height);
    for (let i = 0; i < canvas.width * canvas.height; i++) {
      const rgb = pixelColor(panel, labels.getInt16(2*i, true), depths.getUint16(2*i, true), normals, i);
      image.data.set([...rgb, 255], 4*i);
    }
    context.putImageData(image, 0, 0);
  });
}
slider.oninput = draw;
for (const canvas of canvases) {
  canvas.onpointermove = event => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.min(recording.w - 1, Math.max(0, Math.floor((event.clientX - rect.left) * recording.w / rect.width)));
    const y = Math.min(recording.h - 1, Math.max(0, Math.floor((event.clientY - rect.top) * recording.h / rect.height)));
    const i = y * recording.w + x, id = labels.getInt16(2*i, true);
    document.querySelector('#details').textContent = id < 0 ? 'Background' :
      recording.names[id] + ' | (' + x + ', ' + y + ') | ' + depths.getUint16(2*i, true) + ' mm';
  };
}
draw()
</script></html>"""
    files[1].write_text(page.replace("DATA", data), encoding="utf-8")
    return [
        {"label": p["label"], "time_s": p["t"], "payload_pixels": p["payload_pixels"]}
        for p in panels
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("host", "interpret", "cuda"), default="host")
    args = parser.parse_args()
    context = nullcontext() if args.backend == "host" else ph.TyxRaycaster(args.backend)
    with context as operator:
        print(
            json.dumps(
                observe(
                    json.loads(args.record.read_text(encoding="utf8")),
                    args.output,
                    operator=operator,
                )
            )
        )
        if operator is not None:
            print(json.dumps(operator.stats()))
