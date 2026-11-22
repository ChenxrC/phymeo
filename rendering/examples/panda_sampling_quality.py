"""Capture the fixed R2 scene, change one sampling setting at a time and compare ROI."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image

from _panda_scene import build_scene, cameras
from phymeo_rendering.adapters.panda_replay import PandaReplaySource
from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig, SapienStudio
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import VisualScene

ROOT = Path(__file__).resolve().parents[1]
TRIALS = (11, 31, 73)
LEVELS = (4, 16, 64, 256)
REFERENCES = (1009, 2027)
SIZE = (640, 480)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def camera(frame):
    cam = cameras(frame)[0]
    scale = SIZE[0] / cam.width
    return replace(cam, width=SIZE[0], height=SIZE[1], fx_px=cam.fx_px*scale, fy_px=cam.fy_px*scale,
                   cx_px=cam.cx_px*scale, cy_px=cam.cy_px*scale)


def setup(output):
    fixture = ROOT / "tests/fixtures/panda_replay"
    def read(scene):
        return PandaReplaySource(scene, fixture / "panda.urdf", fixture / "trace.json")
    dummy = read(VisualScene("quality-metadata", (), (), ()))
    geom = dummy.provenance(next(dummy.frames()))["geometry"]
    directory = output / "geometry"
    directory.mkdir()
    scene = build_scene(ROOT / "assets/panda", directory, geom["cube_full_size_m"], geom["table_full_size_m"])
    return scene, tuple(read(scene).frames())


def photo(session, frame, path, channels):
    exposure = camera(frame)
    capture = session.capture(frame, CaptureRequest(exposure, path, channels))
    if capture.frame_key != frame.key or capture.time_s != frame.time_s or capture.camera != exposure:
        raise RuntimeError("source and exposure keys differ")
    return {plane.channel: plane.data for plane in capture.planes}


def render(output, device):
    output.mkdir(parents=True, exist_ok=False)
    scene, frames = setup(output)
    first, frame = frames
    if frame.key.revision != 2159:
        raise ValueError("different source pose")
    path = RenderPath.RASTER
    sensors = SapienBackend(SapienConfig(device=device, path=path)).open_scene(scene)
    try:
        p = photo(sensors, frame, path, (Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID))
        for channel, raw in p.items():
            (output / (channel.value + ".raw")).write_bytes(raw)
    finally:
        sensors.close()
    save_json(output / "sensors.json", sensors.diagnostics())
    path = RenderPath.PATH_TRACE
    settings = [(spp, seed) for spp in LEVELS for seed in TRIALS] + [(1024, seed) for seed in REFERENCES]
    rows = []
    for spp, seed in settings:
        cfg = SapienConfig(device=device, path=path, samples_per_pixel=spp,
                           sampling_mode="shifted-halton", seed=seed,
                           studio=SapienStudio(center_m=(0.18, 0, 0.42), scale_m=0.85))
        engine = SapienBackend(cfg)
        session = engine.open_scene(scene)
        def take(target):
            return photo(session, target, path, (Channel.RGB,))[Channel.RGB]
        try:
            pixels = take(frame)
            cold_hash = sha256(pixels).hexdigest()
            reset = None
            if (spp,seed) == (4,11):
                changed = take(first)
                back = take(frame)
                if changed == pixels or back != pixels:
                    raise ValueError("camera history contaminated the repeated exposure")
                reset = [cold_hash, sha256(changed).hexdigest(), sha256(back).hexdigest()]
        finally:
            session.close()
        if reset is not None:
            session = engine.open_scene(scene)
            try:
                fresh = take(frame)
                if fresh != pixels:
                    raise ValueError("fresh scene and reused camera state disagree")
                reset.append(sha256(fresh).hexdigest())
            finally:
                session.close()
        name = f"spp-{spp}-seed-{seed}"
        Image.frombytes("RGB", SIZE, pixels).save(output / (name + ".png"))
        diag = session.diagnostics()
        save_json(output / (name + ".json"), diag)
        rows.append(dict(name=name, samples=spp, seed=seed, source_record=frame.key.revision,
                         sha256=cold_hash, shader_sha256=diag["shader_sha256"], reset=reset))
        print(name, cold_hash, flush=True)
    save_json(output / "exposures.json", rows)


def spatial_filter(rgb, ids, depth):
    """Two undecimated 5x5 bilateral passes, guided by the matching raster ID and depth."""
    signal = rgb.astype(np.float32) / 255
    d = np.where(np.isfinite(depth), depth, 0)
    h, w = ids.shape
    coefficients = (1,4,6,4,1)
    for step, sigma in ((1,0.16), (2,0.12)):
        pad = 2*step
        extra = ((pad,pad),(pad,pad))
        guard_ids = np.pad(ids, extra, mode="edge")
        guard_depth = np.pad(d, extra, mode="edge")
        colors = np.pad(signal, (*extra,(0,0)), mode="edge")
        accum = np.zeros_like(signal)
        weight_sum = np.zeros_like(d, dtype=np.float32)
        for j, wy in enumerate(coefficients):
            for i, wx in enumerate(coefficients):
                region = (slice(j*step,j*step+h), slice(i*step,i*step+w))
                other = colors[region]
                weight = (ids == guard_ids[region]) * (wx*wy)
                delta = np.mean((signal-other)**2, axis=2)/(2*sigma*sigma)
                dz = (d - guard_depth[region])/(0.004 * step)
                weight = (weight * np.exp(-delta - 0.5*dz*dz)).astype(np.float32)
                accum += other * weight[...,None]
                weight_sum += weight
        signal = accum / weight_sum[...,None]
    return np.floor(signal.clip(0,1)*255 + .5).astype(np.uint8)


def compare(root):
    rows = json.loads((root/"exposures.json").read_bytes())
    if len(rows)!=14 or len({row["sha256"] for row in rows})!=14:
        raise ValueError("missing or repeated exposures across sampling configurations")
    shape = SIZE[::-1]
    ids = np.fromfile(root/(Channel.INSTANCE_ID.value+".raw"),dtype="<u4").reshape(shape)
    depth = np.fromfile(root/(Channel.FORWARD_DEPTH.value+".raw"),dtype="<f4").reshape(shape)
    valid = np.fromfile(root/(Channel.VALID.value+".raw"),dtype="u1").reshape(shape)
    if not np.array_equal(ids!=0, valid==1) or not np.array_equal(np.isfinite(depth),ids!=0):
        raise ValueError("inconsistent ROI geometry")
    masks = {"robot":(ids>=100)&(ids<=158), "cube":ids==1, "table":ids==2, "all_foreground":ids!=0}
    counts = {k:int(v.sum()) for k,v in masks.items()}
    if min(counts.values())<=0:
        raise ValueError("an authored ROI is empty")
    def load(spp,seed):
        stem = f"spp-{spp}-seed-{seed}"
        row = next(r for r in rows if r["name"]==stem)
        arr = np.asarray(Image.open(root/(stem+".png")).convert("RGB"))
        if arr.shape != (*shape,3) or sha256(arr.tobytes()).hexdigest()!=row["sha256"]:
            raise ValueError("saved image differs from device capture")
        return arr
    refs = [load(1024, seed).astype(np.float64)/255 for seed in REFERENCES]
    ref = (refs[0]+refs[1])/2
    def mse(a,b):
        return {k:float(np.mean((a[m]-b[m])**2)) for k,m in masks.items()}
    Image.fromarray(np.floor(ref*255+.5).astype(np.uint8)).save(root/"reference-mean.png")
    results = []
    for spp in LEVELS:
        for seed in TRIALS:
            image = load(spp,seed)
            filtered = spatial_filter(image,ids,depth)
            results.append(dict(spp=spp,seed=seed,raw=mse(image/255.,ref),
                                filtered=mse(filtered/255.,ref)))
            if seed==11:
                Image.fromarray(filtered).save(root/f"filtered-{spp}.png")
    levels = {spp:{mode:{k:float(np.mean([r[mode][k] for r in results if r["spp"]==spp]))
                         for k in masks} for mode in ("raw","filtered")} for spp in LEVELS}
    report = dict(space="public sRGB8/255, mean squared error per channel against mean of two held-out 1024 spp displays",
                  counts=counts, independent_reference_difference=mse(*refs), measurements=results,
                  by_level=levels, raw_foreground_converged=(levels[256]["raw"]["all_foreground"]<levels[4]["raw"]["all_foreground"]),
                  filter="spatial, screen-space; 5x5 two passes; uses clean independent raster ID/forward depth; no temporal history")
    save_json(root/"comparison.json",report)
    print(json.dumps(levels,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("render","compare"))
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.operation=="render":
        render(args.work,args.device)
    else:
        compare(args.work)
