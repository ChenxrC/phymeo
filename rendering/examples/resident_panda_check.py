"""Resident Vulkan full-robot checks against frozen R1 sensors and physical anchors."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from _panda_anchors import measure
from _panda_scene import build_scene, cameras, marker, unit
from panda_replay import render_set
from phymeo_rendering.adapters.panda_replay import PandaReplaySource
from phymeo_rendering.assets import Material
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import RenderPath
from phymeo_rendering.scene import DirectionalLight, Instance, Pose, VisualScene

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    root = args.output
    root.mkdir(parents=True,exist_ok=False)
    geo = root/"geometry"
    geo.mkdir()
    fixture = ROOT/"tests/fixtures/panda_replay"
    def replay(scene):
        return PandaReplaySource(scene,fixture/"panda.urdf",fixture/"trace.json")
    empty = replay(VisualScene("input",(),(),()))
    geom = empty.provenance(next(empty.frames()))["geometry"]
    scene = build_scene(ROOT/"assets/panda",geo,geom["cube_full_size_m"],geom["table_full_size_m"])
    fill = DirectionalLight(tuple(unit((-0.6,0.8,-0.65))), (1.35,1.35,1.35))
    scene = replace(scene,key="panda-r1-raster-fill-v1",lights=(fill,))
    def engine():
        return NativeBackend(library_path=args.library,device_index=args.device_index,validation=True,resident=True)
    raster = RenderPath.RASTER
    source = replay(scene)
    frames = tuple(source.frames())
    actual = render_set(engine(),source,raster,{"fixed","wrist"},root,"resident",True)
    evidence = ROOT/"results/r1/evidence"
    anchor_bytes = (evidence/"source-anchors.json").read_bytes()
    anchors = json.loads(anchor_bytes)
    markers = marker(geo)
    diag = VisualScene("panda-r1-material-fiducials",(markers,),(Material("tag",(0.5,0.5,0.5),0,0.8),),
                       tuple(Instance(10+i,markers.id,"tag",a["node"],Pose(tuple(a["node_point"])))
                             for i,a in enumerate(anchors)), scene.lights)
    tags = render_set(engine(),replay(diag),raster,{"fixed"},root,"markers",True)
    measurements, sensor_comparison = [], []
    with ZipFile(evidence/"sensor-planes.zip") as archive:
        for i, frame in enumerate(frames):
            for cam in cameras(frame)[:2]:
                data = actual[i,cam.id]
                shape = (cam.height,cam.width)
                def plane(source,channel,dtype):
                    return np.frombuffer(source[channel],dtype).reshape(shape)
                ids = plane(data,Channel.INSTANCE_ID,"<u4")
                depth = plane(data,Channel.FORWARD_DEPTH,"<f4")
                valid = plane(data,Channel.VALID,"u1")
                if not np.array_equal(ids!=0,valid==1) or not np.array_equal(np.isfinite(depth),ids!=0):
                    raise AssertionError("visibility mismatch")
                ref = {c:archive.read(f"raster-{i+1}-{cam.id}-{c.value}.raw")
                       for c in (Channel.INSTANCE_ID,Channel.FORWARD_DEPTH)}
                expected = plane(ref,Channel.INSTANCE_ID,"<u4")
                expected_z = plane(ref,Channel.FORWARD_DEPTH,"<f4")
                # Evaluate homogeneous ID interiors separately from raster edge/ownership ties.
                interior = expected!=0
                border = np.pad(expected,2,mode="constant")
                for y in range(5):
                    for x in range(5):
                        interior &= expected==border[y:y+cam.height,x:x+cam.width]
                matched = interior & (ids==expected)
                if not np.any(matched):
                    raise AssertionError("no comparable interior")
                err = abs(depth[matched]-expected_z[matched])
                row = dict(pose=frame.key.revision,camera=cam.id,pixels=ids.size,
                           mismatched_id_pixels=int((ids!=expected).sum()),interior_pixels=int(interior.sum()),
                           interior_owner_disagreements=int((interior & (ids!=expected)).sum()),
                           max_matching_interior_depth_error_m=float(err.max()),
                           p99_matching_interior_depth_error_m=float(np.percentile(err,99)),
                           cube_pixels=int((ids==1).sum()),robot_pixels=int((ids>=100).sum()))
                sensor_comparison.append(row)
                if min(row["robot_pixels"],row["cube_pixels"])==0:
                    raise AssertionError(row)
                if cam.id=="fixed":
                    measurements += measure(anchors,i,ids,depth,plane(tags[i,"fixed"],Channel.INSTANCE_ID,"<u4"))
    audit = json.loads((root/"resident-capture.json").read_bytes())
    report = dict(source_anchors_sha256=sha256(anchor_bytes).hexdigest(),physical_anchors=measurements,
                  compare_to_archived_sapien_raster=sensor_comparison,capture_receipt="resident-capture.json")
    (root/"comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(dict(anchors=len(measurements),max_px=max(x["error_px"] for x in measurements),
                          cameras=len(sensor_comparison),receipt_keys=list(audit)[:5])))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--device-index",type=int,default=1)
    run(p.parse_args())
