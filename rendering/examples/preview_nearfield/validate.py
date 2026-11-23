"""Render geometrically similar rooms to test the selected diffuse range in metres."""
import argparse
from contextlib import closing
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys


def run(prepared, output, device_index):
    from preview_compact.runtime import CompactBackend, activate
    prepared, manifest = activate(prepared)
    if manifest["contract"].get("local_diffuse", {}).get("radius_m") != .12:
        raise ValueError("this fixture is registered for a 12 cm diffuse range")
    import numpy as np
    from PIL import Image
    import closed_room_bounce_calibration as room
    from _panda_scene import write_mesh
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath

    output.mkdir(parents=True, exist_ok=False)
    report = {"source": sha256((prepared/"source-manifest.json").read_bytes()).hexdigest(),
              "status":"started", "cases":{}}
    geometry = output/"geometry"
    scene, frame, camera, studio = room.build_public(geometry)
    lights = replace(studio, panel_strength=0., environment_radiance=.25)
    sensors = {}
    incoming = {}
    try:
        for label, size, interior in (("open",1.,"open"),("far-dark",1.,"dark"),
                                      ("near-dark",.01,"dark"),("near-red",.01,"red")):
            faces = [f for f in room.geometry()[:6] if interior!="open" or f["id"]==5]
            ids = {f["id"] for f in faces}
            names = {f["name"] for f in faces}
            folder = geometry/label
            folder.mkdir()
            meshes = tuple(write_mesh(folder,f["name"],[tuple(v*size for v in point)
                       for point in room.vertices(f)],(f["normal"],)*4,((0,1,2),(0,2,3))) for f in faces)
            materials = tuple(replace(m, base_color=(.6,.6,.6) if m.id=="back" else
                        ((.8,.01,.01) if interior=="red" else (0.,0.,0.)),
                        metallic=0. if m.id=="back" or interior=="red" else 1.,roughness=1.)
                        for m in scene.materials if m.id in names)
            model = replace(scene,key=label,materials=materials,meshes=meshes,
                            instances=tuple(i for i in scene.instances if i.id in ids))
            pose = replace(frame,key=replace(frame.key,scene_key=label),
                           nodes=tuple(n for n in frame.nodes if n.node_id in names))
            eye = tuple(x*size for x in camera.world_pose.position_m)
            cam = replace(camera,frame_key=pose.key,world_pose=replace(camera.world_pose,position_m=eye),
                          near_m=camera.near_m*size,far_m=camera.far_m*size)
            backend = CompactBackend(prepared,device_index=device_index,studio=lights,seed=0,exposure=1.,
                                     reflection_mode="environment",validation=True)
            with closing(backend.open_scene(model)) as session:
                shot = session.capture(pose,CaptureRequest(cam,RenderPath.RASTER,tuple(Channel)))
                planes = {p.channel:p.data for p in shot.planes}
                guide = session.read_incoming_guide()
                low = session.read_low_sensors()
                w,h = guide["width"],guide["height"]
                hit_id = np.frombuffer(low["id"]["data"],"<u4").reshape(h,w)
                values = np.frombuffer(guide["data"],"<f4").reshape(h,w,4)
                y,x = np.mgrid[:h,:w]
                # Central 8x8 guide cells see the back wall. In the large room
                # each ray starts over half a metre away from other surfaces.
                selected = (abs(x-(w-1)/2)<4) & (abs(y-(h-1)/2)<4)
                if selected.sum()!=64 or not np.all(hit_id[selected]==5) or not np.all(
                        values[...,3][selected].astype(np.uint32)&(1<<20)):
                    raise RuntimeError("registered receiver is missing or untraced")
                signal = values[...,:3][selected]
                if not np.isfinite(signal).all():
                    raise RuntimeError("invalid incoming irradiance")
                incoming[label] = signal.copy()
                sensors[label] = {k.value:sha256(v).hexdigest() for k,v in planes.items() if k!=Channel.RGB}
                Image.frombytes("RGB",(cam.width,cam.height),planes[Channel.RGB]).save(output/(label+".png"))
                report["cases"][label] = dict(world_scale=size,anchors=64,minimum=signal.min(axis=0).tolist(),
                        maximum=signal.max(axis=0).tolist(),mean=signal.mean(axis=0).tolist(),sensors=sensors[label])
        if not np.array_equal(incoming["open"],incoming["far-dark"]):
            raise RuntimeError("an obstruction beyond the radius changed the central guide")
        if np.min(incoming["open"])<=.20 or np.max(incoming["near-dark"])>=.10:
            raise RuntimeError("the small closed enclosure did not attenuate incoming sky")
        red=incoming["near-red"].mean(axis=0)
        if red[0]<=red[1]+.035 or red[0]<=red[2]+.035:
            raise RuntimeError("near surfaces did not contribute colored light")
        if sensors["near-red"]!=sensors["near-dark"]:
            raise RuntimeError("recoloring a wall changed geometric sensors")
        report["status"]="complete"
    finally:
        (output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device-index",type=int,default=1)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    run(args.prepared,args.output,args.device_index)
