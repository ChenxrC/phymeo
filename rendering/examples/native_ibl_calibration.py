"""Photograph a known directional light field and check reflected world-space rays."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from reference_hdri_probe import (_DIRECTIONS, _NAMES, _PIXELS, _png, _srgb8,
                                  calibration_hdr, expected_radiance, view_camera, world_ray)
from roughness_calibration import write_sphere
from phymeo_rendering.assets import Material
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.environment import RadianceEnvironment
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene
from phymeo_rendering.studio import StudioLighting


def reflected_direction(camera,u,v,radius):
    # Independent ray intersection with the ideal sphere at the origin.
    eye=np.array(camera.world_pose.position_m)
    d=np.array([(u+.5-camera.cx_px)/camera.fx_px,(camera.cy_px-v-.5)/camera.fy_px,-1])
    d/=np.linalg.norm(d)
    b=d@eye
    t=-b-np.sqrt(b*b-eye@eye+radius*radius)
    n=(eye+t*d)/radius
    return d-2*(d@n)*n


def run(output, library, device_index):
    output.mkdir(parents=True,exist_ok=False)
    source=calibration_hdr()
    image=output/"calibration.hdr"
    image.write_bytes(source)
    sphere=write_sphere(output/"assets")
    empty=VisualScene("sky-directions",(),(),())
    mirror_pixels=[(u,v) for u in (54,64,74,85,95,105) for v in (52,62,70)]
    results={"source_sha256":sha256(source).hexdigest(),"background_exposures":0,"sky_samples":[],
             "mirror_samples":[],"material_exposures":0,"red_values":{},"closed_sessions":[]}
    last_sensors={}
    def shoot(session,frame,camera,name):
        captured=session.capture(frame,CaptureRequest(camera,RenderPath.RASTER,tuple(Channel)))
        planes={p.channel:p.data for p in captured.planes}
        sensor_hash=tuple(sha256(planes[c]).hexdigest() for c in (Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID))
        if camera.id in last_sensors and last_sensors[camera.id]!=sensor_hash:
            raise ValueError(f"lighting or material altered geometry for {camera.id}")
        last_sensors[camera.id]=sensor_hash
        raw=planes[Channel.RGB]
        _png(output/(name+".png"),raw,camera.width,camera.height)
        return np.frombuffer(raw,np.uint8).reshape(camera.height,camera.width,3)
    for yaw in (0.,90.):
        env=RadianceEnvironment(image.resolve().as_uri(),results["source_sha256"],yaw_degrees=yaw)
        studio=StudioLighting(panel_strength=0.,environment_map=env)
        def open_scene(scene):
            return NativeBackend(library_path=library,resident=True,device_index=device_index,studio=studio,
                                 validation=True).open_scene(scene)
        session=open_scene(empty)
        try:
            for i,(axis,name) in enumerate(zip(_DIRECTIONS,_NAMES,strict=True)):
                key=FrameKey(empty.key,"sky",0,i,i,"authored")
                cam,basis=view_camera(axis,key=key)
                rgb=shoot(session,SceneFrame(key,float(i),()),cam,f"sky-{name}-{int(yaw)}")
                results["background_exposures"]+=1
                for u,v in _PIXELS:
                    expected=np.array([_srgb8(x) for x in expected_radiance(world_ray(cam,basis,u,v),yaw_degrees=yaw)])
                    actual=rgb[v,u].astype(int)
                    results["sky_samples"].append(dict(yaw=yaw,axis=name,pixel=[u,v],
                        expected=expected.tolist(),actual=actual.tolist(),error=int(max(abs(actual-expected)))))
        finally:
            session.close()
            results["closed_sessions"].append(session.diagnostics())
        for material in (Material("mirror",(.95,)*3,1.,.01),Material("matte",(.70,)*3,0.,.85)):
            scene=VisualScene("ibl-ball",(sphere,),(material,),(Instance(1,sphere.id,material.id,"ball"),))
            session=open_scene(scene)
            try:
                key=FrameKey(scene.key,"ball",0,0,0,"authored")
                frame=SceneFrame(key,0.,(NodePose("ball",Pose()),))
                cam=Camera("ball",key,Pose((0.,0.,1.)),160,120,160.,160.,80.,60.,.1,5.)
                rgb=shoot(session,frame,cam,f"{material.id}-{int(yaw)}")
                results["material_exposures"]+=1
                results["red_values"].setdefault(material.id,[]).append(float(rgb[45:75,45:70,0].mean()))
                if material.id=="mirror":
                    for u,v in mirror_pixels:
                        direction=reflected_direction(cam,u,v,.25)
                        expected=np.array([_srgb8(.95*x) for x in expected_radiance(direction,yaw_degrees=yaw)])
                        actual=rgb[v,u].astype(int)
                        results["mirror_samples"].append(dict(yaw=yaw,pixel=[u,v],
                            expected=expected.tolist(),actual=actual.tolist(),error=int(max(abs(actual-expected)))))
            finally:
                session.close()
                results["closed_sessions"].append(session.diagnostics())
    sky=max(s["error"] for s in results["sky_samples"])
    mirror=max(s["error"] for s in results["mirror_samples"])
    changes={k:v[1]-v[0] for k,v in results["red_values"].items()}
    results.update(sky_max_rgb8=sky,mirror_max_rgb8=mirror,red_delta_90=changes,
                   sensor_sha256=last_sensors,thresholds=dict(sky_max=3,mirror_max=5,mirror_delta=8,matte_delta=3))
    results["passed"]=sky<=3 and mirror<=5 and changes["mirror"]>=8 and changes["matte"]>=3
    (output/"summary.json").write_text(json.dumps(results,indent=2,allow_nan=False),encoding="utf-8")
    if not results["passed"]:
        raise ValueError(f"directional lighting deviation: sky={sky}, mirror={mirror}, red delta={changes}")
    print(f"108 sky targets and 36 mirror directions; max RGB8 {sky}/{mirror}, red deltas {changes}")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--library",type=Path,required=True)
    cli.add_argument("--device-index",type=int,default=1)
    options=cli.parse_args()
    run(options.output,options.library,options.device_index)
