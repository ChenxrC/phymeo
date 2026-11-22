"""Build a held-out radiance reference in bounded independent GPU exposures."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from PIL import Image

from panda_movies import movie_camera
from panda_realtime_film import ROOT,load_episode
from phymeo_rendering.backends import _sapien_session
from phymeo_rendering.backends.sapien import SapienBackend,SapienConfig,SapienStudio
from phymeo_rendering.environment import RadianceEnvironment
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest,RenderPath


SEEDS=(1709,1129,3331,4591,5003,6011,7001,8017)
ROIS={"whole":(0,0,960,540),"shell":(410,70,490,90),"wood":(690,465,820,500)}


def run(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    scene,frames=load_episode(out/"geometry",args.excerpt,photographic=True,local_light_scale=.125)
    frame=frames[36]
    cam=movie_camera(frame,"materials-wide",36/(len(frames)-1))
    ratio=540/cam.height
    cam=replace(cam,width=960,height=540,cx_px=480.,cy_px=270.,fx_px=cam.fx_px*ratio,fy_px=cam.fy_px*ratio)
    path=RenderPath.PATH_TRACE
    request=CaptureRequest(cam,path,(Channel.RGB,))
    uri=(ROOT/"examples/materials/university_workshop_1k.hdr").as_uri()
    env=RadianceEnvironment(uri,"a125ada38fc602025326f29b68df09531bb16f899f1973b03d9ac5ac98142548",
                            strength=.0625,yaw_degrees=25.,face_size=256)
    studio=SapienStudio(center_m=(.18,0,.42),scale_m=.85,panel_strength=.125,environment_map=env)
    convert=_sapien_session.encode_linear_image
    total=np.zeros((540,960,3),dtype=np.float64)
    captures=[]
    for seed in SEEDS:
        cfg=SapienConfig(path=path,device="cuda:0",studio=studio,exposure=8.,
                         samples_per_pixel=512,seed=seed,sampling_mode="shifted-halton")
        session=SapienBackend(cfg).open_scene(scene)
        raw=[]
        def observe(image,**options):
            raw.append(np.array(image,dtype="<f4",copy=True))
            return convert(image,**options)
        try:
            start=perf_counter()
            with patch.object(_sapien_session,"encode_linear_image",side_effect=observe):
                shot=session.capture(frame,request)
            elapsed=1000*(perf_counter()-start)
        finally:
            session.close()
        if len(raw)!=1 or raw[0].shape!=total.shape or not np.isfinite(raw[0]).all():
            raise ValueError("invalid linear camera readback")
        image=raw.pop()
        total+=image
        dest=out/f"linear-seed-{seed}.npy"
        np.save(dest,image,allow_pickle=False)
        raw_sha=sha256(image.tobytes()).hexdigest()
        if raw_sha in {entry["linear_sha256"] for entry in captures}:
            raise ValueError("repeated random stream in held-out reference")
        captures.append(dict(seed=seed,samples=512,ms=elapsed,linear_sha256=raw_sha,
                             rgb_sha256=sha256(shot.planes[0].data).hexdigest(),raw_file=dest.name))
    linear=total/len(SEEDS)
    np.save(out/"linear-mean.npy",linear,allow_pickle=False)
    # Average unexposed linear radiance in FP64 before the sole display conversion.
    x=np.clip(8.*linear,0.,1.)
    result=np.floor(255.*np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)+.5).astype(np.uint8)
    Image.fromarray(result).save(out/"heldout.png")
    errors={}
    for spp,directory in zip((256,1024,4096),args.sources,strict=True):
        journal=json.loads((directory/"capture.json").read_text())
        with Image.open(directory/"frames/0000.png") as photo:
            original=np.asarray(photo)
        if (journal["source_indices"]!=[frame.key.revision] or journal["source_times_s"]!=[frame.time_s]
            or journal["camera_setup"]!="wide" or journal["size"]!=[960,540]
            or sha256(original.tobytes()).hexdigest()!=journal["frame_sha256"][0]):
            raise ValueError("candidate image does not match the registered exposure")
        diff=(original.astype(np.float64)-result)/255.
        errors[str(spp)]={name:float(np.mean(diff[y0:y1,x0:x1]**2)) for name,(x0,y0,x1,y1) in ROIS.items()}
    summary=dict(samples_per_seed=512,effective_primary_samples=4096,frame=36,revision=frame.key.revision,
        physics_time_s=frame.time_s,display="linear mean, exposure 8, clip and sRGB once",rois=ROIS,
        reference_rgb_sha256=sha256(result.tobytes()).hexdigest(),captures=captures,mse=errors,
        actual_source_sha256=sha256(args.excerpt.read_bytes()).hexdigest())
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(errors))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--excerpt",type=Path,required=True)
    p.add_argument("sources",type=Path,nargs=3,metavar="CANDIDATE")
    run(p.parse_args())
