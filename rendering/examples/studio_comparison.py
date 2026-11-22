"""Compare the resident lighting modes against held-out reference pixels at one physical pose."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from io import BytesIO
from panda_sampling_quality import SIZE, photo, setup, REFERENCES
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import RenderPath
from phymeo_rendering.studio import StudioLighting


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    scene,(_,frame) = setup(args.output)
    studio = StudioLighting(center_m=(0.18,0,0.42),scale_m=.85)
    captures = {}
    diagnostics = {}
    for label,preset in (("key",None),("studio",studio)):
        session = NativeBackend(library_path=args.library,resident=True,device_index=args.device_index,
                                studio=preset).open_scene(scene)
        try:
            captures[label] = photo(session,frame,RenderPath.RASTER,tuple(Channel))
            Image.frombytes("RGB",SIZE,captures[label][Channel.RGB]).save(args.output/(label+".png"))
        finally:
            session.close()
        diagnostics[label] = session.diagnostics()
    planes = [Channel.INSTANCE_ID,Channel.FORWARD_DEPTH,Channel.VALID]
    for channel in planes:
        if captures["key"][channel]!=captures["studio"][channel]:
            raise ValueError("lighting changed geometry: "+channel.value)
    shape = (SIZE[1],SIZE[0])
    with ZipFile(args.sampling) as source:
        refs = [np.asarray(Image.open(BytesIO(source.read(f"spp-1024-seed-{seed}.png"))),
                           dtype=np.float64)/255. for seed in REFERENCES]
        ref = (refs[0]+refs[1])/2
        ids = np.frombuffer(source.read(Channel.INSTANCE_ID.value+".raw"),dtype="<u4").reshape(shape)
    current = np.frombuffer(captures["studio"][Channel.INSTANCE_ID],dtype="<u4").reshape(shape)
    matched = ids==current
    masks = {"foreground":matched&(ids!=0),"robot":matched&(ids>=100)&(ids<=158),
             "payload":matched&(ids==1),"stand_top":matched&(ids==2),"background":matched&(ids==0)}
    report = {}
    for label,cap in captures.items():
        rgb = np.frombuffer(cap[Channel.RGB],dtype=np.uint8).reshape(*shape,3)/255.
        report[label] = {name:dict(pixels=int(mask.sum()),mse=float(np.mean((rgb[mask]-ref[mask])**2)),
                                   mean_rgb=rgb[mask].mean(axis=0).tolist(),
                                   reference_mean_rgb=ref[mask].mean(axis=0).tolist()) for name,mask in masks.items()}
    out = dict(source_revision=frame.key.revision,source_time_s=frame.time_s,
               signal="sRGB8/255 compared against the mean of the two held-out studio references",
               lighting_comparison="key is a lighting ablation; studio is the same declared lights as both references",
               sensor_mismatches_within_native=0,reference_id_disagreements=int(np.count_nonzero(~matched)),
               measurements=report,diagnostics=diagnostics,
               reference_archive_sha256=sha256(args.sampling.read_bytes()).hexdigest())
    (args.output/"comparison.json").write_text(json.dumps(out,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v["foreground"] for k,v in report.items()}))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--library",type=Path,required=True)
    cli.add_argument("--sampling",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--device-index",type=int,default=1)
    run(cli.parse_args())
