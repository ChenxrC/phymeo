"""Score one-bounce moving-occluder exposures against two fixed independent seeds.

Run reference once per scene and reuse its geometry and RGB for new candidates.
"""
import argparse
from contextlib import closing
from dataclasses import asdict
from functools import partial
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


POSES = (0, 15, 24)
REF_BATCHES = 64


def digest(data):
    return sha256(data).hexdigest()


def save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def experiment(args):
    if args.adapter:
        where = args.adapter.resolve(strict=True)
        sys.path.insert(0, str(where.parent.parent))
        impl = importlib.import_module(where.parent.name + ".runtime")
        if Path(impl.__file__).resolve() != where:
            raise ValueError("unexpected adapter module")
        impl.activate(args.prepared)
    import numpy as np
    from PIL import Image
    import guided_motion_study as fixture
    import guided_motion_metrics as metrics
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath

    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    result = dict(schema="phymeo-native-reveal-preview-v1", poses=list(POSES), reference_batches=REF_BATCHES,
                  fixture_sha256=digest(Path(fixture.__file__).read_bytes()), frames={}, scores={}, status="started")
    save(folder/"summary.json", result)
    def native(filtered, seed):
        return NativeBackend(library_path=args.library, shader_directory=args.shaders, device_index=1, resident=True,
                shadow_mode="ray_query", exposure=1., display_transform="pbr-neutral", msaa_samples=1,
                indirect_mode="diffuse_bounce", indirect_seed=seed, indirect_bounces=1, progressive=True,
                reflection_mode="environment", spatial_filter="guided" if filtered else "none")
    def shot(owner, node):
        scene, pose, cam=node
        capture=owner.capture(pose, CaptureRequest(cam,RenderPath.RASTER,tuple(Channel)))
        if capture.frame_key != pose.key or capture.camera != cam:
            raise ValueError("capture does not correspond to request")
        return {plane.channel:plane.data for plane in capture.planes}
    def rgb(planes, camera):
        return np.frombuffer(planes[Channel.RGB],np.uint8).reshape(camera.height,camera.width,3)
    def hashes(planes):
        return {c.value:digest(v) for c,v in planes.items() if c is not Channel.RGB}
    def rmse(a,b,mask):
        delta=metrics.srgb_to_display_linear(a[mask])-metrics.srgb_to_display_linear(b[mask])
        return float(np.sqrt(np.mean(delta*delta)))
    with TemporaryDirectory(prefix="phymeo-reveal-") as tmp:
        try:
            # Private prepared DLLs have a Python shim without bundled meshes. Bind
            # the same hash-verified vendor assets for either implementation.
            visual=fixture.original.panda_visual_assets
            assets=Path(__file__).resolve().parents[1]/"assets"/"panda"
            with patch.object(fixture.original,"panda_visual_assets",partial(visual,assets)):
                nodes=fixture.build_motion(Path(tmp))
            scene=nodes[0][0]
            if args.baseline:
                original=json.loads((args.baseline/'summary.json').read_text(encoding='utf-8'))
                if original['status']!='complete' or original['fixture_sha256']!=result['fixture_sha256']:
                    raise ValueError("reference setup differs")
                adapter_class=getattr(impl,"CompactBackend",None) or impl.SparseBackend
                producer=adapter_class(args.prepared,device_index=1,seed=0,studio=None,exposure=1.,reflection_mode="environment")
                result['prepared_sha256']=digest((args.prepared/'source-manifest.json').read_bytes())
            else:
                producer=native(True,0)
                result['library_sha256']=digest(args.library.read_bytes())
            previous=None
            with closing(producer.open_scene(scene)) as moving:
                for i,node in enumerate(nodes):
                    _,frame,cam=node
                    data=shot(moving,node)
                    geo=fixture._arrays(data)
                    row=dict(sensors=hashes(data),key=asdict(frame.key),time=frame.time_s,
                             rgb_sha256=digest(data[Channel.RGB]))
                    result['frames'][str(i)]=row
                    if args.baseline and any(row[k]!=original['frames'][str(i)][k] for k in ('sensors','key','time')):
                        raise ValueError("candidate geometry/time differs")
                    if i in POSES:
                        rgb8=rgb(data,cam)
                        Image.fromarray(rgb8).save(folder/f"pose-{i}.png")
                        masks=metrics.build_masks(**fixture._mask_inputs(geo,previous))
                        masks["reveal_right"]=masks["reveal"] & (geo["INSTANCE_ID"]==158)
                        if i==15 and int(masks["reveal_right"].sum())!=69:
                            raise ValueError("expected 69 registered right-finger pixels")
                        if args.baseline:
                            baseline_image=np.array(Image.open(args.baseline/f"pose-{i}.png"))
                            if digest(baseline_image.tobytes())!=original['frames'][str(i)]['rgb_sha256']:
                                raise ValueError("baseline RGB changed")
                            refs=[]
                            for seed in (1,2):
                                im=np.array(Image.open(args.baseline/f"seed-{seed}-pose-{i}.png"))
                                if digest(im.tobytes())!=original['references'][f"{seed}-{i}"]['rgb_sha256']:
                                    raise ValueError("reference RGB changed")
                                refs.append(im)
                            result['scores'][str(i)]={}
                            for label,mask in masks.items():
                                if not mask.any():
                                    continue
                                pairs=[dict(candidate=rmse(rgb8,r,mask),baseline=rmse(baseline_image,r,mask)) for r in refs]
                                result['scores'][str(i)][label]=dict(pixels=int(mask.sum()),per_seed=pairs,
                                    within_margin=all(x['candidate']<=x['baseline']+.005 for x in pairs),
                                    reference_disagreement=rmse(refs[0],refs[1],mask))
                        if i==15:
                            Image.fromarray(np.uint8(masks["reveal_right"])*255).save(folder/"exposed-right-finger.png")
                    previous=geo
                result['diagnostics']=moving.diagnostics()
            if args.baseline:
                with closing(producer.open_scene(scene)) as fresh:
                    data=shot(fresh,nodes[15]);row=result['frames']['15']
                    if hashes(data)!=row['sensors'] or digest(data[Channel.RGB])!=row['rgb_sha256']:
                        raise ValueError("moving versus fresh pose 15 differs")
                result['fresh_and_moving_rgb_identical_at_pose15']=True
            else:
                result['references']={}
                for seed in (1,2):
                    with closing(native(False,seed).open_scene(scene)) as reference:
                        for i in POSES:
                            for _ in range(REF_BATCHES):
                                data=shot(reference,nodes[i])
                            diag=reference.diagnostics()
                            if diag['progressive']['batches']!=REF_BATCHES or hashes(data)!=result['frames'][str(i)]['sensors']:
                                raise ValueError("reference did not match pose, sensors or sample count")
                            Image.fromarray(rgb(data,nodes[i][2])).save(folder/f"seed-{seed}-pose-{i}.png")
                            result['references'][f"{seed}-{i}"]=dict(rgb_sha256=digest(data[Channel.RGB]),spp=REF_BATCHES*16)
            result['status']='complete'
        except Exception as error:
            result['status']='failed'
            result['error']=str(error)
            raise
        finally:
            save(folder/'summary.json',result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    for field in ("library","shaders","adapter","baseline","prepared"):
        parser.add_argument("--"+field,type=Path)
    args=parser.parse_args()
    if not (all((args.baseline,args.prepared,args.adapter)) or all((args.library,args.shaders))):
        parser.error("provide a reference library and shaders, or a baseline, prepared package and adapter")
    experiment(args)
