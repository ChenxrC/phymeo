"""Photograph known UV samples beside an independent linear-color material oracle."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image

from _panda_scene import write_mesh
from phymeo_rendering.assets import Material, TextureMap, TexturedMaterial
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import DirectionalLight, FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene

TEXELS = np.array([[(206,40,23),(43,204,67)],[(30,71,213),(152,158,172)]], dtype=np.uint8)
COORDINATES = ((.25,.25),(.75,.25),(.25,.75),(.75,.75),(.5,.25),(-.25,1.25),(1.,.5),(-.08,-.17))
FACTORS = ((1.,1.,1.),(.48,.72,.31))
SIZE = (640,240)


def reference(uv):
    """Linearize first, then interpolate four periodic texels about their centres."""
    color = TEXELS.astype(np.float64)/255.
    color = np.where(color<=.04045,color/12.92,((color+.055)/1.055)**2.4)
    xy = 2.*np.array(uv)-.5
    ix,iy = np.floor(xy).astype(int)
    dx,dy = xy-np.floor(xy)
    row0 = (1-dx)*color[iy%2,ix%2]+dx*color[iy%2,(ix+1)%2]
    row1 = (1-dx)*color[(iy+1)%2,ix%2]+dx*color[(iy+1)%2,(ix+1)%2]
    return (1-dy)*row0+dy*row1


def plates(directory, role="color"):
    directory.mkdir()
    image = directory/"chips.png"
    Image.fromarray(TEXELS).save(image)
    texture = TextureMap(image.resolve().as_uri(),sha256(image.read_bytes()).hexdigest())
    if role == "roughness":
        from phymeo_rendering.assets import TextureChannel

        white = directory/"white-color.png"
        Image.fromarray(np.full_like(TEXELS,255)).save(white)
        white_color = TextureMap(white.resolve().as_uri(),sha256(white.read_bytes()).hexdigest())
    geometries, pairs, positions = [], [], []
    for row,factor in enumerate(FACTORS):
        for col,uv in enumerate(COORDINATES):
            ident = 1+len(geometries)
            cx,cy = 40+col*80,65+row*110
            verts = [((cx+x-320)/300.,(120-cy-y)/300.,0.) for x,y in
                     ((-36,45),(36,45),(36,-45),(-36,-45))]
            normal = (0,0,1)
            if role == "roughness":
                view = np.array([-(cx-320)/300.,(cy-120)/300.,1.])
                half = view/np.linalg.norm(view) + np.array([0.,0.,1.])
                half /= np.linalg.norm(half)
                tangent = np.cross(half,[0.,1.,0.])
                normal = np.cos(.16)*half + np.sin(.16)*tangent/np.linalg.norm(tangent)
            mesh = write_mesh(directory,f"plate-{ident}",verts,[normal]*4,[(0,1,2),(0,2,3)],
                              texcoords=[uv]*4)
            geometries.append(mesh)
            if role == "color":
                actual = TexturedMaterial(f"swatch-{ident}",factor,0.,.67,texture)
                expected = Material(actual.id,tuple(np.array(factor)*reference(uv)),0.,.67)
            else:
                factor_r = (.85,.58)[row]
                actual = TexturedMaterial(f"swatch-{ident}",(.4,.4,.4),.9,factor_r,
                                          white_color if row else None,TextureChannel(texture,"g"))
                # Independent interpolation of original GREEN bytes; no sRGB for data.
                xy = np.asarray(uv)*2.-.5
                dx,dy = xy-np.floor(xy)
                ix,iy = np.floor(xy).astype(int)
                g = TEXELS[:,:,1].astype(float)/255.
                linear_g = sum(weight*g[(iy+y)%2,(ix+x)%2] for x,y,weight in
                               [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)])
                expected = Material(actual.id,actual.base_color,actual.metallic,factor_r*linear_g)
            pairs.append((actual,expected))
            positions.append((cx,cy))
    nodes = tuple(Instance(i+1,g.id,p[0].id,"chart") for i,(g,p) in enumerate(zip(geometries,pairs)))
    lamp = DirectionalLight((0.,0.,-1.),(.8,.8,.8))
    prefix = "uv-chart-" if role == "color" else "roughness-chart-"
    scenes = [VisualScene(prefix+name,tuple(geometries),tuple(pair[idx] for pair in pairs),nodes,(lamp,))
              for idx,name in enumerate(("texture","analytic"))]
    return scenes,positions


def photograph(args):
    args.output.mkdir(parents=True,exist_ok=False)
    scenes,locations = plates(args.output/"geometry",args.role)
    pt = args.backend=="sapien-pt"
    path = RenderPath.PATH_TRACE if pt else RenderPath.RASTER
    channels = (Channel.RGB,) if pt else tuple(Channel)
    shots = []
    diagnostics = []
    for scene in scenes:
        backend = (NativeBackend(library_path=args.library,device_index=args.device_index,
                                 validation=True,resident=True) if args.backend=="native" else
                   SapienBackend(SapienConfig(path=path,device="cuda:0",samples_per_pixel=128)))
        session = backend.open_scene(scene)
        try:
            key = FrameKey(scene.key,"colour-check",0,0,0,"authored")
            frame = SceneFrame(key,0.,(NodePose("chart",Pose()),))
            cam = Camera("fixed",key,Pose((0.,0.,1.)),*SIZE,300.,300.,320.,120.,.01,3.)
            shot = session.capture(frame,CaptureRequest(cam,path,channels))
            planes = {p.channel:p.data for p in shot.planes}
            Image.frombytes("RGB",SIZE,planes[Channel.RGB]).save(args.output/(scene.key+".png"))
            shots.append(planes)
        finally:
            session.close()
        diagnostics.append(session.diagnostics())
    discrepancy = {c.value:shots[0][c]!=shots[1][c] for c in channels if c!=Channel.RGB}
    images = [np.frombuffer(p[Channel.RGB],np.uint8).reshape(SIZE[1],SIZE[0],3) for p in shots]
    cells = []
    for idx,(x,y) in enumerate(locations):
        actual,oracle = [i[y-3:y+4,x-3:x+4].astype(int) for i in images]
        material = scenes[0].materials[idx]
        cells.append(dict(uv=COORDINATES[idx%8],factor=material.roughness if args.role=="roughness" else material.base_color,
                          mean_rgb_actual=actual.mean(axis=(0,1)).tolist(),
                          mean_rgb_oracle=oracle.mean(axis=(0,1)).tolist(),
                          max_absolute_rgb8=int(np.abs(actual-oracle).max())))
    worst = max(c["max_absolute_rgb8"] for c in cells)
    # Minimum contrast protects the comparison against two equally blank images.
    span = float(np.ptp([r["mean_rgb_oracle"] for r in cells],axis=0).max())
    passed = not any(discrepancy.values()) and worst<=3 and span>=40.
    report = dict(backend=args.backend,role=args.role,passed=passed,cells=cells,max_absolute_rgb8=worst,
                  sample_count=len(cells)*49,oracle_color_span=span,sensor_differences=discrepancy,
                  limits=dict(max_rgb8=3,min_span=40),diagnostics=diagnostics)
    (args.output/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(f"{args.backend}: {len(cells)} swatches, worst RGB8 {worst}, span {span:.1f}, passed {passed}")
    if not passed:
        raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend",choices=("native","sapien-raster","sapien-pt"),required=True)
    p.add_argument("--role",choices=("color","roughness"),default="color")
    p.add_argument("--library",type=Path)
    p.add_argument("--device-index",type=int,default=1)
    p.add_argument("--output",type=Path,required=True)
    photograph(p.parse_args())
