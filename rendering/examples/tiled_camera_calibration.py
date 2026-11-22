"""Independent full-versus-tiled public pinhole geometry calibration.

Native: --backend native --library BUILD/phymeo_native.dll --device-index 1 --output NEW
Reference: --backend sapien --sapien-device cuda:0 --output NEW
Optional --tile-width 257 --tile-height 173 (647x419 is not divisible by either).

No physics, image resampling, PT noise acceptance or performance claim. Original
planes are saved unmodified. ID/VALID equality includes every silhouette pixel;
only the separate analytic depth check excludes a declared 2px geometric edge.
"""
from __future__ import annotations
import argparse
import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import zlib

from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import DirectionalLight, FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene


@dataclass(frozen=True)
class Surface:
    identity: int
    center: tuple
    half_size: tuple
    angle: float


def _f32(value): return struct.unpack('<f',struct.pack('<f',value))[0]
def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def _quad(path,half):
    x,y=half; points=((-x,-y,0.),(x,-y,0.),(x,y,0.),(-x,y,0.))
    raw=struct.pack('<12f',*(v for p in points for v in p))+struct.pack('<12f',*((0.,0.,1.)*4))+struct.pack('<6H',0,1,2,0,2,3)
    document={'asset':{'version':'2.0','generator':'authored tiled camera calibration'},
        'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'mesh':0}],
        'meshes':[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1},'indices':2,'mode':4}]}],
        'buffers':[{'byteLength':len(raw),'uri':'data:application/octet-stream;base64,'+base64.b64encode(raw).decode()}],
        'bufferViews':[{'buffer':0,'byteOffset':offset,'byteLength':size} for offset,size in ((0,48),(48,48),(96,12))],
        'accessors':[{'bufferView':0,'componentType':5126,'count':4,'type':'VEC3','min':[-x,-y,0.],'max':[x,y,0.]},
                     {'bufferView':1,'componentType':5126,'count':4,'type':'VEC3'},
                     {'bufferView':2,'componentType':5123,'count':6,'type':'SCALAR'}]}
    _json(path,document)


def build_fixture(directory):
    """Three finite noncoplanar surfaces; independent authored metre declarations."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    declarations=((0x01020304,(0.,0.,-3.2),(1.7,1.05),0.),
                  (0x00A1B2C3,(-.18,.08,-1.8),(.63,.45),0.),
                  (0xF1020304,(-.08,.03,-1.1),(.014,.50),-.42))
    surfaces=tuple(Surface(identity,center,tuple(_f32(v) for v in half),angle)
                   for identity,center,half,angle in declarations)
    meshes=[]; instances=[]; nodes=[]
    for i,surface in enumerate(surfaces):
        name=f'surface-{i}'; path=directory/(name+'.gltf'); _quad(path,surface.half_size)
        meshes.append(Mesh(name,path.resolve().as_uri()))
        instances.append(Instance(surface.identity,name,name,name))
        nodes.append(NodePose(name,Pose(surface.center,(0.,0.,math.sin(surface.angle/2),math.cos(surface.angle/2)))))
    materials=tuple(Material(f'surface-{i}',color,0.,.9) for i,color in
        enumerate(((.18,.22,.28),(.65,.12,.06),(.04,.55,.12))))
    scene=VisualScene('tiled-calibration-v1',tuple(meshes),materials,tuple(instances),
                      (DirectionalLight((-.3/math.hypot(.3,1.),0.,-1./math.hypot(.3,1.)),(2.,2.,2.)),))
    key=FrameKey(scene.key,'calibration',0,0,0,'authored'); frame=SceneFrame(key,0.,tuple(nodes))
    camera=Camera('off-center',key,Pose(),647,419,535.,521.,293.37,226.19,.1,5.)
    return scene,frame,camera,surfaces


def analytic_image(camera,surfaces):
    """Ray/finite rectangle equations; never uses decoded mesh or backend pixels."""
    import numpy as np
    if camera.world_pose!=Pose(): raise ValueError('calibration oracle requires its authored identity camera')
    yy,xx=np.mgrid[:camera.height,:camera.width]; px=xx+.5; py=yy+.5
    rx=(px-camera.cx_px)/camera.fx_px; ry=(camera.cy_px-py)/camera.fy_px
    depth=np.full(px.shape,np.inf); ids=np.zeros(px.shape,dtype='<u4'); edges=np.zeros(px.shape,dtype=bool)
    for surface in surfaces:
        cx,cy,cz=surface.center; d=-cz; c=math.cos(surface.angle); s=math.sin(surface.angle)
        dx=d*rx-cx; dy=d*ry-cy; localx=c*dx+s*dy; localy=-s*dx+c*dy
        hit=(abs(localx)<=surface.half_size[0])&(abs(localy)<=surface.half_size[1])&(d<depth)
        depth[hit]=d; ids[hit]=surface.identity
        corners=[]
        for ax,ay in ((-1,-1),(1,-1),(1,1),(-1,1)):
            u=ax*surface.half_size[0]; v=ay*surface.half_size[1]
            corners.append((camera.cx_px+camera.fx_px*(cx+c*u-s*v)/d,
                            camera.cy_px-camera.fy_px*(cy+s*u+c*v)/d))
        for a,b in zip(corners,corners[1:]+corners[:1]):
            vx,vy=b[0]-a[0],b[1]-a[1]
            t=np.clip(((px-a[0])*vx+(py-a[1])*vy)/(vx*vx+vy*vy),0,1)
            edges|=(px-a[0]-t*vx)**2+(py-a[1]-t*vy)**2<=4.
    edges[:2]=True; edges[-2:]=True; edges[:,:2]=True; edges[:,-2:]=True
    return {'depth':depth,'ids':ids,'edge':edges}


def _planes(capture):
    import numpy as np
    values={p.channel:p.data for p in capture.planes}; shape=(capture.camera.height,capture.camera.width)
    if set(values)!=set(Channel): raise ValueError('calibration requires all four public channels')
    return {Channel.RGB:np.frombuffer(values[Channel.RGB],dtype='u1').reshape(*shape,3),
        Channel.FORWARD_DEPTH:np.frombuffer(values[Channel.FORWARD_DEPTH],dtype='<f4').reshape(shape),
        Channel.INSTANCE_ID:np.frombuffer(values[Channel.INSTANCE_ID],dtype='<u4').reshape(shape),
        Channel.VALID:np.frombuffer(values[Channel.VALID],dtype='u1').reshape(shape)}


def audit_pair(full,tiled,frame,camera,surfaces,tile_width,tile_height):
    """Return honest measurements and mismatch rasters; no data-dependent exclusion."""
    import numpy as np
    for capture in (full,tiled):
        if capture.camera!=camera or capture.frame_key!=frame.key or capture.time_s!=frame.time_s:
            raise ValueError('capture does not bind the declared full exposure')
    a,b=_planes(full),_planes(tiled); reference=analytic_image(camera,surfaces)
    edges=reference['edge']; expected=reference['ids']; inside=(expected!=0)&~edges
    mismatch_id=a[Channel.INSTANCE_ID]!=b[Channel.INSTANCE_ID]
    mismatch_valid=a[Channel.VALID]!=b[Channel.VALID]
    common=(a[Channel.VALID]!=0)&(b[Channel.VALID]!=0)
    depth_delta=np.zeros(expected.shape); depth_delta[common]=abs(a[Channel.FORWARD_DEPTH][common].astype(float)-b[Channel.FORWARD_DEPTH][common])
    seams=np.zeros(expected.shape,dtype=bool); seam_rows=[]
    for axis,size,extent in (('x',tile_width,camera.width),('y',tile_height,camera.height)):
        for coordinate in range(size,extent,size):
            stripe=np.zeros(expected.shape,dtype=bool)
            if axis=='x': stripe[:,coordinate-1:coordinate+1]=True
            else: stripe[coordinate-1:coordinate+1,:]=True
            seams|=stripe
            seam_rows.append({'axis':axis,'coordinate':coordinate,
                'full_foreground':int(np.count_nonzero(stripe&(a[Channel.VALID]!=0))),
                'tiled_foreground':int(np.count_nonzero(stripe&(b[Channel.VALID]!=0))),
                'id_mismatches':int(np.count_nonzero(stripe&mismatch_id)),
                'valid_mismatches':int(np.count_nonzero(stripe&mismatch_valid))})
    analytic=[]
    for label,planes in (('full',a),('tiled',b)):
        errors=abs(planes[Channel.FORWARD_DEPTH][inside].astype(float)-reference['depth'][inside])
        background=(expected==0)&~edges
        analytic.append({'capture':label,'interior_pixels':int(np.count_nonzero(inside)),
            'background_false_hits':int(np.count_nonzero(background&(planes[Channel.VALID]!=0))),
            'surface_interior_counts':{str(s.identity):int(np.count_nonzero(inside&(expected==s.identity))) for s in surfaces},
            'interior_id_mismatches':int(np.count_nonzero(planes[Channel.INSTANCE_ID][inside]!=expected[inside])),
            'interior_max_depth_error_m':float(errors.max()) if np.isfinite(errors).all() else None,
            'edge_oracle_id_mismatches':int(np.count_nonzero(edges&(planes[Channel.INSTANCE_ID]!=expected)))})
    center=(camera.height//2,camera.width//2)
    rgb_error=abs(a[Channel.RGB].astype('i2')-b[Channel.RGB].astype('i2'))
    report={'id_mismatches_all':int(np.count_nonzero(mismatch_id)),
        'valid_mismatches_all':int(np.count_nonzero(mismatch_valid)),
        'id_mismatches_geometric_edge':int(np.count_nonzero(edges&mismatch_id)),
        'id_mismatches_away_from_edge':int(np.count_nonzero(~edges&mismatch_id)),
        'common_foreground_max_depth_difference_m':float(depth_delta[common].max()) if common.any() else None,
        'interior_max_depth_difference_m':float(depth_delta[inside&common].max()) if (inside&common).any() else None,
        'center_full_id':int(a[Channel.INSTANCE_ID][center]),'center_tiled_id':int(b[Channel.INSTANCE_ID][center]),
        'seams':seam_rows,'analytic':analytic,
        'rgb_max_abs_code_difference':int(rgb_error.max()),'rgb_differing_pixels':int(np.count_nonzero(rgb_error.any(axis=2))),
        'rgb_acceptance':'reported only; screen-space/shadow differences are not geometry equivalence',
        'edge_policy':'fixed <=2px to authored projected exterior edges plus outer 2px image border; ID/VALID pair comparison excludes nothing',
        'mismatch_xy_prefix':np.argwhere(mismatch_id|mismatch_valid)[:64,::-1].tolist()}
    report['passed']=(report['id_mismatches_all']==0 and report['valid_mismatches_all']==0
        and report['center_full_id']!=0 and report['center_tiled_id']!=0 and bool(seam_rows)
        and all(row['full_foreground']>0 and row['tiled_foreground']>0 for row in seam_rows)
        and report['interior_max_depth_difference_m'] is not None and report['interior_max_depth_difference_m']<=.002
        and all(row['interior_pixels']>0 and row['interior_id_mismatches']==0 and row['background_false_hits']==0
            and all(count>0 for count in row['surface_interior_counts'].values())
            and row['interior_max_depth_error_m'] is not None and row['interior_max_depth_error_m']<=.002 for row in analytic))
    return report,{'id_difference':mismatch_id,'valid_difference':mismatch_valid,'depth_difference':depth_delta>.002,'edge':edges,'seams':seams}


def _png(path,rgb):
    height,width,_=rgb.shape
    def chunk(kind,data): return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    raw=b''.join(b'\0'+rgb[row].tobytes() for row in range(height))
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
                          +chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))


def _save_capture(output,label,capture):
    import numpy as np
    evidence={}
    for plane in capture.planes:
        path=output/(label+'.'+plane.channel.value+'.raw'); path.write_bytes(plane.data)
        evidence[plane.channel.value]={'file':path.name,'sha256':_sha(path),'bytes':len(plane.data)}
    values=_planes(capture); _png(output/(label+'.rgb.png'),values[Channel.RGB])
    identities=values[Channel.INSTANCE_ID]; colors=np.zeros((*identities.shape,3),dtype='u1')
    for identity,color in ((0x01020304,(70,110,180)),(0x00A1B2C3,(220,90,50)),(0xF1020304,(70,220,100))): colors[identities==identity]=color
    _png(output/(label+'.ids.png'),colors)
    return evidence


def run_calibration(output,*,backend='native',library=None,shaders=None,device_index=1,sapien_device='cuda:0',tile_width=257,tile_height=173):
    import phymeo_rendering.tiled as tiled_module
    capture_tiled=tiled_module.capture_tiled
    if backend not in ('native','sapien'): raise ValueError('backend must be native or sapien')
    if type(tile_width) is not int or type(tile_height) is not int or min(tile_width,tile_height)<1: raise ValueError('positive integer tile dimensions required')
    if tile_width>=647 or tile_height>=419 or 647%tile_width==0 or 419%tile_height==0:
        raise ValueError('this calibration requires non-divisible subdivision in both axes')
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    scene,frame,camera,surfaces=build_fixture(output/'geometry')
    request=CaptureRequest(camera,RenderPath.RASTER,tuple(Channel))
    protocol={'schema':'phymeo-tiled-camera-protocol-v1','scene':asdict(scene),'frame':asdict(frame),'camera':asdict(camera),
        'surfaces':[asdict(s) for s in surfaces],'tile_width':tile_width,'tile_height':tile_height,
        'source_sha256':_sha(__file__),'tiled_module_sha256':_sha(tiled_module.__file__),'geometry_sha256':{p.name:_sha(p) for p in (output/'geometry').glob('*.gltf')},
        'max_interior_depth_error_m':.002,'id_valid_acceptance':'exact full-image equality including geometry and tile boundaries',
        'physics_steps':0,'scope':'raster geometry only; no PT sampling or throughput claim'}
    _json(output/'protocol.json',protocol)
    report={'schema':'phymeo-tiled-camera-calibration-v1','status':'running','backend':backend,
            'protocol_sha256':_sha(output/'protocol.json'),'captures':{},'physics_steps':0}
    owner=None
    try:
        if backend=='native':
            from phymeo_rendering.backends.native import NativeBackend
            owner=NativeBackend(library_path=library,shader_directory=shaders,device_index=device_index,validation=True,resident=True).open_scene(scene)
        else:
            from phymeo_rendering.backends.sapien import SapienBackend,SapienConfig
            owner=SapienBackend(SapienConfig(path=RenderPath.RASTER,device=sapien_device)).open_scene(scene)
        start=time.perf_counter(); full=owner.capture(frame,request)
        report['full_wall_s']=time.perf_counter()-start
        report['captures']['full']=_save_capture(output,'full',full); _json(output/'summary.json',report)
        start=time.perf_counter(); tiled=capture_tiled(owner,frame,request,tile_width,tile_height)
        report['tiled_wall_s']=time.perf_counter()-start
        report['captures']['tiled']=_save_capture(output,'tiled',tiled)
        audit,masks=audit_pair(full,tiled,frame,camera,surfaces,tile_width,tile_height); report['audit']=audit
        import numpy as np
        for name,mask in masks.items(): _png(output/(name+'.png'),np.repeat((mask.astype('u1')*255)[...,None],3,axis=2))
        report['diagnostics']=owner.diagnostics(); report['status']='passed' if audit['passed'] else 'geometry_mismatch'
        report['timing_scope']='single sequential full then tiles, no warmup; not a performance comparison'
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}'); raise
    finally:
        if owner is not None:
            try: owner.close(); report['closed_diagnostics']=owner.diagnostics()
            except BaseException as error:
                report['close_error']=f'{type(error).__name__}: {error}'; report['status']='failed'
        _json(output/'summary.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('native','sapien'),default='native')
    parser.add_argument('--library',type=Path); parser.add_argument('--shaders',type=Path)
    parser.add_argument('--device-index',type=int,default=1); parser.add_argument('--sapien-device',default='cuda:0')
    parser.add_argument('--tile-width',type=int,default=257); parser.add_argument('--tile-height',type=int,default=173)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv); result=run_calibration(**vars(args))
    print(json.dumps({'status':result['status'],'summary':str(args.output.resolve()/'summary.json')}))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__': raise SystemExit(main())
