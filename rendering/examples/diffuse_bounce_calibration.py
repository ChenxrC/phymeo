"""Finite colored-plate to diffuse receiver transport, independently integrated.

No renderer ray queries, RNG, shader code or fitted coefficients form the oracle.
Native RGB8 is inverted with explicit quantization intervals. SAPIEN's real HDR
readback is observed before encoding, with public seeded 0/1-bounce exposures.
The two material transport laws have separate references, not a fitted match.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import statistics
import struct
import sys
from time import perf_counter
import traceback

WIDTH,HEIGHT,FOCAL=160,120,128.
ROI=(64,44,96,76)
SEEDS=(0,1,2,3)
SENSORS=('FORWARD_DEPTH','INSTANCE_ID','VALID')
LUMA=(.2126,.7152,.0722)
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
RED=tuple(map(f32,(.8,.02,.02)))
GRAY=(f32(.185828),)*3
RECEIVER_ALBEDO=f32(.8)
HEIGHT_M=f32(.6)
PATCH=tuple(map(f32,(.4,1.2,-.4,.4)))
LIGHT_E=4.
CASES=('red','gray','blocked-red','blocked-gray')


def protocol():
    return dict(extent=[WIDTH,HEIGHT],focal_px=FOCAL,roi=list(ROI),seeds=list(SEEDS),
        native_samples_per_pixel=16,sapien_samples_per_pixel=512,sapien_max_bounces=[0,1],
        native_mean_relative_limit=.10,sapien_mean_relative_limit=.05,
        native_nrmse_limit=.60,sapien_nrmse_limit=.12,blocked_relative_limit=.03,
        quadrature_relative_limit=.0025,receiver_albedo=RECEIVER_ALBEDO,receiver_roughness=1.,
        plate_bounds=list(PATCH),plate_height=HEIGHT_M,plate_red=RED,plate_equal_linear_luminance_gray=GRAY,
        directional_toward_light=[1/math.sqrt(2),0.,-1/math.sqrt(2)],directional_rgb=[LIGHT_E]*3,
        environment='none',exposure=1.,display_transform='clip',msaa_samples=1,
        native_reference='rho_receiver=.8*(1-.04); finite-area cos*cos/r^2 integral of original terminal (1-F(V,H))*albedo*E*cosL/pi',
        sapien_reference='installed pure Lambert diffuse at both vertices; same finite-area integral; terminal white specular cancels in red-gray',
        color_observable='projection of paired red-gray linear RGB difference on their authored linear albedo difference; same-seed depth1-depth0',
        baffle='opaque black x=.3, y=+/- .6, z=.01.. .59; inactive is translated +3 along X',
        primary_roi='fixed interior receiver window; camera=(0,0,2), looks -Z; no silhouette pixels',
        native_readback='inverse IEC sRGB of RGB8, not raw radiance; half-code interval reported; any ROI code255 rejected',
        sapien_readback='actual HdrColor FP32 before public exposure/display encoder; no denoiser',
        uncertainty='four explicit shifted/native seeds; sample SD and SE descriptive only, never used to widen gates',
        scope='single diffuse transfer, no physics, no fit, no exposure/ROI/threshold tuning')


def roi_points():
    x0,y0,x1,y1=ROI
    return [((x+.5-WIDTH/2)*2/FOCAL,-(y+.5-HEIGHT/2)*2/FOCAL,0.) for y in range(y0,y1) for x in range(x0,x1)]


def rectangle_factor(point,bounds,height):
    x0,x1,y0,y1=bounds
    h=height-point[2]
    if not x0<x1 or not y0<y1 or not h>0 or not all(math.isfinite(v) for v in (*point,*bounds,height)):
        raise ValueError('finite ordered parallel rectangle above receiver required')
    def primitive(x,y):
        ax,ay=math.hypot(x,h),math.hypot(y,h)
        return x/ax*math.atan(y/ax)+y/ay*math.atan(x/ay)
    a,b,c,d=x0-point[0],x1-point[0],y0-point[1],y1-point[1]
    return (primitive(b,d)-primitive(a,d)-primitive(b,c)+primitive(a,c))/(2*math.pi)


def patch_integral(point,n,model):
    if type(n) is not int or n<1 or model not in ('native','sapien','form-factor'):
        raise ValueError('positive quadrature count and named reference required')
    x0,x1,y0,y1=PATCH
    area=(x1-x0)*(y1-y0)/(n*n)
    h=HEIGHT_M-point[2]
    total=0.
    for iy in range(n):
        dy=y0+(iy+.5)*(y1-y0)/n-point[1]
        for ix in range(n):
            dx=x0+(ix+.5)*(x1-x0)/n-point[0]
            r2=dx*dx+dy*dy+h*h
            weight=h*h/r2**2*area/math.pi
            if model=='native':
                # Terminal V points from the plate to this receiver; L=(+X,0,-Z)/sqrt(2).
                vh=math.sqrt(max(0.,(1+(-dx+h)/math.sqrt(2*r2))/2))
                fresnel=.04+.96*(1-vh)**5
                weight*=.96*(1-fresnel)
            total+=weight
    if model!='form-factor':
        total*=RECEIVER_ALBEDO*LIGHT_E/math.sqrt(2)/math.pi
    return total


def reference_maps(n):
    import numpy as np
    x0,x1,y0,y1=PATCH
    xx,yy=np.meshgrid(x0+(np.arange(n)+.5)*(x1-x0)/n,y0+(np.arange(n)+.5)*(y1-y0)/n)
    area=(x1-x0)*(y1-y0)/(n*n)
    result=[]
    for px,py,pz in roi_points():
        dx,dy,h=xx-px,yy-py,HEIGHT_M-pz
        r2=dx*dx+dy*dy+h*h
        vh=np.sqrt(np.maximum(0.,(1+(-dx+h)/np.sqrt(2*r2))/2))
        factor=np.sum(h*h/r2**2*area/math.pi*.96*(1-(.04+.96*(1-vh)**5)))
        result.append(float(factor*RECEIVER_ALBEDO*LIGHT_E/math.sqrt(2)/math.pi))
    exact=[RECEIVER_ALBEDO*LIGHT_E/math.sqrt(2)/math.pi*rectangle_factor(p,PATCH,HEIGHT_M) for p in roi_points()]
    return dict(native=np.array(result).reshape(32,32),sapien=np.array(exact).reshape(32,32))


def baffle_blocks(start,end):
    if end[0]==start[0]:return False
    t=(f32(.3)-start[0])/(end[0]-start[0])
    if not 0<t<1:return False
    y=start[1]+t*(end[1]-start[1]);z=start[2]+t*(end[2]-start[2])
    return -f32(.6)<=y<=f32(.6) and f32(.01)<=z<=f32(.59)


def inverse_srgb(value):
    if not 0<=value<=1:raise ValueError('unit sRGB code required')
    return value/12.92 if value<=.04045 else ((value+.055)/1.055)**2.4


def encode_srgb(value):
    if not 0<=value<=1:raise ValueError('unit display-linear value required')
    code=12.92*value if value<=.0031308 else 1.055*value**(1/2.4)-.055
    return round(255*code)


def decode_code(code):
    if type(code) is not int or not 0<=code<255:
        raise ValueError('unsaturated RGB8 code required')
    return inverse_srgb(code/255),inverse_srgb(max(0,code-.5)/255),inverse_srgb((code+.5)/255)


def numeric_gate(actual,reference,backend):
    if backend not in ('native','sapien') or not actual or len(actual)!=len(reference):
        raise ValueError('matched nonempty maps and named backend required')
    mean_ref=statistics.mean(reference)
    rms_ref=math.sqrt(statistics.mean(x*x for x in reference))
    if mean_ref<=0 or rms_ref<=0 or not all(math.isfinite(x) for x in (*actual,*reference)):
        raise ValueError('finite positive reference required')
    bias=(statistics.mean(actual)-mean_ref)/mean_ref
    mse=statistics.mean((a-r)**2 for a,r in zip(actual,reference))
    nrmse=math.sqrt(mse)/rms_ref
    p=protocol()
    return dict(passed=abs(bias)<=p[backend+'_mean_relative_limit'] and nrmse<=p[backend+'_nrmse_limit'],
        actual_mean=statistics.mean(actual),reference_mean=mean_ref,relative_mean_bias=bias,mse=mse,nrmse=nrmse,
        mean_relative_limit=p[backend+'_mean_relative_limit'],nrmse_limit=p[backend+'_nrmse_limit'])


def blocked_gate(actual,reference):
    if len(actual)!=len(reference) or not actual:raise ValueError('matched maps required')
    ratio=math.sqrt(statistics.mean(x*x for x in actual))/math.sqrt(statistics.mean(x*x for x in reference))
    return dict(passed=math.isfinite(ratio) and ratio<=.03,blocked_rms_over_unblocked_reference_rms=ratio,limit=.03)


def seed_summary(seeds,values):
    if len(seeds)!=4 or len(values)!=4 or len(set(seeds))!=4 or not all(type(s) is int and 0<=s<2**32 for s in seeds):
        raise ValueError('four distinct explicit uint32 seeds required')
    if not all(math.isfinite(v) for v in values):raise ValueError('finite seed observations required')
    sd=statistics.stdev(values)
    return dict(count=4,seeds=list(seeds),values=list(values),mean=statistics.mean(values),sample_sd=sd,
                standard_error=sd/2,note='descriptive variation of four declared seeded streams, not an error bound or relaxed gate')


def sensor_equal(a,b):return all(a[name]==b[name] for name in SENSORS)


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def build_scene(root,case):
    from _panda_scene import write_mesh
    from phymeo_rendering.assets import Material
    from phymeo_rendering.scene import DirectionalLight,FrameKey,Instance,NodePose,Pose,SceneFrame,VisualScene
    from phymeo_rendering.cameras import Camera
    root.mkdir(parents=True,exist_ok=True)
    receiver=write_mesh(root,'receiver',((-.35,-.35,0.),(.35,-.35,0.),(.35,.35,0.),(-.35,.35,0.)),((0.,0.,1.),)*4,((0,1,2),(0,2,3)))
    x0,x1,y0,y1=PATCH
    plate=write_mesh(root,'plate',((x0,y0,HEIGHT_M),(x1,y0,HEIGHT_M),(x1,y1,HEIGHT_M),(x0,y1,HEIGHT_M)),((0.,0.,-1.),)*4,((0,2,1),(0,3,2)))
    baffle=write_mesh(root,'baffle',((.3,-.6,.01),(.3,.6,.01),(.3,.6,.59),(.3,-.6,.59)),((-1.,0.,0.),)*4,((0,2,1),(0,3,2)))
    materials=(Material('receiver',(RECEIVER_ALBEDO,)*3,0.,1.),Material('plate',RED if case.endswith('red') else GRAY,0.,1.),Material('black',(0.,0.,0.),0.,1.))
    scene=VisualScene('diffuse-chart-'+case,(receiver,plate,baffle),materials,
        (Instance(1,'receiver','receiver','receiver'),Instance(2,'plate','plate','plate'),Instance(3,'baffle','black','baffle')),
        (DirectionalLight((-1/math.sqrt(2),0.,1/math.sqrt(2)),(LIGHT_E,)*3),))
    frame=SceneFrame(FrameKey(scene.key,'authored-chart',0,0,0,'authored'),0.,
        (NodePose('receiver',Pose()),NodePose('plate',Pose()),NodePose('baffle',Pose((0.,0.,0.) if case.startswith('blocked') else (3.,0.,0.)))))
    camera=Camera('diffuse-chart',frame.key,Pose((0.,0.,2.)),WIDTH,HEIGHT,FOCAL,FOCAL,WIDTH/2,HEIGHT/2,.01,10.)
    return scene,frame,camera


def _native_closed(diag):
    device=diag['device'];t=device.get('resident_telemetry')
    return bool(device.get('closed') and device.get('validation_errors')==0 and t and t.get('final_after_destroy') and
        all(device.get(k)==0 for k in ('live_contexts','live_frames','live_resources')) and
        t['device_local_bytes_current']==t['host_visible_bytes_current']==0 and t['allocation_count']==t['free_count'])


def _sapien_closed(diag):
    life=diag.get('lifecycle',{})
    after={k:v for k,v in life.items() if k.endswith('_after_clear')}
    return life.get('closed') is True and not life.get('close_error') and bool(after) and all(v==0 for v in after.values())


def run_experiment(library,output,*,shaders=None,device_index=1,validation=False,reference_device='cuda:0'):
    import numpy as np
    from PIL import Image,ImageDraw
    from unittest.mock import patch
    import sapien
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.backends.sapien import SapienBackend,SapienConfig
    from phymeo_rendering.backends import _sapien_session
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    output,library=Path(output),Path(library)
    output.mkdir(parents=True,exist_ok=False)
    started=perf_counter()
    report=dict(schema='phymeo-diffuse-bounce-chart-v1',status='running',physics_steps=0,registration=protocol(),
        library=dict(path=str(library.resolve()),sha256=sha256(library.read_bytes()).hexdigest()),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        public_source=dict(file=inspect.getfile(NativeBackend),sha256=sha256(Path(inspect.getfile(NativeBackend)).read_bytes()).hexdigest()),
        captures=[],cleanup=[],acceptance={k:False for k in ('quadrature_convergence','native_numeric','sapien_numeric',
        'native_blocked','sapien_blocked','direct_control','geometry_bytes','seed_evidence','capability','cleanup')})
    dump(output/'registration.json',report)
    maps={};bounds={};planes_by_case={};rows={};capability=True;direct_ok=True;sensors_ok=True
    x0,y0,x1,y1=ROI
    albedo_difference=np.asarray(RED)-np.asarray(GRAY)
    projection=albedo_difference/float(albedo_difference@albedo_difference)
    original_encoder=_sapien_session.encode_linear_image
    def observe_capture(owner,scene,frame,camera,path,label,backend,seed,depth,case):
        taps=[]
        def tap(values,**options):
            if values.dtype!=np.float32 or values.shape!=(HEIGHT,WIDTH,3):
                raise ValueError('actual FP32 HDR RGB readback required')
            taps.append(np.array(values,dtype='<f4',copy=True))
            return original_encoder(values,**options)
        before=perf_counter()
        if backend=='sapien':
            with patch.object(_sapien_session,'encode_linear_image',side_effect=tap):
                captured=owner.capture(frame,CaptureRequest(camera,path,(Channel.RGB,)))
        else:
            captured=owner.capture(frame,CaptureRequest(camera,path,tuple(Channel)))
        wall=perf_counter()-before
        if captured.frame_key!=frame.key or captured.camera!=camera or captured.time_s!=frame.time_s:
            raise ValueError('source/camera/time changed at public capture boundary')
        planes={p.channel.name:p.data for p in captured.planes}
        rgb=np.frombuffer(planes['RGB'],np.uint8).reshape(HEIGHT,WIDTH,3)
        image=output/(label+'.png');Image.fromarray(rgb).save(image)
        (output/(label+'.rgb8.raw')).write_bytes(planes['RGB'])
        item=dict(label=label,backend=backend,seed=seed,depth=depth,case=case,wall_ms=wall*1000,
            scene=asdict(scene),frame=asdict(frame),camera=asdict(camera),image=image.name,
            png_sha256=sha256(image.read_bytes()).hexdigest(),plane_sha256={k:sha256(v).hexdigest() for k,v in planes.items()},
            diagnostics=owner.diagnostics())
        report['captures'].append(item);rows[(backend,seed,depth,case)]=item
        if backend=='native':
            ids=np.frombuffer(planes['INSTANCE_ID'],'<u4').reshape(HEIGHT,WIDTH)
            if not np.all(ids[y0:y1,x0:x1]==1):raise ValueError('predeclared ROI is not entirely receiver instance 1')
            codes=rgb[y0:y1,x0:x1]
            item['roi_saturated_pixels']=int(np.any(codes==255,axis=2).sum())
            if item['roi_saturated_pixels']:raise ValueError('cannot infer radiance from saturated ROI RGB8')
            lut=np.asarray([decode_code(i) for i in range(255)])
            decoded=lut[codes]
            values=decoded[...,0]
            bounds[(backend,seed,depth,case)]=(decoded[...,1],decoded[...,2])
            item['linear_observation']='inverse sRGB display-linear estimate, not HDR readback'
            item['roi_linear_quantization_halfwidth_max']=float(np.max((decoded[...,2]-decoded[...,1])/2))
            for channel in SENSORS:(output/(label+'.'+channel+'.raw')).write_bytes(planes[channel])
        else:
            if len(taps)!=1 or not np.isfinite(taps[0]).all() or np.any(taps[0]<0):
                raise ValueError('one finite nonnegative real HDR readback required')
            full=taps[0]
            raw=full.tobytes();(output/(label+'.hdr-f32.raw')).write_bytes(raw)
            item.update(hdr_sha256=sha256(raw).hexdigest(),hdr_file=label+'.hdr-f32.raw',hdr_shape=list(full.shape),
                        linear_observation='real HdrColor FP32 before display/exposure')
            values=full[y0:y1,x0:x1].astype(np.float64)
            bounds[(backend,seed,depth,case)]=(values,values)
        maps[(backend,seed,depth,case)]=values
        item['roi_mean_linear_rgb']=np.mean(values,axis=(0,1)).tolist()
        dump(output/'summary.json',report)
        print(json.dumps(dict(capture=label,wall_ms=wall*1000,roi_mean=item['roi_mean_linear_rgb'])),flush=True)
        return planes
    try:
        report['sapien_version']=sapien.__version__
        report['sapien_default_specular']=float(sapien.render.RenderMaterial().specular)
        if sapien.__version__!='3.0.3' or report['sapien_default_specular']!=0.:
            raise ValueError('installed reference material default no longer matches the predeclared Lambert color-difference model')
        low,reference=reference_maps(32),reference_maps(64)
        convergence=float(np.max(np.abs(low['native']-reference['native'])/reference['native']))
        report['quadrature']=dict(coarse=32,fine=64,native_max_relative_change=convergence,
            native_mean=float(reference['native'].mean()),sapien_mean=float(reference['sapien'].mean()),
            expected_backend_mean_ratio=float(reference['native'].mean()/reference['sapien'].mean()),
            baffle_all_roi_corner_segments_blocked=all(baffle_blocks(p,(x,y,HEIGHT_M)) for p in roi_points()
                for x in (PATCH[0],PATCH[1]) for y in (PATCH[2],PATCH[3])))
        report['acceptance']['quadrature_convergence']=convergence<=.0025 and report['quadrature']['baffle_all_roi_corner_segments_blocked']
        for backend,value in reference.items():
            path=output/(backend+'.reference-coefficient-f64.raw');path.write_bytes(value.astype('<f8').tobytes())
        for case in CASES:
            scene,frame,camera=build_scene(output/'geometry',case)
            for seed,depth in [(0,0)]+[(s,1) for s in SEEDS]:
                mode='none' if depth==0 else 'diffuse_bounce'
                label=f'native-{case}-seed{seed}-depth{depth}'
                owner=NativeBackend(library_path=library,shader_directory=shaders,device_index=device_index,
                    resident=True,shadow_mode='ray_query',indirect_mode=mode,indirect_seed=seed,
                    validation=validation,msaa_samples=1).open_scene(scene)
                try:
                    planes=observe_capture(owner,scene,frame,camera,RenderPath.RASTER,label,'native',seed,depth,case)
                    if case not in planes_by_case:planes_by_case[case]=planes
                    sensors_ok &= sensor_equal(planes_by_case[case],planes)
                    if depth:
                        diag=owner.diagnostics();info=diag['device'].get('resident_indirect_info',{})
                        capability &= diag.get('indirect',{}).get('mode')=='diffuse_bounce' and info.get('samples_per_pixel')==16 and info.get('seed')==seed and info.get('enabled')==1
                finally:
                    try:owner.close()
                    finally:
                        diag=owner.diagnostics();report['cleanup'].append(dict(label=label,passed=_native_closed(diag),diagnostics=diag))
                        dump(output/'summary.json',report)
        # Same visible receiver geometry: changing the emitter's reflectance must not change sensors.
        sensors_ok &= sensor_equal(planes_by_case['red'],planes_by_case['gray']) and sensor_equal(planes_by_case['blocked-red'],planes_by_case['blocked-gray'])
        for case in CASES:
            scene,frame,camera=build_scene(output/'geometry',case)
            for seed in SEEDS:
                for depth in (0,1):
                    label=f'sapien-{case}-seed{seed}-depth{depth}'
                    owner=SapienBackend(SapienConfig(path=RenderPath.PATH_TRACE,device=reference_device,
                        samples_per_pixel=512,max_bounces=depth,sampling_mode='shifted-halton',seed=seed,
                        denoiser='none',exposure=1.,display_transform='clip')).open_scene(scene)
                    try:
                        observe_capture(owner,scene,frame,camera,RenderPath.PATH_TRACE,label,'sapien',seed,depth,case)
                        diag=owner.diagnostics()
                        capability &= diag.get('sampling_pattern',{}).get('mode')=='shifted-halton' and diag.get('seed')==seed and diag.get('max_bounces')==depth
                    finally:
                        try:owner.close()
                        finally:
                            diag=owner.diagnostics();report['cleanup'].append(dict(label=label,passed=_sapien_closed(diag),diagnostics=diag))
                            dump(output/'summary.json',report)
        def delta(backend,seed,blocked):
            prefix='blocked-' if blocked else ''
            zero_seed=0 if backend=='native' else seed
            return (maps[(backend,seed,1,prefix+'red')]-maps[(backend,zero_seed,0,prefix+'red')]
                    -maps[(backend,seed,1,prefix+'gray')]+maps[(backend,zero_seed,0,prefix+'gray')])
        results={};coefficient_maps={}
        for backend in ('native','sapien'):
            per_seed=[np.sum(delta(backend,s,False)*projection,axis=-1) for s in SEEDS]
            blocked=[np.sum(delta(backend,s,True)*projection,axis=-1) for s in SEEDS]
            average=np.mean(per_seed,axis=0);average_blocked=np.mean(blocked,axis=0)
            coefficient_maps[backend]=average
            gate=numeric_gate(average.ravel().tolist(),reference[backend].ravel().tolist(),backend)
            block=blocked_gate(average_blocked.ravel().tolist(),reference[backend].ravel().tolist())
            result=dict(numeric=gate,blocked=block,seeds=seed_summary(SEEDS,[float(v.mean()) for v in per_seed]),
                per_seed_numeric=[numeric_gate(v.ravel().tolist(),reference[backend].ravel().tolist(),backend) for v in per_seed],
                per_seed_blocked_mean=[float(v.mean()) for v in blocked])
            # Per-seed tests are observations, not gates on four times fewer samples.
            result['per_seed_scope']='reported without reclassifying the frozen four-seed aggregate gate'
            if backend=='native':
                lows=[];highs=[]
                for seed in SEEDS:
                    lo=np.zeros((32,32,3));hi=np.zeros_like(lo)
                    for sign,key in ((1,('native',seed,1,'red')),(-1,('native',0,0,'red')),
                                     (-1,('native',seed,1,'gray')),(1,('native',0,0,'gray'))):
                        a,b=bounds[key];lo+=sign*(a if sign>0 else b);hi+=sign*(b if sign>0 else a)
                    lows.append(float(np.mean(np.sum(np.where(projection>=0,lo,hi)*projection,axis=-1))))
                    highs.append(float(np.mean(np.sum(np.where(projection>=0,hi,lo)*projection,axis=-1))))
                result['quantization_interval_for_aggregate_mean']=[statistics.mean(lows),statistics.mean(highs)]
                result['quantization_note']='rigorous half-code bounds of each inverse-sRGB sample; numerical gate still uses unchanged point estimate limits'
            results[backend]=result
            for name,array in (('mean',average),('blocked',average_blocked)):
                (output/(backend+'.'+name+'-coefficient-f64.raw')).write_bytes(array.astype('<f8').tobytes())
            report['acceptance'][backend+'_numeric']=gate['passed']
            report['acceptance'][backend+'_blocked']=block['passed']
            for seed in (SEEDS if backend=='sapien' else (0,)):
                for prefix in ('','blocked-'):
                    a=maps[(backend,seed,0,prefix+'red')];b=maps[(backend,seed,0,prefix+'gray')]
                    direct_ok &= bool(np.array_equal(a,b))
        native_unique=len({rows[('native',s,1,'red')]['plane_sha256']['RGB'] for s in SEEDS})
        pt_unique=len({rows[('sapien',s,1,'red')]['hdr_sha256'] for s in SEEDS})
        report['results']=results
        report['seed_observations']=dict(native_unique_red_frames=native_unique,sapien_unique_red_hdr_frames=pt_unique,
            note='explicit different seeds and shader diagnostics bind streams; identical official fixed-seed replays are not counted as independent')
        report['backend_comparison']=dict(measured_native_over_sapien=float(coefficient_maps['native'].mean()/coefficient_maps['sapien'].mean()),
            predicted_native_over_sapien=float(reference['native'].mean()/reference['sapien'].mean()),
            note='separate material transport definitions; do not interpret this expected model bias as sampling error')
        report['acceptance'].update(direct_control=direct_ok,geometry_bytes=sensors_ok,seed_evidence=native_unique==4 and pt_unique==4,
            capability=capability,cleanup=all(x['passed'] for x in report['cleanup']))
        # Fixed display scale, no per-image normalization or denoising. This is a data view, not new rendering.
        sheet=Image.new('RGB',(WIDTH*2,(HEIGHT+20)*4),(24,25,28));draw=ImageDraw.Draw(sheet)
        for row,case in enumerate(CASES):
            for col,backend in enumerate(('native','sapien')):
                label=f'{backend}-{case}-seed0-depth1'
                draw.text((col*WIDTH+3,row*(HEIGHT+20)+3),label,fill=(255,255,255))
                with Image.open(output/(label+'.png')) as image:sheet.paste(image,(col*WIDTH,row*(HEIGHT+20)+20))
        sheet.save(output/'comparison.png')
        report['status']='passed' if all(report['acceptance'].values()) else 'measured-not-passed'
    except BaseException as error:
        report['status']='failed';report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_wall_s']=perf_counter()-started
        report['geometry_sha256']={p.name:sha256(p.read_bytes()).hexdigest() for p in (output/'geometry').glob('*.gltf')} if (output/'geometry').exists() else {}
        report['imported_sources']={name:dict(file=m.__file__,sha256=sha256(Path(m.__file__).read_bytes()).hexdigest())
            for name,m in tuple(sys.modules.items()) if name.startswith('phymeo_rendering') and getattr(m,'__file__',None) and Path(m.__file__).is_file()}
        dump(output/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True);parser.add_argument('--shaders',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device-index',type=int,default=1)
    parser.add_argument('--reference-device',default='cuda:0');parser.add_argument('--validation',action='store_true')
    args=parser.parse_args()
    report=run_experiment(args.library,args.output,shaders=args.shaders,device_index=args.device_index,
                          validation=args.validation,reference_device=args.reference_device)
    print(json.dumps(dict(status=report['status'],acceptance=report['acceptance'])))
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
