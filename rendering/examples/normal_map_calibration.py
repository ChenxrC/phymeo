"""Independent tangent-normal chart for the public Vulkan rendering session.

Three finite opaque panels use planar, rotated and mirrored-U geometry. The
reference derives its TBN from authored FP32 positions/UVs, intersects camera
rays with the actual planes, and evaluates the closed roughness=1 dielectric
GGX expression. It does not import native packing, tangent or shader helpers.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import struct
import sys
from time import perf_counter
import traceback

NORMALS={'flat':(128,128,255),'direction':(204,166,230),'positive-y':(128,204,230)}
LIGHTS={'plus':(.6,.3,math.sqrt(.55)),'minus':(-.6,-.3,math.sqrt(.55))}
SENSOR_NAMES=('FORWARD_DEPTH','INSTANCE_ID','VALID')
WIDTH,HEIGHT=480,240
FOCAL=600.


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def unit(vector):
    length=math.hypot(*vector)
    if not math.isfinite(length) or length<=1e-15:
        raise ValueError('finite nonzero direction required')
    return tuple(value/length for value in vector)


def decode_normal(rgb):
    if len(rgb)!=3 or any(type(value) is not int or not 0<=value<=255 for value in rgb):
        raise ValueError('three linear RGB8 normal components required')
    return unit(tuple(2*value/255-1 for value in rgb))


def tangent_frame(positions,uv,normal):
    if len(positions)<3 or len(uv)!=len(positions):
        raise ValueError('triangle positions and matching UV required')
    a,b=sub(positions[1],positions[0]),sub(positions[2],positions[0])
    u1,v1=sub(uv[1],uv[0]);u2,v2=sub(uv[2],uv[0])
    determinant=u1*v2-v1*u2
    if not math.isfinite(determinant) or abs(determinant)<=1e-12:
        raise ValueError('nondegenerate UV required')
    du=tuple((v2*x-v1*y)/determinant for x,y in zip(a,b))
    dv=tuple((-u2*x+u1*y)/determinant for x,y in zip(a,b))
    n=unit(normal)
    tangent=unit(tuple(value-dot(n,du)*ni for value,ni in zip(du,n)))
    sign=dot(cross(n,tangent),dv)
    if not math.isfinite(sign) or abs(sign)<=1e-12:
        raise ValueError('nondegenerate geometric tangent frame required')
    handedness=1 if sign>0 else -1
    bitangent=tuple(handedness*v for v in cross(n,tangent))
    return tangent,bitangent,n,handedness


def panel_definitions():
    result={}
    f32=lambda value:struct.unpack('<f',struct.pack('<f',value))[0]
    for identity,(name,x,degrees,mirror) in enumerate((('flat',-.48,0.,False),('tilted',0.,25.,False),('mirror',.48,0.,True)),1):
        angle=math.radians(degrees);t=(math.cos(angle),0.,-math.sin(angle));b=(0.,1.,0.)
        n=(math.sin(angle),0.,math.cos(angle));centre=(x,0.,0.)
        positions=tuple(tuple(f32(centre[i]+u*t[i]+v*b[i]) for i in range(3))
                        for u,v in ((-.16,-.24),(.16,-.24),(.16,.24),(-.16,.24)))
        uv=((1.,0.),(0.,0.),(0.,1.),(1.,1.)) if mirror else ((0.,0.),(1.,0.),(1.,1.),(0.,1.))
        cx=round(WIDTH/2+FOCAL*x/2)
        result[name]={'id':identity,'positions':positions,'normal':tuple(map(f32,n)),'uv':uv,
                      'centre':tuple(sum(p[i] for p in positions)/4 for i in range(3)),
                      'roi':(cx-12,HEIGHT//2-12,cx+12,HEIGHT//2+12),'geometry_rotation_y_degrees':degrees}
    return result


def mapped_normal(panel,rgb,*,omit_handedness=False,flip_green=False):
    t,b,n,h=tangent_frame(panel['positions'],panel['uv'],panel['normal'])
    if omit_handedness:
        b=cross(n,t)
    x,y,z=decode_normal(rgb)
    if flip_green:
        y=-y
    return unit(tuple(x*t[i]+y*b[i]+z*n[i] for i in range(3)))


def radiance(normal,view,light):
    """Unit-strength light, fixed intensity 2, albedo .28, metal 0, roughness 1."""
    n,v,l=unit(normal),unit(view),unit(light)
    nv,nl=dot(n,v),dot(n,l)
    if nv<=0 or nl<=0:
        return 0.
    h=unit(tuple(a+b for a,b in zip(v,l)))
    fresnel=.04+.96*(1-max(0.,dot(v,h)))**5
    gv,gl=2*nv/(nv+1),2*nl/(nl+1)
    return 2*((1-fresnel)*.28*nl/math.pi+fresnel*gv*gl/(4*math.pi*nv))


def _encode(value):
    if not 0<=value<1:
        raise ValueError('predeclared chart reference must not clip')
    return round(255*(12.92*value if value<=.0031308 else 1.055*value**(1/2.4)-.055))


def _point(panel,x,y):
    eye=(0.,0.,2.)
    ray=((x+.5-WIDTH/2)/FOCAL,-(y+.5-HEIGHT/2)/FOCAL,-1.)
    a,b,c=panel['positions'][:3]
    geometric=unit(cross(sub(b,a),sub(c,a)))
    parameter=dot(geometric,sub(a,eye))/dot(geometric,ray)
    point=tuple(o+parameter*d for o,d in zip(eye,ray))
    # The fixed ROI must stay inside the actual finite panel, not just its plane.
    horizontal=unit(sub(b,a));vertical=unit(sub(panel['positions'][3],a))
    delta=sub(point,a)
    inside=(0<dot(delta,horizontal)<math.dist(a,b) and 0<dot(delta,vertical)<math.dist(a,panel['positions'][3]))
    return point,inside


def roi_prediction(panel,mode,light,*,omit_handedness=False,flip_green=False):
    n=unit(panel['normal']) if mode=='none' else mapped_normal(panel,NORMALS[mode],
        omit_handedness=omit_handedness,flip_green=flip_green)
    x0,y0,x1,y1=panel['roi'];values=[];encoded=[];all_inside=True
    for y in range(y0,y1):
        for x in range(x0,x1):
            p,inside=_point(panel,x,y);all_inside=all_inside and inside
            value=radiance(n,sub((0.,0.,2.),p),LIGHTS[light])
            values.append(value);encoded.extend((_encode(value),)*3)
    return {'rgb8':bytes(encoded),'linear':tuple(values),'mean_rgb8':sum(encoded)/len(encoded),
            'all_inside':all_inside,'world_normal':n}


def direction_gate(expected_plus,expected_minus,actual_plus,actual_minus):
    values=(expected_plus,expected_minus,actual_plus,actual_minus)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('finite response measurements required')
    expected=expected_plus-expected_minus;actual=actual_plus-actual_minus
    return {'predicted_signed_rgb8_difference':expected,'actual_signed_rgb8_difference':actual,
            'predicted_margin_rgb8':abs(expected),'contrast_error_rgb8':abs(expected-actual),
            'passed':abs(expected)>=8 and expected*actual>0 and abs(expected-actual)<=4}


def flat_gate(unmapped,mapped):
    if not isinstance(unmapped,bytes) or not isinstance(mapped,bytes) or len(unmapped)!=len(mapped) or not unmapped:
        raise ValueError('same-sized nonempty RGB8 byte regions required')
    errors=[abs(a-b) for a,b in zip(unmapped,mapped)]
    return {'max_rgb8_error':max(errors),'mean_absolute_rgb8_error':sum(errors)/len(errors),'passed':max(errors)<=1}


def sensor_equal(a,b):
    if any(name not in a or name not in b for name in SENSOR_NAMES):
        raise ValueError('all three geometry channels required')
    return all(a[name]==b[name] for name in SENSOR_NAMES)


def _dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _closed(diagnostics):
    d=diagnostics['device'];t=d.get('resident_telemetry')
    return bool(d['closed'] and d['validation_errors']==0 and all(d[key]==0 for key in (
        'live_contexts','live_frames','live_resources')) and (t is None or (
        t['final_after_destroy'] and t['device_local_bytes_current']==0 and t['host_visible_bytes_current']==0
        and t['allocation_count']==t['free_count'])))


def run_experiment(library,output,*,shaders=None,device_index=1,validation=True):
    import numpy as np
    from PIL import Image,ImageDraw
    from _panda_scene import write_mesh
    from phymeo_rendering.assets import Material,TexturedMaterial,TextureMap
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.cameras import Camera
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    from phymeo_rendering.scene import DirectionalLight,FrameKey,Instance,NodePose,Pose,SceneFrame,VisualScene
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    library=Path(library);started=perf_counter()
    report={'schema':'phymeo-native-normal-chart-v1','status':'running','physics_steps':0,
        'registration':{'extent':[WIDTH,HEIGHT],'panel_rois':'fixed 24x24 integer squares before rendering',
            'linear_decode':'normalize(2*linear_RGB8/255-1), including the true 128 quantization bias',
            'flat_max_rgb8':1,'analytic_max_rgb8':2,'predicted_min_direction_margin_rgb8':8,
            'direction_contrast_tolerance_rgb8':4,'albedo':.28,'metallic':0.,'roughness':1.,
            'light_intensity':2.,'exposure':1.,'display_transform':'clip','msaa_samples':1,
            'reference':'independent authored triangle/UV tangent frame, actual ray-plane point, closed rough GGX',
            'normal_strength':1.,'normal_convention':'+Y; do not flip green or sample as sRGB'},
        'library':{'path':str(library.resolve()),'sha256':sha256(library.read_bytes()).hexdigest()},
        'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'public_source':{'file':inspect.getfile(NativeBackend),'sha256':sha256(Path(inspect.getfile(NativeBackend)).read_bytes()).hexdigest()},
        'panels':panel_definitions(),'normal_rgb8':NORMALS,'lights_toward_source':LIGHTS,
        'captures':{},'cleanup':[], 'acceptance':{key:False for key in (
            'flat_within_one_rgb8','analytic_within_two_rgb8','direction_and_margin','positive_y_handedness',
            'geometry_bytes_unchanged','request_combinations','cleanup')}}
    geometries=output/'geometry';geometries.mkdir()
    meshes=[]
    for name,panel in report['panels'].items():
        meshes.append(write_mesh(geometries,name,panel['positions'],(panel['normal'],)*4,((0,1,2),(0,2,3)),texcoords=panel['uv']))
    report['geometry_sha256']={path.name:sha256(path.read_bytes()).hexdigest() for path in geometries.glob('*.gltf')}
    normal_maps={}
    for name,rgb in NORMALS.items():
        path=output/(name+'.png');Image.new('RGB',(2,2),rgb).save(path)
        normal_maps[name]=TextureMap(path.resolve().as_uri(),sha256(path.read_bytes()).hexdigest())
    report['normal_map_sources']={name:asdict(value) for name,value in normal_maps.items()}
    readings={};combination_ok=True
    try:
        for mode in ('none','flat','direction','positive-y'):
            material=Material('chart',(.28,.28,.28),0.,1.) if mode=='none' else TexturedMaterial(
                'chart',(.28,.28,.28),0.,1.,normal_map=normal_maps[mode])
            for light,direction in LIGHTS.items():
                label=mode+'-'+light
                scene=VisualScene('normal-chart-'+label,tuple(meshes),(material,),tuple(
                    Instance(panel['id'],name,material.id,name) for name,panel in report['panels'].items()),
                    (DirectionalLight(tuple(-v for v in direction),(2.,2.,2.)),))
                frame=SceneFrame(FrameKey(scene.key,'authored-chart',0,0,0,'authored'),0.,tuple(
                    NodePose(name,Pose()) for name in report['panels']))
                camera=Camera('chart',frame.key,Pose((0.,0.,2.)),WIDTH,HEIGHT,FOCAL,FOCAL,WIDTH/2,HEIGHT/2,.01,10.)
                backend=NativeBackend(library_path=library,shader_directory=shaders,device_index=device_index,
                                      resident=True,validation=validation,msaa_samples=1)
                owner=backend.open_scene(scene)
                try:
                    requests={};times={}
                    for request_name,channels in (('rgb',(Channel.RGB,)),('mixed',tuple(Channel)),
                        ('sensors',(Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID))):
                        before=perf_counter();capture=owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,channels))
                        times[request_name]=1000*(perf_counter()-before)
                        if capture.frame_key!=frame.key or capture.time_s!=0 or capture.camera!=camera:
                            raise ValueError('capture source/camera/time mismatch')
                        requests[request_name]={p.channel.name:p.data for p in capture.planes}
                    mixed=requests['mixed'];readings[label]=mixed
                    agree=requests['rgb']['RGB']==mixed['RGB'] and sensor_equal(requests['sensors'],mixed)
                    combination_ok=combination_ok and agree
                    raw=np.frombuffer(mixed['RGB'],np.uint8).reshape(HEIGHT,WIDTH,3)
                    image=output/(label+'.png');Image.fromarray(raw).save(image)
                    ids=np.frombuffer(mixed['INSTANCE_ID'],'<u4').reshape(HEIGHT,WIDTH)
                    entry={'camera':asdict(camera),'frame':asdict(frame),'material':asdict(material),'scene':asdict(scene),
                        'wall_ms':times,'request_agreement':agree,'image':image.name,'png_sha256':sha256(image.read_bytes()).hexdigest(),
                        'plane_sha256':{name:sha256(data).hexdigest() for name,data in mixed.items()},'panels':{}}
                    for channel in SENSOR_NAMES:
                        (output/(label+'.'+channel+'.raw')).write_bytes(mixed[channel])
                    for panel_name,panel in report['panels'].items():
                        prediction=roi_prediction(panel,mode,light)
                        x0,y0,x1,y1=panel['roi'];actual=raw[y0:y1,x0:x1].tobytes()
                        if not prediction['all_inside'] or not (ids[y0:y1,x0:x1]==panel['id']).all():
                            raise ValueError('predeclared ROI left finite panel or lost the expected instance')
                        errors=np.abs(np.frombuffer(actual,np.uint8).astype(np.int16)-np.frombuffer(prediction['rgb8'],np.uint8).astype(np.int16))
                        entry['panels'][panel_name]={'roi':panel['roi'],'actual_mean_rgb8':sum(actual)/len(actual),
                            'predicted_mean_rgb8':prediction['mean_rgb8'],'analytic_max_rgb8_error':int(errors.max()),
                            'analytic_mae_rgb8':float(errors.mean()),'predicted_world_normal':prediction['world_normal'],
                            'predicted_linear_range':[min(prediction['linear']),max(prediction['linear'])],
                            'predicted_rgb8_sha256':sha256(prediction['rgb8']).hexdigest()}
                        (output/(label+'.'+panel_name+'.predicted.rgb8')).write_bytes(prediction['rgb8'])
                    report['captures'][label]=entry
                finally:
                    try:
                        owner.close()
                    finally:
                        diagnostics=owner.diagnostics()
                        report['cleanup'].append({'label':label,'passed':_closed(diagnostics),'diagnostics':diagnostics})
                        _dump(output/'summary.json',report)
                print(json.dumps({'case':label,'max_analytic_error':max(p['analytic_max_rgb8_error'] for p in entry['panels'].values())}),flush=True)
        flat_results={};directions={};handedness={}
        for light in LIGHTS:
            before=np.frombuffer(readings['none-'+light]['RGB'],np.uint8).reshape(HEIGHT,WIDTH,3)
            after=np.frombuffer(readings['flat-'+light]['RGB'],np.uint8).reshape(HEIGHT,WIDTH,3)
            for name,panel in report['panels'].items():
                x0,y0,x1,y1=panel['roi']
                flat_results[name+'-'+light]=flat_gate(before[y0:y1,x0:x1].tobytes(),after[y0:y1,x0:x1].tobytes())
        for mode in ('direction','positive-y'):
            for name,panel in report['panels'].items():
                plus=report['captures'][mode+'-plus']['panels'][name];minus=report['captures'][mode+'-minus']['panels'][name]
                directions[mode+'-'+name]=direction_gate(plus['predicted_mean_rgb8'],minus['predicted_mean_rgb8'],plus['actual_mean_rgb8'],minus['actual_mean_rgb8'])
        panel=report['panels']['mirror']
        actual_plus=report['captures']['positive-y-plus']['panels']['mirror']['actual_mean_rgb8']
        actual_minus=report['captures']['positive-y-minus']['panels']['mirror']['actual_mean_rgb8']
        for name,options in (('omit_handedness',{'omit_handedness':True}),('flip_green',{'flip_green':True})):
            wrong_plus=roi_prediction(panel,'positive-y','plus',**options)['mean_rgb8']
            wrong_minus=roi_prediction(panel,'positive-y','minus',**options)['mean_rgb8']
            handedness[name]=direction_gate(wrong_plus,wrong_minus,actual_plus,actual_minus)
        sensor_checks={label:sensor_equal(readings['none-plus'],planes) for label,planes in readings.items()}
        report.update(flat_results=flat_results,direction_results=directions,wrong_convention_negative_controls=handedness,sensor_checks=sensor_checks)
        report['acceptance'].update(flat_within_one_rgb8=all(v['passed'] for v in flat_results.values()),
            analytic_within_two_rgb8=all(p['analytic_max_rgb8_error']<=2 for cap in report['captures'].values() for p in cap['panels'].values()),
            direction_and_margin=all(v['passed'] for v in directions.values()),
            positive_y_handedness=directions['positive-y-mirror']['passed'] and all(not v['passed'] for v in handedness.values()),
            geometry_bytes_unchanged=all(sensor_checks.values()),request_combinations=combination_ok,
            cleanup=all(v['passed'] for v in report['cleanup']))
        sheet=Image.new('RGB',(WIDTH*2,(HEIGHT+20)*4),(22,24,28));draw=ImageDraw.Draw(sheet)
        for row,mode in enumerate(('none','flat','direction','positive-y')):
            for column,light in enumerate(LIGHTS):
                label=mode+'-'+light;draw.text((column*WIDTH+5,row*(HEIGHT+20)+3),label,fill=(255,255,255))
                with Image.open(output/(label+'.png')) as image:
                    sheet.paste(image,(column*WIDTH,row*(HEIGHT+20)+20))
        sheet.save(output/'comparison.png')
        report['status']='passed' if all(report['acceptance'].values()) else 'measured-not-passed'
    except BaseException as error:
        report['status']='failed';report['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
        raise
    finally:
        report['elapsed_wall_s']=perf_counter()-started
        report['imported_sources']={name:{'file':module.__file__,'sha256':sha256(Path(module.__file__).read_bytes()).hexdigest()}
            for name,module in tuple(sys.modules.items()) if name.startswith('phymeo_rendering') and getattr(module,'__file__',None) and Path(module.__file__).is_file()}
        _dump(output/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True);parser.add_argument('--shaders',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device-index',type=int,default=1)
    parser.add_argument('--validation',action='store_true');args=parser.parse_args()
    report=run_experiment(args.library,args.output,shaders=args.shaders,device_index=args.device_index,validation=args.validation)
    print(json.dumps({'status':report['status'],'acceptance':report['acceptance']}))
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
