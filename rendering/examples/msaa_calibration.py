"""Independent pixel-area and motion calibration for public native 1x/4x MSAA.

The reference clips convex projected polygons against unit pixel squares. It
never uses Vulkan sample locations, rendered IDs, or shader coverage to choose
its edge region. Camera/vertices are exactly representable in FP32 for these
bounded fixtures. No physical simulation or image reconstruction is involved.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path
import struct
import traceback


def _extent(width,height):
    if type(width) is not int or type(height) is not int or not 0<width<=1024 or not 0<height<=1024:
        raise ValueError('positive integer extent up to 1024 required')


def _cross(a,b,p):
    return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])


def _signed(poly):
    return .5*sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1]))


def _polygon(poly):
    points=tuple(tuple(p) for p in poly)
    if not points:
        return points
    if len(points)<3 or any(len(p)!=2 for p in points):
        raise ValueError('empty or convex polygon with at least three xy vertices required')
    if any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) for p in points for v in p):
        raise ValueError('finite polygon coordinates required')
    if len(set(points))!=len(points) or _signed(points)==0:
        raise ValueError('nondegenerate polygon required')
    signs=[_cross(points[i-1],points[i],points[(i+1)%len(points)]) for i in range(len(points))]
    if min(signs)<0<max(signs):
        raise ValueError('convex non-self-intersecting polygon required')
    oriented=points if _signed(points)>0 else points[::-1]
    if any(_cross(a,b,p)<-1e-12 for a,b in zip(oriented,oriented[1:]+oriented[:1]) for p in oriented):
        raise ValueError('convex non-self-intersecting polygon required')
    return oriented


def _clip(subject,clipper):
    result=tuple(subject)
    for a,b in zip(clipper,clipper[1:]+clipper[:1]):
        if not result:
            break
        output=[]
        previous=result[-1]
        dp=_cross(a,b,previous)
        for current in result:
            dc=_cross(a,b,current)
            if (dc>=0)!=(dp>=0):
                t=dp/(dp-dc)
                output.append((previous[0]+t*(current[0]-previous[0]),
                               previous[1]+t*(current[1]-previous[1])))
            if dc>=0:
                output.append(current)
            previous,dp=current,dc
        result=tuple(output)
    return result


def _coverage(poly,width,height):
    result=[[0.]*width for _ in range(height)]
    if not poly:
        return tuple(map(tuple,result))
    for y in range(max(0,math.floor(min(p[1] for p in poly))),min(height,math.ceil(max(p[1] for p in poly)))):
        for x in range(max(0,math.floor(min(p[0] for p in poly))),min(width,math.ceil(max(p[0] for p in poly)))):
            clipped=_clip(poly,((x,y),(x+1,y),(x+1,y+1),(x,y+1)))
            area=abs(_signed(clipped)) if len(clipped)>=3 else 0.
            if not -1e-10<=area<=1+1e-10:
                raise ArithmeticError('pixel polygon area outside unit square')
            result[y][x]=min(1.,max(0.,area))
    return tuple(map(tuple,result))


def coverage(poly,width,height):
    _extent(width,height)
    return _coverage(_polygon(poly),width,height)


def visible_coverage(back,front,width,height):
    _extent(width,height)
    a,b=_polygon(back),_polygon(front)
    outer=_coverage(a,width,height)
    hidden=_coverage(_clip(a,b),width,height) if a and b else _coverage((),width,height)
    return tuple(tuple(max(0.,x-y) for x,y in zip(ar,br)) for ar,br in zip(outer,hidden))


def linear8(value):
    if not isinstance(value,Real) or not math.isfinite(value) or not 0<=value<=255:
        raise ValueError('finite sRGB8 value required')
    encoded=value/255.
    return encoded/12.92 if encoded<=.04045 else ((encoded+.055)/1.055)**2.4


def srgb8(value):
    if not isinstance(value,Real) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('finite unit display-linear value required')
    encoded=12.92*value if value<=.0031308 else 1.055*value**(1/2.4)-.055
    return round(255*encoded)


def rgb_coverage(rgb,width,height,*,black,white):
    _extent(width,height)
    if not isinstance(rgb,bytes) or len(rgb)!=width*height*3 or len(black)!=3 or len(white)!=3:
        raise ValueError('packed RGB8 and measured three-channel references required')
    lo=tuple(map(linear8,black)); hi=tuple(map(linear8,white))
    contrast=tuple(b-a for a,b in zip(lo,hi))
    if min(contrast)<=1e-6:
        raise ValueError('positive measured contrast in every channel required')
    # Deliberately do not clamp overshoot: it is an error, not ideal coverage.
    values=[sum((linear8(rgb[p+c])-lo[c])/contrast[c] for c in range(3))/3
            for p in range(0,len(rgb),3)]
    return tuple(tuple(values[y*width:(y+1)*width]) for y in range(height))


def _matrix(values):
    result=tuple(tuple(row) for row in values)
    if not result or not result[0] or any(len(row)!=len(result[0]) for row in result):
        raise ValueError('nonempty rectangular matrix required')
    if any(not isinstance(v,Real) or not math.isfinite(v) for row in result for v in row):
        raise ValueError('finite matrix required')
    return result


def _shape(values):
    return len(values),len(values[0])


def edge_mask(references):
    matrices=tuple(_matrix(ref) for ref in references)
    if not matrices or any(_shape(m)!=_shape(matrices[0]) for m in matrices):
        raise ValueError('aligned nonempty references required')
    h,w=_shape(matrices[0])
    return tuple(tuple(any(1e-12<m[y][x]<1-1e-12 for m in matrices) for x in range(w)) for y in range(h))


def spatial_error(actual,reference,mask):
    a,r,m=_matrix(actual),_matrix(reference),_matrix(mask)
    if _shape(a)!=_shape(r) or _shape(a)!=_shape(m):
        raise ValueError('aligned actual/reference/geometric mask required')
    errors=[abs(x-y) for ar,rr,mr in zip(a,r,m) for x,y,keep in zip(ar,rr,mr) if keep]
    if not errors:
        raise ValueError('empty geometric measurement region')
    return {'mae':sum(errors)/len(errors),'max_error':max(errors),'pixels':len(errors)}


def temporal_error(actual,reference,mask):
    a,r=tuple(map(_matrix,actual)),tuple(map(_matrix,reference))
    if len(a)!=len(r) or len(a)<2:
        raise ValueError('at least two aligned motion frames required')
    if any(_shape(m)!=_shape(a[0]) for m in (*a,*r)):
        raise ValueError('motion frame shapes differ')
    errors=[]
    for i in range(1,len(a)):
        delta=tuple(tuple(b-x for x,b in zip(p,n)) for p,n in zip(a[i-1],a[i]))
        ideal=tuple(tuple(b-x for x,b in zip(p,n)) for p,n in zip(r[i-1],r[i]))
        errors.append(spatial_error(delta,ideal,mask)['mae'])
    return {'mae':sum(errors)/len(errors),'transitions':len(errors),'per_transition_mae':errors}


def improvement(baseline,actual,required):
    if any(not isinstance(v,Real) or not math.isfinite(v) or v<0 for v in (baseline,actual,required)) or required>1:
        raise ValueError('finite nonnegative errors and unit fractional threshold required')
    fraction=1-actual/baseline if baseline else None
    return {'baseline':baseline,'actual':actual,'required_reduction':required,'reduction':fraction,
            'passed':fraction is not None and fraction>=required}


def centre_truth(layers,width,height):
    _extent(width,height)
    prepared=[]
    for poly,depth,identity in layers:
        if not math.isfinite(depth) or depth<=0 or type(identity) is not int or identity<=0:
            raise ValueError('positive plane depth and stable ID required')
        prepared.append((_polygon(poly),depth,identity))
    ids=[]; depths=[]; boundary=[]
    for y in range(height):
        row_ids=[]; row_depths=[]
        for x in range(width):
            point=(x+.5,y+.5); nearest=float('inf'); found=0
            for polygon,depth,identity in prepared:
                distances=[_cross(a,b,point) for a,b in zip(polygon,polygon[1:]+polygon[:1])]
                if distances and min(distances)>=-1e-10:
                    if min(abs(d) for d in distances)<=1e-10:
                        boundary.append((x,y))
                    if depth<nearest:
                        nearest,found=depth,identity
            row_ids.append(found); row_depths.append(nearest)
        ids.append(tuple(row_ids)); depths.append(tuple(row_depths))
    return {'ids':tuple(ids),'depth':tuple(depths),
            'valid':tuple(tuple(int(value!=0) for value in row) for row in ids),
            'boundary_pixels':tuple(sorted(set(boundary)))}


def _dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _fixture(directory):
    from _panda_scene import write_mesh
    from phymeo_rendering.assets import Material
    from phymeo_rendering.scene import DirectionalLight,Instance,VisualScene
    polygons={
        'edge': ((-8.,-8.),(42.125,-8.),(91.375,104.),(-8.,104.)),
        'rod': ((40.125,8.125),(40.75,8.125),(57.75,88.125),(57.125,88.125)),
        'back': ((12.125,10.25),(116.25,10.25),(116.25,86.125),(12.125,86.125)),
        'front': ((44.125,6.25),(87.375,16.25),(78.875,92.25),(35.625,82.25)),
        'solid': ((-8.,-8.),(136.,-8.),(136.,104.),(-8.,104.)),
    }
    for name,fraction in (('quarter',.25),('half',.5),('three-quarter',.75)):
        polygons[name]=((40.,16.),(40.+fraction,16.),(40.+fraction,80.),(40.,80.))
    meshes=[]; instances=[]; declarations={}
    for identity,(name,poly) in enumerate(polygons.items(),1):
        depth=.75 if name=='front' else 1.
        # Write and independently retain precisely the FP32 source vertex values.
        f32=lambda value: struct.unpack('<f',struct.pack('<f',value))[0]
        vertices=tuple((f32((x-64)*depth/128),f32((48-y)*depth/128),f32(1-depth)) for x,y in poly)
        mesh=write_mesh(directory,name,vertices,((0.,0.,1.),)*4,((0,2,1),(0,3,2)))
        meshes.append(mesh)
        instances.append(Instance(identity,mesh.id,'black' if name=='front' else 'white',name))
        declarations[name]={'id':identity,'vertices':vertices,'depth':depth,'white':name!='front'}
    scene=VisualScene('authored-msaa-calibration-v1',tuple(meshes),
        (Material('white',(1.,1.,1.),0.,1.),Material('black',(0.,0.,0.),1.,1.)),
        tuple(instances),(DirectionalLight((0.,0.,-1.),(16.,16.,16.)),))
    return scene,declarations


def _cases():
    return [('clear',(),0.),('solid',('solid',),0.),('edge',('edge',),0.)]+[
        (f'rod-{i:02d}',('rod',),i/16) for i in range(16)]+[
        ('occlusion',('back','front'),0.),('quarter',('quarter',),0.),
        ('half',('half',),0.),('three-quarter',('three-quarter',),0.)]


def _exposure(scene,declarations,case,index,*,width=128,height=96,epoch=0):
    from phymeo_rendering.cameras import Camera
    from phymeo_rendering.scene import FrameKey,NodePose,Pose,SceneFrame
    name,active,shift=case
    nodes=tuple(NodePose(label,Pose((shift/128 if label=='rod' else 0.,0.,0.) if label in active
                                  else (1024.,0.,0.))) for label in declarations)
    frame=SceneFrame(FrameKey(scene.key,'authored-calibration',epoch,index,index,'authored'),index/60,nodes)
    camera=Camera('coverage',frame.key,Pose((0.,0.,1.)),width,height,128.,128.,width/2,height/2,.01,10.)
    layers=[]
    for label in active:
        entry=declarations[label]
        polygon=tuple((width/2+128*(x+(shift/128 if label=='rod' else 0.))/(1-z),
                       height/2-128*y/(1-z)) for x,y,z in entry['vertices'])
        layers.append((polygon,entry['depth'],entry['id']))
    reference=visible_coverage(layers[0][0],layers[1][0],width,height) if len(layers)==2 else coverage(
        layers[0][0] if layers else (),width,height)
    return frame,camera,layers,reference


def _sensor_check(planes,layers,width,height):
    from phymeo_rendering.outputs import Channel
    truth=centre_truth(layers,width,height)
    ids=struct.unpack('<'+'I'*(width*height),planes[Channel.INSTANCE_ID])
    depths=struct.unpack('<'+'f'*(width*height),planes[Channel.FORWARD_DEPTH])
    valid=planes[Channel.VALID]
    boundary=set(truth['boundary_pixels']); bad=[]
    for y in range(height):
        for x in range(width):
            if (x,y) in boundary:
                continue
            i=y*width+x; expected=truth['depth'][y][x]
            depth_ok=(math.isinf(depths[i]) if math.isinf(expected) else abs(depths[i]-expected)<=1e-6)
            if ids[i]!=truth['ids'][y][x] or valid[i]!=truth['valid'][y][x] or not depth_ok:
                bad.append((x,y))
    return {'passed':not bad,'tested_pixels':width*height-len(boundary),
            'boundary_pixels_excluded':sorted(boundary),'mismatch_count':len(bad),'first_mismatches':bad[:16]}


def _cleanup(diagnostics):
    device=diagnostics['device']
    okay=device['closed'] and device['validation_errors']==0 and all(
        device[name]==0 for name in ('live_contexts','live_frames','live_resources'))
    telemetry=device.get('resident_telemetry')
    if telemetry is not None:
        okay=okay and telemetry['final_after_destroy'] and telemetry['device_local_bytes_current']==0 and (
            telemetry['host_visible_bytes_current']==0 and telemetry['allocation_count']==telemetry['free_count'])
    return bool(okay)


def run_experiment(library,output,*,shaders=None,device_index=1,validation=True):
    """Run public sessions only. Return real failures as false gates, never retune them."""
    import inspect
    import sys
    from time import perf_counter
    import numpy as np
    from PIL import Image
    import phymeo_rendering
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath

    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    geometry=output/'geometry'; geometry.mkdir()
    library=Path(library)
    report={'schema':'phymeo-native-msaa-calibration-v1','status':'running','physics_steps':0,
        'registration':{'extent':[128,96],'edge_required_reduction':.25,'motion_required_reduction':.20,
            'rod_phases_px':[i/16 for i in range(16)],'motion_transitions':15,
            'edge_region':'fractional ideal polygon coverage, fixed before rendering',
            'temporal_region':'union of fractional ideal coverage over all 16 translated rod states',
            'metric_domain':'inverse IEC sRGB, normalized by measured stable internal foreground/background colors',
            'linear_resolve_rgb8_tolerance':2,'sensor_depth_tolerance_m':1e-6,
            'reset_scope':'authored epoch change and close/reopen owner; no physics reset or simulation'},
        'library':{'path':str(library.resolve()),'sha256':sha256(library.read_bytes()).hexdigest()},
        'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'public_backend_source':{'file':inspect.getfile(NativeBackend),
                                'sha256':sha256(Path(inspect.getfile(NativeBackend)).read_bytes()).hexdigest()},
        'package_file':phymeo_rendering.__file__,'modes':{},'cleanup':[],
        'acceptance':{name:False for name in ('edge_mae_25_percent','motion_error_20_percent',
                      'linear_resolve','centre_sensors','request_combinations','resize_and_reset','cleanup')}}
    started=perf_counter()
    scene,declarations=_fixture(geometry)
    report['scene']=asdict(scene); report['authored_vertices']=declarations
    report['mesh_sha256']={path.name:sha256(path.read_bytes()).hexdigest() for path in geometry.glob('*.gltf')}
    _dump(output/'summary.json',report)
    raw_modes={}; references={}
    sensor_channels=(Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID)
    request_sets=(('rgb',(Channel.RGB,)),('mixed',tuple(Channel)),('sensors',sensor_channels))

    def take(owner,frame,camera):
        result={}
        metadata={}
        for label,channels in request_sets:
            before=perf_counter()
            capture=owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,channels))
            elapsed=perf_counter()-before
            if capture.frame_key!=frame.key or capture.camera!=camera or capture.time_s!=frame.time_s:
                raise ValueError('public capture changed the submitted frame/camera/time')
            result[label]={plane.channel:plane.data for plane in capture.planes}
            metadata[label]={'wall_ms':1000*elapsed,'sha256':{
                channel.value:sha256(data).hexdigest() for channel,data in result[label].items()}}
        return result,metadata

    def finish(owner,label):
        try:
            owner.close()
        finally:
            diagnostics=owner.diagnostics()
            report['cleanup'].append({'owner':label,'passed':_cleanup(diagnostics),'diagnostics':diagnostics})
        return diagnostics

    try:
        for samples in (1,4):
            mode=str(samples); mode_dir=output/('samples-'+mode); mode_dir.mkdir()
            backend=NativeBackend(library_path=library,shader_directory=shaders,device_index=device_index,
                                  resident=True,validation=validation,msaa_samples=samples)
            owner=backend.open_scene(scene)
            entries={}; readings={}; same_requests=True; sensor_ok=True
            report['modes'][mode]={'records':entries}
            try:
                for index,case in enumerate(_cases()):
                    name=case[0]
                    frame,camera,layers,reference=_exposure(scene,declarations,case,index)
                    # Geometry oracle and ROI are computed before this capture.
                    references[name]=reference
                    observed,metadata=take(owner,frame,camera)
                    mixed=observed['mixed']
                    agree=observed['rgb'][Channel.RGB]==mixed[Channel.RGB] and all(
                        observed['sensors'][channel]==mixed[channel] for channel in sensor_channels)
                    same_requests=same_requests and agree
                    checked=_sensor_check(mixed,layers,camera.width,camera.height)
                    sensor_ok=sensor_ok and checked['passed']
                    image=mode_dir/(name+'.png')
                    Image.frombytes('RGB',(camera.width,camera.height),mixed[Channel.RGB]).save(image)
                    for channel in sensor_channels:
                        (mode_dir/(name+'.'+channel.name+'.raw')).write_bytes(mixed[channel])
                    entries[name]={'frame':asdict(frame),'camera':asdict(camera),'projected_layers':layers,
                        'request_receipts':metadata,'request_agreement':agree,'centre_truth':checked,
                        'png':str(image.relative_to(output)),'png_sha256':sha256(image.read_bytes()).hexdigest()}
                    readings[name]=mixed
                # Change targets, restore the original extent, then change epoch.
                reset_agreement=True
                for label,width,height,epoch,index in (
                    ('resized',160,112,0,500),('restored_extent',128,96,0,501),('new_epoch',128,96,1,0)):
                    frame,camera,layers,reference=_exposure(scene,declarations,('edge',('edge',),0.),index,
                                                           width=width,height=height,epoch=epoch)
                    observed,metadata=take(owner,frame,camera)
                    mixed=observed['mixed']; readings[label]=mixed
                    checked=_sensor_check(mixed,layers,width,height)
                    sensor_ok=sensor_ok and checked['passed']
                    agree=observed['rgb'][Channel.RGB]==mixed[Channel.RGB] and all(
                        observed['sensors'][channel]==mixed[channel] for channel in sensor_channels)
                    same_requests=same_requests and agree
                    if label!='resized':
                        reset_agreement=reset_agreement and mixed==readings['edge']
                    entries[label]={'frame':asdict(frame),'camera':asdict(camera),'projected_layers':layers,
                                    'request_receipts':metadata,'request_agreement':agree,'centre_truth':checked}
                # Uniform measured controls must actually be spatially stable.
                black=np.frombuffer(readings['clear'][Channel.RGB],np.uint8).reshape(96,128,3)[8:-8,8:-8]
                white=np.frombuffer(readings['solid'][Channel.RGB],np.uint8).reshape(96,128,3)[8:-8,8:-8]
                if np.ptp(black.reshape(-1,3),axis=0).max()>1 or np.ptp(white.reshape(-1,3),axis=0).max()>1:
                    raise ValueError('foreground/background reference is not a stable internal color')
                black_color=tuple(map(float,black.mean(axis=(0,1))))
                white_color=tuple(map(float,white.mean(axis=(0,1))))
                actual={name:rgb_coverage(planes[Channel.RGB],128,96,black=black_color,white=white_color)
                        for name,planes in readings.items() if name not in ('resized','restored_extent','new_epoch')}
                edge_region=edge_mask((references['edge'],))
                rod_names=[f'rod-{i:02d}' for i in range(16)]
                motion_reference=tuple(references[name] for name in rod_names)
                motion_region=edge_mask(motion_reference)
                edge_metric=spatial_error(actual['edge'],references['edge'],edge_region)
                temporal_metric=temporal_error(tuple(actual[name] for name in rod_names),motion_reference,motion_region)
                resolve={}
                for name,fraction in (('quarter',.25),('half',.5),('three-quarter',.75)):
                    rgb=np.frombuffer(readings[name][Channel.RGB],np.uint8).reshape(96,128,3)[20:76,40,:]
                    expected=tuple(srgb8((1-fraction)*linear8(b)+fraction*linear8(w)) for b,w in zip(black_color,white_color))
                    delta=np.abs(rgb.astype(np.int16)-np.asarray(expected,dtype=np.int16))
                    resolve[name]={'ideal_coverage':fraction,'expected_srgb8':expected,
                        'observed_unique_srgb8':np.unique(rgb,axis=0).tolist(),
                        'max_rgb8_error':int(delta.max()),'passed':bool(delta.max()<=2)}
                report['modes'][mode].update(measured_black_srgb8=black_color,measured_white_srgb8=white_color,
                    edge=edge_metric,motion=temporal_metric,resolve=resolve,request_combinations=same_requests,
                    centre_sensors=sensor_ok,resize_and_epoch=reset_agreement,
                    geometric_motion_region_pixels=sum(sum(row) for row in motion_region))
                raw_modes[mode]=readings
                if samples==4:
                    reference_image=bytes(srgb8(value) for row in references['edge'] for value in row for _ in range(3))
                    Image.frombytes('RGB',(128,96),reference_image).save(output/'edge-ideal-area.png')
                    a=np.frombuffer(raw_modes['1']['edge'][Channel.RGB],np.uint8).astype(np.int16)
                    b=np.frombuffer(readings['edge'][Channel.RGB],np.uint8).astype(np.int16)
                    Image.fromarray(np.minimum(255,4*np.abs(a-b)).astype(np.uint8).reshape(96,128,3)).save(
                        output/'edge-difference-rgb8-times4.png')
            finally:
                diagnostics=finish(owner,'samples-'+mode)
                report['modes'][mode]['diagnostics_after_close']=diagnostics
                _dump(output/'summary.json',report)
            # Real destruction/recreation; same source geometry must remain reproducible.
            recreated=backend.open_scene(scene)
            try:
                frame,camera,_,_=_exposure(scene,declarations,('edge',('edge',),0.),0,epoch=2)
                observed,metadata=take(recreated,frame,camera)
                report['modes'][mode]['recreated_owner']={'equal':observed['mixed']==readings['edge'],
                                                         'request_receipts':metadata,'frame':asdict(frame)}
            finally:
                finish(recreated,'recreated-'+mode)
            print(json.dumps({'mode':samples,'edge_mae':edge_metric['mae'],'motion_mae':temporal_metric['mae'],
                              'centre_sensors':sensor_ok,'requests_agree':same_requests}),flush=True)

        one,four=report['modes']['1'],report['modes']['4']
        report['edge_improvement']=improvement(one['edge']['mae'],four['edge']['mae'],.25)
        report['motion_improvement']=improvement(one['motion']['mae'],four['motion']['mae'],.20)
        cross={name:all(raw_modes['1'][name][channel]==raw_modes['4'][name][channel]
                       for channel in sensor_channels) for name in raw_modes['1']}
        report['all_pixels_cross_mode_sensor_bytes']=cross
        diagnostic_contract=True
        for mode in ('1','4'):
            msaa=report['modes'][mode]['diagnostics_after_close'].get('msaa',{})
            diagnostic_contract=diagnostic_contract and msaa.get('requested_samples')==int(mode) and (
                msaa.get('actual_color_samples')==int(mode) and msaa.get('sensor_samples')==1 and
                msaa.get('sensor_sampling')=='deterministic pixel center')
            if mode=='4':
                diagnostic_contract=diagnostic_contract and msaa.get('resolve_space')=='display-linear then IEC sRGB once'
        report['diagnostic_contract']=bool(diagnostic_contract)
        report['acceptance'].update(
            edge_mae_25_percent=report['edge_improvement']['passed'],
            motion_error_20_percent=report['motion_improvement']['passed'],
            linear_resolve=all(value['passed'] for value in four['resolve'].values()),
            centre_sensors=bool(one['centre_sensors'] and four['centre_sensors'] and all(cross.values())),
            request_combinations=bool(one['request_combinations'] and four['request_combinations']),
            resize_and_reset=bool(one['resize_and_epoch'] and four['resize_and_epoch'] and
                                  one['recreated_owner']['equal'] and four['recreated_owner']['equal']),
            cleanup=all(record['passed'] for record in report['cleanup']))
        report['status']='passed' if all(report['acceptance'].values()) and diagnostic_contract else 'measured-not-passed'
        _dump(output/'ideal-coverage.json',references)
    except BaseException as error:
        report['status']='failed'
        report['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
        raise
    finally:
        report['elapsed_wall_s']=perf_counter()-started
        report['imported_sources']={name:{'file':module.__file__,'sha256':sha256(Path(module.__file__).read_bytes()).hexdigest()}
            for name,module in tuple(sys.modules.items()) if name.startswith('phymeo_rendering') and
            getattr(module,'__file__',None) and Path(module.__file__).is_file()}
        _dump(output/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--shaders',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device-index',type=int,default=1)
    parser.add_argument('--validation',action='store_true')
    args=parser.parse_args()
    result=run_experiment(args.library,args.output,shaders=args.shaders,device_index=args.device_index,validation=args.validation)
    print(json.dumps({'status':result['status'],'acceptance':result['acceptance']}))
    if result['status']!='passed':
        raise SystemExit(1)


if __name__=='__main__':
    main()

