"""True six-wall room: separate bounce-order bias and equal-budget seed variance.

The SAPIEN reference is an explicitly independent SDK scene, NOT the public PT
adapter (whose all-vertices-in-front admission excludes an interior camera).
The SDK's shipped seeded shader derivative is used, with actual HDR/segmentation
readback. Native measurements use its public resident progressive session and
owned, scene-linear RGBA32F mean readback. No physics, exposure fit or denoiser.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import struct
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import traceback

WIDTH=HEIGHT=96
FOCAL=75.
NEAR,FAR=.01,10.
EYE=(0.,-.8,1.)
ROIS={'bright':(14,44,34,64),'shadow':(72,44,84,64),'red':(22,71,32,79)}
SENSORS=('FORWARD_DEPTH','INSTANCE_ID','VALID')
LUMA=(.2126,.7152,.0722)


def dot(a,b):return sum(x*y for x,y in zip(a,b))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def unit(a):
    d=math.hypot(*a)
    return tuple(x/d for x in a)


def geometry():
    values=[
        ('floor',(-1.,-1.,0.),(2.,0.,0.),(0.,2.,0.),(.45,.45,.45)),
        ('ceiling',(-1.,1.,2.),(2.,0.,0.),(0.,-2.,0.),(.55,.55,.55)),
        ('left',(-1.,-1.,0.),(0.,2.,0.),(0.,0.,2.),(.50,.55,.60)),
        ('right',(1.,1.,0.),(0.,-2.,0.),(0.,0.,2.),(.60,.55,.45)),
        ('back',(-1.,1.,0.),(2.,0.,0.),(0.,0.,2.),(.6,.6,.6)),
        ('front',(1.,-1.,0.),(-2.,0.,0.),(0.,0.,2.),(.6,.6,.6)),
        ('red',(-.7,.99,.15),(.4,0.,0.),(0.,0.,.4),(.8,.02,.02)),
        ('baffle',(0.,.2,.4),(.2,0.,0.),(0.,0.,.9),(.12,.12,.12)),
    ]
    return [dict(id=i+1,name=name,origin=p,u=u,v=v,normal=unit(cross(u,v)),albedo=c)
            for i,(name,p,u,v,c) in enumerate(values)]


def vertices(face):
    return [tuple(face['origin'][i]+a*face['u'][i]+b*face['v'][i] for i in range(3)) for a,b in ((0,0),(1,0),(1,1),(0,1))]


def inside_room(p):return -1<p[0]<1 and -1<p[1]<1 and 0<p[2]<2


def panels():
    result=[]
    for name,offset,size,radiance in (('key',(-1.2,-1.4,1.8),(.8,.55),6.),('fill',(1.4,.7,1.2),(.65,.8),2.5)):
        center=tuple(a+.35*b for a,b in zip((0.,0.,.75),offset));direction=unit(tuple(-x for x in offset))
        k=cross((1.,0.,0.),direction)
        def rotate(v):
            a,b=cross(k,v),cross(k,cross(k,v))
            return tuple(v[i]+a[i]+b[i]/(1+direction[0]) for i in range(3))
        result.append(dict(name=name,center=center,direction=direction,
            half_u=rotate((0.,.35*size[0],0.)),half_v=rotate((0.,0.,.35*size[1])),
            half_size=(.35*size[0],.35*size[1]),radiance=radiance*.20))
    return result


def panel_corners(panel):
    return [tuple(panel['center'][i]+a*panel['half_u'][i]+b*panel['half_v'][i] for i in range(3))
            for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))]


def primary_ray(x,y):return ((x+.5-WIDTH/2)/FOCAL,1.,-(y+.5-HEIGHT/2)/FOCAL)


def primary_hit(x,y):
    direction=primary_ray(x,y);hits=[]
    for f in geometry():
        denominator=dot(direction,f['normal'])
        if abs(denominator)<1e-14:continue
        t=dot(sub(f['origin'],EYE),f['normal'])/denominator
        if t<=0:continue
        p=tuple(EYE[i]+t*direction[i] for i in range(3));d=sub(p,f['origin'])
        a,b=dot(d,f['u'])/dot(f['u'],f['u']),dot(d,f['v'])/dot(f['v'],f['v'])
        if 0<=a<=1 and 0<=b<=1:hits.append((t,f['id']))
    if not hits:raise ValueError('open room or missed primary ray')
    depth,identity=min(hits)
    return identity,depth


def roi_pixels(roi):
    x0,y0,x1,y1=roi
    return [(x,y) for y in range(y0,y1) for x in range(x0,x1)]


def protocol():
    return dict(extent=[WIDTH,HEIGHT],eye=EYE,focal_px=FOCAL,near=NEAR,far=FAR,rois=ROIS,
        geometry=geometry(),panels=panels(),pt_seeds=[0,1,2,3],pt_depths=[0,1,2,6],pt_spp=512,
        native_seeds=[0,1],native_batches=[1,4,16],native_bounces=[1,2],native_samples_per_batch=16,
        energy_aggregation='mean of two independent groups, explicitly 32/128/512 total samples; each individual group is 16/64/256 and its MSE is reported separately',
        native_area_light_grid=8,environment_radiance=0.,exposure=1.,display_transform='clip',msaa_samples=1,
        reference_uses_public_pt_admission=False,
        reference_scope='independent actual SAPIEN SDK scene; the public PT adapter still rejects interior cameras',
        native_material='opaque metal0 roughness1: existing dielectric GGX/Fresnel model retained, not mathematically pure Lambert',
        reference_material='SDK opaque metal0 roughness1 specular0; no free correction factor to match native',
        fixed_energy_bias_limit=.35,noise_ratio_4_limit=.80,noise_ratio_16_limit=.55,
        pt_second_order_min_fraction_of_deep=.02,native_second_order_min_fraction_of_deep=.01,
        energy_order_tolerance_fraction=.02,deep_reference='finite maxDepth6, not infinite-bounce convergence',
        lighting_scope='two invisible finite parallelogram emitters completely inside six opaque walls; no wall removed for lighting',
        seed_scope='independent base seeds; 1/4/16 progressive batches have equal budgets across base-seed groups; no temporal motion',
        statistics_scope='seed-pair difference estimates stochastic noise; distance to PT also contains material/quadrature/truncation bias')


def pair_noise(a,b):
    if not a or len(a)!=len(b):raise ValueError('matched nonempty samples required')
    return math.sqrt(statistics.mean((x-y)**2 for x,y in zip(a,b))/2)


def errors(actual,reference):
    if not actual or len(actual)!=len(reference):raise ValueError('matched nonempty samples required')
    mean=statistics.mean(reference);rms=math.sqrt(statistics.mean(x*x for x in reference))
    if mean<=0 or rms<=0:raise ValueError('positive reference required')
    mse=statistics.mean((a-b)**2 for a,b in zip(actual,reference))
    return dict(mean=statistics.mean(actual),reference_mean=mean,relative_mean_bias=(statistics.mean(actual)-mean)/mean,
                mse=mse,rmse=math.sqrt(mse),nrmse=math.sqrt(mse)/rms)


def variance_gate(noise):
    if set(noise)!={1,4,16} or noise[1]<=0 or not all(math.isfinite(x) and x>=0 for x in noise.values()):
        raise ValueError('positive first-batch noise and complete finite sample budgets required')
    r4,r16=noise[4]/noise[1],noise[16]/noise[1]
    return dict(passed=r4<=.8 and r16<=.55,rmse_ratio_4=r4,rmse_ratio_16=r16,
                variance_ratio_4=r4*r4,variance_ratio_16=r16*r16,noise_by_batches=noise,
                note='difference of two independent equal-budget seed means divided by sqrt(2); constant bias cancels')


def order_gate(pt,native):
    scale=pt[6]
    if scale<=0:raise ValueError('nonzero finite deep reference required')
    monotone=all(pt[b]>=pt[a]-.02*scale for a,b in ((0,1),(1,2),(2,6)))
    p,n=(pt[2]-pt[1])/scale,(native[2]-native[1])/scale
    return dict(passed=monotone and p>=.02 and n>=.01,pt_monotone_with_registered_tolerance=monotone,
                pt_second_increment_over_deep=p,native_second_increment_over_deep=n,
                deep_minus_second=pt[6]-pt[2],deep_minus_second_fraction=(pt[6]-pt[2])/scale)


def validate_mean(payload,width,height,batches):
    required=dict(width=width,height=height,batches=batches,primary_samples_per_pixel=16*batches,
                  format='RGBA32F_LE',space='scene-linear-before-exposure')
    if any(payload.get(k)!=v for k,v in required.items()) or type(payload.get('data')) is not bytes or len(payload['data'])!=width*height*16:
        raise ValueError('actual owned scene-linear mean extent/budget/format mismatch')


def sensor_equal(a,b):return all(a[k]==b[k] for k in SENSORS)


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def build_public(root):
    from _panda_scene import write_mesh
    from phymeo_rendering.assets import Material
    from phymeo_rendering.cameras import Camera
    from phymeo_rendering.scene import FrameKey,Instance,NodePose,Pose,SceneFrame,VisualScene
    from phymeo_rendering.studio import StudioLighting
    root.mkdir(parents=True,exist_ok=True)
    faces=geometry();meshes=[]
    for f in faces:meshes.append(write_mesh(root,f['name'],vertices(f),(f['normal'],)*4,((0,1,2),(0,2,3))))
    scene=VisualScene('closed-diffuse-room',tuple(meshes),tuple(Material(f['name'],f['albedo'],0.,1.) for f in faces),
        tuple(Instance(f['id'],f['name'],f['name'],f['name']) for f in faces))
    frame=SceneFrame(FrameKey(scene.key,'authored-room',0,0,0,'authored'),0.,tuple(NodePose(f['name'],Pose()) for f in faces))
    camera=Camera('inside-room',frame.key,Pose(EYE,(math.sqrt(.5),0.,0.,math.sqrt(.5))),WIDTH,HEIGHT,FOCAL,FOCAL,WIDTH/2,HEIGHT/2,NEAR,FAR)
    studio=StudioLighting(center_m=(0.,0.,.75),scale_m=.35,panel_strength=.20,environment_radiance=0.)
    return scene,frame,camera,studio


def _roi_values(rgb,name,np):
    if name=='white':return np.concatenate([_roi_values(rgb,n,np) for n in ('bright','shadow')],axis=0)
    x0,y0,x1,y1=ROIS[name]
    return rgb[y0:y1,x0:x1].reshape(-1,3)


def _energy(rgb,name,np):return float(np.mean(_roi_values(rgb,name,np)@np.asarray(LUMA)))


def _metrics(actual,reference,name,np):
    a,b=_roi_values(actual,name,np),_roi_values(reference,name,np)
    result=errors(a.ravel().tolist(),b.ravel().tolist())
    ya,yb=float(np.mean(a@np.asarray(LUMA))),float(np.mean(b@np.asarray(LUMA)))
    result.update(luminance=ya,reference_luminance=yb,luminance_relative_bias=(ya-yb)/yb)
    return result


def run_reference(output,*,device='cuda:0'):
    import numpy as np
    import sapien
    from PIL import Image
    from phymeo_rendering.backends._sapien_sampling import make_seeded_pack
    from phymeo_rendering.backends._sapien_pixels import encode_linear_image
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='phymeo-closed-room-sdk-reference-v1',status='running',physics_steps=0,registration=protocol(),
        script_sha256=digest(__file__),sdk_version=sapien.__version__,sdk_file=sapien.__file__,device_requested=device,
        interface='direct public SAPIEN SDK rendering components, NOT public Phymeo PT session',captures=[])
    dump(output/'registration.json',report)
    if sapien.__version__!='3.0.3':raise ValueError('fixed SDK version required')
    started=perf_counter();system=sapien.render.RenderSystem(device);scene=sapien.Scene([system]);system.ambient_light=(0.,0.,0.)
    actual_ids={};means={};samples={};geometry_receipts=[]
    def counts():
        return dict(entities=len(scene.get_entities()),cameras=len(system.cameras),bodies=len(system.render_bodies),lights=len(system.lights))
    try:
        for face in geometry():
            material=sapien.render.RenderMaterial(base_color=(*face['albedo'],1.),metallic=0.,roughness=1.,specular=0.)
            positions=np.asarray(vertices(face),dtype=np.float32);normals=np.asarray([face['normal']]*4,np.float32)
            indices=np.asarray(((0,1,2),(0,2,3)),np.uint32)
            shape=sapien.render.RenderShapeTriangleMesh(positions,indices,normals,np.zeros((4,2),np.float32),material)
            body=sapien.render.RenderBodyComponent();body.attach(shape)
            entity=sapien.Entity();entity.name=face['name'];entity.add_component(body);scene.add_entity(entity)
            actual_ids[int(shape.per_scene_id)]=face['id']
            geometry_receipts.append(dict(id=face['id'],name=face['name'],vertices_f32=positions.tolist(),triangles=indices.tolist(),
                normals_f32=normals.tolist(),vertex_sha256=sha256(positions.tobytes()).hexdigest(),index_sha256=sha256(indices.tobytes()).hexdigest(),
                material=dict(base_color=list(map(float,material.base_color)),metallic=float(material.metallic),roughness=float(material.roughness),specular=float(material.specular))))
        actual_lights=[]
        for p in panels():
            light=sapien.render.RenderParallelogramLightComponent();light.color=(p['radiance'],)*3
            light.set_shape(*p['half_size'],math.pi/2)
            entity=sapien.Entity();entity.name=p['name'];entity.add_component(light)
            rotation=sapien.math.shortest_rotation((1.,0.,0.),p['direction'])
            entity.pose=sapien.Pose(p['center'],rotation);scene.add_entity(entity)
            actual_lights.append(dict(name=p['name'],position=list(map(float,entity.pose.p)),quaternion_wxyz=list(map(float,entity.pose.q)),
                half_width=float(light.half_width),half_height=float(light.half_height),angle=float(light.angle),
                color=list(map(float,light.color)),declared=p))
        report.update(geometry=geometry_receipts,actual_lights=actual_lights,attachments=counts(),device=str(system.device),
            source_clipping_proof=dict(camera_inside=inside_room(EYE),primary_center_depth_range=[min(primary_hit(x,y)[1] for y in range(HEIGHT) for x in range(WIDTH)),
                max(primary_hit(x,y)[1] for y in range(HEIGHT) for x in range(WIDTH))],
                behind_camera_vertices=sum(p[1]<EYE[1] for f in geometry() for p in vertices(f)),
                public_adapter_not_used='all-vertices clip slab would reject these real rear-wall vertices; SDK rays are directly observed below'))
        if counts()['bodies']!=8 or counts()['lights']!=2:raise ValueError('reference lost a real wall, material patch, occluder or finite emitter')
        original=Path(sapien.__file__).resolve().parent/'vulkan_shader'/'rt'
        report['official_shader_sha256']={p.name:digest(p) for p in sorted(original.iterdir()) if p.is_file()}
        with TemporaryDirectory(prefix='phymeo-closed-reference-') as temporary:
            for seed in protocol()['pt_seeds']:
                pack=make_seeded_pack(original,Path(temporary)/f'seed-{seed}',seed)
                shader_hash={p.name:digest(p) for p in sorted(pack.iterdir()) if p.is_file()}
                for depth in protocol()['pt_depths']:
                    label=f'pt-depth{depth}-seed{seed}'
                    camera=sapien.render.RenderCameraComponent(WIDTH,HEIGHT,str(pack));entity=sapien.Entity();entity.add_component(camera)
                    # SDK +X forward/+Y left/+Z up: +90 degrees about world Z.
                    entity.pose=sapien.Pose(EYE,(math.sqrt(.5),0.,0.,math.sqrt(.5)))
                    camera.set_perspective_parameters(NEAR,FAR,FOCAL,FOCAL,WIDTH/2,HEIGHT/2,0.)
                    for key,value in (('spp',512),('maxDepth',depth),('aperture',0.),('focusPlane',1.),('russianRoulette',0)):
                        camera.set_property(key,value)
                    msaa,denoiser=sapien.render.get_msaa(),sapien.render.get_ray_tracing_denoiser()
                    try:
                        sapien.render.set_msaa(1);sapien.render.set_ray_tracing_denoiser('none');scene.add_entity(entity)
                    finally:
                        sapien.render.set_msaa(msaa);sapien.render.set_ray_tracing_denoiser(denoiser)
                    try:
                        before=perf_counter();scene.update_render();camera.take_picture()
                        hdr=np.array(camera.get_picture('HdrColor'),dtype='<f4',copy=True)
                        position=np.array(camera.get_picture('Position'),dtype='<f4',copy=True)
                        segmentation=np.array(camera.get_picture('Segmentation'),dtype='<u4',copy=True)
                        wall=perf_counter()-before
                        if hdr.shape!=(HEIGHT,WIDTH,4) or not np.isfinite(hdr).all() or np.any(hdr[...,:3]<0):raise ValueError('invalid real HDR readback')
                        roi_identity={}
                        for name,roi in ROIS.items():
                            x0,y0,x1,y1=roi;expected=7 if name=='red' else 5
                            visible=segmentation[y0:y1,x0:x1,0]
                            mapped=np.vectorize(actual_ids.__getitem__)(visible)
                            depths=-position[y0:y1,x0:x1,2]
                            predicted=1.79 if name=='red' else 1.8
                            good=bool(np.all(mapped==expected) and np.all(depths>NEAR) and np.all(depths<FAR) and np.max(np.abs(depths-predicted))<1e-4)
                            roi_identity[name]=dict(passed=good,expected_instance=expected,observed_instances=np.unique(mapped).tolist(),
                                forward_depth_min=float(depths.min()),forward_depth_max=float(depths.max()),expected_forward_depth=predicted)
                            if not good:raise ValueError('independent ROI ray/identity/depth proof failed')
                        raw=hdr.tobytes();(output/(label+'.hdr-rgba-f32.raw')).write_bytes(raw)
                        (output/(label+'.position-f32.raw')).write_bytes(position.tobytes())
                        (output/(label+'.segmentation-u32.raw')).write_bytes(segmentation.tobytes())
                        rgb=encode_linear_image(hdr[...,:3],exposure=1.,display_transform='clip')
                        Image.frombytes('RGB',(WIDTH,HEIGHT),rgb).save(output/(label+'.png'))
                        samples[(depth,seed)]=hdr[...,:3].astype(np.float64)
                        entry=dict(label=label,depth=depth,seed=seed,spp=512,wall_ms=wall*1000,hdr_file=label+'.hdr-rgba-f32.raw',
                            hdr_sha256=sha256(raw).hexdigest(),png_sha256=digest(output/(label+'.png')),roi_proof=roi_identity,
                            roi_luminance={n:_energy(samples[(depth,seed)],n,np) for n in (*ROIS,'white')},
                            shader_sha256=shader_hash,camera_entity_position=list(map(float,entity.pose.p)),camera_entity_quaternion_wxyz=list(map(float,entity.pose.q)),
                            sampling='same validated local shifted-halton derivative; distinct explicit seed, fresh camera and no history',attachments=counts())
                        report['captures'].append(entry);dump(output/'summary.json',report)
                        print(json.dumps(dict(reference=label,white_luminance=entry['roi_luminance']['white'])),flush=True)
                    finally:scene.remove_entity(entity)
        for depth in protocol()['pt_depths']:
            mean=np.mean([samples[(depth,s)] for s in protocol()['pt_seeds']],axis=0)
            means[depth]=mean;(output/f'pt-depth{depth}.mean-rgb-f64.raw').write_bytes(mean.astype('<f8').tobytes())
        report['energy']={str(d):{n:_energy(means[d],n,np) for n in (*ROIS,'white')} for d in means}
        for e in report['energy'].values():e['bright_minus_shadow']=e['bright']-e['shadow']
        report['seed_statistics']={str(d):{n:dict(values=[_energy(samples[(d,s)],n,np) for s in range(4)],
            sample_sd=statistics.stdev(_energy(samples[(d,s)],n,np) for s in range(4)),
            standard_error=statistics.stdev(_energy(samples[(d,s)],n,np) for s in range(4))/2) for n in (*ROIS,'white')} for d in means}
        report['errors_to_depth6']={str(d):{n:_metrics(means[d],means[6],n,np) for n in (*ROIS,'white')} for d in (0,1,2)}
        report['depth6_minus_depth2']={n:_energy(means[6]-means[2],n,np) for n in (*ROIS,'white')}
        report['distinct_seed_frames']={str(d):len({c['hdr_sha256'] for c in report['captures'] if c['depth']==d}) for d in means}
        report['status']='captured-reference'
    except BaseException as error:
        report['status']='failed';report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());raise
    finally:
        scene.clear();report['cleanup']=counts();report['elapsed_wall_s']=perf_counter()-started
        dump(output/'summary.json',report)
    return report



def load_reference(root):
    import numpy as np
    root=Path(root);report=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    if report['status']!='captured-reference' or json.dumps(report['registration'],sort_keys=True)!=json.dumps(protocol(),sort_keys=True):
        raise ValueError('reference is incomplete or uses a different frozen scene/protocol')
    if report['attachments']['bodies']!=8 or report['attachments']['lights']!=2 or any(report['cleanup'].values()):
        raise ValueError('reference geometry/lights/cleanup proof is incomplete')
    values={}
    for item in report['captures']:
        key=(item['depth'],item['seed']);path=root/item['hdr_file']
        if key in values or digest(path)!=item['hdr_sha256'] or not all(v['passed'] for v in item['roi_proof'].values()):
            raise ValueError('reference readback identity or SHA mismatch')
        data=np.frombuffer(path.read_bytes(),'<f4').reshape(HEIGHT,WIDTH,4)[...,:3].astype(np.float64)
        if not np.isfinite(data).all() or np.any(data<0):raise ValueError('invalid reference HDR')
        values[key]=data
    expected={(d,s) for d in (0,1,2,6) for s in range(4)}
    if set(values)!=expected:raise ValueError('reference seed/depth set incomplete')
    return report,{d:np.mean([values[(d,s)] for s in range(4)],axis=0) for d in (0,1,2,6)}


def run_native(library,output,reference_root,*,shaders=None,device_index=1,validation=False):
    import numpy as np
    from PIL import Image,ImageDraw
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    from diffuse_bounce_calibration import _native_closed,inverse_srgb
    reference_report,references=load_reference(reference_root)
    output,library=Path(output),Path(library);output.mkdir(parents=True,exist_ok=False)
    scene,frame,camera,studio=build_public(output/'geometry')
    report=dict(schema='phymeo-closed-room-progressive-v1',status='running',physics_steps=0,registration=protocol(),
        script_sha256=digest(__file__),library=dict(path=str(library.resolve()),sha256=digest(library)),
        reference=dict(path=str(Path(reference_root).resolve()),summary_sha256=digest(Path(reference_root)/'summary.json'),
            interface=reference_report['interface'],sdk_version=reference_report['sdk_version'],device=reference_report['device']),
        scene=asdict(scene),frame=asdict(frame),camera=asdict(camera),studio=asdict(studio),
        captures=[],cleanup=[],acceptance={k:False for k in ('primary_identity','sensor_bytes','hdr_budget','order_energy',
            'variance_reduction','matched_depth_energy_bias','cleanup')})
    dump(output/'registration.json',report);started=perf_counter();images={};sensor_reference=None
    sensor_ok=True;identity_ok=True;budget_ok=True
    def save_capture(owner,label,bounces,seed,batch,progressive):
        nonlocal sensor_reference,sensor_ok,identity_ok,budget_ok
        before=perf_counter();capture=owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,tuple(Channel)))
        capture_wall=perf_counter()-before
        if capture.frame_key!=frame.key or capture.camera!=camera or capture.time_s!=frame.time_s:raise ValueError('native source/camera/time mismatch')
        planes={p.channel.name:p.data for p in capture.planes}
        if sensor_reference is None:sensor_reference=planes
        sensor_ok &= sensor_equal(sensor_reference,planes)
        ids=np.frombuffer(planes['INSTANCE_ID'],'<u4').reshape(HEIGHT,WIDTH)
        for name,roi in ROIS.items():
            x0,y0,x1,y1=roi;identity_ok &= bool(np.all(ids[y0:y1,x0:x1]==(7 if name=='red' else 5)))
        for name,data in planes.items():(output/(label+'.'+name+'.raw')).write_bytes(data)
        Image.frombytes('RGB',(WIDTH,HEIGHT),planes['RGB']).save(output/(label+'.png'))
        if progressive:
            prior=owner.diagnostics()['progressive']['batches'];before=perf_counter();payload=owner.read_linear_mean()
            read_wall=perf_counter()-before;validate_mean(payload,WIDTH,HEIGHT,batch)
            after=owner.diagnostics()['progressive']['batches'];budget_ok &= prior==after==batch
            raw=payload['data'];rgba=np.frombuffer(raw,'<f4').reshape(HEIGHT,WIDTH,4)
            if not np.isfinite(rgba).all() or np.any(rgba[...,:3]<0):raise ValueError('invalid real scene-linear mean')
            rgb=rgba[...,:3].astype(np.float64);(output/(label+'.linear-rgba-f32.raw')).write_bytes(raw)
            linear_meta={k:v for k,v in payload.items() if k!='data'}
            linear_meta.update(sha256=sha256(raw).hexdigest(),file=label+'.linear-rgba-f32.raw',read_does_not_advance_batch=prior==after)
        else:
            # Direct-control display only; quantitative 1/2-bounce results below use real raw means.
            codes=np.frombuffer(planes['RGB'],np.uint8).reshape(HEIGHT,WIDTH,3)
            rgb=np.asarray([inverse_srgb(i/255) for i in range(256)])[codes]
            read_wall=0.;linear_meta=dict(space='inverse-sRGB direct-control estimate; excluded from raw mean error/variance gates')
        diag=owner.diagnostics()
        if progressive:
            gpu=diag['device'].get('resident_progressive_info') or {}
            budget_ok &= gpu.get('enabled')==1 and gpu.get('bounces')==bounces and gpu.get('batches')==batch and gpu.get('samples_per_batch')==16
        item=dict(label=label,bounces=bounces,seed=seed,batches=batch,samples_per_pixel=16*batch if progressive else 0,
            capture_wall_ms=capture_wall*1000,linear_readback_wall_ms=read_wall*1000,
            plane_sha256={k:sha256(v).hexdigest() for k,v in planes.items()},png_sha256=digest(output/(label+'.png')),
            linear=linear_meta,progressive=diag.get('progressive'),device_progressive=diag['device'].get('resident_progressive_info'),device_profile=diag['device'].get('resident_profile'),
            roi_luminance={n:_energy(rgb,n,np) for n in (*ROIS,'white')},
            display_clipped_pixels=int(np.any(np.frombuffer(planes['RGB'],np.uint8).reshape(HEIGHT,WIDTH,3)==255,axis=2).sum()))
        item['bright_minus_shadow']=item['roi_luminance']['bright']-item['roi_luminance']['shadow']
        report['captures'].append(item);dump(output/'summary.json',report)
        print(json.dumps(dict(native=label,white_luminance=item['roi_luminance']['white'])),flush=True)
        return rgb
    def create(**options):
        return NativeBackend(library_path=library,shader_directory=shaders,resident=True,shadow_mode='ray_query',
            area_light_grid=8,studio=studio,device_index=device_index,validation=validation,msaa_samples=1,**options).open_scene(scene)
    try:
        owner=create(indirect_mode='none')
        try:save_capture(owner,'native-direct',0,0,0,False)
        finally:
            try:owner.close()
            finally:
                diag=owner.diagnostics();report['cleanup'].append(dict(label='direct',passed=_native_closed(diag),diagnostics=diag))
        for bounces in (1,2):
            for seed in (0,1):
                owner=create(indirect_mode='diffuse_bounce',indirect_seed=seed,indirect_bounces=bounces,progressive=True)
                try:
                    for batch in range(1,17):
                        label=f'native-depth{bounces}-seed{seed}-batch{batch:02d}'
                        rgb=save_capture(owner,label,bounces,seed,batch,True)
                        if batch in (1,4,16):images[(bounces,seed,batch)]=rgb
                finally:
                    try:owner.close()
                    finally:
                        diag=owner.diagnostics();report['cleanup'].append(dict(label=f'depth{bounces}-seed{seed}',passed=_native_closed(diag),diagnostics=diag))
                        dump(output/'summary.json',report)
        energies={};metrics={};noise_results={};bias_ok=True
        for depth in (1,2):
            noises={};region_noises={name:{} for name in ROIS}
            for batches in (1,4,16):
                a,b=images[(depth,0,batches)],images[(depth,1,batches)];average=(a+b)/2
                key=f'depth{depth}-batch{batches}'
                energies[key]={n:_energy(average,n,np) for n in (*ROIS,'white')}
                energies[key]['bright_minus_shadow']=energies[key]['bright']-energies[key]['shadow']
                metrics[key]=dict(matched_depth={n:_metrics(average,references[depth],n,np) for n in (*ROIS,'white')},
                                  deep6={n:_metrics(average,references[6],n,np) for n in (*ROIS,'white')},
                                  averaging='two independent seed groups, each at exactly the named per-pixel sample count',
                                  per_group_samples_per_pixel=16*batches,two_group_mean_total_samples_per_pixel=32*batches,
                                  individual_seeds={str(seed):dict(matched_depth={n:_metrics(value,references[depth],n,np) for n in (*ROIS,'white')},
                                      deep6={n:_metrics(value,references[6],n,np) for n in (*ROIS,'white')}) for seed,value in ((0,a),(1,b))})
                metrics[key]['equal_budget_rmse']={n:math.sqrt(sum(metrics[key]['individual_seeds'][str(s)]['matched_depth'][n]['mse'] for s in (0,1))/2) for n in (*ROIS,'white')}
                if batches==16:bias_ok &= all(abs(metrics[key]['matched_depth'][n]['luminance_relative_bias'])<=.35 for n in (*ROIS,'white'))
                noises[batches]=pair_noise(_roi_values(a,'white',np).ravel().tolist(),_roi_values(b,'white',np).ravel().tolist())
                for name in ROIS:region_noises[name][batches]=pair_noise(_roi_values(a,name,np).ravel().tolist(),_roi_values(b,name,np).ravel().tolist())
                (output/(key+'.two-seed-mean-rgb-f64.raw')).write_bytes(average.astype('<f8').tobytes())
            noise_results[str(depth)]=dict(white=variance_gate(noises),regions={n:variance_gate(v) for n,v in region_noises.items()})
        pt_energy={d:_energy(references[d],'white',np) for d in (0,1,2,6)}
        native_energy={d:energies[f'depth{d}-batch16']['white'] for d in (1,2)}
        order=order_gate(pt_energy,native_energy)
        report.update(energy=energies,errors=metrics,variance=noise_results,order=order,
            pt_energy={str(d):{n:_energy(references[d],n,np) for n in (*ROIS,'white')} for d in (0,1,2,6)},
            pt_depth6_minus_depth2={n:_energy(references[6]-references[2],n,np) for n in (*ROIS,'white')},
            per_seed_energy={f'{d}-{s}-{b}':{n:_energy(v,n,np) for n in (*ROIS,'white')} for (d,s,b),v in images.items()})
        report['acceptance'].update(primary_identity=identity_ok,sensor_bytes=sensor_ok,hdr_budget=budget_ok,
            order_energy=order['passed'],variance_reduction=all(x['white']['passed'] for x in noise_results.values()),
            matched_depth_energy_bias=bias_ok,cleanup=all(x['passed'] for x in report['cleanup']))
        sheet=Image.new('RGB',(WIDTH*4,(HEIGHT+18)*2),(20,22,25));draw=ImageDraw.Draw(sheet)
        from phymeo_rendering.backends._sapien_pixels import encode_linear_image
        for row,depth in enumerate((1,2)):
            for col,batches in enumerate((1,4,16)):
                label=f'native-depth{depth}-seed0-batch{batches:02d}'
                draw.text((col*WIDTH+2,row*(HEIGHT+18)+2),f'{depth} hop / {batches*16}spp',fill=(255,255,255))
                with Image.open(output/(label+'.png')) as image:sheet.paste(image,(col*WIDTH,row*(HEIGHT+18)+18))
            draw.text((3*WIDTH+2,row*(HEIGHT+18)+2),f'PT depth {depth}',fill=(255,255,255))
            rendered=encode_linear_image(references[depth].astype(np.float32),exposure=1.,display_transform='clip')
            sheet.paste(Image.frombytes('RGB',(WIDTH,HEIGHT),rendered),(3*WIDTH,row*(HEIGHT+18)+18))
        sheet.save(output/'comparison.png')
        report['status']='passed' if all(report['acceptance'].values()) else 'measured-not-passed'
    except BaseException as error:
        report['status']='failed';report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());raise
    finally:
        report['elapsed_wall_s']=perf_counter()-started
        report['imported_sources']={n:dict(file=m.__file__,sha256=digest(m.__file__)) for n,m in tuple(sys.modules.items())
            if n.startswith('phymeo_rendering') and getattr(m,'__file__',None) and Path(m.__file__).is_file()}
        dump(output/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);subparsers=parser.add_subparsers(dest='operation',required=True)
    reference=subparsers.add_parser('reference');reference.add_argument('--output',type=Path,required=True);reference.add_argument('--device',default='cuda:0')
    native=subparsers.add_parser('native');native.add_argument('--output',type=Path,required=True);native.add_argument('--reference',type=Path,required=True)
    native.add_argument('--library',type=Path,required=True);native.add_argument('--shaders',type=Path)
    native.add_argument('--device-index',type=int,default=1);native.add_argument('--validation',action='store_true')
    args=parser.parse_args()
    if args.operation=='reference':report=run_reference(args.output,device=args.device)
    else:report=run_native(args.library,args.output,args.reference,shaders=args.shaders,device_index=args.device_index,validation=args.validation)
    print(json.dumps(dict(status=report['status'],acceptance=report.get('acceptance'))))
    if report['status'] not in ('passed','captured-reference'):raise SystemExit(1)


if __name__=='__main__':main()


