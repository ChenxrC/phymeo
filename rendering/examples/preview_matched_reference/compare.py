"""One actual Panda pose, common authored materials, three explicit renderers.

Uses public session APIs and the existing strict tiled helper. No renderer
fallback, material edit, exposure fit, denoiser or performance acceptance.
"""
import argparse
from dataclasses import asdict,replace
from hashlib import sha256
import html
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import traceback
import zlib

MODES=('near','full','sapien-geometry','sapien')
ROI_IDS={'white-shell':(150,151,154),'fingers':(156,158),'rubber':(2,)}


def protocol():
    return dict(schema='phymeo-preview-matched-reference-v1',pose=60,scene_index=0,normal_map=False,
        extent=[1280,720],exposure=8.,display_transform='pbr-neutral',native_batches=64,native_primary_spp=1024,
        pt_spp=1024,pt_max_bounces=4,pt_seed=1,pt_sampling_mode='shifted-halton',pt_tile=[320,180],
        pt_sampling_scope='same pinhole rays via principal-point crop; tile-local stochastic sequence, not full-dispatch byte equivalence',
        material_scope='actual scenes[0] from panda_render_performance.bind_scene; both retain scanned base/roughness, metallic payload, all original lights/HDR',
        comparison_scope='SAPIEN global transport/BRDF and native one-hop+single-bounce specular differ; not ground truth or per-pixel correctness acceptance',
        native_near_scope='one current compact finite-range exposure, no temporal history',
        native_full_scope='64 progressive batches of 16 main paths, full-distance one-hop diffuse, native one-bounce specular',
        denoiser='none',performance_claim=False,physics_steps=0,roi_ids=ROI_IDS)


def tile_plan(width,height,tw,th):
    if any(type(x) is not int or x<=0 for x in (width,height,tw,th)):raise ValueError('positive integer tile dimensions required')
    return [(x,y,min(tw,width-x),min(th,height-y)) for y in range(0,height,th) for x in range(0,width,tw)]


def check_match(records):
    if not records or any(x.get('status')!='complete' for x in records):raise ValueError('complete render records required')
    if len({x['input_sha256'] for x in records})!=1:raise ValueError('actual common scene/material/light/frame/camera input differs')


def closed_ok(mode,diag):
    if mode in ('near','full'):
        d=diag.get('device',{});t=d.get('resident_telemetry') or {}
        return (d.get('closed') is True and all(d.get(k)==0 for k in ('live_contexts','live_frames','live_resources','validation_errors'))
            and t.get('final_after_destroy') is True and t.get('device_local_bytes_current')==0
            and t.get('host_visible_bytes_current')==0 and t.get('allocation_count') is not None
            and t.get('allocation_count')==t.get('free_count'))
    life=diag.get('lifecycle',{})
    return life.get('closed') is True and life.get('attached_entities_after_clear')==0 and life.get('studio_temporary_files_exist') is False

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def capture(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='phymeo-preview-matched-frame-v1',status='started',mode=args.mode,protocol=protocol(),script_sha256=digest(__file__))
    save(out/'report.json',report);owner=None
    try:
        sys.path.insert(0,str(args.examples.resolve()))
        if args.mode=='near':
            from preview_compact.runtime import activate,CompactBackend
            prepared,manifest=activate(args.prepared)
            contract=manifest['contract']
            if 'local_diffuse' not in contract or 'rigid_history' in contract:raise ValueError('explicit current finite-near prepared package required')
            if contract.get('area_light_grid')!=2:raise ValueError('registered common native area quadrature is grid2')
            for name,expected in manifest['generated_sha256'].items():
                path=(prepared/name).resolve()
                if not path.is_relative_to(prepared) or digest(path)!=expected:raise ValueError('prepared source changed')
            report.update(prepared=str(prepared),manifest_sha256=digest(prepared/'source-manifest.json'),contract=contract,
                dll_sha256=digest(prepared/'build/phymeo_native.dll'),shaders={p.name:digest(p) for p in (prepared/'build').glob('*.spv')})
        else:sys.path.insert(0,str(args.runtime_src.resolve()))
        import numpy as np
        import PIL
        from PIL import Image
        import phymeo_rendering
        from phymeo_rendering.outputs import Channel
        from phymeo_rendering.pipeline import CaptureRequest,RenderPath
        from phymeo_rendering.mesh_data import load_mesh
        report['dependency_versions']=dict(python=sys.version,numpy=np.__version__,pillow=PIL.__version__)
        import panda_render_performance as factory
        geometry=out/'geometry' # factory load_episode owns creation of this new directory
        scenes,frames,studio,trace=factory.bind_scene(geometry);scene=scenes[0]
        if any(getattr(m,'normal_map',None) is not None for m in scene.materials):raise ValueError('common material subset unexpectedly contains a normal map')
        frame,original=factory.make_inputs(scene,frames)[60];camera=original.camera
        if (camera.width,camera.height)!=(1280,720):raise ValueError('original fixed camera extent differs')
        actual_scene=asdict(scene);canonical_scene=dict(actual_scene);meshes=[]
        for mesh in scene.meshes:
            data=load_mesh(mesh)
            decoded=dict(positions=data.positions,normals=data.normals,triangles=data.triangles,texcoords=getattr(data,'texcoords',None))
            meshes.append(dict(id=mesh.id,decoded_sha256=sha256(canonical(decoded)).hexdigest()))
        canonical_scene['meshes']=meshes
        binding=dict(scene=canonical_scene,frame=asdict(frame),camera=asdict(camera),studio=asdict(studio),
            exposure=8.,display_transform='pbr-neutral',trace_sha256=digest(trace),pose_index=60,
            factory_sha256=digest(factory.__file__),film_factory_sha256=digest(factory.film.__file__))
        save(out/'inputs.json',dict(binding=binding,actual_scene=actual_scene))
        report.update(input_sha256=sha256(canonical(binding)).hexdigest(),inputs_file='inputs.json',inputs_file_sha256=digest(out/'inputs.json'),
            imported_runtime=str(Path(phymeo_rendering.__file__).resolve()),imported_factory=str(Path(factory.__file__).resolve()))
        save(out/'report.json',report)
        channels=tuple(Channel) if args.mode in ('near','full') else ((Channel.RGB,) if args.mode=='sapien' else tuple(c for c in Channel if c is not Channel.RGB))
        path=RenderPath.PATH_TRACE if args.mode=='sapien' else RenderPath.RASTER
        request=CaptureRequest(camera,path,channels)
        if args.mode=='near':backend=CompactBackend(prepared,seed=0,studio=studio,exposure=8.,display_transform='pbr-neutral',device_index=args.device_index,validation=True)
        elif args.mode=='full':
            from phymeo_rendering.backends.native import NativeBackend
            backend=NativeBackend(library_path=args.library,shader_directory=args.shaders,resident=True,shadow_mode='ray_query',
                reflection_mode='single_bounce',indirect_mode='diffuse_bounce',indirect_bounces=1,indirect_seed=1,
                progressive=True,spatial_filter='none',area_light_grid=2,msaa_samples=1,studio=studio,
                exposure=8.,display_transform='pbr-neutral',device_index=args.device_index,validation=True)
            report.update(dll_sha256=digest(args.library),shaders={p.name:digest(p) for p in args.shaders.glob('*.spv')})
        else:
            from phymeo_rendering.backends.sapien import SapienBackend,SapienConfig
            cfg=SapienConfig(path=path,device=args.sapien_device,samples_per_pixel=1024,max_bounces=4,
                seed=1 if args.mode=='sapien' else None,sampling_mode='shifted-halton' if args.mode=='sapien' else 'official',
                denoiser='none',studio=studio if args.mode=='sapien' else None,exposure=8.,display_transform='pbr-neutral')
            backend=SapienBackend(cfg)
            report['sensor_companion_scope']='separate SAPIEN raster geometry-only capture, not PT AOV; studio omitted because RGB not requested'
        started=perf_counter();owner=backend.open_scene(scene);report['open_wall_ms']=1000*(perf_counter()-started)
        if not owner.supports(request):raise ValueError('public backend refuses unchanged common input/request')
        raw=None;raw_format=None;started=perf_counter()
        if args.mode=='sapien':
            from unittest.mock import patch
            from phymeo_rendering.backends import _sapien_session
            from phymeo_rendering.tiled import capture_tiled
            original_encoder=_sapien_session.encode_linear_image;taps=[]
            def observe(values,**options):
                if values.dtype!=np.float32 or options!=dict(exposure=8.,display_transform='pbr-neutral'):raise ValueError('unexpected actual HDR display boundary')
                taps.append(np.array(values,dtype='<f4',copy=True))
                return original_encoder(values,**options)
            with patch.object(_sapien_session,'encode_linear_image',side_effect=observe):
                shot=capture_tiled(owner,frame,request,320,180)
            plan=tile_plan(1280,720,320,180)
            if len(taps)!=len(plan):raise ValueError('actual HDR tile count differs')
            merged=np.empty((720,1280,3),dtype='<f4')
            for (x,y,w,h),values in zip(plan,taps):
                if values.shape!=(h,w,3):raise ValueError('actual HDR tile extent differs')
                merged[y:y+h,x:x+w]=values
            raw=merged.tobytes();raw_format='RGB32F_LE scene-linear actual SAPIEN HdrColor before exposure'
            report['tiles']=plan
        else:
            repeats=64 if args.mode=='full' else 1
            for _ in range(repeats):shot=owner.capture(frame,request)
        report['capture_wall_ms']=1000*(perf_counter()-started)
        if shot.frame_key!=frame.key or shot.time_s!=frame.time_s or shot.camera!=camera:raise ValueError('actual publication identity differs')
        if args.mode in ('near','full'):
            value=owner.read_reconstructed_linear() if args.mode=='near' else owner.read_linear_mean()
            if (value.get('width'),value.get('height'),value.get('format'))!=(1280,720,'RGBA32F_LE') or type(value.get('data')) is not bytes or len(value['data'])!=1280*720*16:raise ValueError('actual owned linear readback extent/format differs')
            if args.mode=='full' and (value['batches']!=64 or value['primary_samples_per_pixel']!=1024 or value['space']!='scene-linear-before-exposure'):raise ValueError('full-distance sampling budget/space differs')
            raw=value['data'];raw_format=value['format']+' '+value['space']
        report['planes']={}
        for plane in shot.planes:
            filename=plane.channel.name+'.raw.zlib';(out/filename).write_bytes(zlib.compress(plane.data,6))
            report['planes'][plane.channel.name]=dict(file=filename,sha256=sha256(plane.data).hexdigest(),bytes=len(plane.data))
            if plane.channel is Channel.RGB:
                Image.frombytes('RGB',(1280,720),plane.data).save(out/'rgb.png')
                report.update(rgb_png='rgb.png',rgb_png_sha256=digest(out/'rgb.png'),rgb_sha256=sha256(plane.data).hexdigest())
        if raw is not None:
            (out/'linear.raw.zlib').write_bytes(zlib.compress(raw,6))
            report['linear']=dict(file='linear.raw.zlib',format=raw_format,sha256=sha256(raw).hexdigest(),bytes=len(raw))
        report.update(status='complete',before_close=owner.diagnostics())
    except BaseException as error:report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        if owner is not None:
            try:
                owner.close();report['after_close']=owner.diagnostics()
                report['cleanup_passed']=closed_ok(args.mode,report['after_close'])
                if not report['cleanup_passed']:raise ValueError('actual renderer teardown diagnostics failed')
            except BaseException as error:report.update(status='failed',close_error=repr(error))
        save(out/'report.json',report)
    return report


def assemble(root,records):
    import numpy as np
    from PIL import Image
    images=[records[m] for m in ('near','full','sapien')]
    check_match(images)
    def plane(mode,name):
        item=records[mode]['planes'][name];data=zlib.decompress((root/mode/item['file']).read_bytes())
        if sha256(data).hexdigest()!=item['sha256']:raise ValueError('saved original plane changed')
        return data
    current_ids=np.frombuffer(plane('near','INSTANCE_ID'),'<u4').reshape(720,1280)
    report=dict(input_sha256=images[0]['input_sha256'],native_sensors_equal={name:plane('near',name)==plane('full',name) for name in ('FORWARD_DEPTH','INSTANCE_ID','VALID')},
        reference_is_ground_truth=False,crops={})
    if records.get('sapien-geometry',{}).get('status')=='complete':
        check_match([images[0],records['sapien-geometry']])
        other=np.frombuffer(plane('sapien-geometry','INSTANCE_ID'),'<u4').reshape(720,1280)
        report['sapien_raster_id_disagreement_pixels']=int(np.count_nonzero(other!=current_ids))
        report['sapien_geometry_scope']='separate raster companion; all boundary/ID differences retained, not PT coverage oracle'
    rgb={mode:Image.open(root/mode/'rgb.png').convert('RGB') for mode in ('near','full','sapien')}
    labels={'near':'Native finite-near current','full':'Native full-distance one hop, 1024 primary spp','sapien':'SAPIEN PT 1024 spp, maxDepth 4, strict tiles'}
    body=['<!doctype html><meta charset="utf-8"><title>Matched Panda frame 60</title><style>body{font:16px system-ui;background:#17191c;color:#eee;margin:24px}.row{display:flex;gap:12px;overflow:auto}figure{margin:0}img{max-width:none}figcaption{padding:8px 0}p{max-width:1000px}</style><h1>Same Panda frame 60 · common authored materials</h1><p>1280×720 · exposure 8 · PBR Neutral · both without rubber normal map. Native finite-range, full-distance one-hop and SAPIEN global path tracing have different transport/material models. These are actual pixels, not a ground-truth quality ranking or a speed comparison.</p><div class="row">']
    for mode in rgb:body.append(f'<figure><figcaption>{html.escape(labels[mode])}</figcaption><a href="{mode}/rgb.png"><img style="width:640px" src="{mode}/rgb.png"></a></figure>')
    body.append('</div><h2>Identical original-pixel crops</h2><p>Crops are bounding rectangles of predeclared IDs in the native geometry, shared unchanged across all three images. Crop pixels are not resampled.</p>')
    for name,identities in ROI_IDS.items():
        yy,xx=np.nonzero(np.isin(current_ids,identities))
        if not len(xx):report['crops'][name]=dict(status='absent');continue
        box=(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)
        report['crops'][name]=dict(box=box,identity_ids=list(identities),visible_pixels=len(xx),files={})
        body.append('<h3>'+html.escape(name)+'</h3><div class="row">')
        for mode,image in rgb.items():
            file=f'{name}-{mode}.png';image.crop(box).save(root/file)
            report['crops'][name]['files'][mode]=dict(file=file,sha256=digest(root/file))
            body.append(f'<figure><figcaption>{html.escape(labels[mode])}</figcaption><img src="{file}"></figure>')
        body.append('</div>')
    (root/'index.html').write_text('\n'.join(body),encoding='utf-8')
    return report


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    report=dict(schema='phymeo-preview-matched-comparison-v1',status='running',protocol=protocol(),records={},runs=[])
    save(root/'registration.json',report)
    for mode in MODES:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'capture','--mode',mode,'--output',str(root/mode),
            '--examples',str(args.examples.resolve()),'--runtime-src',str(args.runtime_src.resolve()),'--prepared',str(args.prepared.resolve()),
            '--library',str(args.library.resolve()),'--shaders',str(args.shaders.resolve()),'--device-index',str(args.device_index)]
        if args.sapien_device:command+=['--sapien-device',args.sapien_device]
        for extra in args.extra_site:command+=['--extra-site',str(extra.resolve())]
        result=subprocess.run(command,capture_output=True,text=True)
        (root/(mode+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        report['runs'].append(dict(mode=mode,command=command,returncode=result.returncode))
        path=root/mode/'report.json'
        report['records'][mode]=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else dict(status='failed',error='worker exited before receipt')
        save(root/'summary.json',report)
    try:report.update(comparison=assemble(root,report['records']),status='complete')
    except BaseException as error:report.update(status='partial_or_unsupported',error=repr(error),scope='all successful original captures retained; no substituted reference')
    save(root/'summary.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('run','capture'):
        p=sub.add_parser(command)
        for name in ('examples','runtime-src','prepared','library','shaders','output'):p.add_argument('--'+name,type=Path,required=True)
        p.add_argument('--device-index',type=int,default=1);p.add_argument('--sapien-device',default=None)
        p.add_argument('--extra-site',type=Path,action='append',default=[])
        if command=='capture':p.add_argument('--mode',choices=MODES,required=True)
    args=parser.parse_args()
    for extra in args.extra_site:
        path=extra.resolve(strict=True)
        if not path.is_dir():raise ValueError('extra site must be an existing local directory')
        sys.path.append(str(path)) # append only: never shadow the environment NumPy/SAPIEN
    value=run(args) if args.command=='run' else capture(args)
    return 0 if value['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())