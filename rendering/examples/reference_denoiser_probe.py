"""Bounded capability probe for the installed SAPIEN camera denoiser.

One explicit mode per process. This intentionally uses the SDK camera attachment
API on the existing scene adapter; it does not change the product's none-only
contract. Run only in a granted device slot. No denoiser is substituted.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def registration():
    return json.loads((ROOT/'tests/fixtures/denoiser_probe/registration.json').read_text(encoding='utf-8-sig'))


@contextmanager
def sdk_camera(delegate, public, mode, spp, report):
    from phymeo_rendering.backends._sapien_session import _CAMERA_DEFAULTS_LOCK, _ENTITY_CAMERA_IN_PUBLIC
    from phymeo_rendering.scene import compose_pose
    render = delegate.sapien.render
    entity = delegate.sapien.Entity()
    camera = render.RenderCameraComponent(public.width,public.height,str(delegate.shader_directory))
    entity.add_component(camera)
    camera.set_perspective_parameters(public.near_m,public.far_m,public.fx_px,public.fy_px,
                                     public.cx_px,public.cy_px,0.)
    entity.pose = delegate._pose(compose_pose(public.world_pose,_ENTITY_CAMERA_IN_PUBLIC))
    for key,value in (('spp',spp),('maxDepth',delegate.config.max_bounces),('aperture',0.),
                      ('focusPlane',1.),('russianRoulette',0)):
        camera.set_property(key,value)
    attached = False
    try:
        with _CAMERA_DEFAULTS_LOCK:
            prior = render.get_ray_tracing_denoiser()
            msaa = render.get_msaa()
            report['global_before'] = prior
            try:
                render.set_msaa(1)
                render.set_ray_tracing_denoiser(mode)
                report['global_requested_readback'] = render.get_ray_tracing_denoiser()
                if report['global_requested_readback'] != mode:
                    raise RuntimeError('requested SDK denoiser did not roundtrip')
                delegate.scene.add_entity(entity)
                attached = True
            finally:
                render.set_msaa(msaa)
                render.set_ray_tracing_denoiser(prior)
                report['global_restored'] = render.get_ray_tracing_denoiser()
        yield camera
    finally:
        if attached:
            delegate.scene.remove_entity(entity)


def run(args):
    # Appending a pre-existing Pillow installation must not override environment NumPy.
    if args.pillow_site:
        sys.path.append(str(args.pillow_site.resolve()))
    import numpy as np
    from PIL import Image
    from panda_realtime_film import load_episode
    from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig, SapienStudio
    from phymeo_rendering.backends._sapien_pixels import encode_linear_image
    from phymeo_rendering.cameras import Camera
    from phymeo_rendering.environment import RadianceEnvironment
    from phymeo_rendering.pipeline import RenderPath
    from phymeo_rendering.scene import Pose
    from dataclasses import replace
    args.output.mkdir(parents=True,exist_ok=False)
    registered = registration()
    report = dict(schema='phymeo-reference-denoiser-probe-v1',status='initializing',mode=args.mode,
        stage='source',registration=registered,probe_sha256=digest(__file__),
        scope='actual SDK camera capability probe, not public product denoiser support',physics_steps=0)
    report_path = args.output/'report.json'
    save(report_path,report)
    owner = None
    began = perf_counter()
    try:
        if digest(args.scout) != registered['scout_sha256']:
            raise ValueError('scout receipt changed from registration')
        scout = json.loads(args.scout.read_text(encoding='utf-8'))
        if scout['size'] != [1920,1080] or scout['source_indices'] != [1972] or scout['source_times_s'] != [1.233125]:
            raise ValueError('foreign source/camera reference')
        if scout['scene_radiance_unit_scale'] != .125 or scout['display_exposure'] != 8.:
            raise ValueError('foreign source radiance/exposure')
        if digest(ROOT/'tests/fixtures/panda_film/trace.json') != scout['source_excerpt_sha256']:
            raise ValueError('episode excerpt differs from original scout')
        scene,frames = load_episode(args.output/'geometry',ROOT/'tests/fixtures/panda_film/trace.json',
            photographic=scout['photographic_worktop'],local_light_scale=.125,
            surface_roughness=scout['scanned_linear_roughness'])
        if frames[36].key.revision != 1972 or frames[36].time_s != 1.233125:
            raise ValueError('episode does not reproduce scout state')
        frame = frames[args.frame]
        source_camera = scout['cameras'][0]
        fx,fy,cx,cy = source_camera['intrinsics']
        camera = Camera(source_camera['id'],frame.key,
            Pose(tuple(source_camera['position_m']),tuple(source_camera['orientation_xyzw'])),
            480,270,fx/4,fy/4,cx/4,cy/4,*source_camera['clip_m'])
        source_env = scout['backend']['studio']['environment_map']
        environment = RadianceEnvironment((ROOT/'examples/materials/university_workshop_1k.hdr').as_uri(),
            source_env['sha256'],strength=source_env['strength'],yaw_degrees=source_env['yaw_degrees'],
            face_size=source_env['face_size'])
        studio = replace(SapienStudio(center_m=(.18,0,.42),scale_m=.85),environment_map=environment,panel_strength=.125)
        config = SapienConfig(path=RenderPath.PATH_TRACE,device=args.device,samples_per_pixel=16,
            max_bounces=scout['backend']['max_bounces'],studio=studio,exposure=8.,display_transform='pbr-neutral')
        report.update(camera=asdict(camera),frame=asdict(frame),scout_sha256=digest(args.scout),
            scene=asdict(scene),stage='owner_create',source_excerpt_sha256=digest(ROOT/'tests/fixtures/panda_film/trace.json'))
        save(report_path,report)
        start = perf_counter(); owner = SapienBackend(config).open_scene(scene)
        report['owner_open_ms'] = 1000*(perf_counter()-start)
        delegate = owner._delegate
        report['sdk_version'] = delegate.sapien.__version__
        report['numpy_path'] = np.__file__
        report['sdk_package_path'] = delegate.sapien.__file__
        sdk = Path(delegate.sapien.__file__).parent
        report['installed_binary_sha256'] = {name:digest(sdk/name) for name in
            ('svulkan2.dll','OpenImageDenoise.dll','OpenImageDenoise_core.dll') if (sdk/name).is_file()}
        report['product_owner_diagnostics'] = owner.diagnostics()
        report['diagnostic_note'] = 'owner diagnostics describe none-only public configuration; actual experimental camera mode is separately requested/read back below'
        points,poses,scale = delegate._world_vertices(frame)
        delegate._admit_pt(points,camera,scale)
        for entity,pose in poses: entity.pose = pose
        report['stage'] = 'camera_attach'
        save(report_path,report)
        start = perf_counter()
        with sdk_camera(delegate,camera,args.mode,16,report) as native:
            report['camera_attach_ms'] = 1000*(perf_counter()-start)
            report['stage'] = 'capture_and_readback'; save(report_path,report)
            start = perf_counter(); delegate.scene.update_render(); native.take_picture()
            names = native.get_picture_names()
            report['picture_names'] = names
            required = registered['capability']['required_targets']
            missing = sorted(set(required)-set(names))
            if missing: raise RuntimeError('SDK camera lacks required raw/guide targets: '+','.join(missing))
            arrays = {name:np.array(native.get_picture(name),copy=True) for name in required}
            report['capture_and_all_readbacks_ms'] = 1000*(perf_counter()-start)
            for name,array in arrays.items():
                if array.shape != (270,480,4) or not np.isfinite(array).all():
                    raise ValueError('invalid SDK target '+name)
                if name != 'Segmentation' and array.dtype != np.float32:
                    raise ValueError('SDK HDR/guide is not actual FP32 '+name)
            report['targets'] = {name:dict(shape=list(array.shape),dtype=str(array.dtype),
                sha256=sha256(array.tobytes()).hexdigest(),minimum=float(array.min()),maximum=float(array.max()))
                for name,array in arrays.items()}
            raw,filtered = arrays['Radiance'][...,:3],arrays['HdrColor'][...,:3]
            report.update(raw_equals_hdr=bool(np.array_equal(raw,filtered)),
                hdr_difference_max=float(np.abs(raw.astype(float)-filtered.astype(float)).max()),
                hdr_changed_components=int(np.count_nonzero(raw!=filtered)),
                raw_after_exposure_max=float(raw.max()*8.),
                render_id_to_public_id={str(k):v for k,v in delegate.ids.items()})
            start = perf_counter()
            np.savez_compressed(args.output/'targets.npz',**arrays)
            for name in ('Radiance','HdrColor'):
                rgb = encode_linear_image(arrays[name][...,:3],exposure=8.,display_transform='pbr-neutral')
                Image.frombytes('RGB',(480,270),rgb).save(args.output/(name+'.png'))
            report['archive_and_display_ms'] = 1000*(perf_counter()-start)
        if args.mode == 'none' and not report['raw_equals_hdr']:
            raise RuntimeError('none camera does not expose equal raw and HDR targets')
        report['status'] = ('raw_baseline_captured' if args.mode=='none' else
                            'filter_difference_observed_pending_raw_binding' if not report['raw_equals_hdr'] else
                            'no_observable_denoising')
    except BaseException as error:
        report.update(status='unsupported_or_failed',error_type=type(error).__name__,error=str(error))
    finally:
        if owner is not None:
            try:
                owner.close(); report['closed_owner_diagnostics'] = owner.diagnostics()
            except BaseException as error:
                report.update(status='cleanup_failed',cleanup_error=repr(error))
        report['total_wall_ms'] = 1000*(perf_counter()-began)
        save(report_path,report)
    print(report['status'],flush=True)
    return 0 if report['status'] in ('raw_baseline_captured','filter_difference_observed_pending_raw_binding') else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scout',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('none','oidn','optix'),required=True)
    parser.add_argument('--frame',type=int,choices=(20,36),default=36)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--pillow-site',type=Path)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
