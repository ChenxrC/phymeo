"""Bounded direct-area-light quadrature experiment; no simulation or image filtering.

Run --library DLL --shaders DIR --baseline-library OLD_DLL --baseline-shaders OLD_DIR
    --assets rendering/assets/panda --device-index 1 --output NEW_DIRECTORY.
The analytic emitter uses a private, recorded NativeStudio ABI fixture; all images
are obtained through the normal checked RenderSession.capture path. Panda uses the
public StudioLighting recipe. The experiment does not expose a new lighting API.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time


def load_registration():
    path = Path(__file__).resolve().parents[1] / 'tests/fixtures/area_light_quadrature/registration.json'
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rectangle_irradiance(x0, x1, y0, y1, height):
    """Integral h²/(x²+y²+h²)² dA from the rectangle boundary divergence."""
    if not all(math.isfinite(v) for v in (x0, x1, y0, y1, height)) or height <= 0 or x0 > x1 or y0 > y1:
        raise ValueError('finite ordered rectangle and positive height required')
    if x0 == x1 or y0 == y1:
        return 0.
    def edge(a, lo, hi):
        r = math.hypot(height, a)
        return a/r * (math.atan(hi/r)-math.atan(lo/r))
    return .5*(edge(x1, y0, y1)-edge(x0, y0, y1)+edge(y1, x0, x1)-edge(y0, x0, x1))


def visible_emitter_bounds(receiver_x, *, bias_m=1e-5, phase_m=0.):
    if not all(math.isfinite(v) for v in (receiver_x, bias_m, phase_m)) or not 0 <= bias_m < .5:
        raise ValueError('invalid receiver or shadow origin')
    fraction = (.5-bias_m)/(1-bias_m)
    cutoff = (phase_m-(1-fraction)*receiver_x)/fraction
    return (min(.5, max(-.5, cutoff)), .5)


def area_coordinates(grid):
    if type(grid) is not int or grid not in (2, 4, 8):
        raise ValueError('grid must be 2, 4 or 8')
    return tuple((2*(i+.5)/grid-1, 2*(j+.5)/grid-1) for j in range(grid) for i in range(grid))


def profile_metrics(actual, reference, *, clear_peak):
    if not actual or len(actual) != len(reference) or not math.isfinite(clear_peak) or clear_peak <= 0:
        raise ValueError('aligned nonempty profiles and positive fixed clear peak required')
    if not all(math.isfinite(v) for v in (*actual, *reference)):
        raise ValueError('nonfinite profile')
    errors = [abs(a-b) for a, b in zip(actual, reference, strict=True)]
    return dict(normalized_mean_absolute_error=sum(errors)/len(errors)/clear_peak,
                normalized_max_absolute_error=max(errors)/clear_peak,
                normalized_max_adjacent_jump=max((abs(b-a) for a,b in zip(actual, actual[1:])), default=0.)/clear_peak)


def error_acceptance(error, baseline):
    if not all(math.isfinite(v) and v >= 0 for v in (error, baseline)):
        raise ValueError('finite nonnegative errors required')
    reduction = 1-error/baseline if baseline > 0 else None
    return dict(absolute_error_passed=error <= .04, relative_improvement=reduction,
                relative_improvement_passed=reduction is not None and reduction >= .5)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _linear(rgb):
    import numpy as np
    s = np.asarray(rgb, dtype=float)/255
    return np.where(s <= .04045, s/12.92, ((s+.055)/1.055)**2.4)


def _reference(xs, order, *, phase=0., clear=False):
    """Independent physical direct BRDF integral; not the shader's MIS estimator.

    Integrate only the analytically visible rectangle. Diffuse Fresnel and GGX
    specular match the declared material, but use no shader area/VNDF samples.
    Arrays are bounded to one receiver point at a time (order <= 128).
    """
    import numpy as np
    nodes, weights = np.polynomial.legendre.leggauss(order)
    h = 1-1e-5
    output, lambert = [], []
    for x in xs:
        lo, hi = (-.5, .5) if clear else visible_emitter_bounds(float(x), phase_m=phase)
        if lo == hi:
            output.append(0.); lambert.append(0.); continue
        u = (lo+hi)/2 + (hi-lo)/2*nodes[:, None]
        v = .5*nodes[None, :]
        dx = np.broadcast_to(u-x, (order, order))
        dy = np.broadcast_to(v, (order, order))
        distance = np.sqrt(dx*dx+dy*dy+h*h)
        lx, ly, nl = dx/distance, dy/distance, h/distance
        view_norm = math.hypot(float(x), .25)
        vx, nv = -x/view_norm, .25/view_norm
        hx, hy, hz = lx+vx, ly, nl+nv
        hn = np.sqrt(hx*hx+hy*hy+hz*hz)
        vh = np.clip((vx*hx+nv*hz)/hn, 0, 1)
        fresnel = .04+.96*(1-vh)**5
        # roughness=1 => alpha²=1, D=1/pi; exact Smith G1=2c/(1+c).
        gv = 2*nv/(1+nv)
        gl = 2*nl/(1+nl)
        brdf_cos = (1-fresnel)*.8*nl/math.pi + fresnel*gv*gl/(4*math.pi*nv)
        measure = h/(distance**3) * ((hi-lo)/2)*.5*weights[:, None]*weights[None, :]
        output.append(float(np.sum(brdf_cos*measure)))
        lambert.append(rectangle_irradiance(lo-x, hi-x, -.5, .5, h)*.8/math.pi)
    return np.array(output), np.array(lambert)


def _crossing(xs, values, threshold):
    for i in range(1, len(xs)):
        if values[i-1] < threshold <= values[i]:
            a = (threshold-values[i-1])/(values[i]-values[i-1])
            return float(xs[i-1]+a*(xs[i]-xs[i-1]))
    return None


def _transition(xs, values, clear):
    normalized = [a/b for a,b in zip(values, clear, strict=True)]
    crossings = [_crossing(xs, normalized, threshold) for threshold in (.1, .5, .9)]
    return dict(x10_m=crossings[0], x50_m=crossings[1], x90_m=crossings[2],
                width_m=None if crossings[0] is None or crossings[2] is None else crossings[2]-crossings[0])


def _backend(args, mode, grid, *, studio=None, baseline=False):
    from phymeo_rendering.backends.native import NativeBackend
    return NativeBackend(library_path=args.baseline_library if baseline else args.library,
        shader_directory=(args.baseline_shaders or args.baseline_library.parent) if baseline else args.shaders,
        device_index=args.device_index, validation=True, resident=True, shadow_mode=mode,
        area_light_grid=grid, studio=studio, exposure=.25 if studio is not None else 1.)


def _capture(owner, frame, camera, output, name):
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    from PIL import Image
    capture = owner.capture(frame, CaptureRequest(camera, RenderPath.RASTER, tuple(Channel)))
    planes = {p.channel: p.data for p in capture.planes}
    Image.frombytes('RGB', (camera.width, camera.height), planes[Channel.RGB]).save(output/(name+'.png'))
    evidence = dict(name=name, camera=asdict(camera), frame=asdict(frame),
                    channels={k.value: _digest(v) for k,v in planes.items()})
    return planes, evidence


def _closed(owner):
    owner.close()
    result = owner.diagnostics()
    device = result['device']
    telemetry = device.get('resident_telemetry')
    if device['validation_errors'] or telemetry is None:
        raise RuntimeError('validation errors or missing real teardown telemetry')
    if telemetry['device_local_bytes_current'] or telemetry['host_visible_bytes_current'] or telemetry['allocation_count'] != telemetry['free_count']:
        raise RuntimeError('native allocations did not return to zero')
    return result


def _panel_fixture(owner, *, backface=False):
    from phymeo_rendering.backends._native_studio import NativeStudio
    import ctypes as ct
    value = NativeStudio()
    value.environment[:] = (0, 0, 0, 1)
    panel = value.panels[0]
    panel.position[:] = (0, 0, 1, 0)
    panel.half_u[:] = (.5, 0, 0, 0)
    panel.half_v[:] = (0, .5 if backface else -.5, 0, 0)
    panel.emission[:] = (1, 1, 1, 0)
    # Experimental authored emitter only: explicit private native fixture, not
    # an unimplemented public arbitrary-light feature or a fake capture.
    owner._delegate.device.set_studio(value)
    return dict(scope='experimental NativeStudio ABI fixture before first capture',
                payload_f32le_sha256=_digest(bytes(value)), bytes=ct.sizeof(value),
                environment=[0, 0, 0], panels=[dict(position=[0,0,1], half_u=list(panel.half_u),
                half_v=list(panel.half_v), emission=[1,1,1]), dict(emission=[0,0,0])])


def analytic(args, output, report):
    import numpy as np
    from _panda_scene import write_mesh
    from phymeo_rendering.assets import Material
    from phymeo_rendering.cameras import Camera
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene
    def quad(name, x0, x1, y0, y1):
        return write_mesh(output, name, [(x0,y0,0),(x1,y0,0),(x1,y1,0),(x0,y1,0)],
                          [(0,0,1)]*4, [(0,1,2),(0,2,3)])
    scene = VisualScene('area-quadrature-analytic',
        (quad('receiver',-1,1,-1,1), quad('blocker',-2,0,-2,2)),
        (Material('matte',(.8,.8,.8),0.,1.),),
        (Instance(1,'receiver','matte','receiver'),Instance(2,'blocker','matte','blocker')))
    report['analytic_scene'] = asdict(scene)
    xs = (np.arange(641)-320)*(.25/128)
    clear, _ = _reference(xs,128,clear=True)
    denominator = float(clear.max())
    references, witnesses = {}, {}
    phases = report['registration']['analytic_scene']['phase_offsets_m']
    for phase in phases:
        high, lambert = _reference(xs,128,phase=phase)
        low, _ = _reference(xs,64,phase=phase)
        discrepancy = float(np.max(np.abs(high-low))/denominator)
        if discrepancy > .0001:
            raise ValueError('independent reference failed preregistered convergence')
        references[phase] = high
        witnesses[str(phase)] = dict(reference_linear=high.tolist(), lambert_geometry_only=lambert.tolist(),
            convergence_max_over_clear_peak=discrepancy, transition=_transition(xs,high,clear))
    report['analytic_reference'] = dict(x_m=xs.tolist(), clear_linear=clear.tolist(),
        clear_peak=denominator, phase=witnesses, integration='FP64 Gauss 64/128, analytically clipped rectangle')
    sensors, baseline_rgb, baseline_metrics = {}, {}, {}
    rows = report.setdefault('analytic_captures', [])
    jobs = [(mode, grid, False) for mode in ('software','ray_query') for grid in (2,4,8)]
    if args.baseline_library:
        jobs += [(mode,2,True) for mode in ('software','ray_query')]
    for mode, grid, is_baseline in jobs:
        owner = _backend(args,mode,grid,baseline=is_baseline).open_scene(scene)
        fixture = _panel_fixture(owner)
        try:
            # Additional controls retain the identical material and camera.
            cases = [(f'phase-{i}', phase, .5) for i,phase in enumerate(phases)]
            cases += [('clear',-3.,.5),('blocked',1.,.5),('beyond-emitter',0.,1.5)]
            clear_rgb = None
            for index,(label, phase, blocker_z) in enumerate(cases):
                frame = SceneFrame(FrameKey(scene.key,'authored',0,index,index,'authored'),float(index),
                    (NodePose('receiver',Pose()),NodePose('blocker',Pose((phase,0,blocker_z)))))
                camera = Camera('profile',frame.key,Pose((0,0,.25)),641,129,128.,128.,320.5,64.5,.005,5.)
                name = f'analytic-{mode}-{grid}-{label}' + ('-baseline' if is_baseline else '')
                report['active_capture'] = dict(name=name,mode=mode,grid=grid,case=label)
                planes, evidence = _capture(owner,frame,camera,output,name)
                if label == 'clear': clear_rgb = planes[Channel.RGB]
                if label == 'blocked' and any(planes[Channel.RGB]):
                    raise AssertionError('fully blocked receiver received direct light')
                if label == 'beyond-emitter' and planes[Channel.RGB] != clear_rgb:
                    raise AssertionError('blocker beyond finite emitter changed direct light')
                reference_sensors = tuple(_digest(planes[c]) for c in (Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID))
                sensor_key = label
                if sensor_key not in sensors: sensors[sensor_key] = reference_sensors
                if sensors[sensor_key] != reference_sensors: raise AssertionError('area grid/traversal changed sensor planes')
                key = (mode,label)
                if grid == 2 and not is_baseline: baseline_rgb[key] = planes[Channel.RGB]
                if is_baseline and planes[Channel.RGB] != baseline_rgb[key]:
                    raise AssertionError('default 2x2 output changed against original shader artifact')
                encoded = np.frombuffer(planes[Channel.RGB],np.uint8).reshape(129,641,3)[64,:,0]
                linear = _linear(encoded)
                expected = references[phase] if index < len(phases) else (np.zeros_like(clear) if label == 'blocked' else clear)
                metrics = profile_metrics(linear.tolist(),expected.tolist(),clear_peak=denominator)
                upper = _linear(np.minimum(encoded.astype(float)+.5,255))-linear
                lower = linear-_linear(np.maximum(encoded.astype(float)-.5,0))
                quantization = float(np.max(np.maximum(upper,lower)))
                evidence.update(mode=mode,grid=grid,original_artifact=is_baseline,fixture=fixture,
                    sensor_equivalent=True, linear_profile=linear.tolist(), metrics=metrics,
                    saturated_samples=int(np.count_nonzero(encoded==255)),
                    quantization_half_code_linear_max=quantization)
                if index < len(phases):
                    transition = _transition(xs,linear,clear)
                    truth = witnesses[str(phase)]['transition']
                    width_error = None if transition['width_m'] is None else abs(transition['width_m']/truth['width_m']-1)
                    evidence.update(transition=transition,transition_width_relative_error=width_error,
                        half_energy_position_error_m=None if transition['x50_m'] is None else abs(transition['x50_m']-truth['x50_m']))
                    if grid == 2 and not is_baseline: baseline_metrics[key] = metrics['normalized_mean_absolute_error']
                    evidence['error_acceptance'] = error_acceptance(metrics['normalized_mean_absolute_error'],baseline_metrics[key])
                    evidence['all_preregistered_gates_passed'] = bool(
                        evidence['error_acceptance']['absolute_error_passed'] and evidence['error_acceptance']['relative_improvement_passed']
                        and metrics['normalized_max_absolute_error'] <= .12 and metrics['normalized_max_adjacent_jump'] <= .15
                        and width_error is not None and width_error <= .1)
                rows.append(evidence)
        finally:
            report.setdefault('sessions',[]).append(_closed(owner))
        _write(output/'summary.json',report)
    # Emission backface is a separate immutable light configuration.
    for mode in ('software','ray_query'):
        owner = _backend(args,mode,8).open_scene(scene)
        fixture = _panel_fixture(owner,backface=True)
        try:
            frame = SceneFrame(FrameKey(scene.key,'backface',0,0,0,'authored'),0.,
                (NodePose('receiver',Pose()),NodePose('blocker',Pose((-3,0,.5)))))
            camera = Camera('profile',frame.key,Pose((0,0,.25)),641,129,128.,128.,320.5,64.5,.005,5.)
            planes,evidence = _capture(owner,frame,camera,output,f'backface-{mode}')
            if any(planes[Channel.RGB]): raise AssertionError('backfacing emitter lit receiver')
            report.setdefault('backface_controls',[]).append(dict(evidence,fixture=fixture,passed=True))
        finally: report['sessions'].append(_closed(owner))
    report['default_output_preserved'] = bool(args.baseline_library)
    report['quality_passed'] = all(row['all_preregistered_gates_passed'] for row in rows
        if row['grid']==8 and 'all_preregistered_gates_passed' in row)


def panda(args, output, report):
    import numpy as np
    from _panda_scene import build_scene, cameras
    from phymeo_rendering.adapters.panda_replay import PandaReplaySource
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    from phymeo_rendering.scene import VisualScene
    from phymeo_rendering.studio import StudioLighting
    fixture = Path(__file__).resolve().parents[1]/'tests/fixtures/panda_replay'
    empty = PandaReplaySource(VisualScene('geometry',(),(),()),fixture/'panda.urdf',fixture/'trace.json')
    seed = empty.provenance(next(empty.frames()))
    frozen = report['registration']['panda']
    for key in ('archive_sha256','model_sha256','excerpt_sha256'):
        if seed[key] != frozen[key]: raise ValueError('Panda source differs from registration')
    scene = build_scene(args.assets,output,seed['geometry']['cube_full_size_m'],seed['geometry']['table_full_size_m'])
    source = PandaReplaySource(scene,fixture/'panda.urdf',fixture/'trace.json')
    frames = tuple(source.frames())
    if [f.key.revision for f in frames] != frozen['source_record_indices']: raise ValueError('foreign Panda poses')
    materials = {m.id:m for m in scene.materials}
    white = [i.id for i in scene.instances if i.node_id in frozen['white_shell_nodes']
             and min(materials[i.material_id].base_color)>=frozen['white_shell_material_min_linear_rgb']]
    if not white: raise ValueError('registered white-shell ROI has no source instances')
    report['panda_source'] = dict(scene=asdict(scene),provenance=[source.provenance(f) for f in frames],
        white_shell_instance_ids=white,table_instance_id=2,
        asset_manifest_sha256=_digest((args.assets/'manifest.json').read_bytes()))
    refs = {}
    for mode in ('software','ray_query'):
        for grid in (2,4,8):
            owner = _backend(args,mode,grid,studio=StudioLighting(panel_strength=1.,environment_radiance=0.)).open_scene(scene)
            try:
                for frame in frames:
                    for camera in cameras(frame):
                        if camera.id not in frozen['cameras']: continue
                        key = (frame.key.revision,camera.id)
                        report['active_capture'] = dict(mode=mode,grid=grid,record=key[0],camera=key[1])
                        planes,evidence = _capture(owner,frame,camera,output,f'panda-{mode}-{grid}-{key[0]}-{key[1]}')
                        sensor_hash = [_digest(planes[c]) for c in (Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID)]
                        rgb = np.frombuffer(planes[Channel.RGB],np.uint8).reshape(camera.height,camera.width,3)
                        ids = np.frombuffer(planes[Channel.INSTANCE_ID],'<u4').reshape(camera.height,camera.width)
                        if key not in refs: refs[key] = (sensor_hash,rgb.copy())
                        if sensor_hash != refs[key][0]: raise AssertionError('Panda sensor changed across grid/traversal')
                        table = ids == 2
                        inner = table.copy()
                        for _ in range(frozen['table_visible_edge_band_pixels']):
                            padded = np.pad(inner,1,constant_values=False)
                            inner = padded[1:-1,1:-1]&padded[:-2,1:-1]&padded[2:,1:-1]&padded[1:-1,:-2]&padded[1:-1,2:]
                        rois = {}
                        for label,mask in (('white_shell',np.isin(ids,white)),('table_visible_edge',table&~inner)):
                            count = int(mask.sum())
                            delta = np.abs(rgb.astype(float)-refs[key][1].astype(float))[mask]
                            # Adjacent differences include true geometry/BRDF changes;
                            # report as image structure, not ground-truth error.
                            adjacent = mask[:,1:] & mask[:,:-1]
                            jump = np.abs(_linear(rgb[:,1:]).mean(2)-_linear(rgb[:,:-1]).mean(2))[adjacent]
                            rois[label] = dict(pixels=count,saturated_pixels=int(np.count_nonzero(np.any(rgb==255,axis=2)&mask)),
                                mean_rgb8_change_vs_software2=None if not count else float(delta.mean()),
                                adjacent_linear_difference_p95=None if not jump.size else float(np.percentile(jump,95)),
                                scope='fixed source-ID ROI image variation, not analytic Panda error')
                        evidence.update(mode=mode,grid=grid,rois=rois,sensor_equivalent=True)
                        report.setdefault('panda_captures',[]).append(evidence)
                frame = frames[1]
                camera = next(c for c in cameras(frame) if c.id=='fixed')
                request = CaptureRequest(camera,RenderPath.RASTER,tuple(Channel))
                for _ in range(3): owner.capture(frame,request)
                before = owner.diagnostics()['device']
                latencies = []
                for _ in range(20):
                    start = time.perf_counter_ns(); owner.capture(frame,request)
                    latencies.append(time.perf_counter_ns()-start)
                after = owner.diagnostics()['device']
                gpu_before,gpu_after = before.get('resident_profile'),after.get('resident_profile')
                gpu_delta = None if not gpu_before or not gpu_after or not gpu_after['timestamp_supported'] else {
                    k:gpu_after[k]-gpu_before[k] for k in ('gpu_render_ns','gpu_copy_ns','gpu_total_ns','timestamp_samples')}
                report.setdefault('cost',[]).append(dict(mode=mode,grid=grid,source_record=frame.key.revision,
                    camera=asdict(camera),warmup=3,completed=20,wall_ns=latencies,
                    wall_p50_ms=float(np.percentile(latencies,50)/1e6),wall_p95_ms=float(np.percentile(latencies,95)/1e6),
                    gpu_delta=gpu_delta,gpu_scope=None if not gpu_after else gpu_after['gpu_scope']))
            finally: report['sessions'].append(_closed(owner))
            _write(output/'summary.json',report)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    record = load_registration()
    report = dict(schema='phymeo-area-light-quadrature-v1',status='running',registration=record,
        registration_sha256=_digest(json.dumps(record,sort_keys=True,separators=(',',':')).encode()),
        runner_sha256=_digest(Path(__file__).read_bytes()),
        physics_steps=0,scope='direct-light sampling experiment, primary raster unchanged; no postfilter or resolution reduction')
    _write(output/'summary.json',report)
    try:
        report['stage'] = 'analytic'
        analytic(args,output,report)
        report['stage'] = 'panda'
        panda(args,output,report)
        report['stage'] = 'complete'
        report['active_capture'] = None
        report['status'] = 'measured-quality-pass' if report['quality_passed'] else 'measured-counterexample'
        if not report['default_output_preserved']: report['status'] = 'measured-default-compatibility-pending'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        _write(output/'summary.json',report)
        raise
    _write(output/'summary.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--shaders',type=Path,required=True)
    parser.add_argument('--baseline-library',type=Path)
    parser.add_argument('--baseline-shaders',type=Path)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--device-index',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    result = run(parser.parse_args())
    print(result['status'],flush=True)
    return 0 if result['status']=='measured-quality-pass' else 2


if __name__ == '__main__':
    raise SystemExit(main())
