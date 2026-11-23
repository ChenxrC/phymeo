"""Independent finite-mirror chart for one-bounce near-field reflection.

The reference reflects finite source rectangles across y=0, projects their
virtual images, and independently intersects the reflected camera ray. It
never reads renderer hit IDs to choose its ROI. World/epoch changes below are
application-supplied replay identities: this example performs no physics reset.
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

WIDTH, HEIGHT, FOCAL = 320, 240, 256.
SENSORS = ('FORWARD_DEPTH', 'INSTANCE_ID', 'VALID')
MARGIN = 16


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def project(point):
    x, y, z = point
    depth = y + 2
    if not all(math.isfinite(v) for v in point) or depth <= 0:
        raise ValueError('point must be finite and in front of the fixed camera')
    return WIDTH / 2 + FOCAL * x / depth, HEIGHT / 2 - FOCAL * (z - 1) / depth


def fixture(state):
    if state not in ('base', 'moved', 'occluded', 'miss'):
        raise ValueError('unknown predeclared chart state')
    scale_near, scale_far = 2.6 / 2.75, 3. / 2.75
    targets = [
        dict(id=1, name='red', center=(-.45, -.75, 1.8), width=.30, height=.24),
        dict(id=2, name='green', center=(-.45 * scale_near, -.6, 1 + .8 * scale_near), width=.14, height=.12),
        dict(id=3, name='blue', center=(-.45 * scale_far, -1., 1 + .8 * scale_far), width=.30 * scale_far, height=.24 * scale_far),
    ]
    for target in targets:
        x, y, z = target['center']
        if state == 'moved' and target['id'] == 1:
            x = .35
        if state == 'miss' or (state != 'occluded' and target['id'] != 1):
            x = 10. + target['id']
        target['center'] = tuple(map(f32, (x, y, z)))
        target['width'] = f32(target['width'])
        target['height'] = f32(target['height'])
    return targets


def corners(target):
    x, y, z = target['center']
    u, v = f32(target['width'] / 2), f32(target['height'] / 2)
    return tuple((f32(x + dx), y, f32(z + dz)) for dx, dz in ((-u, -v), (u, -v), (u, v), (-u, v)))


def mirror_contains(x, y):
    # Pixel centers, y=0 at forward depth 2; finite mesh z=.1..1.9.
    mx, mz = (x + .5 - WIDTH / 2) * 2 / FOCAL, 1 - (y + .5 - HEIGHT / 2) * 2 / FOCAL
    return -1 < mx < 1 and f32(.1) < mz < f32(1.9)


def ray_identity(x, y, targets):
    if not mirror_contains(x, y):
        return 255
    dx, dz = (x + .5 - WIDTH / 2) / FOCAL, -(y + .5 - HEIGHT / 2) / FOCAL
    # Primary direction=(dx,1,dz); reflected=(dx,-1,dz).
    mx, mz = 2 * dx, 1 + 2 * dz
    hits = []
    for target in targets:
        distance = -target['center'][1]
        px, pz = mx + distance * dx, mz + distance * dz
        lo, _, hi, _ = corners(target)
        if distance > 0 and lo[0] <= px < hi[0] and lo[2] <= pz < hi[2]:
            hits.append((distance, target['id']))
    return min(hits)[1] if hits else 0


def expected_labels(state):
    # Independent virtual-image rectangle projection; nearest virtual depth wins.
    projected = []
    for target in fixture(state):
        pixels = [project((p[0], -p[1], p[2])) for p in corners(target)]
        projected.append((-target['center'][1], target['id'],
            min(p[0] for p in pixels), max(p[0] for p in pixels),
            min(p[1] for p in pixels), max(p[1] for p in pixels)))
    projected.sort()
    result = bytearray()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if not mirror_contains(x, y):
                result.append(255)
                continue
            result.append(next((identity for _, identity, x0, x1, y0, y1 in projected
                if x0 <= x + .5 < x1 and y0 < y + .5 <= y1), 0))
    return bytes(result)


def allowed_labels(labels, x, y):
    return {labels[yy * WIDTH + xx] for yy in range(max(0, y-1), min(HEIGHT, y+2))
            for xx in range(max(0, x-1), min(WIDTH, x+2))}


def evaluate_rgb(rgb, labels):
    if len(rgb) != WIDTH * HEIGHT * 3 or len(labels) != WIDTH * HEIGHT:
        raise ValueError('exact chart extent required')
    failures, interiors, edges = [], 0, 0
    counts = {str(i): 0 for i in range(4)}
    for y in range(HEIGHT):
        for x in range(WIDTH):
            index = y * WIDTH + x
            if labels[index] == 255:
                continue
            color = rgb[3*index:3*index+3]
            channel = max(range(3), key=lambda c: color[c])
            observed = channel + 1 if color[channel] - max(color[c] for c in range(3) if c != channel) >= MARGIN else 0
            allowed = allowed_labels(labels, x, y)
            if len(allowed) == 1:
                interiors += 1
            else:
                edges += 1
            counts[str(observed)] += 1
            if observed not in allowed:
                failures.append((x, y, labels[index], observed, list(color)))
    return dict(passed=not failures, mismatched_pixels=len(failures), first_failures=failures[:12],
                interior_pixels=interiors, one_pixel_boundary_band_pixels=edges, observed_identity_counts=counts,
                dominant_channel_minimum_rgb8=MARGIN, boundary_tolerance_px=1)


def miss_equal(a, b, labels):
    if len(a) != WIDTH * HEIGHT * 3 or len(b) != len(a):
        raise ValueError('exact chart RGB extent required')
    for y in range(HEIGHT):
        for x in range(WIDTH):
            i = y * WIDTH + x
            if labels[i] == 0 and allowed_labels(labels, x, y) == {0} and a[3*i:3*i+3] != b[3*i:3*i+3]:
                return False
    return True


def sensor_equal(a, b):
    return all(a[name] == b[name] for name in SENSORS)


def replay_checks(readings):
    get = lambda i: readings[('single_bounce', i)]
    return dict(b_matches_independent_moved=get(4) == get(1), a_reset_matches_base=get(5) == get(0),
                b_survives_a_reset=get(6) == get(4), a_and_b_are_distinct=get(5)['RGB'] != get(6)['RGB'])


def offscreen_check(planes, labels):
    if len(planes['INSTANCE_ID']) != WIDTH * HEIGHT * 4 or len(planes['RGB']) != WIDTH * HEIGHT * 3:
        raise ValueError('full primary ID and RGB planes required')
    primary_red = sum(value == 1 for (value,) in struct.iter_unpack('<I', planes['INSTANCE_ID']))
    reflected_red = 0
    for y in range(HEIGHT):
        for x in range(WIDTH):
            i = y * WIDTH + x
            if labels[i] == 1 and allowed_labels(labels, x, y) == {1}:
                r, g, b = planes['RGB'][3*i:3*i+3]
                reflected_red += r - max(g, b) >= MARGIN
    return dict(passed=primary_red == 0 and reflected_red > 0,
                primary_red_pixels=primary_red, observed_reflected_red_interior_pixels=reflected_red)

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def closed(diagnostics):
    device = diagnostics['device']
    telemetry = device.get('resident_telemetry')
    return (device.get('closed') is True and device.get('validation_errors') == 0 and
        all(device.get(key) == 0 for key in ('live_contexts', 'live_frames', 'live_resources')) and
        (telemetry is not None and telemetry.get('final_after_destroy') is True and
         telemetry['device_local_bytes_current'] == telemetry['host_visible_bytes_current'] == 0 and
         telemetry['allocation_count'] == telemetry['free_count']))


def run_experiment(library, output, *, shaders=None, device_index=1, validation=False):
    import numpy as np
    from PIL import Image, ImageDraw
    from _panda_scene import look_at, write_mesh
    from phymeo_rendering.assets import Material
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.cameras import Camera
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    from phymeo_rendering.scene import DirectionalLight, FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene
    from phymeo_rendering.studio import StudioLighting
    output, library = Path(output), Path(library)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    report = dict(schema='phymeo-near-reflection-chart-v1', status='running', physics_steps=0,
        registration=dict(extent=[WIDTH, HEIGHT], focal_px=FOCAL, boundary_tolerance_px=1,
            dominant_channel_minimum_rgb8=MARGIN, boundary_rule='Chebyshev distance <=1 from independently predicted identity boundary; all remaining mirror pixels exact identity',
            oracle='virtual finite rectangle image and independent reflected camera ray; authored FP32 coordinates',
            mirror_y=0., mirror_x=[-1., 1.], mirror_z=[.1, 1.9], metallic=1., roughness=.04,
            environment_radiance=.05, panel_strength=0., directional_travel=[0., -1., 0.], directional_rgb=[2., 2., 2.],
            display_transform='clip', exposure=1., msaa_samples=1,
            reset_scope='authored world/epoch key and complete current transforms, not physics reset',
            reset_sequence='B=moved matches independent moved; A reset=base; B reread=moved and distinct from A',
            offscreen_observation='full primary ID plane contains no red target; predicted strict mirror interior contains measured red',
            cleanup_requires='actual final resident telemetry, zero live allocation bytes and balanced allocation/free counts',
            no_tuning='fixed geometry, identity margin and one-pixel boundary band before device execution'),
        library=dict(path=str(library.resolve()), sha256=sha256(library.read_bytes()).hexdigest()),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        public_source=dict(file=inspect.getfile(NativeBackend), sha256=sha256(Path(inspect.getfile(NativeBackend)).read_bytes()).hexdigest()),
        fixtures={s:fixture(s) for s in ('base','moved','occluded','miss')}, captures={}, cleanup=[],
        acceptance={k:False for k in ('analytic_boundary_and_identity','offscreen_source','nearest_occluder','motion',
            'miss_equals_environment','geometry_bytes','request_combinations','world_epoch_replay','reopen','capability','cleanup')})
    dump(output/'registration.json', report)
    geometry = output/'geometry'
    geometry.mkdir()
    mirror = write_mesh(geometry, 'mirror', ((-1.,0.,.1),(1.,0.,.1),(1.,0.,1.9),(-1.,0.,1.9)),
                        ((0.,-1.,0.),)*4, ((0,1,2),(0,2,3)))
    meshes = [mirror]
    for target in fixture('base'):
        u,v = f32(target['width']/2),f32(target['height']/2)
        meshes.append(write_mesh(geometry,target['name'],((-u,0.,-v),(u,0.,-v),(u,0.,v),(-u,0.,v)),
                                 ((0.,-1.,0.),)*4,((0,1,2),(0,2,3))))
    materials = (Material('mirror',(1.,1.,1.),1.,.04), Material('red',(.9,.02,.02),0.,1.),
                 Material('green',(.02,.9,.02),0.,1.), Material('blue',(.02,.02,.9),0.,1.))
    scene = VisualScene('near-reflection-chart',tuple(meshes),materials,
        (Instance(10,'mirror','mirror','mirror'),)+tuple(Instance(t['id'],t['name'],t['name'],t['name']) for t in fixture('base')),
        (DirectionalLight((0.,-1.,0.),(2.,2.,2.)),))
    report['scene'] = asdict(scene)
    report['geometry_sha256'] = {p.name:sha256(p.read_bytes()).hexdigest() for p in geometry.glob('*.gltf')}
    schedule = [('base','world-a',0),('moved','world-a',0),('occluded','world-a',0),('miss','world-a',0),
                ('moved','world-b',0),('base','world-a',1),('moved','world-b',0)]
    readings = {}
    request_ok = True
    def capture(owner, state, world, epoch, tick, channels):
        frame = SceneFrame(FrameKey(scene.key,world,epoch,tick,tick,'authored'),tick/30.,
            (NodePose('mirror',Pose()),)+tuple(NodePose(t['name'],Pose(t['center'])) for t in fixture(state)))
        camera = Camera('mirror-chart',frame.key,look_at((0.,-2.,1.),(0.,0.,1.),(0.,0.,1.)),
                        WIDTH,HEIGHT,FOCAL,FOCAL,WIDTH/2,HEIGHT/2,.01,100.)
        start = perf_counter()
        result = owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,channels))
        wall = perf_counter()-start
        if result.frame_key != frame.key or result.camera != camera or result.time_s != frame.time_s:
            raise ValueError('published frame, camera or time differs from submitted source')
        return {p.channel.name:p.data for p in result.planes},frame,camera,wall
    def backend(mode):
        return NativeBackend(library_path=library,shader_directory=shaders,device_index=device_index,resident=True,
            validation=validation,shadow_mode='ray_query',reflection_mode=mode,msaa_samples=1,
            studio=StudioLighting(panel_strength=0.,environment_radiance=.05))
    try:
        for mode in ('environment','single_bounce'):
            owner = backend(mode).open_scene(scene)
            try:
                for tick,(state,world,epoch) in enumerate(schedule):
                    label = mode+'-'+str(tick)+'-'+state
                    planes,frame,camera,wall = capture(owner,state,world,epoch,tick,tuple(Channel))
                    readings[(mode,tick)] = planes
                    image = output/(label+'.png')
                    Image.fromarray(np.frombuffer(planes['RGB'],np.uint8).reshape(HEIGHT,WIDTH,3)).save(image)
                    for channel in SENSORS:
                        (output/(label+'.'+channel+'.raw')).write_bytes(planes[channel])
                    labels = expected_labels(state)
                    (output/(label+'.expected-identities.u8')).write_bytes(labels)
                    analysis = evaluate_rgb(planes['RGB'],labels) if mode == 'single_bounce' else None
                    entry = dict(state=state,frame=asdict(frame),camera=asdict(camera),wall_ms=wall*1000,
                        image=image.name,png_sha256=sha256(image.read_bytes()).hexdigest(),
                        plane_sha256={k:sha256(v).hexdigest() for k,v in planes.items()},analytic=analysis,
                        expected_counts={str(i):labels.count(i) for i in (0,1,2,3,255)})
                    report['captures'][label] = entry
                    if tick == 0:
                        rgb,*_ = capture(owner,state,world,epoch,tick,(Channel.RGB,))
                        sensors,*_ = capture(owner,state,world,epoch,tick,(Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID))
                        request_ok &= rgb['RGB'] == planes['RGB'] and sensor_equal(sensors,planes)
                    dump(output/'summary.json',report)
                    print(json.dumps(dict(capture=label,analytic=analysis)),flush=True)
            finally:
                try:
                    owner.close()
                finally:
                    diag = owner.diagnostics()
                    report['cleanup'].append(dict(mode=mode,passed=closed(diag),diagnostics=diag))
                    dump(output/'summary.json',report)
        owner = backend('single_bounce').open_scene(scene)
        try:
            planes,*_ = capture(owner,'base','world-a',0,0,tuple(Channel))
            report['acceptance']['reopen'] = planes == readings[('single_bounce',0)]
        finally:
            try:
                owner.close()
            finally:
                diag = owner.diagnostics()
                report['cleanup'].append(dict(mode='single_bounce-reopen',passed=closed(diag),diagnostics=diag))
        analyses = [c['analytic'] for c in report['captures'].values() if c['analytic'] is not None]
        report['miss_checks'] = {str(i):miss_equal(readings[('environment',i)]['RGB'],readings[('single_bounce',i)]['RGB'],expected_labels(s)) for i,(s,_,_) in enumerate(schedule)}
        report['sensor_checks'] = {str(i):sensor_equal(readings[('environment',i)],readings[('single_bounce',i)]) for i in range(len(schedule))}
        report['replay_checks'] = replay_checks(readings)
        report['offscreen_observation'] = offscreen_check(readings[('single_bounce',0)],expected_labels('base'))
        enabled_diags = [c['diagnostics'] for c in report['cleanup'] if c['mode'].startswith('single_bounce')]
        capability = all(d.get('reflection',{}).get('mode') == 'single_bounce' and
            d['device'].get('resident_reflection_info',{}).get('enabled') == 1 for d in enabled_diags)
        report['acceptance'].update(analytic_boundary_and_identity=all(a['passed'] for a in analyses),
            offscreen_source=all(project(p)[1]<0 for p in corners(fixture('base')[0])) and report['offscreen_observation']['passed'] and analyses[0]['passed'],
            nearest_occluder=report['captures']['single_bounce-2-occluded']['analytic']['passed'],
            motion=report['captures']['single_bounce-1-moved']['analytic']['passed'] and readings[('single_bounce',0)]['RGB'] != readings[('single_bounce',1)]['RGB'],
            miss_equals_environment=all(report['miss_checks'].values()), geometry_bytes=all(report['sensor_checks'].values()),
            request_combinations=request_ok, world_epoch_replay=all(report['replay_checks'].values()),capability=capability,
            cleanup=all(c['passed'] for c in report['cleanup']))
        sheet = Image.new('RGB',(WIDTH*2,(HEIGHT+20)*4),(20,22,26))
        draw = ImageDraw.Draw(sheet)
        for row in range(4):
            for col,mode in enumerate(('environment','single_bounce')):
                label = mode+'-'+str(row)+'-'+schedule[row][0]
                draw.text((col*WIDTH+5,row*(HEIGHT+20)+3),label,fill=(255,255,255))
                with Image.open(output/(label+'.png')) as image:
                    sheet.paste(image,(col*WIDTH,row*(HEIGHT+20)+20))
        sheet.save(output/'comparison.png')
        report['status'] = 'passed' if all(report['acceptance'].values()) else 'measured-not-passed'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_wall_s'] = perf_counter()-started
        report['imported_sources'] = {name:dict(file=m.__file__,sha256=sha256(Path(m.__file__).read_bytes()).hexdigest())
            for name,m in tuple(sys.modules.items()) if name.startswith('phymeo_rendering') and getattr(m,'__file__',None) and Path(m.__file__).is_file()}
        dump(output/'summary.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--shaders',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device-index',type=int,default=1)
    parser.add_argument('--validation',action='store_true')
    args = parser.parse_args()
    report = run_experiment(args.library,args.output,shaders=args.shaders,device_index=args.device_index,validation=args.validation)
    print(json.dumps(dict(status=report['status'],acceptance=report['acceptance'])))
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

