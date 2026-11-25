"""Native strict-sensor regressions for the original ten-object room and two shared Pandas.

Run one case per process with --case ten or --case robots. The thirty-object
film has its own native acceptance entry, film_home_collection.py --preview.
"""

import argparse
from contextlib import closing
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from _home_film_quality import _capture
from _home_showcase_room import ROOM_EXPOSURE, room_camera, room_lighting
from _panda_scene import box, cameras as panda_cameras, look_at, pose_matrix
from film_home_showcase import FPS, SIZE, assemble
from phymeo_rendering import StudioRenderer
from phymeo_rendering.adapters.panda_replay import PandaReplaySource
from phymeo_rendering.assets import Material
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.panda_visuals import panda_visual_assets
from phymeo_rendering.scene import DirectionalLight, FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene, compose_pose


ROOT = Path(__file__).resolve().parents[1]
TEN_RECEIPT = ROOT.parent/'artifacts/home-showcase-20261125/quality-v2/final/film.json'
PLANES = (Channel.RGB, Channel.FORWARD_DEPTH, Channel.INSTANCE_ID, Channel.VALID)


def capture_set(renderer, subject, views, inspect):
    captures = []
    with closing(renderer.open_scene(subject)) as session:
        for frame, camera in views:
            data = _capture(session, frame, camera, PLANES)
            dims = camera.height, camera.width
            ids = np.frombuffer(data[Channel.INSTANCE_ID], '<u4').reshape(dims)
            depth = np.frombuffer(data[Channel.FORWARD_DEPTH], '<f4').reshape(dims)
            valid = np.frombuffer(data[Channel.VALID], 'u1').reshape(dims)
            rgb = np.frombuffer(data[Channel.RGB], 'u1').reshape(*dims, 3)
            hit = ids != 0
            if (not hit.any() or not np.array_equal(valid, hit)
                    or not np.all(np.isposinf(depth[~hit])) or not np.all(np.isfinite(depth[hit]))
                    or np.any(depth[hit] < camera.near_m) or np.any(depth[hit] > camera.far_m)):
                raise AssertionError('depth, instance ownership and validity disagree')
            captures.append(dict(camera=camera.id, revision=frame.key.revision,
                                 hashes={ch.value: sha256(data[ch]).hexdigest() for ch in PLANES},
                                 measurements=inspect(frame, camera, ids, depth, rgb)))
        diagnostics = session.diagnostics()
        device = diagnostics['device']
        if device['validation_errors']:
            raise AssertionError(f"Vulkan validation reported {device['validation_errors']} errors")
    return dict(captures=captures, triangles=diagnostics['triangles'], device=device['device_name'])


def parity(renderer, scene, views, cache, inspect):
    native = capture_set(renderer, scene, views, inspect)
    # Omitting usage must select strict sensor semantics for all camera roles.
    prepared = renderer.prepare_scene(scene, cameras=views, cache_dir=cache)
    report = prepared.report
    if (report['usage'] != 'sensor' or report['adapted'] or prepared.scene != scene
            or report['geometry']['before'] != report['geometry']['after']):
        raise AssertionError('an in-budget sensor scene was changed')
    bound = [prepared.bind_view(frame, camera) for frame, camera in views]
    if bound != views:
        raise AssertionError('strict preparation changed exposure identity or calibrated pose')
    selected = capture_set(renderer, prepared, bound, inspect)
    if native != selected:
        for reference, actual in zip(native['captures'], selected['captures'], strict=True):
            if reference != actual:
                raise AssertionError({'original': reference, 'prepared': actual})
        raise AssertionError('scene triangle count or physical device changed')
    return dict(preparation=report, original=native, prepared=selected)


def ten(args, options):
    if not args.assets:
        raise ValueError('--assets is required for the ten-object case')
    scene, poses, ids, sources = assemble(args.assets.resolve(), args.cache/'ten')
    archive_bytes = args.ten_receipt.read_bytes()
    archive = json.loads(archive_bytes)
    if set(sources) != set(archive['sources']) or len(sources) != 10:
        raise AssertionError('the archived ten categories differ')
    geometry = {}
    for name, current in sources.items():
        prior = archive['sources'][name]
        a, b = current['import_details'], prior['import_details']
        if (a['source_sha256'] != b['source_sha256'] or current['source'] != prior['source']
                or tuple(current['position_m']) != tuple(prior['position_m'])
                or a['parts'] != b['parts']):
            raise AssertionError(f'original geometry, asset or placement changed for {name}')
        if any(row['retained_triangles'] != row['source_triangles'] - row['zero_area_removed'] for row in a['parts']):
            raise AssertionError(f'{name} omitted a non-degenerate source triangle')
        geometry[name] = dict(glb_sha256=a['source_sha256'], triangles=sum(row['retained_triangles'] for row in a['parts']))
    views = []
    for i in (0, 345, 651):
        key = FrameKey(scene.key, 'ten-native-parity', 0, i, i, 'authored')
        frame = SceneFrame(key, i/FPS, poses)
        views.append((frame, room_camera(key, frame.time_s, *SIZE)))

    def inspect(frame, camera, labels, depth, rgb):
        counts = {name: int(np.isin(labels, numbers).sum()) for name, numbers in ids.items()}
        if frame.key.revision == 0 and min(counts.values()) <= 0:
            raise AssertionError(f'the complete room misses an establishing-shot subject: {counts}')
        return counts

    renderer = StudioRenderer(**options, studio=room_lighting(), exposure=ROOM_EXPOSURE, panel_samples=64)
    result = parity(renderer, scene, views, args.cache/'prepared-ten', inspect)
    archived_triangles = archive['vulkan']['triangles']
    if result['original']['triangles'] != archived_triangles:
        raise AssertionError(f'the integrated room triangle count differs from the archived {archived_triangles}')
    return dict(**result, sources=geometry, archived_scene_triangles=archived_triangles,
                archive_file=str(args.ten_receipt), archive_sha256=sha256(archive_bytes).hexdigest())


def cube_measurement(camera, pose, size, rgb, labels, depth, owner, color):
    """Compare same-owner interior pixels to ray intersections with the published oriented box."""
    mask = labels == owner
    interior = mask.copy()
    padded = np.pad(mask, 1)
    for y in range(3):
        for x in range(3):
            interior &= padded[y:y+camera.height, x:x+camera.width]
    ys, xs = np.nonzero(interior)
    if not len(xs):
        return None
    rays = np.stack(((xs+.5-camera.cx_px)/camera.fx_px,
                     (camera.cy_px-ys-.5)/camera.fy_px, -np.ones(len(xs))), axis=1)
    transform = np.linalg.inv(pose_matrix(pose)) @ pose_matrix(camera.world_pose)
    origin = transform[:3, 3]
    direction = rays @ transform[:3, :3].T
    half = np.asarray(size)/2
    parallel = np.abs(direction) < 1e-12
    if np.any(parallel & (np.abs(origin) > half)):
        raise AssertionError('a labelled cube pixel misses the analytic parallel slab')
    safe = np.where(parallel, 1., direction)
    first, second = (-half-origin)/safe, (half-origin)/safe
    near = np.where(parallel, -np.inf, np.minimum(first, second)).max(axis=1)
    far = np.where(parallel, np.inf, np.maximum(first, second)).min(axis=1)
    if np.any(near > far) or np.any(near <= 0):
        raise AssertionError('a labelled cube pixel misses its own physical box')
    error = float(np.max(np.abs(depth[interior]-near)))
    median = np.median(rgb[interior], axis=0)
    if error > .001 or np.argmax(median) != color or median[color] - np.partition(median, -2)[-2] < 8:
        raise AssertionError(dict(camera=camera.id, id=owner, max_depth_error_m=error, rgb_median=median.tolist()))
    return dict(interior_pixels=int(len(xs)), max_depth_error_m=error, median_srgb=median.tolist())


def robots(args, options):
    fixture = ROOT/'tests/fixtures/panda_replay'
    replay = PandaReplaySource(VisualScene('panda-source', (), (), ()), fixture/'panda.urdf', fixture/'trace.json')
    records = tuple(replay.frames())
    geometry = replay.provenance(records[0])['geometry']
    dimensions = geometry['cube_full_size_m']
    meshes, materials, geometry_parts = panda_visual_assets(ROOT/'assets/panda')
    model_dir = args.cache/'panda'
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = box(model_dir, 'test-payload', dimensions)
    offsets = (Pose((0., -.46, 0.)), Pose((0., .46, 0.)))
    colors = (Material('test-red', (.78,.045,.025), 0., .65), Material('test-blue', (.028,.042,.78), 0., .65))
    instances = []
    robot_ids = []
    for arm, material in enumerate(colors):
        parts = tuple(replace(part, id=part.id+1000*arm, node_id=f'{arm}/{part.node_id}') for part in geometry_parts)
        instances.extend(parts)
        robot_ids.append([part.id for part in parts])
        instances.append(Instance(1+1000*arm, payload.id, material.id, f'{arm}/cube'))
    scene = VisualScene('two-shared-real-pandas', meshes+(payload,), materials+colors, tuple(instances),
                        (DirectionalLight((.36,.48,-.8), (1.3,1.3,1.3)),
                         DirectionalLight((-.8,0.,-.6), (.9,.9,.9)),
                         DirectionalLight((0.,-.8,-.6), (.7,.7,.7))))
    views = []
    for step, right in enumerate((records[0], records[-1])):
        states = (records[0], right)
        key = FrameKey(scene.key, 'shared-arm-scene', 0, step, step, 'authored')
        nodes = tuple(NodePose(f'{arm}/{node.node_id}', compose_pose(offsets[arm], node.world_pose))
                      for arm, state in enumerate(states) for node in state.nodes)
        frame = SceneFrame(key, right.time_s, nodes)
        overview = Camera('overview', key, look_at((1.77,-.28,1.21),(.15,0.,.43),(0.,0.,1.)),
                          960,720,850.,850.,480.,360.,.005,5.)
        views.append((frame, overview))
        for arm, state in enumerate(states):
            wrist = panda_cameras(state)[1]
            views.append((frame, replace(wrist, id=f'wrist-{arm}', frame_key=key,
                                         world_pose=compose_pose(offsets[arm], wrist.world_pose))))

    def inspect(frame, camera, ids, depth, rgb):
        poses = {node.node_id: node.world_pose for node in frame.nodes}
        counts = [int(np.isin(ids, arm).sum()) for arm in robot_ids]
        samples = [cube_measurement(camera, poses[f'{i}/cube'], dimensions, rgb, ids, depth, 1+1000*i, color)
                   for i, color in enumerate((0,2))]
        required = (0,1) if camera.id == 'overview' else (int(camera.id[-1]),)
        if any(counts[i] < 32 or samples[i] is None or samples[i]['interior_pixels'] < 6 for i in required):
            raise AssertionError(dict(camera=camera.id, robot_pixels=counts, cube_measurements=samples))
        # The same A exposure is repeated while only B advances. Compare A's
        # ownership and depth independently of the moving arm's background.
        own = np.isin(ids, robot_ids[0]+[1])
        return dict(robot_pixels=counts, payloads=samples, arm_a_ids_sha256=sha256(np.where(own,ids,0).tobytes()).hexdigest(),
                    arm_a_depth_sha256=sha256(np.where(own, depth, np.inf).tobytes()).hexdigest())

    renderer = NativeBackend(**options, resident=True)
    result = parity(renderer, scene, views, args.cache/'prepared-robots', inspect)
    rows = result['original']['captures']
    a1, a2 = rows[1]['measurements'], rows[4]['measurements']
    if any(a1[k] != a2[k] for k in ('arm_a_ids_sha256', 'arm_a_depth_sha256')):
        raise AssertionError('moving robot B changed the stationary robot A wrist sensor')
    if all(rows[2]['hashes'][ch.value] == rows[5]['hashes'][ch.value] for ch in (Channel.FORWARD_DEPTH, Channel.INSTANCE_ID)):
        raise AssertionError('robot B did not publish a changing wrist exposure')
    counts = result['preparation']['geometry']['before']
    # Right and left Panda fingers also share surfaces within a single arm.
    if counts['max_instanced_triangles'] < 2*counts['unique_triangles']:
        raise AssertionError('two instances did not reference the same geometry resources')
    return dict(**result, robot_ids=robot_ids, archive_sha256=replay.provenance(records[0])['archive_sha256'],
                right_source_records=[records[0].key.revision, records[-1].key.revision], stationary_left=records[0].key.revision)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('ten','robots'), required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--device-index', type=int, default=1)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON receipt path')
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--ten-receipt', type=Path, default=TEN_RECEIPT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.cache = args.cache.resolve()
    library = args.library.resolve(strict=True)
    options = dict(library_path=library, shader_directory=library.parent, device_index=args.device_index, validation=True)
    audit = (ten if args.case == 'ten' else robots)(args, options)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
    print(f'{args.case} passed: {args.output}', flush=True)
