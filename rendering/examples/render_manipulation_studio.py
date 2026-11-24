"""Render the same accepted eight-world endpoint under diagnostic and photography lighting."""
import argparse
from contextlib import closing
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from PIL import Image

from _heterogeneous_camera_worlds import WORLD_KEYS, open_worlds
from _r4_archive import _pose
from _manipulation_photo_studio import (
    STUDIO_DISPLAY, STUDIO_EXPOSURE, studio_lighting, style_manipulation_studio,
)
from _studio_camera_gallery import build_comparison, build_gallery
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.device_pose_packing import DeviceVisualBatchBinding, DeviceVisualBinding
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import NodePose, SceneFrame


def studio_variant(world, directory):
    scenes, bindings, nodes = {}, {}, {}
    for key in world.keys:
        scenes[key], set_nodes = style_manipulation_studio(world.scenes[key], world.models[key], directory / str(key))
        nodes[key] = {**world.node_bodies[key], **set_nodes}
        bindings[key] = DeviceVisualBinding(scenes[key], world.workspace, key, nodes[key],
                                            camera_mounts=(world.mounts[key],))
    return replace(world, scenes=scenes, bindings=bindings, node_bodies=nodes,
                   batch_binding=DeviceVisualBatchBinding(tuple(bindings.values())))


def capture(world, snapshot, args, studio, baseline=None, body_rows=None):
    lighting = studio_lighting() if studio else None
    grid = args.area_grid if studio else 2
    direct = body_rows is None
    renderer = NativeBackend(library_path=args.library, shader_directory=args.library.parent,
                             device_index=args.device_index, validation=args.validation, resident=True,
                             shared_cuda_device=args.device if direct else None,
                             shared_camera_capacity=1 if direct else 0,
                             shadow_mode='ray_query', area_light_grid=grid, msaa_samples=4, studio=lighting,
                             exposure=STUDIO_EXPOSURE if studio else 1.,
                             display_transform=STUDIO_DISPLAY if studio else 'clip')
    if direct:
        packed = world.pack(snapshot)
    shots, counters = {}, {}
    for index, key in enumerate(world.keys):
        with closing(renderer.open_scene(world.scenes[key])) as session:
            if direct:
                with session.publish_packed_device_poses(packed, world.workspace, key) as active:
                    views = world.requests(key, session, active, size=(args.width, args.height))
                    batch = session.capture_device_poses_device_batch(active, views, consumer_stream=world.stream)
                batch.wait(world.stream)
                rendered = batch.readback_async(world.stream).result()
            else:
                lo, hi = snapshot.body_offsets[index:index + 2]
                bodies = body_rows[lo:hi]
                nodes = tuple(NodePose(name, _pose(bodies[i, :, 3], bodies[i, :, :3].astype(float)))
                              for name, i in world.node_bodies[key].items())
                frame = SceneFrame(world.bindings[key].frame_key(snapshot),
                                   world.workspace.accepted_frame_time(snapshot.frame, key), nodes)
                cameras = [replace(baseline[key, view].camera, frame_key=frame.key) for view in ('fixed', 'wrist')]
                rendered = [session.capture(frame, CaptureRequest(cam, RenderPath.RASTER, tuple(Channel)))
                            for cam in cameras]
            for shot in rendered:
                shots[key, shot.camera.id] = shot
            diag = session.diagnostics()
            gpu = diag['device']
            env = gpu.get('resident_environment_info') or {}
            ray = gpu['resident_ray_query_info']
            if (diag['msaa']['actual_color_samples'] != 4 or not ray['enabled'] or diag['area_light_grid'] != grid or
                    ray['shadow_captures'] != 2 or gpu['validation_errors']):
                raise RuntimeError('missing 4x MSAA, ray-traced shadows or valid camera capture')
            if studio and not env.get('loaded'):
                raise RuntimeError('studio HDR environment did not load')
            counters[key] = dict(gpu=gpu['device_name'], msaa=4, ray_query_shots=ray['shadow_captures'],
                                 sky_draws=env.get('sky_draws', 0), environment_loaded=bool(env.get('loaded')),
                                 validation_errors=gpu['validation_errors'], area_samples_per_panel=grid**2,
                                 pose_path='shared-cuda' if direct else 'host-snapshot',
                                 fragment_sha256=diag['shader_sha256']['frag'])
    return shots, counters


def read_planes(shot):
    planes = {p.channel: p.data for p in shot.planes}
    h, w = shot.camera.height, shot.camera.width
    rgb = np.frombuffer(planes[Channel.RGB], 'u1').reshape(h, w, 3)
    labels = np.frombuffer(planes[Channel.INSTANCE_ID], '<u4').reshape(h, w)
    depth = np.frombuffer(planes[Channel.FORWARD_DEPTH], '<f4').reshape(h, w)
    valid = np.frombuffer(planes[Channel.VALID], 'u1').reshape(h, w)
    if not (np.array_equal(valid, labels != 0) and np.array_equal(valid, np.isfinite(depth))):
        raise ValueError('depth, labels and validity disagree')
    return rgb, labels, depth


def compare(before, after, model):
    a, b = before.frame_key, after.frame_key
    if (replace(before.camera, frame_key=b) != after.camera or replace(a, scene_key=b.scene_key) != b
            or before.time_s != after.time_s or b.tick != 32):
        raise ValueError('the two looks were captured with different cameras or physics')
    ar, ai, az = read_planes(before)
    br, bi, bz = read_planes(after)
    row = dict(world=b.world_id, camera=after.camera.id, tick=b.tick, time_s=after.time_s,
               original_scene=a.scene_key, studio_scene=b.scene_key, camera_pose=asdict(after.camera.world_pose))
    ids = np.arange(1, model.ngeom + 1)
    names = [model.geom(i).name for i in range(model.ngeom)]
    for name, selection in [('robot', [s != 'support' and not s.startswith('object-') for s in names]),
                            ('workpiece', [s.startswith('object-') for s in names])]:
        original_hit = np.isin(ai, ids[selection])
        hit = original_hit & (ai == bi)
        if not np.any(hit):
            raise ValueError(f'{name} invisible in matched view {b.world_id}/{after.camera.id}')
        depth_error = float(np.abs(az[hit] - bz[hit]).max())
        # Compare both looks with the same pose publication path. Switching
        # paths alone can alter intersections on nearly tangential surfaces.
        if depth_error > 2e-5:
            raise ValueError(f'{name} surface depth changed by {depth_error} m in {b.world_id}/{after.camera.id}')
        def brightness(rgb):
            return float(np.median(rgb[hit] @ np.array([.2126, .7152, .0722])))
        row[name] = dict(matched_pixels=int(hit.sum()), original_pixels=int(original_hit.sum()),
                         median_srgb_luma_before=brightness(ar), median_srgb_luma_studio=brightness(br),
                         depth_delta_max_m=depth_error)
    row['black_percent_before'] = float(np.mean(ar.max(axis=2) <= 8) * 100)
    row['black_percent_studio'] = float(np.mean(br.max(axis=2) <= 8) * 100)
    return row, Image.fromarray(ar), Image.fromarray(br)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True, help='native DLL alongside the resident SPIR-V files')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--area-grid', type=int, choices=(2, 4, 8), default=8,
                        help='samples per panel axis for the studio look; 2 uses the live shared-buffer path')
    parser.add_argument('--keys', nargs='+', type=int, default=WORLD_KEYS, choices=WORLD_KEYS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assets = args.output / 'meshes'
    with open_worlds(assets, args.keys, device=args.device) as world:
        world.step(32)
        snapshot = world.snapshot()
        live, live_info = capture(world, snapshot, args, False)
        rows = None
        baseline, old_info = live, live_info
        if args.area_grid > 2:
            world.stream.synchronize()
            rows = snapshot.transforms.cpu().numpy()
            baseline, old_info = capture(world, snapshot, args, False, live, rows)
        styled = studio_variant(world, assets / 'studio')
        photographs, info = capture(styled, snapshot, args, True, live, rows)
        names = {row['world_key']: row['model_key'] for row in world.manifest['worlds']}
        images, observations = {}, []
        hero = world.keys[0]
        for k, key in photographs:
            row, old, new = compare(baseline[k, key], photographs[k, key], world.models[k])
            if rows is not None:
                _, host_id, host_depth = read_planes(baseline[k, key])
                _, live_id, live_depth = read_planes(live[k, key])
                common = (host_id == live_id) & (host_id != 0)
                row['baseline_host_vs_shared'] = dict(instance_disagreements=int((host_id != live_id).sum()),
                    max_depth_delta_on_same_id_m=float(np.abs(host_depth[common] - live_depth[common]).max()))
            observations.append(row)
            images[k, key] = new
            if k == hero:
                new.save(args.output / f'studio-{key}.png')
                if key == 'fixed':
                    build_comparison(old, new, args.output / 'same-frame.png')
            else:
                views = args.output / 'views'
                views.mkdir(exist_ok=True)
                new.save(views / f'{k}-{key}.png')
        build_gallery(images, names, args.output / 'studio-cameras.png')
        report = dict(size=[args.width, args.height], views=observations, device_diagnostic=old_info,
                      device_studio=info, device_live=live_info, exposure=STUDIO_EXPOSURE, display=STUDIO_DISPLAY,
                      source_step=32, source_revision=snapshot.frame.revision,
                      note='Both looks share the same single accepted CUDA body-pose snapshot.')
        (args.output / 'capture.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(f'Rendered {len(images)} studio views of the same 32 ms physics endpoint.')


if __name__ == '__main__':
    main()
