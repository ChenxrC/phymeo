"""Verify the R1 conveyor reader on freshly rendered authored stop exposures."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from phymeo_rendering.backends.native import NativeBackend, _instance_matrices
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, NodePose, Pose, SceneFrame, validate_scene_frame

from .sorting_device_reader import classify_station_rgb_cuda
from .sorting_layout import ITEM_KINDS, ITEM_NODES, LAYOUT
from .sorting_motion import _allocate
from .sorting_reader import classify_station_rgb, station_camera


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / 'artifacts/rendering'
ORDERS = {'recorded': (0, 1, 2, 3), 'permuted': (1, 0, 3, 2)}


def build_scene(args):
    from .finish_details import apply_details
    from .finishes import apply_finish
    from .model import load_model
    from .polished_normals import PAINTED_SHELLS, smooth_painted_shells
    from .sorting_labels import apply_sorting_labels
    from .sorting_set import append_sorting_set
    from .workshop_surfaces import apply_workshop_surfaces

    model = load_model(args.asset_dir, args.cache_dir)
    generated = args.scene_assets or args.output / 'assets'
    if generated.resolve() == args.archive.resolve():
        raise ValueError('keep regenerated scene assets outside the archived film directory')
    scene = apply_details(apply_finish(model.scene, 'polished'), generated / 'finish-assets')
    scene = smooth_painted_shells(scene, generated / 'normals', targets=PAINTED_SHELLS)
    scene = append_sorting_set(scene, generated / 'sorting-meshes', LAYOUT, ITEM_KINDS)
    scene = apply_workshop_surfaces(scene, generated / 'surface-assets')
    return apply_sorting_labels(scene, generated / 'markings', LAYOUT), model.diagnostics['revision']


def compare_scene_assets(archive, generated):
    """Compare rebuilt meshes and material texture samples with the film's local asset set."""
    from PIL import Image

    groups = {}
    for group in ('finish-assets', 'normals', 'sorting-meshes', 'surface-assets', 'markings'):
        def indices(directory):
            return {(re.sub(r'-[0-9a-f]{18}$', '', p.stem), p.suffix): p
                    for p in (directory / group).iterdir() if p.suffix in ('.gltf', '.png')}
        old, new = indices(archive), indices(generated)
        if old.keys() != new.keys():
            raise ValueError(f'rebuilt {group} has different render assets from the archived sorting cell')
        reencoded = []
        for (stem, ext), src in old.items():
            other = new[stem, ext]
            if sha256(src.read_bytes()).digest() == sha256(other.read_bytes()).digest():
                continue
            if ext != '.png':
                raise ValueError(f'rebuilt R1 mesh bytes differ: {group}/{stem}')
            with Image.open(src) as a, Image.open(other) as b:
                if a.size != b.size or a.convert('RGBA').tobytes() != b.convert('RGBA').tobytes():
                    raise ValueError(f'rebuilt R1 material texels differ: {group}/{stem}')
            reencoded.append(stem)
        groups[group] = dict(meshes=sum(k[1] == '.gltf' for k in old),
                             textures=sum(k[1] == '.png' for k in old), reencoded_identical_texels=sorted(reencoded))
    return groups


def read_stops(archive):
    manifest = json.loads((archive / 'render.json').read_bytes())
    raw = (archive / 'trajectory.json').read_bytes()
    if sha256(raw).hexdigest() != manifest['trajectory_sha256']:
        raise ValueError('R1 trajectory digest differs from the archived run')
    track = json.loads(raw)
    if tuple(track['orders']) != ITEM_KINDS:
        raise ValueError('expected the registered four workpiece geometries')
    stops = [track['frames'][entry['frame']] for entry in manifest['station_reads']]
    if len(stops) != 4 or any(s['phase'] != 'inspect' for s in stops):
        raise ValueError('expected four stopped, open-jaw source exposures')
    return manifest, stops


def authored_stop(scene, stop, order_name, index, completed):
    # The archived robot, rollers, witness marks and physical reader are fixed
    # to their original observation. Remap the *actual mesh nodes* onto the
    # indexed conveyor positions so a changed order presents different geometry.
    world = 'r1-indexed-conveyor' if order_name == 'recorded' else 'r1-indexed-conveyor-permuted'
    key = FrameKey(scene.key, world, 0, stop['frame'], stop['frame'], 'authored')
    original = {name: Pose(tuple(p['position_m']), tuple(p['orientation_xyzw']))
                for name, p in stop['poses'].items()}
    poses = dict(original)
    for arrival, identity in enumerate(ORDERS[order_name]):
        poses[ITEM_NODES[identity]] = completed[arrival] if arrival < index else original[ITEM_NODES[arrival]]
    incoming = poses[ITEM_NODES[ORDERS[order_name][index]]]
    expected = (*LAYOUT.pick_xy, LAYOUT.support_z + LAYOUT.payload_size[2] / 2)
    if not np.allclose(incoming.position_m, expected, rtol=0., atol=1e-7):
        raise ValueError('incoming part is not stopped at the physical reader')
    return SceneFrame(key, stop['time_s'], tuple(NodePose(k, v) for k, v in poses.items()))


def station_request(frame, archive_camera):
    from _panda_scene import pose_matrix
    camera = station_camera(frame.key, LAYOUT)
    for key in ('id', 'width', 'height', 'fx_px', 'fy_px', 'cx_px', 'cy_px', 'near_m', 'far_m'):
        if getattr(camera, key) != archive_camera[key]:
            raise ValueError(f'reader {key} differs: {getattr(camera, key)} vs archived {archive_camera[key]}')
    stored = archive_camera['world_pose']
    registered = Pose(tuple(stored['position_m']), tuple(stored['orientation_xyzw']))
    actual = camera.world_pose
    rotation_error = float(np.abs(pose_matrix(actual)[:3, :3] - pose_matrix(registered)[:3, :3]).max())
    position_error = float(np.abs(np.subtract(actual.position_m, registered.position_m)).max())
    if rotation_error > 2e-12 or position_error > 1e-12:
        raise ValueError(f'reader extrinsics differ: position {position_error} m, rotation matrix {rotation_error}')
    audit = dict(derived_xyzw=actual.orientation_xyzw, archived_xyzw=registered.orientation_xyzw,
                 raw_quaternion_max_difference=float(np.abs(np.subtract(actual.orientation_xyzw,
                                                                        registered.orientation_xyzw)).max()),
                 rotation_matrix_max_difference=rotation_error, position_max_difference_m=position_error)
    return CaptureRequest(replace(camera, world_pose=registered), RenderPath.RASTER, (Channel.RGB,)), audit


def authored_records(scene, frame):
    """Pack the host-authored transforms using the resident FP32 world/inverse convention."""
    validate_scene_frame(scene, frame)
    world = np.asarray(_instance_matrices(scene, frame)).reshape(-1, 4, 4)
    # world is column-major. Invert the rounded FP32 linear part first and
    # compute the inverse translation from that rounded inverse as in resident.inc.
    columns = world[:, :3, :3].astype(np.float64)
    a, b, c = columns.transpose(1, 0, 2)
    bc = np.cross(b, c)
    det = np.einsum('ij,ij->i', a, bc)
    inverse = np.zeros_like(world)
    cofactor = np.stack((bc, np.cross(c, a), np.cross(a, b)), axis=1)
    inverse[:, :3, :3] = (cofactor / det[:, None, None]).transpose(0, 2, 1)
    inverse[:, 3, :3] = -np.einsum('bi,bij->bj', world[:, 3, :3].astype(np.float64),
                                    inverse[:, :3, :3].astype(np.float64))
    inverse[:, 3, 3] = 1.
    return np.stack((world, inverse), axis=1)


def read_host(session, frame, request, board_images):
    result = session.capture(frame, request)
    raw = result.planes[0].data
    board_images.append(raw)
    rgb = np.frombuffer(raw, dtype='u1').reshape(request.camera.height, request.camera.width, 3)
    decision, measured = classify_station_rgb(rgb)
    x0, y0, x1, y1 = measured['roi_xyxy']
    count = (x1 - x0) * (y1 - y0)
    # CPU classifier reports exact integer counts divided by the ROI size.
    counts = {kind: round(f * count) for kind, f in measured['fraction'].items()}
    return decision, dict(**measured, pixel_count=counts, rgb_sha256=sha256(raw).hexdigest(),
                          public_rgb_bytes=len(raw), authored_record_torch_h2d_bytes=0)


def read_device(session, frame, request, scene, streams, cuda_device):
    import torch

    producer, delivery, consumer = streams
    records = authored_records(scene, frame)
    with torch.cuda.stream(producer):
        packed = torch.from_numpy(records).to(producer.device)
    # The public physics publication validates a committed workspace snapshot.
    # These frames are authored, so use the renderer's actual matrix exchange
    # and attachment ABI directly, preserving their source identity in the report.
    delegate = session._delegate
    if not session.supports(request):
        raise ValueError('unsupported station RGB request')
    exchange = delegate.device.open_shared_matrices(cuda_device=cuda_device)
    with exchange.publish(packed, producer_stream=producer) as lease:
        (planes,) = delegate.capture_shared_device_batch((request,), (None,), lease, delivery)
    rgb = planes[Channel.RGB]
    decision, measured = classify_station_rgb_cuda(rgb, producer_stream=delivery, consumer_stream=consumer)
    return decision, dict(**measured, authored_record_torch_h2d_bytes=records.nbytes, tensor_stride=tuple(rgb.stride()))


def run_order(scene, session, stops, manifest, name, method, streams, cuda_device, board_images):
    from preview_compact_rigid_history.state import frame_digest

    occupied, placed, records = Counter(), [], []
    for index, (stop, identity) in enumerate(zip(stops, ORDERS[name], strict=True)):
        frame = authored_stop(scene, stop, name, index, placed)
        request, optics = station_request(frame, manifest['station_reads'][index]['camera'])
        if method == 'host':
            decision, evidence = read_host(session, frame, request, board_images)
        else:
            decision, evidence = read_device(session, frame, request, scene, streams, cuda_device)
        # The class and slot come only from the optical decision. Geometry is
        # consulted afterwards to check the decision, including permuted arrivals.
        route = _allocate(identity, decision, occupied, LAYOUT)
        if decision != ITEM_KINDS[identity]:
            raise ValueError(f'{method}/{name} at stop {index}: measured {decision} for {ITEM_NODES[identity]}')
        placed.append(Pose(route['center_m']))
        digest = frame_digest(frame)
        records.append(dict(frame=frame.key.tick, source_pose_sha256=digest, node=route['node'],
                            matches_archived_pose=digest == manifest['station_reads'][index]['source_pose_sha256'],
                            optics=optics,
                            decision=decision, bin='A' if decision == 'block' else 'B', slot=route['slot_index'],
                            slot_center_m=route['center_m'], **evidence))
    if occupied != Counter(block=2, hex=2):
        raise ValueError('four optical decisions did not fill the two bin pairs')
    return records


def station_board(path, images, rows):
    from PIL import Image, ImageDraw, ImageFont

    def font(size):
        for name in ('C:/Windows/Fonts/segoeui.ttf', 'DejaVuSans.ttf'):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
        return ImageFont.load_default()

    w, h, margin, gap = 256, 154, 18, 9
    board = Image.new('RGB', (2*margin + 4*w + 3*gap, 2*margin + 2*(h+58) + gap), '#172026')
    draw = ImageDraw.Draw(board)
    header, label = font(17), font(14)
    for row_index, order in enumerate(ORDERS):
        y = margin + row_index * (h + 58 + gap)
        draw.text((margin, y), order.upper() + ' ARRIVAL ORDER', font=header, fill='#edf1f2')
        for i, (raw, measured) in enumerate(zip(images[order], rows[order], strict=True)):
            x = margin + i*(w+gap)
            image = Image.frombytes('RGB', (320, 192), raw).resize((w, h), Image.Resampling.LANCZOS)
            board.paste(image, (x, y + 26))
            text = f'{measured["node"]}   {measured["decision"]}   /   Bin {measured["bin"]}, slot {measured["slot"]+1}'
            draw.text((x, y+30+h), text, font=label, fill='#edf1f2')
    board.save(path)


def run(args):
    import torch
    from .workshop import workshop_lighting

    args.output.mkdir(parents=True, exist_ok=True)
    if any(p.name not in ('assets', 'station-reads.png') for p in args.output.iterdir()):
        raise ValueError('use a new station audit directory, or retry an incomplete station run')
    manifest, stops = read_stops(args.archive)
    scene, revision = build_scene(args)
    if revision != manifest['model_revision']:
        raise ValueError('reconstructed R1 differs from the pinned visual revision')
    appearance = compare_scene_assets(args.archive, args.scene_assets or args.output / 'assets')
    common = dict(resident=True, library_path=args.library, shader_directory=args.shader_directory or args.library.parent,
                  device_index=args.device_index, validation=args.validation, shadow_mode='ray_query',
                  msaa_samples=1, studio=workshop_lighting(), exposure=8., display_transform='pbr-neutral')
    rows, diagnostics = {}, {}
    board_images = {name: [] for name in ORDERS}
    with torch.cuda.device(args.device):
        streams = tuple(torch.cuda.Stream(device=args.device) for _ in range(3))
        if len({int(s.cuda_stream) for s in streams} | {0}) != 4:
            raise ValueError('require distinct nondefault pose, attachment and classification streams')
        for method in ('host', 'device'):
            extra = dict(shared_cuda_device=args.device) if method == 'device' else {}
            session = NativeBackend(**common, **extra).open_scene(scene)
            try:
                rows[method] = {name: run_order(scene, session, stops, manifest, name, method, streams, args.device,
                                                board_images[name])
                                for name in ORDERS}
            finally:
                session.close()
            device = session.diagnostics()['device']
            diagnostics[method] = {k: device[k] for k in ('device_name', 'device_uuid', 'resident_captures',
                'resident_telemetry', 'resident_readback', 'resident_shared_matrices', 'resident_device_output',
                'resident_host_transform_input_bytes', 'live_contexts', 'live_frames', 'live_resources',
                'validation_errors', 'validation_warnings')}
            if any(device[k] for k in ('validation_errors', 'live_resources', 'live_frames', 'live_contexts')):
                raise RuntimeError(f'{method}: native session reported a validation or resource failure')
            if method == 'host':
                station_board(args.output / 'station-reads.png', board_images, rows['host'])
            # The delegate retains decoded mesh geometry. Drop it before opening the next full R1 scene.
            del session
    for order in ORDERS:
        for host, gpu in zip(rows['host'][order], rows['device'][order], strict=True):
            for key in ('source_pose_sha256', 'node', 'frame', 'roi_xyxy', 'pixel_count', 'fraction',
                        'decision', 'bin', 'slot', 'slot_center_m'):
                if host[key] != gpu[key]:
                    raise ValueError(f'{order} at frame {host["frame"]}: {key} differs between CPU and direct CUDA')
    host, gpu = diagnostics['host'], diagnostics['device']
    bridge = gpu['resident_device_output']
    captures = sum(len(order) for order in ORDERS.values())
    expected_bytes = captures * 320 * 192 * 4
    if (gpu['device_uuid'] != host['device_uuid'] or host['resident_readback']['copied_bytes'] != expected_bytes
            or gpu['resident_telemetry']['readback_bytes'] or gpu['resident_readback']['copied_bytes']
            or bridge['batches'] != captures or bridge['attachment_d2d_bytes'] != expected_bytes
            or bridge['image_d2h_bytes'] or bridge['image_h2d_bytes'] or bridge['imports_live']
            or bridge['imports_created'] != bridge['imports_destroyed']):
        raise RuntimeError('R1 station transport or device identity accounting differs')
    report = dict(source='R1 full sorting cell; archived authored robot/conveyor stop poses; rebuilt live RGB',
                  image_board='station-reads.png', board_source='Eight newly rendered CPU-reference RGB attachments',
                  original_trajectory_sha256=manifest['trajectory_sha256'], model_revision=revision,
                  scene=scene.key, archived_scene=manifest['scene'], scene_key_matches_archive=scene.key == manifest['scene'],
                  archived_mesh_and_texture_comparison=appearance,
                  meshes=len(scene.meshes), instances=len(scene.instances), materials=len(scene.materials),
                  camera=manifest['station_reads'][0]['camera'],
                  renderer='native resident direct lighting, 2x2 area panels, hardware ray queries, MSAA1, PBR Neutral',
                  exposure=8., studio=asdict(common['studio']), same_settings_both_routes=True,
                  archive_renderer='sealed stable lighting film pipeline; classification compared to fresh host captures',
                  pose_source='CPU-authored stops. World/inverse FP32 records uploaded for CUDA delivery, no physics steps.',
                  route_scope='Actual reordered workpiece meshes at four stops; assigned slots position earlier items. '
                              'The permuted arm motion between observations is not simulated.',
                  orders=rows, diagnostics=diagnostics,
                  decision_d2h_bytes=sum(r['d2h_bytes'] for order in rows['device'].values() for r in order),
                  authored_record_torch_h2d_bytes=sum(r['authored_record_torch_h2d_bytes']
                                                      for order in rows['device'].values() for r in order))
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--shader-directory', type=Path)
    parser.add_argument('--asset-dir', type=Path, default=ASSETS / 'galaxea-r1-assets')
    parser.add_argument('--cache-dir', type=Path, default=ASSETS / 'film-aa-r1/r1-cache')
    parser.add_argument('--scene-assets', type=Path, help='Shared scratch directory for reconstructed scene assets')
    parser.add_argument('--archive', type=Path, default=ASSETS / 'r1-sorting/master01')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--validation', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
