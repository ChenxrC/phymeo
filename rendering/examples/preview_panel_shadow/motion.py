"""Compare the first moving-shadow frames with fresh no-history Vulkan captures."""
import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from preview_compact_rigid_history.runtime import activate
from preview_compact_rigid_history.state import bind_presentations
from preview_lowres_filter._helpers import digest, save
from preview_surface_quality.runtime import SurfaceQualityBackend


WIDTH, HEIGHT = 128, 64
ROI = (20, 20, 44, 44)


def build_scene(out):
    from _panda_scene import write_mesh
    from phymeo_rendering.assets import Material
    from phymeo_rendering.cameras import Camera
    from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene

    def quad(name, left, right, extent):
        return write_mesh(out, name, [(left,-extent,0), (right,-extent,0),
                          (right,extent,0), (left,extent,0)], [(0,0,1)]*4, [(0,1,2),(0,2,3)])

    out.mkdir()
    scene = VisualScene('moving-panel-shadow', (quad('receiver', -1, 1, 1), quad('shutter', -2, 0, 2)),
                        (Material('white', (.8, .8, .8), 0., 1.), Material('black', (0., 0., 0.), 0., 1.)),
                        (Instance(1, 'receiver', 'white', 'receiver'), Instance(2, 'shutter', 'black', 'shutter')))
    frames, cameras = [], []
    for i, shutter_x in enumerate((0., 0., -3., 0.)):
        key = FrameKey(scene.key, 'panel-shadow-motion', 0, i, i, 'authored')
        frames.append(SceneFrame(key, i/30, (NodePose('receiver', Pose()),
                                             NodePose('shutter', Pose((shutter_x, 0., .5))))))
        # Camera below the shutter, looking down at the receiver.
        cameras.append(Camera('receiver', key, Pose((0., 0., .25)), WIDTH, HEIGHT,
                              64., 64., WIDTH/2, HEIGHT/2, .005, 5.))
    return scene, frames, cameras


def run(args):
    package, manifest = activate(args.package)
    from area_light_quadrature import _panel_fixture
    from panda_motion_preview import verify_close
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    from PIL import Image

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    scene, frames, cameras = build_scene(out / 'geometry')
    save(out / 'inputs.json', dict(scene=asdict(scene), frames=list(map(asdict, frames)),
                                   cameras=list(map(asdict, cameras)), roi=ROI))
    bindings = bind_presentations(frames, digest(out / 'inputs.json'))
    x, y, right, bottom = ROI
    roi = np.s_[y:bottom, x:right]
    layout = np.dtype([('incoming', '<f4', 4), ('geometry', '<u4', 4), ('primary', '<f4', 4)])
    report = dict(package_seal_sha256=digest(package / 'surface-seal.json'),
                  panels=manifest['contract']['surface_quality']['panels'], roi=ROI,
                  scope='Current primary linear RGB includes diffuse and specular panel lighting. '
                        'No HDR, directional lights or reflected-environment radiance in this fixture. '
                        'Displayed RGB is checked separately and includes reconstructed indirect light.',
                  states=['partly blocked', 'partly blocked', 'clear', 'partly blocked'],
                  records=[], comparisons=[])
    sensor_hashes = None

    def capture_sequence(indices, seed, reuse, label):
        nonlocal sensor_hashes
        session = SurfaceQualityBackend(package, device_index=args.device_index, seed=seed, exposure=1.,
                                        display_transform='clip', reflection_mode='environment',
                                        reuse_history=reuse, validation=True).open_scene(scene)
        values = []
        try:
            # The existing analytic area-light control installs one 1x1 m panel
            # at z=1 through the native immutable studio ABI before any capture.
            report['emitter'] = _panel_fixture(session.underlying)
            for i in indices:
                request = CaptureRequest(cameras[i], RenderPath.RASTER, tuple(Channel))
                shot = session.capture(frames[i], request, presentation=bindings[i])
                if shot.frame_key != frames[i].key or shot.camera != cameras[i]:
                    raise RuntimeError('capture identity differs')
                planes = {p.channel: p.data for p in shot.planes}
                hashes = {ch.name: sha256(planes[ch]).hexdigest() for ch in Channel if ch != Channel.RGB}
                if sensor_hashes is None:
                    sensor_hashes = hashes
                if (hashes != sensor_hashes or not np.all(np.frombuffer(planes[Channel.INSTANCE_ID], '<u4') == 1)
                        or not np.all(np.frombuffer(planes[Channel.VALID], 'u1') == 1)
                        or not np.all(np.frombuffer(planes[Channel.FORWARD_DEPTH], '<f4') == .25)):
                    raise RuntimeError('receiver visibility or depth changed')
                packet = session.read_history_incoming()
                if packet['format'] != 'F32x4_U32x4_F32x4_LE':
                    raise RuntimeError('missing full-resolution current primary lighting')
                pixels = np.frombuffer(packet['data'], layout).reshape(HEIGHT, WIDTH)
                primary = pixels['primary'][roi][..., :3].copy()
                if not np.isfinite(primary).all() or primary.min() < 0:
                    raise RuntimeError('invalid primary lighting')
                rgb = np.frombuffer(planes[Channel.RGB], 'u1').reshape(HEIGHT, WIDTH, 3)
                effective = session.diagnostics()['history']['effective_seed']
                report['records'].append(dict(label=label, frame=i, reuse_history=reuse,
                    effective_seed=effective, primary_roi_mean=float(primary.mean()),
                    primary_roi_sha256=sha256(primary.tobytes()).hexdigest(),
                    max_history_age=int(pixels['incoming'][roi][..., 3].max())))
                values.append((primary, rgb[roi].copy(), effective))
                Image.fromarray(rgb).save(out / f'{label}-{i}.png')
        finally:
            session.close()
            verify_close(session.diagnostics()['device'])
        return values

    try:
        moving = capture_sequence(range(4), 0, True, 'moving')
        for i in (2, 3):
            previous, current = moving[i-1][0], moving[i][0]
            fresh = capture_sequence([i], moving[i][2], False, 'fresh')[0]
            delta = (current - previous) * (1 if i == 2 else -1)
            displayed_error = np.abs(moving[i][1].astype(int) - fresh[1].astype(int))
            report['comparisons'].append(dict(frame=i, primary_matches_fresh=np.array_equal(current, fresh[0]),
                max_displayed_rgb8_difference=int(displayed_error.max()),
                min_signed_primary_change=float(delta.min()), mean_signed_primary_change=float(delta.mean())))
        report['passed'] = all(c['primary_matches_fresh'] and c['max_displayed_rgb8_difference'] == 0
                               and c['min_signed_primary_change'] > .01 for c in report['comparisons'])
    finally:
        report['full_frame_sensor_sha256'] = sensor_hashes
        save(out / 'report.json', report)
    if not report['passed']:
        raise RuntimeError('moving shadow comparison differs; see report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device-index', type=int, default=1)
    run(parser.parse_args())
