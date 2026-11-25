"""Record a continuous motor-driven MuJoCo motion once for both studio camera cuts."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from _heterogeneous_camera_worlds import WORLD_KEYS
from _studio_contact_evidence import ContactEvidence
from _studio_grasp_control import grasp_setup, studio_grasp_mjcf
from _studio_pusht_control import PushController
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.manipulation_tasks import H, manipulation_scenes


@dataclass
class StudioMotion:
    models: dict
    rows: dict
    ticks: np.ndarray
    times: np.ndarray
    report: dict


def _model(scene, *, contact_demonstration=True):
    source = studio_grasp_mjcf(scene) if contact_demonstration and scene.finger_geom_names else scene.mjcf
    return compile_mujoco_xml(source, require_asset=False).fresh_native_model()


def _record(scene, model, ticks, end_tick):
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    if model.opt.timestep != H:
        raise ValueError('studio playback requires the registered 1 ms models')
    controller = grasp_setup(scene, model, data) if scene.finger_geom_names else PushController(scene, model, data)
    evidence = ContactEvidence(scene, model, controller.setup)
    pose_rows = np.empty((len(ticks), model.nbody, 3, 4))
    object_id = model.body(scene.object_key).id
    free_joint = model.body_jntadr[object_id]
    if model.body_parentid[object_id] != 0 or model.jnt_type[free_joint] != mj.mjtJoint.mjJNT_FREE:
        raise ValueError('the studio needs a standalone free object')
    adr = model.jnt_qposadr[free_joint]
    object_path = np.empty((end_tick + 1, 3))
    object_path[0] = data.qpos[adr:adr + 3]
    times = np.empty(len(ticks))
    index = tick = 0
    # Hold controls over four original physics steps. Sample boundaries can
    # split a hold without refreshing the control early for a specific camera.
    while tick <= end_tick:
        if index < len(ticks) and tick == ticks[index]:
            mj.mj_kinematics(model, data)
            pose_rows[index, :, :, :3] = data.xmat.reshape(-1, 3, 3)
            pose_rows[index, :, :, 3] = data.xpos
            times[index] = data.time
            index += 1
        if tick == end_tick:
            break
        if tick % 4 == 0:
            controller.apply(data, tick * H)
        remaining = ticks[index] - tick if index < len(ticks) else end_tick - tick
        steps = min(4 - tick % 4, int(remaining), end_tick - tick)
        for _ in range(steps):
            mj.mj_step(model, data)
            evidence.inspect(data, tick)
            tick += 1
            object_path[tick] = data.qpos[adr:adr + 3]
    mj.mj_kinematics(model, data)
    final = np.concatenate((data.xmat.reshape(-1, 3, 3), data.xpos[:, :, None]), axis=2)
    if (index != len(ticks) or not np.isfinite(pose_rows).all() or not np.isfinite(final).all()
            or np.max(np.abs(times - ticks * H)) > 1e-8 or abs(data.time - end_tick * H) > 1e-8):
        raise RuntimeError(f'{scene.key}: nonfinite pose or discontinuous source clock')
    # Check all native integration ticks, including the unsampled final tick.
    # Bounding spheres include the offset pieces of compound objects and
    # conservatively cover the rotating shapes in the tabletop projection.
    geoms = np.flatnonzero(model.geom_bodyid == object_id)
    radius = float(np.max(np.linalg.norm(model.geom_pos[geoms], axis=1) + model.geom_rbound[geoms]))
    minimum, maximum = object_path.min(axis=0), object_path.max(axis=0)
    limits = ((-.30, .75), (-.34, .34))
    table_ok = np.isfinite(object_path).all() and all(
        minimum[i] - radius >= lo and maximum[i] + radius <= hi for i, (lo, hi) in enumerate(limits))
    tool = pose_rows[:, model.body('tool').id, :, 3]
    mj.mj_forward(model, data)
    stats = dict(task=controller.setup, interaction=evidence.result(data), model_key=scene.key, mjcf_sha256=scene.source_sha256,
                 body_names=[model.body(i).name for i in range(model.nbody)],
                 actuated_joints=list(scene.robot_joint_names), actuator_names=list(scene.actuator_names),
                 end_time_s=float(data.time), tool_axis_travel_m=np.ptp(tool, axis=0).tolist(),
                 object_axis_travel_m=(maximum - minimum).tolist(),
                 workpiece_bounds=dict(table_xy_m=limits, enclosing_radius_m=radius, checked_states=end_tick + 1,
                                       centre_min_m=minimum.tolist(), centre_max_m=maximum.tolist()),
                 observed_contact_warnings=[int(w.number) for w in data.warning])
    stats['interaction']['stayed_on_set'] = bool(table_ok)
    tail_xy = object_path[-min(end_tick+1, int(round(1./H))+1):, :2]
    stats['interaction']['last_second_max_xy_drift_m'] = float(np.linalg.norm(tail_xy-tail_xy[0], axis=1).max())
    stats['interaction']['passed'] &= bool(table_ok and max(stats['tool_axis_travel_m']) >= .01
                                          and not any(stats['observed_contact_warnings']))
    return pose_rows, times, final, stats


def _schedule(fps, seconds):
    if type(fps) is not int or not 1 <= fps <= 1000 or seconds <= 0 or not np.isfinite(seconds):
        raise ValueError('expected positive duration and integer FPS in [1, 1000]')
    n = int(round(fps * seconds))
    end = int(round(seconds / H))
    if n < 1 or abs(n - fps * seconds) > 1e-8 or abs(end * H - seconds) > 1e-8:
        raise ValueError('duration must contain whole video frames and whole physics ticks')
    # Sample at the closest exact simulation tick to each constant-rate PTS.
    return (np.arange(n, dtype=np.int64) * 2000 + fps) // (2 * fps), end


def record_motion(directory, keys, fps=30, seconds=30):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive, metadata = directory / 'motion.npz', directory / 'motion.json'
    if archive.exists() or metadata.exists():
        raise FileExistsError('motion archive already exists; use load_motion')
    ticks, end = _schedule(fps, seconds)
    keys = tuple(keys)
    catalog = dict(zip(WORLD_KEYS, manipulation_scenes(), strict=True))
    if not keys or set(keys) - catalog.keys() or len(set(keys)) != len(keys):
        raise ValueError('expected a nonempty subset of the eight registered world keys')
    models, rows, arrays, evidence = {}, {}, {}, []
    times = None
    for key in keys:
        scene = catalog[key]
        models[key] = _model(scene)
        trajectory, clock, final, stats = _record(scene, models[key], ticks, end)
        if times is not None and not np.array_equal(times, clock):
            raise RuntimeError('world clocks differ')
        times = clock
        rows[key] = arrays[f'rows_{key}'] = trajectory
        arrays[f'endpoint_{key}'] = final
        evidence.append(dict(world=key, **stats))
    failures = {row['model_key']: row['interaction'] for row in evidence if not row['interaction']['passed']}
    if failures:
        (directory/'failed-interactions.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
        raise RuntimeError('task contact or final-state verification failed: ' + ', '.join(failures))
    np.savez_compressed(archive, ticks=ticks, times=times, **arrays)
    report = dict(format='studio-motion-v2', backend='MuJoCo CPU mj_step', mujoco_version=mj.__version__,
                  control='joint PD with gravity/bias compensation through the original motors, 4 ms hold',
                  fps=fps, seconds=seconds, frames=len(ticks), step_s=H, end_tick=end, worlds=evidence,
                  archive_sha256=sha256(archive.read_bytes()).hexdigest())
    metadata.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return StudioMotion(models, rows, ticks, times, report)


def load_motion(directory):
    """Read the same physical observations for a second camera or a resumed render."""
    directory = Path(directory)
    archive = directory / 'motion.npz'
    report = json.loads((directory / 'motion.json').read_text(encoding='utf-8'))
    if report['format'] not in ('studio-motion-v1', 'studio-motion-v2') or sha256(archive.read_bytes()).hexdigest() != report['archive_sha256']:
        raise ValueError('different motion format or archive checksum')
    catalog = dict(zip(WORLD_KEYS, manipulation_scenes(), strict=True))
    models, rows = {}, {}
    with np.load(archive, allow_pickle=False) as data:
        ticks, times = data['ticks'], data['times']
        expected, end = _schedule(report['fps'], report['seconds'])
        if (not np.array_equal(ticks, expected) or report['end_tick'] != end or times.shape != ticks.shape
                or not np.allclose(times, ticks * H, rtol=0, atol=1e-8)):
            raise ValueError('source ticks or exposure times differ from the archive schedule')
        for entry in report['worlds']:
            key = entry['world']
            scene = catalog[key]
            model = _model(scene, contact_demonstration=report['format'] == 'studio-motion-v2')
            expected_source = studio_grasp_mjcf(scene) if scene.finger_geom_names else scene.mjcf
            if report['format'] == 'studio-motion-v2' and scene.finger_geom_names:
                if entry['task']['studio_mjcf_sha256'] != sha256(expected_source.encode()).hexdigest():
                    raise ValueError('studio jaw geometry differs from recorded contact geometry')
            if (key in models or scene.source_sha256 != entry['mjcf_sha256'] or scene.key != entry['model_key']
                    or entry['body_names'] != [model.body(i).name for i in range(model.nbody)]):
                raise ValueError('duplicate world, changed source scene or body order')
            poses = data[f'rows_{key}']
            if poses.shape != (len(ticks), model.nbody, 3, 4) or not np.isfinite(poses).all():
                raise ValueError('unexpected pose archive shape or nonfinite transform')
            models[key], rows[key] = model, poses
    return StudioMotion(models, rows, ticks, times, report)
