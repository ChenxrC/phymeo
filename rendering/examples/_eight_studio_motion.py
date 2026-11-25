"""Record a continuous motor-driven MuJoCo motion once for both studio camera cuts."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from _heterogeneous_camera_worlds import WORLD_KEYS
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.manipulation_tasks import H, manipulation_scenes


@dataclass
class StudioMotion:
    models: dict
    rows: dict
    ticks: np.ndarray
    times: np.ndarray
    report: dict


def _model(scene):
    return compile_mujoco_xml(scene.mjcf, require_asset=False).fresh_native_model()


def _targets(scene):
    """Joint travel, in radians/metres, for slow reach, close, traverse and release beats."""
    count = len(scene.robot_joint_names)
    home = np.zeros(count)
    reach, close, lift, traverse = (home.copy() for _ in range(4))
    if scene.finger_geom_names:
        if 'P' in scene.robot_kinds:
            # The freely rolling ball needs an overhead pre-grasp. Raise the
            # vertical slide before bringing the pads together and traversing.
            reach[scene.robot_kinds.index('P')] = .025
            lift[scene.robot_kinds.index('P')] = .06
        else:
            reach[1:3] = (.28, -.18)
            lift[1:3] = (-.31, .24)
        close[:] = reach
        close[-2:] = .052
        lift[-2:] = close[-2:]
        traverse[:] = lift
        traverse[0] = .50
    else:
        # Approach from the side, sweep laterally, retract, then return above
        # the same work area. Only original joint motors act on the mechanism.
        if scene.robot_kinds == 'PPPR':
            reach[:3] = (.083, -.055, -.020)
            close[:3] = (.113, -.024, -.020)
            lift[:3] = (.105, .055, -.020)
            traverse[:3] = (.015, .085, .025)
        else:
            straight = (-.44, .82) if count == 2 else (-.54, 1.13, -.54)
            reach[:] = straight
            reach[0] -= .14
            close[:] = straight
            lift[:] = close
            lift[0] += .15
            traverse[0] = .34
    return np.asarray((home, reach, close, lift, traverse, lift, close, reach, home))


def _record(scene, model, ticks, end_tick, world_index):
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    if model.opt.timestep != H:
        raise ValueError('studio playback requires the registered 1 ms models')
    joints = np.array([model.joint(n).id for n in scene.robot_joint_names])
    actors = np.array([model.actuator(n).id for n in scene.actuator_names])
    q, v = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    slide = model.jnt_type[joints] == mj.mjtJoint.mjJNT_SLIDE
    kp, kd = np.where(slide, 230., 28.), np.where(slide, 10., 2.4)
    spans = np.array((0., .16, .25, .41, .56, .69, .79, .88, 1.))
    targets = _targets(scene)
    cycle_s = 7.3 + .41 * world_index
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
            phase = (tick * H / cycle_s) % 1.
            seg = np.searchsorted(spans, phase, side='right') - 1
            u = (phase - spans[seg]) / (spans[seg + 1] - spans[seg])
            smooth = u * u * (3. - 2. * u)
            desired = targets[seg] + smooth * (targets[seg + 1] - targets[seg])
            torque = kp * (desired - data.qpos[q]) - kd * data.qvel[v] + data.qfrc_bias[v]
            data.ctrl[actors] = np.clip(torque, *model.actuator_ctrlrange[actors].T)
        remaining = ticks[index] - tick if index < len(ticks) else end_tick - tick
        steps = min(4 - tick % 4, int(remaining), end_tick - tick)
        for _ in range(steps):
            mj.mj_step(model, data)
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
    if (not np.isfinite(object_path).all()
            or any(minimum[i] - radius < lo or maximum[i] + radius > hi for i, (lo, hi) in enumerate(limits))):
        raise RuntimeError(f'{scene.key}: workpiece leaves the overview table, centre range {minimum} to {maximum}')
    tool = pose_rows[:, model.body('tool').id, :, 3]
    stats = dict(model_key=scene.key, mjcf_sha256=scene.source_sha256,
                 body_names=[model.body(i).name for i in range(model.nbody)],
                 actuated_joints=list(scene.robot_joint_names), actuator_names=list(scene.actuator_names),
                 end_time_s=float(data.time), tool_axis_travel_m=np.ptp(tool, axis=0).tolist(),
                 object_axis_travel_m=(maximum - minimum).tolist(), cycle_s=cycle_s,
                 workpiece_bounds=dict(table_xy_m=limits, enclosing_radius_m=radius, checked_states=end_tick + 1,
                                       centre_min_m=minimum.tolist(), centre_max_m=maximum.tolist()),
                 observed_contact_warnings=[int(w.number) for w in data.warning])
    if max(stats['tool_axis_travel_m']) < .01 and end_tick >= 2000:
        raise RuntimeError(f'{scene.key}: less than 1 cm tool travel')
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
        trajectory, clock, final, stats = _record(scene, models[key], ticks, end, WORLD_KEYS.index(key))
        if times is not None and not np.array_equal(times, clock):
            raise RuntimeError('world clocks differ')
        times = clock
        rows[key] = arrays[f'rows_{key}'] = trajectory
        arrays[f'endpoint_{key}'] = final
        evidence.append(dict(world=key, **stats))
    np.savez_compressed(archive, ticks=ticks, times=times, **arrays)
    report = dict(format='studio-motion-v1', backend='MuJoCo CPU mj_step', mujoco_version=mj.__version__,
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
    if report['format'] != 'studio-motion-v1' or sha256(archive.read_bytes()).hexdigest() != report['archive_sha256']:
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
            model = _model(scene)
            if (key in models or scene.source_sha256 != entry['mjcf_sha256'] or scene.key != entry['model_key']
                    or entry['body_names'] != [model.body(i).name for i in range(model.nbody)]):
                raise ValueError('duplicate world, changed source scene or body order')
            poses = data[f'rows_{key}']
            if poses.shape != (len(ticks), model.nbody, 3, 4) or not np.isfinite(poses).all():
                raise ValueError('unexpected pose archive shape or nonfinite transform')
            models[key], rows[key] = model, poses
    return StudioMotion(models, rows, ticks, times, report)
