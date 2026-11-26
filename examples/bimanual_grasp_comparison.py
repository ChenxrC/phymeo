"""Serial, resumable scheduling of frozen bilateral-grasp comparisons.

    python -B examples/bimanual_grasp_comparison.py --source-dir FROZEN/examples \
        --out NEW --stage development --controls face-alignment firm-grip stable-grasp
    python -B examples/bimanual_grasp_comparison.py --source-dir FROZEN/examples \
        --out SAME --stage holdout --controls stable-grasp

A completed job is skipped only after rechecking its receipts and all hashes.
Failed/interrupted jobs are retained and never overwritten or silently restarted.
Run the existing independent bimanual_grasp_score.py on a completed stage for
physical slip/rotation scoring; this scheduler performs no physics or FK itself.
"""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET

CONTROLS = ('face-alignment', 'firm-grip', 'stable-grasp')
DEVELOPMENT = (701, 88201)
HOLDOUT = tuple(range(91001, 91009))
LAYOUTS = ('6x7', '7x6')
SCHEMA = 'bimanual-grasp-comparison-v1'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('duplicate JSON key: ' + k)
            result[k] = v
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + x)))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def jobs(stage):
    seeds = DEVELOPMENT if stage == 'development' else HOLDOUT
    layouts = ('6x7',) if stage == 'development' else LAYOUTS
    return [(seed, layout, control) for seed in seeds for layout in layouts for control in CONTROLS]


def source_binding(directory):
    directory = directory.resolve()
    needed = ('bimanual_flat_gripper.py', 'bimanual_policy_replay.py', 'bimanual_musa_train.py',
              'bimanual_musa_env.py', 'bimanual_wrist_render.py', '_bimanual_visual_policy.py',
              '_bimanual_image_pose.py', '_bimanual_fiducial.py', '_bimanual_stable_grasp.py')
    for name in needed:
        require((directory / name).is_file(), 'missing frozen source: ' + name)
    files = sorted(directory.glob('*bimanual*.py'))
    return {'directory': str(directory), 'sha256': {p.name: digest(p) for p in files}}


def protocol(args):
    return dict(schema=SCHEMA, source=source_binding(args.source_dir),
                orchestrator_sha256=digest(__file__), python=str(Path(sys.executable).resolve()),
                pythonpath=os.environ.get('PYTHONPATH', ''), physics=args.mode,
                library=None if args.library is None else dict(path=str(args.library.resolve()), sha256=digest(args.library)),
                device_index=args.device_index, cache_dir=str(args.cache_dir.resolve()),
                policy='same source-bound untrained PRIOR; no training or base-policy override',
                active_arms='both', black_rgb=False, command_xy=args.command_xy,
                image_size=96, action_repeat=8, horizon=400,
                controls=list(CONTROLS), development_seeds=list(DEVELOPMENT), development_layouts=['6x7'],
                holdout_seeds=list(HOLDOUT), holdout_layouts=list(LAYOUTS),
                stage_scope='both development seeds are development only; holdout aggregation requires all 48 jobs',
                control_scope='face-alignment baseline; firm-grip clamp-only change; stable-grasp; implementation bound to source hashes')


@contextmanager
def lock(root):
    path = root / '.scheduler.lock'
    try:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(dict(pid=os.getpid(), note='exclusive serial scheduler; inspect stale owner before manual removal'), stream)
    except FileExistsError as error:
        raise RuntimeError('scheduler lock exists; do not run concurrent jobs; inspect owner before removing stale lock') from error
    try:
        yield
    finally:
        path.unlink()


def task_fixture(xml, q):
    world = ET.fromstring(xml).find('worldbody')
    require(world is not None, 'missing worldbody')
    payload = world.find("body[@name='payload']")
    goal = world.find("site[@name='task_goal']")
    require(payload is not None and goal is not None, 'missing payload/task goal')
    require(payload.find("freejoint[@name='payload_free']") is not None, 'unknown free-payload convention')
    joints = list(world.iter('joint')) + list(world.iter('freejoint'))
    require(sum(j.tag == 'freejoint' for j in joints) == 1, 'unknown free-joint layout')
    # This frozen environment declares payload first, then the two serial arms.
    moving = [b for b in world.findall('body') if list(b.iter('joint')) or list(b.iter('freejoint'))]
    require(moving and moving[0] is payload, 'payload no longer first in qpos order')
    return dict(payload=ET.tostring(payload, encoding='unicode'),
                target=ET.tostring(goal, encoding='unicode'),
                target_markers=[ET.tostring(g, encoding='unicode') for g in world.findall('geom')
                                if g.get('name', '').startswith('goal_marker_')],
                initial_payload_qpos_sha256=sha256(q[:7].tobytes()).hexdigest())


def inspect_job(folder, seed, layout, control, bound):
    import numpy as np
    stem = folder / f'{seed}-four-contact'
    row = read(stem.with_suffix('.json'))
    report = read(folder / 'flat-jaw.json')
    require((row['seed'], row['arm_dofs'], row['control'], row['physics'], row['tool_profile']) ==
            (seed, [int(x) for x in layout.split('x')], control, bound['physics'], 'four-contact'), 'job identity differs')
    require(row['task'] == 'carry' and row['job']['arms'] == 'both' and not row['job']['black'], 'task/arm/image control differs')
    require(row['goal_input'] == ('registered_command_xy' if bound['command_xy'] else 'wrist_rgb'), 'goal input differs')
    require(report['policy_kind'] == 'untrained_prior' and report['base_policy_sha256'] is None and
            report['base_checkpoint_sha256'] is None, 'unexpected transferred or trained weights')
    require(report['adapter_sha256'] == bound['source']['sha256']['bimanual_flat_gripper.py'], 'adapter differs')
    require(Path(report['source_dir']).resolve() == Path(bound['source']['directory']), 'source directory differs')
    require(all(bound['source']['sha256'].get(k) == v for k, v in report['sources'].items()), 'episode source hashes differ')
    require(len(report['episodes']) == 1 and report['episodes'][0]['receipt'] == stem.with_suffix('.json').name,
            'unexpected child episode inventory')
    require(row['terminated'] or row['truncated'], 'unfinished episode')
    xml = stem.with_suffix('.xml').read_bytes()
    require(sha256(xml).hexdigest() == row['scene_sha256'], 'model hash differs')
    require(digest(stem.with_suffix('.npz')) == row['trajectory_sha256'], 'trajectory hash differs')
    n = row['decisions']
    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as data:
        q, ticks, times = data['qpos'], data['ticks'], data['time_s']
        require(0 < n <= 400 and q.shape == (n + 1, 20) and np.isfinite(q).all(), 'trajectory extent/values differ')
        require(data['model_sha256'].item() == row['scene_sha256'], 'trajectory model differs')
        require(ticks.shape == times.shape == (n + 1,) and ticks[0] == 0 and times[0] == 0 and
                np.isfinite(times).all() and np.all(np.diff(ticks) > 0) and np.all(np.diff(ticks) <= 8) and
                np.all(np.diff(times) > 0) and ticks[-1] == row['physics_ticks'], 'trajectory clocks differ')
        require(data['action'].shape == (n, 14) and np.isfinite(data['action']).all(), 'action dimensions differ')
        require(np.array_equal(data['rewards'], row['rewards']), 'rewards differ')
        first = q[0].copy()
    require(len(row['decision_diagnostics']) == n, 'missing decisions')
    require(np.array_equal(ticks[:-1], [d['input_tick'] for d in row['decision_diagnostics']]) and
            np.array_equal(ticks[1:], [d['output_tick'] for d in row['decision_diagnostics']]), 'decision tick binding differs')
    files = {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'completed.json'}
    return dict(seed=seed, layout=layout, control=control, files=files,
                scene_sha256=row['scene_sha256'], initial_qpos_dtype=first.dtype.str,
                initial_qpos_sha256=sha256(first.tobytes()).hexdigest(), task_fixture=task_fixture(xml, first),
                weights=row['job']['weights'], success=bool(row['success']), decisions=n,
                carried_airborne=bool(row['final_metrics']['carried_airborne']),
                final_goal_error_xy_m=row['final_metrics']['goal_error_xy_m'])


def pairing(rows):
    for seed in {r['seed'] for r in rows}:
        same_seed = [r for r in rows if r['seed'] == seed]
        require(all(r['task_fixture'] == same_seed[0]['task_fixture'] for r in same_seed), 'swapped layouts changed initial task/target')
        for layout in {r['layout'] for r in same_seed}:
            group = [r for r in same_seed if r['layout'] == layout]
            for key in ('scene_sha256', 'initial_qpos_dtype', 'initial_qpos_sha256', 'weights'):
                require(all(r[key] == group[0][key] for r in group), 'paired initial model/qpos/weights differ: ' + key)
    if rows:
        require(all(r['weights'] == rows[0]['weights'] for r in rows), 'prior weights differ across trials')


def completed(folder, job, bound, binding_sha):
    path = folder / 'completed.json'
    require(path.is_file(), f'incomplete output retained at {folder}; refusing overwrite or automatic reset')
    saved = read(path)
    require(saved['protocol_sha256'] == binding_sha, 'completed job belongs to another protocol')
    actual = inspect_job(folder, *job, bound)
    require(saved['result'] == actual, 'completed data changed since verification')
    return actual


def command(args, job, folder):
    seed, layout, control = job
    cmd = [sys.executable, '-B', str(args.source_dir.resolve() / 'bimanual_flat_gripper.py'),
           '--source-dir', str(args.source_dir.resolve()), '--out', str(folder), '--seed', str(seed),
           '--arm-layout', layout, '--control', control, '--arms', 'both', '--mode', args.mode,
           '--action-repeat', '8', '--image-size', '96', '--horizon', '400', '--cache-dir', str(args.cache_dir.resolve())]
    if args.command_xy:
        cmd.append('--command-xy')
    if args.library is not None:
        cmd += ['--library', str(args.library.resolve())]
    if args.device_index is not None:
        cmd += ['--device-index', str(args.device_index)]
    return cmd


def run(args):
    args.out = args.out.resolve()
    if args.cache_dir is None:
        args.cache_dir = args.out / 'cache'
    bound = protocol(args)
    args.out.mkdir(parents=True, exist_ok=True)
    with lock(args.out):
        binding = args.out / 'protocol.json'
        if binding.exists():
            require(read(binding) == bound, 'frozen protocol/source/options changed; use a distinct experiment root')
        else:
            require(not any(p.name != '.scheduler.lock' for p in args.out.iterdir()), 'new protocol requires an empty root')
            write_new(binding, bound)
        binding_sha = digest(binding)
        rows = []
        for job in jobs(args.stage):
            seed, layout, control = job
            folder = args.out / args.stage / str(seed) / layout / control
            if (folder / 'completed.json').is_file():
                rows.append(completed(folder, job, bound, binding_sha))
            elif control in args.controls:
                require(not folder.exists(), f'incomplete output retained at {folder}; refusing overwrite or automatic reset')
                require(not (folder.parent / (control + '.launch.json')).exists(), f'previous attempt retained for {folder}; refusing a silent retry')
                require(source_binding(args.source_dir) == bound['source'], 'source changed during schedule')
                folder.parent.mkdir(parents=True, exist_ok=True)
                binding_path = folder.parent / (control + '.launch.json')
                cmd = command(args, job, folder)
                write_new(binding_path, dict(protocol_sha256=binding_sha, command=cmd, status='launched; final status in completed.json or failure.json'))
                log_path = folder.parent / (control + '.log')
                child = None
                try:
                    with log_path.open('xb') as log:
                        child = subprocess.Popen(cmd, cwd=args.source_dir.resolve(), stdout=log, stderr=subprocess.STDOUT)
                        code = child.wait()
                    require(code == 0, f'child exit {code}; original output/log retained: {log_path}')
                    require(source_binding(args.source_dir) == bound['source'], 'source changed during child')
                    result = inspect_job(folder, *job, bound)
                    pairing(rows + [result])
                    write_new(folder / 'completed.json', dict(protocol_sha256=binding_sha, result=result))
                    rows.append(result)
                except BaseException as error:
                    if child is not None and child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill(); child.wait()
                    failure = folder.parent / (control + '.failure.json')
                    if not failure.exists():
                        write_new(failure, dict(protocol_sha256=binding_sha, error_type=type(error).__name__,
                                               message=str(error), returncode=None if child is None else child.returncode,
                                               output_retained=str(folder), log=str(log_path)))
                    raise
            pairing(rows)
        expected = jobs(args.stage)
        complete = len(rows) == len(expected)
        print(json.dumps(dict(stage=args.stage, verified_completed=len(rows), required=len(expected), complete=complete)), flush=True)
        if complete:
            # No holdout rates or aggregate are published before all 48 paired jobs exist.
            summary = dict(schema=SCHEMA, stage=args.stage, protocol_sha256=binding_sha, complete=True,
                           pairing_verified=True, rows=rows,
                           by_control={c: dict(trials=sum(r['control'] == c for r in rows),
                                               successes=sum(r['control'] == c and r['success'] for r in rows)) for c in CONTROLS},
                           scope='task success counts only; independent grasp-score performs physical slip/rotation assessment')
            destination = args.out / args.stage / 'aggregate.json'
            if destination.exists():
                require(read(destination) == summary, 'existing aggregate changed')
            else:
                write_new(destination, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('development', 'holdout'), required=True)
    parser.add_argument('--controls', choices=CONTROLS, nargs='+', default=list(CONTROLS))
    parser.add_argument('--mode', choices=('musa', 'cpu'), default='musa')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--device-index', type=int)
    parser.add_argument('--command-xy', action='store_true')
    args = parser.parse_args()
    require(len(set(args.controls)) == len(args.controls), 'duplicate controls')
    run(args)


if __name__ == '__main__':
    main()
