"""Installed ROB2a/N4 observation consumer. No checkout/test/private-state imports.

The caller supplies the installation and dependencies and validates module
provenance before running this copied file outside the source tree. This file
never changes sys.path. Authored only; run each backend in the sole serial slot.
"""
import argparse
import base64
import copy
import gzip
import hashlib
import json
import math
import struct
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

H = .001
KEYS = (7, 91)
ACT_RTOL, ACT_ATOL = 3e-5, 3e-6
FORCE_RTOL, FORCE_ATOL = 3e-4, 2e-4
Q_ATOL, V_ATOL, TIME_ATOL = 5e-5, 3e-4, 2e-6
XML = '''<mujoco model="installed-activation"><compiler angle="radian"/>
<option timestep=".001" gravity=".1 -.2 -1" integrator="Euler"/>
<worldbody><body name="base" pos=".04 .02 .3">
 <joint name="hinge" type="hinge" axis="0 1 0" armature=".04" damping=".12"/>
 <inertial pos=".03 .01 -.02" mass="1.1" diaginertia=".04 .05 .06"/>
 <geom size=".04" contype="0" conaffinity="0"/>
 <body name="tip" pos=".2 .03 .04" quat=".9238795325 0 0 .3826834324">
  <joint name="slide" type="slide" axis="1 0 0" armature=".05" damping=".11"/>
  <inertial pos=".03 -.02 .01" mass=".8" diaginertia=".02 .03 .04"/>
  <geom size=".03" contype="0" conaffinity="0"/>
 </body></body></worldbody>
<actuator>
 <motor name="direct" joint="hinge" gear="-.5"/>
 <general name="filter-slide" joint="slide" gear="-2" dyntype="filterexact"
  dynprm=".004" gainprm="1.2" biastype="affine" biasprm=".01 -.3 -.04" actearly="true"/>
 <general name="filter-hinge" joint="hinge" gear="1.5" dyntype="filterexact" dynprm=".002" gainprm=".8"/>
 <general name="second-hinge" joint="hinge" gear="-.25" dyntype="filterexact" dynprm=".008" gainprm=".6"/>
</actuator></mujoco>'''


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def encode(value):
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "object array in evidence")
        return dict(dtype=value.dtype.str, shape=list(value.shape), bytes_b64=base64.b64encode(value.tobytes()).decode())
    if isinstance(value, np.generic):
        return dict(dtype=value.dtype.str, scalar_b64=base64.b64encode(value.tobytes()).decode())
    if is_dataclass(value):
        return dict(type=type(value).__module__+'.'+type(value).__name__, fields={f.name: encode(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, Mapping):
        return dict(mapping_entries=[[encode(k), encode(v)] for k, v in value.items()])
    if isinstance(value, tuple):
        return dict(tuple=[encode(v) for v in value])
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, bytes):
        return dict(bytes_b64=base64.b64encode(value).decode())
    if type(value) is float:
        return dict(float64_bits=struct.pack('>d', value).hex())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError('unhandled evidence value '+type(value).__name__)


def exact(left, right, label):
    require(encode(left) == encode(right), label)


class Journal:
    def __init__(self, path):
        self.path = path
        self.stream = gzip.open(path, 'wb')
        self.sha = hashlib.sha256()
        self.size = self.events = 0

    def emit(self, event, **payload):
        line = (json.dumps(encode(dict(event=event, **payload)), allow_nan=False, separators=(',', ':'))+'\n').encode()
        self.stream.write(line)
        self.stream.flush()
        self.sha.update(line)
        self.size += len(line)
        self.events += 1

    def close(self):
        self.stream.close()
        return dict(path=str(self.path), events=self.events, logical_bytes=self.size,
                    logical_sha256=self.sha.hexdigest(), compressed_bytes=self.path.stat().st_size,
                    compressed_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())


def modules():
    return {name: str(Path(module.__file__).resolve()) for name, module in sorted(sys.modules.items())
            if (name == 'phymeo' or name.startswith('phymeo.')) and getattr(module, '__file__', None)}


def set_control(data, packet):
    for name in ('ctrl', 'qfrc_applied', 'xfrc_applied'):
        getattr(data, name)[:] = np.asarray(getattr(packet, name), np.float32)


def native_state(data, *, force_phase="last-step-input-dynamics"):
    return dict(force_phase=force_phase, q=data.qpos.copy(), v=data.qvel.copy(), act=data.act.copy(), time=float(data.time),
                ctrl=data.ctrl.copy(), qfrc_applied=data.qfrc_applied.copy(),
                xfrc_applied=data.xfrc_applied.copy(), qfrc_actuator=data.qfrc_actuator.copy(),
                qacc=data.qacc.copy(), actuator_force=data.actuator_force.copy(), act_dot=data.act_dot.copy())


def analytic(model, entry, packet):
    output = entry.act.copy()
    derivative = np.zeros(model.na)
    control = np.empty(model.nu)
    selected = np.empty(model.nu)
    raw = np.empty(model.nu)
    force = np.empty(model.nu)
    generalized = np.zeros(model.nv)
    for actor in range(model.nu):
        u = float(np.float32(packet.ctrl[actor]))
        if model.actuator_ctrllimited[actor]:
            u = float(np.clip(u, *model.actuator_ctrlrange[actor]))
        control[actor] = u
        adr = int(model.actuator_actadr[actor])
        disabled = bool(model.opt.disableactuator & (1 << int(model.actuator_group[actor])))
        value = u
        if adr >= 0:
            tau = max(1e-15, float(model.actuator_dynprm[actor, 0]))
            derivative[adr] = (u-entry.act[adr])/tau
            value_next = entry.act[adr] if disabled else entry.act[adr]+(u-entry.act[adr])*-math.expm1(-H/tau)
            if model.actuator_actlimited[actor]:
                value_next = float(np.clip(value_next, *model.actuator_actrange[actor]))
            output[adr] = value_next
            value = value_next if model.actuator_actearly[actor] else entry.act[adr]
        selected[actor] = value
        joint = int(model.actuator_trnid[actor, 0])
        qid, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        gear = float(model.actuator_gear[actor, 0])
        bias = model.actuator_biasprm[actor, :3] if int(model.actuator_biastype[actor]) else np.zeros(3)
        raw[actor] = 0 if disabled else float(model.actuator_gainprm[actor, 0])*value + bias[0]+bias[1]*gear*entry.q[qid]+bias[2]*gear*entry.v[dof]
        force[actor] = raw[actor]
        if model.actuator_forcelimited[actor]:
            force[actor] = np.clip(force[actor], *model.actuator_forcerange[actor])
        generalized[dof] += gear*force[actor]
    return dict(output_act=output, activation_derivative=derivative, clamped_control=control,
                force_input=selected, raw_force=raw, scalar_force=force, generalized_force=generalized)


def observations(runtime, mj):
    full = runtime.snapshot()
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU forward/step used for integration observation')
    with patch.object(mj, 'mj_step', forbidden), patch.object(mj, 'mj_forward', forbidden):
        integration = runtime.integration_snapshot()
    require(tuple(full) == tuple(integration), 'observation world identity')
    for key in full:
        for name in ('world', 'q', 'v', 'act', 'ctrl', 'time', 'tick', 'qfrc_applied',
                     'xfrc_applied', 'held_control', 'last_step', 'coordinate_system'):
            exact(getattr(full[key], name), getattr(integration[key], name), 'full/integration '+name)
        require(integration[key].observation_kind == 'accepted-integration-state', 'explicit observation kind')
    return full, integration


def advance(runtime, mj, model, packets, reference, journal, label):
    full_entry, entry = observations(runtime, mj)
    expected, analytic_expected = {}, {}
    for key in runtime.world_keys:
        native = mj.MjData(model)
        native.qpos[:], native.qvel[:], native.act[:] = entry[key].q, entry[key].v, entry[key].act
        native.time = entry[key].time
        set_control(native, packets[key])
        mj.mj_forward(model, native)
        matrix = np.zeros((model.nv, model.nv))
        mj.mj_fullM(model, matrix, native.qM)
        dynamics = dict(mass_matrix=matrix, bias=native.qfrc_bias.copy(),
                        passive=native.qfrc_passive.copy(),
                        applied=native.qfrc_smooth.copy()-native.qfrc_passive+native.qfrc_bias,
                        qacc=native.qacc.copy(), scalar_force=native.actuator_force.copy(),
                        generalized_force=native.qfrc_actuator.copy(), act_dot=native.act_dot.copy())
        native_input = native_state(native, force_phase="same-input-forward-dynamics")
        mj.mj_step(model, native)
        set_control(reference[key], packets[key])
        mj.mj_step(model, reference[key])
        expected[key] = dict(input=native_input, dynamics=dynamics,
                             output=native_state(native), continuous=native_state(reference[key]))
        analytic_expected[key] = analytic(model, entry[key], packets[key])
    journal.emit('attempt', label=label, full_entry=full_entry, integration_entry=entry,
                 requested_controls=packets, native=expected, analytic=analytic_expected)
    runtime.set_controls(packets)
    counts_before = dict(runtime.counters)
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU forward/step used for device physics/integration observation')
    with patch.object(mj, 'mj_step', forbidden), patch.object(mj, 'mj_forward', forbidden):
        accepted = runtime.step(H)
        immediate = runtime.integration_snapshot()
    full, current = observations(runtime, mj)
    counts_after = dict(runtime.counters)
    journal.emit('step', label=label, accepted=accepted, immediate=immediate,
                 full_output=full, integration_output=current, counters_before=counts_before,
                 counters_after=counts_after)
    exact(immediate, current, 'observation did not alter accepted state')
    require(dict(accepted) == dict.fromkeys(runtime.world_keys, True), 'complete accepted worlds')
    require(counts_after['completed_kernels']-counts_before['completed_kernels'] == 4, 'four real activation launches')
    for key, state in current.items():
        receipt = state.last_step
        a = receipt.actuation
        require(state.tick == entry[key].tick+1, 'continuous accepted tick')
        require(abs(state.time-entry[key].time-H) <= TIME_ATOL, 'step clock')
        np.testing.assert_array_equal(a.input_act, entry[key].act)
        np.testing.assert_array_equal(a.output_act, state.act)
        for name, target in analytic_expected[key].items():
            act_channel = name in ('output_act', 'activation_derivative')
            np.testing.assert_allclose(getattr(a, name), target,
                                       rtol=ACT_RTOL if act_channel else FORCE_RTOL,
                                       atol=ACT_ATOL if act_channel else FORCE_ATOL)
        expected_entry = expected[key]['dynamics']
        np.testing.assert_allclose(a.scalar_force, expected_entry['scalar_force'], rtol=FORCE_RTOL, atol=FORCE_ATOL)
        np.testing.assert_allclose(a.generalized_force, expected_entry['generalized_force'], rtol=FORCE_RTOL, atol=FORCE_ATOL)
        np.testing.assert_array_equal(a.generalized_force, receipt.qfrc_actuator)
        for name, rtol, atol in (('mass_matrix', 2e-4, 3e-5), ('bias', 3e-4, 2e-4),
                                ('passive', 3e-4, 2e-4), ('applied', 3e-4, 2e-4)):
            np.testing.assert_allclose(getattr(receipt.dynamics, name), expected_entry[name], rtol=rtol, atol=atol)
        np.testing.assert_allclose(receipt.qacc, expected_entry['qacc'], rtol=5e-4, atol=8e-4)
        for lane in ('output', 'continuous'):
            native = expected[key][lane]
            np.testing.assert_allclose(state.q, native['q'], rtol=0, atol=Q_ATOL)
            np.testing.assert_allclose(state.v, native['v'], rtol=0, atol=V_ATOL)
            np.testing.assert_allclose(state.act, native['act'], rtol=ACT_RTOL, atol=ACT_ATOL)
            require(abs(state.time-native['time']) <= TIME_ATOL, 'native clock')
    return current


def run(mode, output, journal):
    import mujoco as mj
    from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
    from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    require(mj.__version__ == '3.3.7', 'fixed independent MuJoCo oracle version')
    model = mj.MjModel.from_xml_string(XML)
    compiled = compile_mujoco_xml(XML, require_asset=False)
    initial = {key: MuJoCoInitialState(model.qpos0+np.array([.06, .025])+slot*.003,
               np.array([-.035, .04])-slot*.002, act=np.array([.34, .25, -.14])+slot*.01,
               time=.125) for slot, key in enumerate(KEYS)}
    def fresh_native(key):
        data = mj.MjData(model)
        for name, value in (('qpos', initial[key].q), ('qvel', initial[key].v), ('act', initial[key].act)):
            getattr(data, name)[:] = np.asarray(value, np.float32)
        data.time = initial[key].time
        return data
    reference = {key: fresh_native(key) for key in KEYS}
    def packet(tick, slot):
        command = .8 if tick < 6 else (-.75 if tick < 12 else 0.)
        wrench = np.zeros((model.nbody, 6))
        wrench[-1] = [.02, -.015, .01, .005, -.003, .004]
        # No residual command offset during the predeclared unloading phase.
        ctrl = np.full(model.nu, command*(1.+slot*.025))
        return MuJoCoControl(ctrl, qfrc_applied=np.full(model.nv, .012), xfrc_applied=wrench)
    def packets(tick):
        return {key: packet(tick, slot) for slot, key in enumerate(KEYS)}

    journal.emit('declaration', source_xml=XML, initial=initial, worlds=KEYS, steps=16,
                 gates=dict(act=(ACT_RTOL, ACT_ATOL), force=(FORCE_RTOL, FORCE_ATOL),
                            q_atol=Q_ATOL, v_atol=V_ATOL, time_atol=TIME_ATOL),
                 native_version=mj.__version__, modules=modules())
    runtime = MuJoCoRuntime(compiled, world_keys=KEYS, mode=mode, implementation='tyx', initial_states=initial)
    clone = None
    main_calls = clone_calls = 0
    try:
        journal.emit('initial', capabilities=runtime.capabilities, observations=observations(runtime, mj))
        for tick in range(16):
            advance(runtime, mj, model, packets(tick), reference, journal, 'main-'+str(tick))
            main_calls += 1
        held = packets(16)
        runtime.set_controls(held)
        checkpoint = runtime.checkpoint()
        raw = checkpoint.to_bytes()
        (output/'accepted.checkpoint').write_bytes(raw)
        capsule = MuJoCoTyxCheckpoint.from_bytes((output/'accepted.checkpoint').read_bytes())
        require(capsule.binding['schema'] == 'mujoco-tyx-exact-checkpoint-v2', 'stateful checkpoint version')
        saved_state = runtime.integration_snapshot()
        saved_reference = {key: copy.copy(value) for key, value in reference.items()}
        journal.emit('checkpoint', checkpoint_bytes=raw, checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                     binding=capsule.binding, state=saved_state, native={key: native_state(value) for key, value in reference.items()})
        clone = runtime.clone()
        exact(clone.integration_snapshot(), saved_state, 'clone accepted activation/state')
        expected = advance(runtime, mj, model, held, reference, journal, 'successor')
        main_calls += 1
        clone_reference = {key: copy.copy(value) for key, value in saved_reference.items()}
        clone_result = advance(clone, mj, model, held, clone_reference, journal, 'clone-successor')
        clone_calls += 1
        exact(clone_result, expected, 'clone real successor')
        runtime.restore(capsule, expected_revision=runtime.checkpoint_revision)
        exact(runtime.integration_snapshot(), saved_state, 'restore exact accepted state')
        reference = {key: copy.copy(value) for key, value in saved_reference.items()}
        repeated = advance(runtime, mj, model, held, reference, journal, 'restored-successor')
        main_calls += 1
        exact(repeated, expected, 'restored real successor')
        before_reset = runtime.integration_snapshot()
        runtime.reset((KEYS[-1],))
        after_reset = runtime.integration_snapshot()
        exact(after_reset[KEYS[0]], before_reset[KEYS[0]], 'unselected world retained')
        reset = after_reset[KEYS[-1]]
        np.testing.assert_array_equal(reset.act, np.asarray(initial[KEYS[-1]].act, np.float32))
        require(reset.tick == 0 and reset.last_step is None, 'selected reset publication')
        reference[KEYS[-1]] = fresh_native(KEYS[-1])
        journal.emit('partial-reset', before=before_reset, after=after_reset, selected=(KEYS[-1],))
        final = advance(runtime, mj, model, held, reference, journal, 'partial-reset-successor')
        main_calls += 1
        counts, clone_counts = dict(runtime.counters), dict(clone.counters)
        require(main_calls == 19 and clone_calls == 1, 'actual calls independent of rewound ticks')
        require(counts['accepted_world_steps'] == 38 and clone_counts['accepted_world_steps'] == 2, 'actual accepted world-step accounting')
        require(counts['failed_batches'] == 0 and clone_counts['failed_batches'] == 0, 'no physics refusal')
        journal.emit('final', state=final, full=runtime.snapshot(), counters=counts, clone_counters=clone_counts)
        return dict(main_calls=main_calls, clone_calls=clone_calls, accepted_world_steps=40,
                    main_counters=counts, clone_counters=clone_counts,
                    checkpoint_sha256=hashlib.sha256(raw).hexdigest(), final_modules=modules())
    finally:
        try:
            if clone is not None:
                clone.close()
        finally:
            runtime.close()
            journal.emit('closed', main_calls=main_calls, clone_calls=clone_calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('interpret', 'cuda'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute(), 'output path must be absolute')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output/'actuation-observation.jsonl.gz')
    summary = dict(status='RUNNING', mode=args.mode, consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   scope='installed T1 filterexact + full/integration observation + exact continuation',
                   excluded='contacts, floating roots, ball joints, training, G1')
    try:
        summary.update(run(args.mode, output, journal))
        summary['status'] = 'PASS'
    except BaseException as error:
        summary['status'] = 'FAIL'
        summary['exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        journal.emit('failure', summary=summary)
        raise
    finally:
        journal.emit('summary', summary=summary)
        summary['trace'] = journal.close()
        (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(summary, allow_nan=False))


if __name__ == '__main__':
    main()
