"""Batch the full grasp/transfer/insert experiment over independent reset states.

Cases, record indices and open_reference refer to list positions. Sparse native
world IDs appear in the audit. An open-reference pair replays arm targets with
each arm solving its own measured-state PD, while both open jaws stay commanded
open. No renderer is imported by this module.
"""
from collections.abc import Mapping
from operator import index

import numpy as np

from _panda_grasp_insert_control import ARM_KD, ARM_KP, DURATION_S, GraspInsertController
from _panda_grasp_insert_metrics import SHAPES, _episode, _trace, evaluate_position_control
from _panda_pickup_resets import initial_state
from panda_grasp_insert import Observer
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime


def _indices(cases, record, open_reference):
    def checked(i):
        if isinstance(i,(bool,np.bool_)):
            raise ValueError("case indices must be integers")
        i = index(i)
        if i < 0 or i >= len(cases):
            raise IndexError(f"case {i} outside this batch")
        return i
    saved = {checked(i) for i in record}
    if open_reference is None:
        pairs = {}
    elif isinstance(open_reference,Mapping):
        pairs = {checked(o):checked(c) for o,c in open_reference.items()}
    else:
        o,c = open_reference
        pairs = {checked(o):checked(c)}
    if set(pairs).intersection(pairs.values()):
        raise ValueError("an open negative cannot supply reference arm targets")
    return saved,pairs


def _storage(n, record):
    # The original episode scorer consumes every physics sample. Allocate one
    # contiguous float64 array per channel rather than a Python object per tick.
    a = {key:np.empty((n,*shape),dtype=np.float64) for key,shape in SHAPES.items()}
    a["time_s"] = np.empty(n)
    a["phase"] = np.empty(n,dtype="U12")
    if record:
        for name,width in (("arm_target_q",14),("arm_bias_nms",14),("contact_qpos",23)):
            a[name] = np.empty((n,width))
        a["contact_time_s"] = np.empty(n)
        a["contact_phase"] = np.empty(n,dtype="U12")
    return a


def run_batch(scene,compiled,cases,*,mode,cache,record=(),open_reference=None,progress=None):
    """Run all cases for 6000 original steps. Return (results in input order, audit).

    Each case is (reset_dict, PickupPolicy or None). Set record=(0,1) and
    open_reference={1:0} for two complete film/metrics traces and the paired
    position-control report at results[1]["position_control"]. progress, when
    supplied, is called as progress(completed_steps, total_steps).
    """
    cases = list(cases)
    if not cases or mode not in ("cpu","musa"):
        raise ValueError("require at least one case and mode='cpu' or 'musa'")
    saved,pairs = _indices(cases,record,open_reference)
    if compiled.source.primary_sha256 != scene.xml_sha256:
        raise ValueError("compiled MJCF differs from the scene used to reset and score the trials")
    model = compiled.mj_model
    dt = float(scene.timestep)
    if dt != .002 or float(model.opt.timestep) != dt or DURATION_S != 12.:
        raise ValueError("the original episode requires twelve seconds at 2 ms")
    ticks = round(DURATION_S/dt)
    keys = tuple(17+13*i for i in range(len(cases)))
    init = {k:initial_state(scene,reset) for k,(reset,_) in zip(keys,cases,strict=True)}
    for o,c in pairs.items():
        left,right = init[keys[o]],init[keys[c]]
        if left.time != right.time or not np.array_equal(left.q,right.q) or not np.array_equal(left.v,right.v):
            raise ValueError("matched open and closed cases must start at the same physical state")
    closed = [i for i in range(len(cases)) if i not in pairs]
    order = closed+list(pairs)
    controllers = [GraspInsertController(scene,model,open_jaw=i in pairs,policy=policy)
                   for i,(_,policy) in enumerate(cases)]
    traces = [_storage(ticks+1,i in saved) for i in range(len(cases))]
    phase = ["initial"]*len(cases)
    efforts = np.zeros((len(cases),model.nu))
    targets = np.array([init[k].q[:14] for k in keys])
    bias = np.zeros_like(targets)
    step_input = {i:np.array(init[keys[i]].q,copy=True) for i in saved}
    observations = [None]*len(cases)
    weight = float(model.body("assembly_part").mass[0]*np.linalg.norm(model.opt.gravity))
    observer = Observer(model)
    opts = dict(cache_root=cache,contact_capacity=64,constraint_capacity=256) if mode=="musa" else {}
    with MuJoCoRuntime(compiled,mode=mode,world_keys=keys,initial_states=init,**opts) as rt:
        snapshot = rt.integration_snapshot if mode=="musa" else rt.snapshot
        before = rt.musa_device_report() if mode=="musa" else {}
        for tick in range(ticks+1):
            states = snapshot()
            for i,k in enumerate(keys):
                state = states[k]
                a = traces[i]
                row = observer.measure(state,efforts[i],phase[i])
                for name,value in row.items():
                    a[name][tick] = value
                if i in saved:
                    report = state.last_step
                    a["contact_time_s"][tick] = (float(state.time) if report is None else
                        report.input_time if report.contact_time is None else report.contact_time)
                    a["contact_qpos"][tick] = step_input[i]
                    a["contact_phase"][tick] = phase[i]
                    a["arm_target_q"][tick] = targets[i]
                    a["arm_bias_nms"][tick] = bias[i]
            if progress is not None and (tick%500==0 or tick==ticks):
                progress(tick,ticks)
            if tick==ticks:
                break
            commands = {}
            for i in order:
                k = keys[i]
                state = states[k]
                control = controllers[i]
                ref = targets[pairs[i]] if i in pairs else None
                chosen = control.advance(state.q,state.v,tick*dt,reference_arm_target=ref)
                diag = chosen.diagnostics
                if observations[i] is None and diag.get("pickup_observation") is not None:
                    observations[i] = diag["pickup_observation"]
                phase[i] = chosen.phase
                targets[i] = diag["arm_target_q"]
                efforts[i] = chosen.ctrl
                commands[k] = MuJoCoControl(chosen.ctrl)
                if i in saved:
                    step_input[i][:] = state.q
                    bias[i] = control.data.qfrc_bias[:14]
            rt.set_controls(commands)
            rt.step(dt)
        after = rt.musa_device_report() if mode=="musa" else {}
    completed = None
    if mode=="musa":
        completed = after["completed_kernels"]-before["completed_kernels"]
        if "S4000" not in after["device_name"] or completed<=0 or after["launched_kernels"] != after["completed_kernels"]:
            raise RuntimeError("could not verify completed S4000 execution")
    for i in saved:
        traces[i].update(arm_kp=ARM_KP.copy(),arm_kd=ARM_KD.copy(),
                         arm_ctrlrange=controllers[i].limits[:14].copy(),model_sha256=scene.xml_sha256)
    comparisons = {o:evaluate_position_control(traces[c],traces[o],dt,weight)
                   for o,c in pairs.items() if o in saved and c in saved}
    scores = {}
    for o,comparison in comparisons.items():
        scores[o] = comparison["opened"]
        scores[pairs[o]] = comparison["closed"]
    results = []
    for i,k in enumerate(keys):
        trace = traces[i]
        score = scores[i] if i in scores else _episode(_trace(trace,dt),weight)
        end_close = np.flatnonzero(trace["phase"]=="close")[-1]
        delta = trace["wrist_xyz"][end_close]-trace["part_xyz"][end_close]
        diagnostics = dict(pickup_observation=observations[i],end_close_tcp_minus_part_m=delta.tolist(),
                           end_close_pad_normals_n=trace["pad_forces"][end_close].tolist())
        results.append(dict(world_id=k,score=score,trace=trace if i in saved else None,
                            position_control=comparisons.get(i),diagnostics=diagnostics))
    audit = dict(mode=mode,device_name=after.get("device_name","CPU MuJoCo"),world_keys=list(keys),
                 steps_per_world=ticks,timestep_s=dt,seconds_per_world=ticks*dt,weight_n=weight,
                 model_sha256=scene.xml_sha256,completed_kernels=completed)
    if mode=="musa":
        audit["device_report_before"] = before
        audit["device_report_after"] = after
    return results,audit
