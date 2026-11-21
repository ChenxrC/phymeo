"""One-step-lag load feedforward for the declared public Push-T experiment.

The original active route and PD policy stay unchanged. Previous accepted area
point forces on the tool are projected onto the route's horizontal direction;
half the opposing load is mapped through the current pusher-centre Jacobian.
This HOST controller does no integration, planning or CUDA action transport.
"""

from collections.abc import Mapping
from dataclasses import asdict

import mujoco as mj
import numpy as np

from phymeo.control.manipulation_diagnostic import NativeJointPD
from phymeo.control.manipulation_push_planner import NativePushRoute
from phymeo.geometry.area_pressure import AreaPressureIdentity, AreaPressureQuadrature
from phymeo.solvers.area_friction import LAW

from .pusht_public_scene import IDENTITY

GAIN = 0.5
EFFORT_LIMIT = 19.9
SOURCE_SCHEMA = "mujoco-tyx-area-friction-resident-native-v1"


def _require(ok, reason):
    if not ok:
        raise ValueError("push load compensation: " + reason)


def _array(value, shape, name):
    values = np.asarray(value)
    _require(
        values.shape == shape and values.dtype.kind in "fiu" and np.isfinite(values).all(),
        "finite " + name + " with shape " + str(shape),
    )
    return np.asarray(values, dtype=np.float64)


def _layout(entry):
    """Prove the original PD's two scalar efforts map to unit direct motors."""
    model, pd = entry.model, entry.pd
    _require(
        isinstance(model, mj.MjModel)
        and isinstance(pd, NativeJointPD)
        and pd.model is model
        and entry.driver.model is model
        and model.nu == 2,
        "one original model and two direct native motors required",
    )
    names = tuple(pd.joint_names)
    _require(
        len(names) == 2 and len(set(names)) == 2 and names == tuple(entry.scene.robot_joint_names),
        "declared controlled joints",
    )
    joints = tuple(int(model.joint(name).id) for name in names)
    qa = tuple(int(model.jnt_qposadr[j]) for j in joints)
    va = tuple(int(model.jnt_dofadr[j]) for j in joints)
    actors = tuple(pd.actuator_ids)
    _require(
        all(int(model.jnt_type[j]) == int(mj.mjtJoint.mjJNT_HINGE) for j in joints)
        and tuple(pd.qpos_addresses) == qa
        and tuple(pd.dof_addresses) == va
        and sorted(actors) == [0, 1],
        "two hinge-joint PD/motor address mapping",
    )
    for joint, actor in zip(joints, actors, strict=True):
        _require(
            int(model.actuator_trntype[actor]) == int(mj.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_trnid[actor, 0]) == joint
            and int(model.actuator_dyntype[actor]) == int(mj.mjtDyn.mjDYN_NONE)
            and int(model.actuator_gaintype[actor]) == int(mj.mjtGain.mjGAIN_FIXED)
            and model.actuator_gainprm[actor, 0] == 1.0
            and int(model.actuator_biastype[actor]) == int(mj.mjtBias.mjBIAS_NONE),
            "stateless direct unit-gain joint motor required",
        )
        gear = np.asarray(model.actuator_gear[actor])
        _require(
            np.isfinite(gear).all() and gear[0] == 1.0 and not np.any(gear[1:]),
            "unit scalar motor gear required",
        )
        for limited, interval in (
            (model.actuator_ctrllimited[actor], model.actuator_ctrlrange[actor]),
            (model.actuator_forcelimited[actor], model.actuator_forcerange[actor]),
            (model.jnt_actfrclimited[joint], model.jnt_actfrcrange[joint]),
        ):
            _require(
                not limited
                or (
                    np.isfinite(interval).all()
                    and interval[0] <= -EFFORT_LIMIT
                    and interval[1] >= EFFORT_LIMIT
                ),
                "original motor limits do not admit the declared +/-19.9 bound",
            )
    order = tuple(int(i) for i in np.argsort(actors))
    return names, qa, va, actors, order


class PushLoadCompensation:
    """A fixed half-load correction, evaluated from each trial's own snapshot.

    Returned efforts and Jacobian columns use native actuator order. ``gain``
    is always the preregistered 0.5; ``enabled`` denotes entry/push phase, while
    ``applied`` also requires a strictly resisting net load. Neither zero load
    nor approach permits skipping source-receipt validation.
    """

    def __init__(self, entry, *, identity=IDENTITY):
        _require(isinstance(identity, AreaPressureIdentity), "registered area material identity")
        self._identity = identity
        self._entry, self._driver = entry, entry.driver
        self._layout = _layout(entry)
        self._model = entry.model
        self._scratch = mj.MjData(self._model)
        self._bodies = (
            int(entry.model.body(identity.compliant_body).id),
            int(entry.model.body(identity.rigid_body).id),
        )
        self._pusher = int(entry.model.geom("pusher").id)
        _require(
            self._bodies[0] != self._bodies[1]
            and int(entry.model.geom_bodyid[self._pusher]) == self._bodies[0],
            "declared pusher geometry and tool/load body ownership",
        )

    def _previous_force(self, snapshot):
        receipt = snapshot.last_step
        _require(receipt is not None, "previous accepted receipt required")
        _require(
            type(snapshot.tick) is int
            and snapshot.tick > 0
            and type(receipt.tick) is int
            and receipt.tick == snapshot.tick
            and receipt.world == snapshot.world
            and receipt.output_time == snapshot.time,
            "previous receipt local tick/world/output time binding",
        )
        _require(
            np.isfinite(receipt.input_time)
            and np.isfinite(receipt.output_time)
            and receipt.input_time < receipt.output_time
            and receipt.contact_time == receipt.input_time
            and receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
            and receipt.state_precision == "fp32-device-integration"
            and receipt.backend in ("mujoco-tyx-cuda", "mujoco-tyx-interpret")
            and receipt.h == self._entry.scene.timestep_s,
            "previous accepted force evaluation phase/backend",
        )
        raw = receipt.raw_contact_fields
        _require(isinstance(raw, Mapping), "raw accepted area publication required")
        pairs, solver = raw["area_friction"], raw["area_friction_solver"]
        _require(
            isinstance(pairs, (tuple, list))
            and len(pairs) == 1
            and solver["schema"] == SOURCE_SCHEMA
            and solver["law"] == LAW
            and solver["precision"] == "actual-packed-point-fp32"
            and type(solver["solver_status"]) is int
            and solver["solver_status"] == 0
            and np.array_equal(solver["output_velocity"], snapshot.v),
            "unique converged declared area solver publication",
        )
        pair = pairs[0]
        _require(
            pair["identity"] == self._identity
            and tuple(pair["body_ids"]) == self._bodies
            and pair["law"] == LAW,
            "ordered declared tool/load material pair",
        )
        quad = pair["source_quadrature"]
        _require(
            isinstance(quad, AreaPressureQuadrature)
            and quad.identity == self._identity
            and pair["stamp"] == quad.stamp
            and quad.stamp.world_id == str(snapshot.world)
            and quad.stamp.step_index == snapshot.tick - 1
            and quad.stamp.sample_time_s == receipt.input_time
            and tuple(pair["sample_ids"]) == quad.sample_ids
            and tuple(pair["source_ids"]) == quad.source_ids,
            "previous source quadrature/sample/input stamp binding",
        )
        count = len(quad.sample_ids)
        fa = _array(pair["force_on_a"], (count, 3), "actual point forces on tool")
        fb = _array(pair["force_on_b"], (count, 3), "actual point forces on load")
        fn = _array(pair["normal_force"], (count,), "actual point normal loads")
        _require(
            np.all(fn >= 0) and np.array_equal(fa, -fb),
            "nonnegative normal loads and actual bilateral force ownership",
        )
        force = fa.sum(axis=0, dtype=np.float64)
        _require(np.isfinite(force).all(), "finite previous net force")
        return receipt, quad, force

    def __call__(self, entry, snapshot, diagnostic):
        _require(
            entry is self._entry
            and entry.model is self._model
            and entry.driver is self._driver
            and _layout(entry) == self._layout,
            "controller belongs to another entry or motor layout",
        )
        names, qa, va, actors, order = self._layout
        model = self._model
        q = _array(snapshot.q, (model.nq,), "current q")
        v = _array(snapshot.v, (model.nv,), "current v")
        _require(
            snapshot.observation_kind == "accepted-integration-state"
            and isinstance(diagnostic, Mapping)
            and diagnostic["input_tick"] == entry.driver.tick
            and diagnostic["local_tick"] == snapshot.tick
            and diagnostic["input_time"] == snapshot.time
            and np.array_equal(entry.driver.last_q, q)
            and np.array_equal(diagnostic["measurement_q"], q)
            and np.array_equal(diagnostic["measurement_v"], v),
            "current accepted controller/PD state binding",
        )
        pd = diagnostic["pd"]
        _require(
            tuple(pd["joint_names"]) == names
            and tuple(pd["qpos_addresses"]) == qa
            and tuple(pd["dof_addresses"]) == va
            and tuple(pd["actuator_ids"]) == actors
            and np.array_equal(pd["position"], q[list(qa)])
            and np.array_equal(pd["velocity"], v[list(va)])
            and np.array_equal(pd["target"], diagnostic["target_q"])
            and np.array_equal(pd["target_velocity"], diagnostic["target_v"]),
            "original PD diagnostic addresses/state/reference",
        )
        pd_raw = _array(pd["unclipped_effort"], (2,), "original unclipped PD effort")
        pd_effort = _array(pd["effort"], (2,), "original clipped PD effort")
        _require(
            np.array_equal(pd_effort, np.clip(pd_raw, -EFFORT_LIMIT, EFFORT_LIMIT)),
            "original PD effort limit contract",
        )
        active = entry.driver.active
        _require(
            isinstance(active, tuple)
            and len(active) == 2
            and isinstance(active[1], NativePushRoute)
            and active[1].proposal is not None
            and active[1].status == "sampled_reference_ready",
            "original active ordinary route",
        )
        start, route = active
        elapsed = (entry.driver.tick - start) * entry.scene.timestep_s
        _require(
            diagnostic["route"]["route_start_tick"] == start
            and diagnostic["route"]["route_elapsed_s"] == elapsed
            and elapsed >= 0,
            "active route source time binding",
        )
        stage = route.nodes[-1][1]
        for end, phase, _ in route.nodes[1:]:
            if elapsed <= end:
                stage = phase
                break
        _require(diagnostic["stage"] == stage, "actual active route stage")
        direction = _array(route.proposal.direction_world, (3,), "push direction")
        _require(
            direction[2] == 0.0
            and abs(float(direction @ direction) - 1.0) <= 64 * np.finfo(float).eps,
            "horizontal unit push direction required without renormalization",
        )
        # Always bind the previous receipt, even when phase gating disables the
        # correction or the correctly published source quadrature is empty.
        receipt, quad, force = self._previous_force(snapshot)
        resisting = max(0.0, -float(force @ direction))
        _require(np.isfinite(resisting), "finite resisting force projection")
        data = self._scratch
        data.qpos[:], data.qvel[:] = q, v
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        jacp = np.empty((3, model.nv))
        mj.mj_jacGeom(model, data, jacp, None, self._pusher)
        dofs = tuple(va[i] for i in order)
        jacobian = jacp[:, list(dofs)].copy()
        _require(
            np.isfinite(jacobian).all() and np.array_equal(data.qpos, q),
            "finite current-configuration pusher Jacobian",
        )
        enabled = stage in ("entry", "push")
        applied = enabled and resisting > 0.0
        correction = (
            jacobian.T @ (GAIN * resisting * direction)
            if applied
            else np.zeros(2, dtype=np.float64)
        )
        original_raw = pd_raw[list(order)]
        combined = original_raw + correction
        _require(
            np.isfinite(correction).all() and np.isfinite(combined).all(),
            "finite combined original PD and load compensation",
        )
        effort = np.clip(combined, -EFFORT_LIMIT, EFFORT_LIMIT)
        report = dict(
            schema="pusht-load-compensation-half-v1",
            source_local_tick=receipt.tick,
            source_output_time=receipt.output_time,
            source_input_local_tick=quad.stamp.step_index,
            source_input_time=receipt.input_time,
            source_world=int(receipt.world),
            source_backend=receipt.backend,
            source_schema=SOURCE_SCHEMA,
            source_identity=asdict(self._identity),
            source_body_ids=list(self._bodies),
            source_sample_count=len(quad.sample_ids),
            force_phase="previous-accepted-step-input-geometry",
            previous_force_on_tool_world_n=force.tolist(),
            push_direction_world=direction.tolist(),
            jacobian_world=jacobian.tolist(),
            jacobian_configuration="current-accepted-input",
            controlled_joint_names=[names[i] for i in order],
            qpos_addresses=[qa[i] for i in order],
            dof_addresses=list(dofs),
            actuator_ids=[0, 1],
            effort_order="native-actuator-order",
            stage=stage,
            resisting_force_n=resisting,
            gain=GAIN,
            enabled=enabled,
            applied=applied,
            correction_nm=correction.tolist(),
            original_pd_unclipped_nm=original_raw.tolist(),
            original_pd_effort_nm=pd_effort[list(order)].tolist(),
            combined_unclipped_nm=combined.tolist(),
            combined_effort_nm=effort.tolist(),
        )
        return effort.copy(), report
