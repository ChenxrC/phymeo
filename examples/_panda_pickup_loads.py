"""Forces on the free workpiece at co-timed native MuJoCo contact solves.

Write the physical control, run mj_forward(model, data), then sample before
integration. Plain mj_step advances time/q/v beyond its cached contact solve.
The caller records a full native trajectory or explicitly labels a native
reconstruction of another backend's states.
"""

import mujoco as mj
import numpy as np

from _panda_grasp_insert_scene import PAD_NAMES, RAIL_NAMES
from _panda_grasp_insert_metrics import _runs


SOURCES = (*PAD_NAMES, *RAIL_NAMES, "other")
PARTS = ("part_handle", "part_shoulder", "assembly_pin")
MIN_NORMAL_WEIGHT = .02
UNLOADED_WEIGHT = .05
BEARING_WEIGHT = .8
LIFT_M = .025
DWELL_S = .150
SLIP_M = .005
SOLVERS = {"native_mujoco_cpu": 1, "native_mujoco_reconstruction": 1, "musa_device_same_solve": 2}


class PickupLoads:
    def __init__(self, model):
        self.model = model
        self.part_body = model.body("assembly_part").id
        joint = int(model.body_jntadr[self.part_body])
        if joint < 0 or model.body_jntnum[self.part_body] != 1 or model.jnt_type[joint] != mj.mjtJoint.mjJNT_FREE:
            raise ValueError("assembly_part must be a free body")
        self.part_dof = int(model.jnt_dofadr[joint])
        self.source_ids = {model.geom(name).id: i for i, name in enumerate(SOURCES[:-1])}
        self.part_ids = {model.geom(name).id: i for i, name in enumerate(PARTS)}
        if set(self.part_ids) != set(np.flatnonzero(model.geom_bodyid == self.part_body)):
            raise ValueError("unaccounted free-part geometry")
        self.part_site = model.site("part_grasp_center").id
        self.wrist_site = model.site("right_grasp_center").id
        self.jaws = [model.jnt_qposadr[model.joint(f"right_panda_finger_joint{i}").id] for i in (1, 2)]
        gravity = np.asarray(model.opt.gravity)
        strength = np.linalg.norm(gravity)
        if not np.isfinite(strength) or strength <= 0.:
            raise ValueError("a nonzero finite gravity vector is required to define load")
        self.up = -gravity / strength
        self.weight_n = float(model.body_mass[self.part_body] * strength)

    def sample(self, data):
        """Each force acts ON the part. Rows are SOURCES, columns are PARTS."""
        force = np.zeros((5, 3, 3))
        normal_vectors = np.zeros_like(force)
        normal = np.zeros((5, 3))
        count = np.zeros((5, 3), dtype=np.int32)
        wrench = np.empty(6)
        for i in range(data.ncon):
            contact = data.contact[i]
            a, b = map(int, contact.geom)
            if contact.efc_address < 0 or (a in self.part_ids) == (b in self.part_ids):
                continue
            # MuJoCo's row-basis normal points geom[0] -> geom[1].
            # mj_contactForce expresses the wrench ON geom[1] in that frame.
            part, other, sign = (b, a, 1.) if b in self.part_ids else (a, b, -1.)
            src, dst = self.source_ids.get(other, 4), self.part_ids[part]
            mj.mj_contactForce(self.model, data, i, wrench)
            frame = contact.frame.reshape(3, 3)
            force[src, dst] += sign * (frame.T @ wrench[:3])
            normal_vectors[src, dst] += sign * wrench[0] * frame[0]
            normal[src, dst] += wrench[0]
            count[src, dst] += 1
        p, w = data.site_xpos[[self.part_site, self.wrist_site]]
        rot = data.site_xmat[self.wrist_site].reshape(3, 3)
        dofs = slice(self.part_dof, self.part_dof + 6)
        return dict(time_s=float(data.time), contact_solver_code=np.int8(1),
                    qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy(),
                    finger_opening_m=data.qpos[self.jaws].copy(), part_xyz=p.copy(), wrist_xyz=w.copy(),
                    part_in_hand_m=rot.T @ (p - w), force_world_n=force, normal_world_n=normal_vectors,
                    normal_n=normal, up_n=force @ self.up, contact_count=count,
                    part_noncontact_applied=np.r_[data.xfrc_applied[self.part_body], data.qfrc_applied[dofs],
                                                  data.qfrc_actuator[dofs]])

    def _trace(self, trace):
        t = np.asarray(trace["time_s"], dtype=float)
        if (t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all()
                or not np.allclose(np.diff(t), self.model.opt.timestep, rtol=.001, atol=1e-9)):
            raise ValueError("provide every solved physics input state in time order")
        shapes = dict(qpos=(self.model.nq,), qvel=(self.model.nv,), ctrl=(self.model.nu,),
                      finger_opening_m=(2,), part_xyz=(3,), wrist_xyz=(3,), part_in_hand_m=(3,),
                      force_world_n=(5, 3, 3), normal_n=(5, 3), up_n=(5, 3), part_noncontact_applied=(18,))
        rows = {"time_s": t}
        for name, shape in shapes.items():
            values = np.asarray(trace[name])
            if values.shape != (len(t), *shape) or not np.isfinite(values).all():
                raise ValueError(f"invalid solved-state channel: {name}")
            rows[name] = values
        if (np.any(rows["normal_n"] < -1e-9) or
                not np.allclose(rows["force_world_n"] @ self.up, rows["up_n"], rtol=2e-6, atol=1e-7)):
            raise ValueError("contact loads disagree with their normal magnitude or world force projection")
        return rows

    @staticmethod
    def _source(trace, label):
        codes = trace.get("contact_solver_code")
        if codes is not None:
            codes = np.asarray(codes)
            if codes.ndim != 1 or len(codes) != len(trace["time_s"]) or not len(codes) or not np.all(codes == codes[0]):
                raise ValueError("each sample must declare the same contact solver")
            if label is None:
                label = {1: "native_mujoco_cpu", 2: "musa_device_same_solve"}.get(codes[0])
        if label not in SOLVERS or (codes is not None and codes[0] != SOLVERS[label]):
            raise ValueError("declare the actual contact_source; it must match any recorded solver code")
        return label

    def score(self, trace, *, contact_source=None):
        """Require support transfer and co-motion during one uninterrupted bilateral grip."""
        source = self._source(trace, contact_source)
        d = self._trace(trace)
        t, n, up = d["time_s"], d["normal_n"], d["up_n"]
        p, w, local = d["part_xyz"], d["wrist_xyz"], d["part_in_hand_m"]
        weight = self.weight_n
        threshold = MIN_NORMAL_WEIGHT * weight
        pads = n[:, :2].sum(axis=2)
        handle = n[:, :2, 0] >= threshold
        both = handle.all(axis=1)
        rail_up = up[:, 2:4].sum(axis=(1, 2))
        fork_supported = rail_up >= BEARING_WEIGHT * weight
        supported_once = np.maximum.accumulate(fork_supported)
        unloaded = n[:, 2:4].sum(axis=(1, 2)) <= UNLOADED_WEIGHT * weight
        others_clear = n[:, 4].sum(axis=1) <= UNLOADED_WEIGHT * weight
        pad_up = up[:, :2].sum(axis=2)
        vertical_load = pad_up.sum(axis=1) >= BEARING_WEIGHT * weight
        height = (p - p[0]) @ self.up

        def first(mask):
            idx = np.flatnonzero(mask)
            return float(t[idx[0]]) if len(idx) else None

        windows = []
        for start, stop in _runs(both):
            acquired = np.flatnonzero((np.abs(height[start:stop]) <= SLIP_M)
                                      & supported_once[start:stop])
            if not len(acquired):
                continue
            grasp = int(start + acquired[0])
            ids = slice(grasp, stop)
            slip = np.linalg.norm(local[ids] - local[grasp], axis=1)
            hand_rise = (w[ids] - w[grasp]) @ self.up
            carried = (unloaded[ids] & others_clear[ids] & vertical_load[ids] & (height[ids] >= LIFT_M)
                       & (hand_rise >= LIFT_M - SLIP_M) & (np.maximum.accumulate(slip) <= SLIP_M))
            for begin, end in _runs(carried):
                k, last = int(grasp + begin), int(grasp + end - 1)
                windows.append(dict(grasp_time_s=float(t[grasp]), start_s=float(t[k]), end_s=float(t[last]),
                                    dwell_s=float(t[last] - t[k]), min_part_lift_m=float(height[k:last+1].min()),
                                    min_wrist_rise_m=float(hand_rise[begin:end].min()),
                                    peak_slip_since_grasp_m=float(slip[:end].max()),
                                    min_handle_normal_by_pad_n=n[k:last+1, :2, 0].min(axis=0).tolist(),
                                    mean_up_by_pad_n=pad_up[k:last+1].mean(axis=0).tolist()))
        longest = max((x["dwell_s"] for x in windows), default=0.)
        no_external = not np.any(d["part_noncontact_applied"])
        return dict(success=bool(no_external and longest + 1e-9 >= DWELL_S),
                    contact_source=source, force_time="co-timed control and physical input to contact solve",
                    force_axes=dict(sources=list(SOURCES), part_geometries=list(PARTS),
                                    upward_world_unit_vector=self.up.tolist()),
                    weight_n=weight, no_direct_applied_part_effort=bool(no_external),
                    thresholds=dict(min_handle_normal_weight_per_pad=MIN_NORMAL_WEIGHT,
                                    min_combined_pad_up_weight=BEARING_WEIGHT, unloaded_normal_weight=UNLOADED_WEIGHT,
                                    min_lift_m=LIFT_M, min_dwell_s=DWELL_S, max_hand_frame_slip_m=SLIP_M,
                                    min_wrist_rise_m=LIFT_M-SLIP_M),
                    first_fork_support_s=first(fork_supported), first_fork_unloaded_after_support_s=first(supported_once & unloaded),
                    first_pad_load_s=[first(pads[:, i] >= threshold) for i in range(2)],
                    first_pad_handle_load_s=[first(handle[:, i]) for i in range(2)],
                    first_bilateral_handle_s=first(both), first_one_sided_only_s=first(handle.sum(axis=1) == 1),
                    first_bilateral_still_on_fork_s=first(both & fork_supported),
                    first_pin_lift_s=first(height >= LIFT_M), peak_lift_m=float(height.max()),
                    peak_part_displacement_m=float(np.linalg.norm(p - p[0], axis=1).max()),
                    peak_normal_by_source_part_n=n.max(axis=0).tolist(),
                    min_up_by_source_part_n=up.min(axis=0).tolist(), max_up_by_source_part_n=up.max(axis=0).tolist(),
                    longest_loaded_dwell_s=longest, loaded_windows=windows)

    def compare(self, closed, opened, *, control_mode, contact_source=None):
        """Score an actual negative trajectory and verify its specified intervention.

        open_jaw verifies both measured openings, motors_off every submitted
        effort, one_sided the observed pad loads, fork_supported the rail load.
        Arm target or torque equality is named separately when evidenced.
        """
        pos = self.score(closed, contact_source=contact_source)
        neg = self.score(opened, contact_source=contact_source)
        c, o = self._trace(closed), self._trace(opened)
        matched = np.array_equal(c["time_s"], o["time_s"])
        same_start = all(np.array_equal(c[k][0], o[k][0]) for k in ("qpos", "qvel"))
        n = o["normal_n"][:, :2].sum(axis=2) >= MIN_NORMAL_WEIGHT * self.weight_n
        if control_mode == "motors_off":
            intervention = not np.any(o["ctrl"])
            evidence = "all submitted motor efforts zero"
        elif control_mode == "open_jaw":
            intervention = np.all(o["finger_opening_m"] > .01202)
            evidence = "both measured finger positions clear the nominal handle half-width"
        elif control_mode == "one_sided":
            intervention = np.any(n.any(axis=1)) and not np.any(n.all(axis=1))
            evidence = "observed load on only one pad at a time"
        elif control_mode == "fork_supported":
            intervention = np.all(o["up_n"][:, 2:4].sum(axis=(1, 2)) >= BEARING_WEIGHT * self.weight_n)
            evidence = "observed fork reaction throughout the trajectory"
        else:
            raise ValueError(f"unknown physical control comparison: {control_mode}")
        gates = dict(same_contact_source=pos["contact_source"] == neg["contact_source"],
                     same_initial_state=bool(same_start), same_clock=bool(matched), loaded_pickup=pos["success"],
                     negative_intervention_verified=bool(intervention), negative_no_loaded_pickup=not neg["success"],
                     negative_no_direct_applied_part_effort=neg["no_direct_applied_part_effort"],
                     negative_part_did_not_lift=neg["peak_lift_m"] < LIFT_M)
        arms_equal = bool(matched and c["ctrl"].shape[1] == 16 and c["ctrl"].dtype == o["ctrl"].dtype
                          and c["ctrl"][:, :14].tobytes() == o["ctrl"][:, :14].tobytes())
        return dict(success=all(gates.values()), control_mode=control_mode, negative_evidence=evidence, gates=gates,
                    observed_bit_equal_arm_efforts=arms_equal, closed=pos, negative=neg)
