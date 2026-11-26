"""Trial lift and in-plane pitch feedback for the four-contact carry actor."""

import numpy as np


def _pitch(forward):
    return np.arctan2(-forward[..., 2], forward[..., 0])


def _angle_delta(current, initial):
    return np.arctan2(np.sin(current-initial), np.cos(current-initial))


class _RecoveryVision:
    """Keep the last optical fit available for braking when stereo drops out."""

    def __init__(self, sensor, guard):
        self.sensor, self.guard, self.last = sensor, guard, None

    def __getattr__(self, key):
        return getattr(self.sensor, key)

    @property
    def pose(self):
        current = self.sensor.pose
        if current is not None:
            self.last = current
            return current
        if self.guard.stage in ("trial", "verified", "recovery"):
            return self.last
        return None


class _StableGrasp:
    def __init__(self, actor):
        self.base = actor.fiducial._carry
        actor.fiducial._carry = self.carry
        actor.fiducial.tracker = _RecoveryVision(actor.fiducial.tracker, self)
        self.stage = "acquiring"
        self.anchor = None
        self.trial_target_z = None
        self.last_tool_z = None
        self.hold_s = self.alarm_s = self.sample_age_s = 0.
        self.pitch = self.pitch_rate = 0.
        self.rel_angle = np.zeros(2)
        self.drop = np.zeros(2)
        self.previous_delta = np.zeros((2, 7))
        self.reason = None
        self.last_measurement = None
        self.seen_this_step = False

    def carry(self, actor, tools, wrist, position, r, yaw, obj, goal, gains, fresh):
        c, dt = actor.fiducial, actor.dt
        self.seen_this_step = True
        full = fresh and c.tracker.rotation_age == 0
        rotations = np.stack([actor.data.site_xmat[s].reshape(3, 3) for s, _ in actor.sites])
        # Per-wrist box pose in the encoder-derived tool frame. The calibration
        # is saved while clamped, before the first airborne measurement.
        local_forward = np.einsum("nji,j->ni", rotations, r[:, 0])
        local_height = np.einsum("ni,ni->n", rotations[:, :, 2], position-tools)
        relative_pitch = _pitch(local_forward)
        world_pitch = np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0]))
        loads = -wrist[:, 2]
        normals = np.array([1., -1.])*(wrist@r[:, 1])
        bottom = None if obj is None else float(obj[3])
        supported = (hasattr(c, "weight") and min(loads) >= .25*c.weight
                     and loads.sum() >= .84*c.weight and min(normals) >= 17.)
        vz = (np.zeros(2) if self.last_tool_z is None else (tools[:, 2]-self.last_tool_z)/dt)
        self.last_tool_z = tools[:, 2].copy()
        if c.phase in ("acquire", "clamp") and full and self.stage == "acquiring":
            self.anchor = (relative_pitch.copy(), local_height.copy(), float(world_pitch))
            self.floor_tool_z = tools[:, 2].copy()
            self.resting_bottom = bottom
            self.last_measurement = float(world_pitch)
            self.pitch_rate = self.sample_age_s = 0.
        trial_height = None if bottom is None or self.anchor is None else bottom-self.resting_bottom
        if (self.stage == "trial" and self.trial_target_z is None and full and supported
                and trial_height is not None and trial_height >= .003):
            # Transfer of load changes jaw seating even when the box stays
            # level. Establish the relative grip once at measured liftoff.
            # Retain the pre-lift world-level reference throughout the carry.
            self.anchor = (relative_pitch.copy(), local_height.copy(), self.anchor[2])
            self.trial_target_z = tools[:, 2]+max(0., .010-trial_height)
        self.sample_age_s += dt
        if self.anchor is not None and full:
            self.rel_angle = _angle_delta(relative_pitch, self.anchor[0])
            self.drop = self.anchor[1]-local_height
            self.pitch = float(_angle_delta(world_pitch, self.anchor[2]))
            # Differentiate actual exposures, including gaps in the fit.
            rate = _angle_delta(world_pitch, self.last_measurement)/self.sample_age_s
            gain = -np.expm1(-self.sample_age_s/.055)
            self.pitch_rate += gain*(float(np.clip(rate, -4., 4.))-self.pitch_rate)
            self.last_measurement = float(world_pitch)
            self.sample_age_s = 0.
        if self.stage in ("trial", "verified") and c.phase in ("lift", "transport"):
            stale = self.sample_age_s >= .096
            # Confirm across exposures. A rapid rotation, loss of bilateral
            # support, or measured downward migration relative to either jaw
            # interrupts translation before the base actor's lost-grip timer.
            moving = self.trial_target_z is not None
            # A symmetric displacement of the box relative to both jaws can
            # reflect load transfer. Interrupt on downward displacement with
            # independent evidence that the supported box is falling/tipping.
            tipping = abs(self.pitch) > .065 and self.pitch*self.pitch_rate > .038
            falling = c.speed[2] < -.055 or loads.sum() < .84*c.weight
            slip = full and moving and (
                (np.max(self.drop) > .013 and (falling or tipping))
                or np.max(abs(self.rel_angle)) > .15
                or (min(loads) < .12*c.weight and min(normals) < 13.))
            if full:
                self.alarm_s = self.alarm_s+dt if slip else 0.
            if stale or self.alarm_s >= .030:
                self.stage, self.reason = "recovery", "stale_attitude" if stale else "grasp_motion"
                self.recovery_tools = tools.copy()
                self.recovery_last_z = tools[:, 2].copy()
                self.recovery_side = r[:, 1].copy()
                c.phase, c.regrasp = "lower", True
                c.ref[:2] = tools.mean(axis=0)[:2]
                c.carry_trim_xy.fill(0.)

        if self.stage == "trial":
            # Verify a supported, nearly stationary trial before allowing full
            # lift. No floor-contact flag or load/pose truth enters this test.
            steady = (full and self.trial_target_z is not None and .003 <= trial_height <= .018
                      and supported and np.max(abs(self.rel_angle)) <= .10
                      and np.max(self.drop) <= .008 and abs(self.pitch_rate) < .40
                      and abs(c.speed[2]) < .055)
            self.hold_s = self.hold_s+dt if steady else 0.
            if self.hold_s >= .080:
                self.stage = "verified"

        # The parent computes its bounded z reference from the installed floor
        # datum plus a gain-dependent height. Substitute the trial height for
        # this calculation only; restore the datum before returning.
        trial = self.stage in ("acquiring", "trial")
        floor = c.floor_center_z
        if trial:
            c.floor_center_z = floor+.010-np.clip(.065*gains[1], .055, .078)
        measured_grasp = fresh and (self.stage != "acquiring" or self.anchor is not None)
        if (self.stage == "recovery" and c.phase == "lower" and supported and trial_height is not None
                and trial_height > .003):
            # At a low trial lift, the base's absolute near-floor threshold
            # could unload immediately while both jaws still bear the weight.
            measured_grasp = False
        try:
            target, effort, angle = self.base(actor, tools, wrist, position, r, yaw, obj,
                                              None if trial or self.stage == "recovery" else goal,
                                              gains, measured_grasp)
        finally:
            c.floor_center_z = floor
        if self.stage == "acquiring" and c.phase == "lift":
            self.stage = "trial"
        if self.stage == "trial" and c.phase == "lift" and self.trial_target_z is not None:
            # Hold actual encoder-derived jaw heights during the trial. A
            # force servo acting on box height alone lets the jaws creep up
            # the walls while the box hovers below its height reference.
            target[:, 2] = self.trial_target_z
            balance = effort[:, 2]-effort[:, 2].mean()
            effort[:, 2] = c.lift_feed+balance+np.clip(
                650.*(self.trial_target_z-tools[:, 2])-31.*vz, -10., 10.)
        if self.stage == "recovery" and c.phase == "lower":
            c.ref[:2] = self.recovery_tools.mean(axis=0)[:2]
            target[:, :2] = self.recovery_tools[:, :2]
            axis = self.recovery_side[:2]
            axis = axis/max(np.linalg.norm(axis), 1e-9)
            # Retain opposing normal preload; drop translation and yaw drive.
            preload = max(0., .5*(effort[1, :2]-effort[0, :2])@axis)
            effort[:, :2] = np.array([-1., 1.])[:, None]*preload*axis
            self.recovery_tools[:, 2] += np.clip(self.floor_tool_z-self.recovery_tools[:, 2],
                                                 -.055*dt, 0.)
            if not measured_grasp:
                # No new visual fit: use measured TCP heights to descend
                # only as far as the recorded pre-lift clamp height. Floor
                # handoff and release still require a live optical fit.
                vz = (tools[:, 2]-self.recovery_last_z)/dt
                target[:, 2] = self.recovery_tools[:, 2]
                effort[:, 2] = c.lift_feed+np.clip(260.*(target[:, 2]-tools[:, 2])-12.*vz, -9., 7.)
            self.recovery_last_z = tools[:, 2].copy()
        if self.stage == "recovery" and c.phase == "clamp":
            self.stage, self.anchor, self.reason = "acquiring", None, None
            self.trial_target_z = None
            self.hold_s = self.alarm_s = 0.
        return target, effort, angle

    def correct(self, actor, observation, action):
        """Apply after face-normal alignment, inside the remaining motor range."""
        c, dt = actor.fiducial, actor.dt
        if not self.seen_this_step:
            self.sample_age_s += dt
            self.hold_s = self.alarm_s = 0.
        self.seen_this_step = False
        applied = np.asarray(action, float).reshape(2, 7).copy()
        velocity = np.asarray(observation[14:28]).reshape(2, 7)
        enabled = np.asarray(observation[28:]).reshape(2, 7)
        active = (self.anchor is not None and self.sample_age_s < .048
                  and c.phase in ("lift", "transport", "lower"))
        moments, scales = [], []
        offset = 0
        for i, ((_, va, _), (tool, _)) in enumerate(zip(actor.joints, actor.sites, strict=True)):
            n = len(va)
            change = np.zeros(n)
            moment = 0.
            if active and enabled[i, 0]:
                jp, jr = np.zeros((3, actor.model.nv)), np.zeros((3, actor.model.nv))
                actor.mj.mj_jacSite(actor.model, actor.data, jp, jr, tool)
                j, angular = jp[:, va], jr[:, va]
                axis = actor.data.site_xmat[tool].reshape(3, 3)[:, 1]
                rate = axis@(angular@velocity[i, :n])
                # Counter the box pitch in world coordinates. Relative pitch
                # detects and resists the box rolling through a level gripper.
                moment = float(np.clip(-7.*self.pitch-2.*self.rel_angle[i]
                                       -.50*self.pitch_rate-.18*rate, -1.2, 1.2))
                tau = angular.T@(moment*axis)
                change = tau-j.T@np.linalg.solve(j@j.T+1e-7*np.eye(3), j@tau)
            lo, hi = actor.model.actuator_ctrlrange[offset:offset+n].T
            offset += n
            base = applied[i, :n]*np.where(applied[i, :n] >= 0., hi, -lo)
            previous = self.previous_delta[i, :n]
            change = previous+np.clip(change-previous, -14.*dt, 14.*dt)
            bounds = np.ones(n)
            np.divide(hi-base, change, out=bounds, where=change > 1e-9)
            np.divide(lo-base, change, out=bounds, where=change < -1e-9)
            scale = float(np.clip(bounds.min(), 0., 1.))
            change = scale*change*enabled[i, :n]
            self.previous_delta[i, :n] = change
            total = np.clip(base+change, lo, hi)
            applied[i, :n] = total/np.where(total >= 0., hi, -lo)
            moments.append(moment)
            scales.append(scale)
        actor.features["stable_grasp"] = dict(stage=self.stage, recovery_reason=self.reason,
            loaded_grip_reference=self.trial_target_z is not None,
            trial_tcp_z_m=None if self.trial_target_z is None else self.trial_target_z.tolist(),
            verified_hold_s=self.hold_s, full_attitude_age_s=self.sample_age_s,
            relative_pitch_rad=self.rel_angle.tolist(), local_drop_m=self.drop.tolist(),
            level_pitch_error_rad=self.pitch, pitch_rate_rad_s=self.pitch_rate,
            requested_moment_nm=moments, headroom_fraction=scales)
        return (applied*enabled).ravel()


def make_controller(actor, mode):
    return _StableGrasp(actor) if mode == "stable-grasp" else None
