"""Bounded gripper targets driven by accepted joint and bilateral-load observations.

This controller never advances physics. ``contact_ready`` certifies measured
bilateral contact, not payload support, frictional stability, or grasp success.
"""

from dataclasses import dataclass, replace
from numbers import Integral
from threading import RLock

import numpy as np

from ..builders._robot_asset import RobotAsset
from ..constraints.collision_shapes import CollisionKey
from ..core.validation import finite_real
from ..dynamics.joint_mimic import MimicRobotAsset
from ..runtime.robot_scene import RobotCommand, RobotSceneObservation


def _vector(value, count, name):
    a = np.asarray(value)
    if a.shape != (count,) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"{name} requires {count} finite real entries")
    return np.frombuffer(np.asarray(a, dtype=float).tobytes(), dtype=float)


def _command_signature(command):
    if command is None:
        return None
    if not isinstance(command, RobotCommand):
        raise TypeError("accepted command must be a RobotCommand")
    # Optional request fields are significant, including omitted gravity feedforward.
    return tuple(
        None if value is None else tuple(value)
        for value in (
            command.target_q,
            command.target_velocity,
            command.effort_limit,
            command.feedforward,
        )
    )


@dataclass(frozen=True, eq=False)
class AcquisitionProposal:
    """A command for the next step, with actual evidence from the observed frame.

    q/v and time_s are accepted end-state observations. bilateral_N belongs to
    load_time_s, the beginning of the preceding physical step. Phases concern
    the finger contact protocol; a separate payload motion test proves holding.
    """

    command: RobotCommand
    phase: str
    contact_ready: bool
    bilateral_N: np.ndarray
    step_number: int
    time_s: float
    load_time_s: float | None
    contact_duration_s: float
    loss_duration_s: float
    actual_effort_saturated: tuple[bool, ...]
    reason: str | None


@dataclass(frozen=True)
class _History:
    step: int
    time: float
    load_time: float | None
    signature: tuple
    phase: str
    contact_duration: float
    loss_duration: float
    release_duration: float
    empty_duration: float
    ever_ready: bool
    attempt_start: float
    reason: str | None = None


class GripperAcquisition:
    """One accepted-step protocol for configured independent gripper coordinates.

    Coordinates/speeds retain their URDF units (m, rad, m/s, rad/s). Selected
    coordinates follow open_q toward closed_q, which may have either sign.
    Limits and motor ratings come from the supplied asset; gains remain in the
    plant. Repeated observations do not integrate the reference or dwell timers.
    An input sequence must include every accepted step controlled by this object.
    """

    def __init__(
        self,
        asset,
        *,
        robot,
        payload_key,
        pad_groups,
        joint_indices,
        open_q,
        closed_q,
        speed_limit,
        effort_limit,
        force_target_N,
        force_band_N,
        contact_min_N,
        dwell_s,
        loss_dwell_s,
        stop_speed,
        timeout_s,
    ):
        if not isinstance(asset, (RobotAsset, MimicRobotAsset)):
            raise TypeError("expected a RobotAsset or MimicRobotAsset")
        address = tuple(robot)
        if len(address) != 2 or any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or i < 0 for i in address
        ):
            raise ValueError("robot must be a nonnegative (world, registration) address")
        self.robot = tuple(map(int, address))
        self._n = len(asset.joint_names)
        indices = tuple(joint_indices)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(
                isinstance(i, (bool, np.bool_))
                or not isinstance(i, Integral)
                or not 0 <= i < self._n
                for i in indices
            )
        ):
            raise ValueError("joint_indices must select distinct independent coordinates")
        self._ids = np.array(indices, dtype=np.intp)
        count = len(indices)
        self._lower = _vector(asset.lower, self._n, "asset lower")
        self._upper = _vector(asset.upper, self._n, "asset upper")
        self._open = _vector(open_q, count, "open_q")
        self._closed = _vector(closed_q, count, "closed_q")
        self._speed = _vector(speed_limit, count, "speed_limit")
        self._stop = _vector(stop_speed, count, "stop_speed")
        self._cap = _vector(effort_limit, self._n, "effort_limit")
        self._direction = np.sign(self._closed - self._open)
        if np.any(self._direction == 0):
            raise ValueError("each selected closing coordinate needs a nonzero stroke")
        for endpoint in (self._open, self._closed):
            if np.any(endpoint < self._lower[self._ids]) or np.any(
                endpoint > self._upper[self._ids]
            ):
                raise ValueError("gripper stroke exceeds the asset interval")
        hard_speed = np.array([np.inf if x is None else x for x in asset.velocity_limit])
        hard_effort = np.array([np.inf if x is None else x for x in asset.effort_limit])
        if np.any(self._speed <= 0) or np.any(self._speed > hard_speed[self._ids]):
            raise ValueError("speed_limit must be positive and within the asset rating")
        if np.any(self._stop < 0) or np.any(self._stop >= self._speed):
            raise ValueError("stop_speed must be nonnegative and below speed_limit")
        if np.any(self._cap < 0) or np.any(self._cap > hard_effort):
            raise ValueError("effort_limit must stay within the asset rating")
        groups = tuple(tuple(g) for g in pad_groups)
        if len(groups) != 2 or any(not g for g in groups):
            raise ValueError("pad_groups needs two nonempty sides")
        flat = tuple(k for group in groups for k in group)
        if (
            not isinstance(payload_key, CollisionKey)
            or any(not isinstance(k, CollisionKey) for k in flat)
            or len(set(flat)) != len(flat)
            or payload_key in flat
            or any(k.world != address[0] for k in (*flat, payload_key))
        ):
            raise ValueError("distinct same-world pads and payload keys are required")
        self.payload_key, self.pad_groups = payload_key, groups
        self._sides = {key: i for i, group in enumerate(groups) for key in group}
        values = {}
        for name, value in (
            ("force_target_N", force_target_N),
            ("force_band_N", force_band_N),
            ("contact_min_N", contact_min_N),
            ("dwell_s", dwell_s),
            ("loss_dwell_s", loss_dwell_s),
            ("timeout_s", timeout_s),
        ):
            value = finite_real(value, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            values[name] = value
        if values["force_target_N"] - values["force_band_N"] < values["contact_min_N"]:
            raise ValueError("the force target band must lie above the bilateral contact threshold")
        self._target, self._band, self._minimum = (
            values["force_target_N"],
            values["force_band_N"],
            values["contact_min_N"],
        )
        self._dwell, self._loss_dwell, self._timeout = (
            values["dwell_s"],
            values["loss_dwell_s"],
            values["timeout_s"],
        )
        self._history = self._pending = None
        self._pending_dt = None
        self._pending_history = None
        self._lock = RLock()

    def _read(self, observation):
        if (
            not isinstance(observation, RobotSceneObservation)
            or self.robot not in observation.robots
        ):
            raise TypeError("expected the registered robot's accepted scene observation")
        step = observation.step_number
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, Integral) or step < 0:
            raise ValueError("invalid accepted step number")
        time = finite_real(observation.time_s, "observation time")
        load_time = observation.load_time_s
        if time < 0 or (step == 0 and (time != 0 or load_time is not None)):
            raise ValueError("initial observation time must be zero, without a load sample")
        if step:
            if load_time is None:
                raise ValueError("accepted steps need a physical load time")
            load_time = finite_real(load_time, "load time")
            if not 0 <= load_time < time:
                raise ValueError("load time cannot be future or equal to end-state time")
        joint = observation.robots[self.robot]
        q, v = _vector(joint.q, self._n, "actual q"), _vector(joint.v, self._n, "actual v")
        effort = _vector(joint.drive_effort, self._n, "actual effort")
        if np.any(q < self._lower) or np.any(q > self._upper):
            raise ValueError("actual q is outside the asset interval")
        normal = np.zeros(2)
        for contact in observation.contacts:
            if contact["channel"] != "robot-pad":
                continue
            a, b = contact["first"], contact["second"]
            other = b if a == self.payload_key else a if b == self.payload_key else None
            if other not in self._sides:
                continue
            force = finite_real(contact["normal_load_N"], "normal contact load")
            if force < 0:
                raise ValueError("normal contact load must be nonnegative")
            normal[self._sides[other]] += force
        if not np.isfinite(normal).all():
            raise ValueError("bilateral load sum is not finite")
        signature = (
            time,
            load_time,
            tuple(q),
            tuple(v),
            tuple(normal),
            tuple(effort),
            _command_signature(joint.command),
            _command_signature(joint.requested_command),
        )
        return (
            int(step),
            time,
            load_time,
            q,
            v,
            normal,
            effort,
            joint.command,
            joint.requested_command,
            signature,
        )

    def _accepted(self, row):
        step, time, load_time, q, v, normal, _, _, requested, signature = row
        old = self._history
        if old is None:
            return _History(step, time, load_time, signature, "closing", 0, 0, 0, 0, False, time)
        if step < old.step or time < old.time:
            raise ValueError("observation step/time moved backwards")
        if step == old.step:
            if signature != old.signature:
                raise ValueError("same accepted step has changed observation data or time")
            return old
        if step != old.step + 1 or self._pending is None:
            raise ValueError("observations must include each accepted controlled step")
        expected_time = old.time + self._pending_dt
        tolerance = 8 * np.finfo(float).eps * max(1.0, abs(expected_time))
        if abs(time - expected_time) > tolerance or load_time != old.time:
            raise ValueError("future or mismatched accepted step time")
        if _command_signature(requested) != _command_signature(self._pending.command):
            raise ValueError("accepted robot request differs from the proposed command")
        h = time - old.time
        history = self._pending_history
        phase = history.phase
        ready = np.all(normal >= self._minimum) and np.all(abs(v[self._ids]) <= self._stop)
        contact = history.contact_duration + h if ready else 0.0
        loss = (
            history.loss_duration + h if history.ever_ready and min(normal) < self._minimum else 0.0
        )
        released = (
            np.all(normal <= 1e-8)
            and np.all(abs(q[self._ids] - self._open) <= 1e-4)
            and np.all(abs(v[self._ids]) <= self._stop)
        )
        release = history.release_duration + h if released and phase == "releasing" else 0.0
        at_closed = (
            np.all(abs(q[self._ids] - self._closed) <= 1e-4)
            and np.all(abs(v[self._ids]) <= self._stop)
            and min(normal) < self._minimum
        )
        empty = history.empty_duration + h if at_closed and phase == "closing" else 0.0
        ever = history.ever_ready
        reason = history.reason
        if phase in {"closing", "force_hold"}:
            if contact >= self._dwell:
                phase, ever = "force_hold", True
            if ever and loss >= self._loss_dwell:
                phase, reason = "lost", "bilateral_load_lost"
            elif phase == "closing" and empty >= self._dwell:
                phase, reason = "empty", "closure_limit_without_bilateral_load"
            elif phase == "closing" and time - history.attempt_start >= self._timeout:
                phase, reason = "timeout", "acquisition_timeout"
        elif phase == "releasing" and release >= self._dwell:
            phase, reason = "released", None
        return _History(
            step,
            time,
            load_time,
            signature,
            phase,
            contact,
            loss,
            release,
            empty,
            ever,
            history.attempt_start,
            reason,
        )

    def propose(self, observation, *, dt_s, posture_target=None, intent="acquire"):
        """Return the next bounded target. Rejected physics does not consume a proposal.

        The same observation and arguments produce the same command and timers.
        A caller may change the intent or requested dt before a physical step;
        the next accepted observation must match the last returned command and dt.
        """
        with self._lock:
            h = finite_real(dt_s, "dt_s")
            if h <= 0:
                raise ValueError("dt_s must be positive")
            if intent not in {"acquire", "release"}:
                raise ValueError("intent must be acquire or release")
            row = self._read(observation)
            step, time, load_time, q, v, normal, effort, actual_command, _, _ = row
            history = self._accepted(row)
            base = q if actual_command is None else actual_command.target_q
            target = np.array(
                base
                if posture_target is None
                else _vector(posture_target, self._n, "posture_target"),
                dtype=float,
                copy=True,
            )
            if np.any(target < self._lower) or np.any(target > self._upper):
                raise ValueError("posture target exceeds the asset interval")
            reference = np.array(base[self._ids], copy=True)
            lo, hi = np.minimum(self._open, self._closed), np.maximum(self._open, self._closed)
            if np.any(reference < lo - 1e-12) or np.any(reference > hi + 1e-12):
                raise ValueError("accepted gripper reference lies outside the configured stroke")
            proposal_history = history
            if intent == "release":
                if history.phase != "released":
                    proposal_history = replace(
                        history,
                        phase="releasing",
                        contact_duration=0.0,
                        release_duration=history.release_duration
                        if history.phase == "releasing"
                        else 0.0,
                        reason=None,
                    )
            elif history.phase in {"releasing", "released"}:
                proposal_history = replace(
                    history,
                    phase="closing",
                    contact_duration=0.0,
                    loss_duration=0.0,
                    release_duration=0.0,
                    empty_duration=0.0,
                    ever_ready=False,
                    attempt_start=time,
                    reason=None,
                )
            phase = proposal_history.phase
            if phase == "releasing":
                factor = -1.0
            elif phase in {"closing", "force_hold"}:
                if max(normal) > self._target + self._band:
                    factor = -min(1.0, (max(normal) - self._target) / self._band)
                elif min(normal) < self._target - self._band:
                    factor = min(1.0, (self._target - min(normal)) / self._band)
                else:
                    factor = 0.0
            else:
                factor = 0.0
            selected_effort = effort[self._ids]
            saturated = abs(selected_effort) >= self._cap[self._ids] - 1e-12
            # Stop reference wind-up at a real motor cap while the plant cannot follow.
            ahead = (reference - q[self._ids]) * self._direction > 1e-4
            stalled = abs(v[self._ids]) <= self._stop
            if factor > 0 and np.any(saturated & ahead & stalled):
                factor = 0.0
            signed_speed = v[self._ids] * self._direction
            if (factor > 0 and np.any(signed_speed > self._speed)) or (
                factor < 0 and np.any(-signed_speed > self._speed)
            ):
                factor = 0.0
            next_reference = np.clip(reference + h * factor * self._direction * self._speed, lo, hi)
            target[self._ids] = next_reference
            rate = np.zeros(self._n)
            rate[self._ids] = (next_reference - reference) / h
            command = RobotCommand(target, target_velocity=rate, effort_limit=self._cap)
            result = AcquisitionProposal(
                command,
                phase,
                phase == "force_hold" and history.contact_duration >= self._dwell,
                _vector(normal, 2, "bilateral load"),
                step,
                time,
                load_time,
                proposal_history.contact_duration,
                proposal_history.loss_duration,
                tuple(bool(x) for x in saturated),
                proposal_history.reason,
            )
            # No history mutation before all input checks and allocations succeed.
            self._history, self._pending = history, result
            self._pending_dt, self._pending_history = h, proposal_history
            return result
