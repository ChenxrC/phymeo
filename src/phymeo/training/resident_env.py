"""A NumPy vector-environment boundary using one real resident batch step.

Tasks supply controls, observations and reward/termination evaluation. Physics
stays in the batch. This first interface reads accepted states to HOST; it is
not a device-tensor interop or large-scale throughput claim.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .resident_stepper import PhysicsStepFailure, ResidentContactStepper, _exact_acceptance


def _owned(value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class VectorEnvStepError(PhysicsStepFailure):
    """No env transition was emitted; the mask says which physics advanced.

    Known all-False permits retry. Partial or unknown acceptance and post-step
    callback failures require full reset; physical rollback is not claimed.
    """

    def __init__(self, message, *, accepted_worlds=None, acceptance_known=True):
        if type(acceptance_known) is not bool:
            raise TypeError("acceptance_known must be boolean")
        if accepted_worlds is None:
            # Preserve the original message-only exception API. Without a mask
            # it cannot prove that physics did not publish.
            accepted_worlds, acceptance_known = {}, False
        super().__init__(
            message, accepted_worlds=accepted_worlds, acceptance_known=acceptance_known
        )


@dataclass(frozen=True)
class VectorStep:
    observation: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    final_observation: np.ndarray
    final_mask: np.ndarray
    episode_id: np.ndarray
    episode_length: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _owned(getattr(self, name)))


class ResidentVectorEnv:
    """Borrow a batch and task; auto-reset completed worlds after saving final obs.

    task.controls(actions, keys) returns the batch's public control mapping.
    task.observe(states, keys) returns a finite [world, observation] matrix.
    task.evaluate(before, after, actions, keys) returns finite rewards [world]
    and boolean terminated [world]. Callbacks cannot advance physical state.
    Deterministic reset uses the batch's original state. Randomized reset and
    seed/normalizer checkpointing are separate follow-up capabilities.
    stepper(h) may explicitly select contact physics; None borrows batch.step.
    The backend protocol is snapshot(keys=None), set_controls(mapping),
    step(h)->exact bool mapping, reset(keys). No backend is owned or closed.
    """

    def __init__(self, batch, task, world_keys, *, action_size, dt, horizon, stepper=None):
        keys = tuple(world_keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("world keys must be nonempty and unique")
        if (
            isinstance(action_size, bool)
            or not isinstance(action_size, Integral)
            or action_size < 1
        ):
            raise ValueError("action_size must be a positive integer")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        limits = np.asarray(horizon)
        if limits.dtype.kind not in "iu" or np.any(limits <= 0):
            raise ValueError("horizon must contain positive integers")
        if limits.ndim == 0:
            limits = np.full(len(keys), int(limits))
        if limits.shape != (len(keys),):
            raise ValueError("horizon must be scalar or one value per world")
        selected_stepper = batch.step if stepper is None else stepper
        if not callable(selected_stepper):
            raise TypeError("stepper must be callable with one physical dt")
        if (
            isinstance(selected_stepper, ResidentContactStepper)
            and selected_stepper.batch is not batch
        ):
            raise ValueError("contact stepper belongs to a different environment batch owner")
        self._stepper = selected_stepper
        self.batch, self.task, self.world_keys = batch, task, keys
        self.action_size, self.dt = int(action_size), float(dt)
        self._horizon = _owned(limits, np.int64)
        self._episode_id = np.zeros(len(keys), dtype=np.int64)
        self._length = np.zeros(len(keys), dtype=np.int64)
        self._poisoned = False
        initial = batch.snapshot()
        if set(initial) != set(keys):
            raise ValueError("environment worlds must cover the complete borrowed batch")
        self._observe(initial)

    @property
    def needs_reset(self):
        return self._poisoned

    def _observe(self, states):
        value = np.asarray(self.task.observe(states, self.world_keys))
        if value.ndim != 2 or len(value) != len(self.world_keys) or not np.isfinite(value).all():
            raise ValueError("task observation must be a finite [world, observation] matrix")
        return _owned(value)

    def reset(self, indices=None):
        if indices is None:
            selected = np.arange(len(self.world_keys))
        else:
            selected = np.asarray(indices)
            if selected.ndim != 1 or selected.dtype.kind not in "iu":
                raise ValueError("reset indices must be an integer vector")
            if (
                len(set(selected.tolist())) != len(selected)
                or np.any(selected >= len(self.world_keys))
                or np.any(selected < 0)
            ):
                raise ValueError("reset indices must be unique and in range")
        if self._poisoned and len(selected) != len(self.world_keys):
            raise RuntimeError("reset all worlds after a post-accept task failure")
        try:
            self.batch.reset(tuple(self.world_keys[int(index)] for index in selected))
            observation = self._observe(self.batch.snapshot(self.world_keys))
        except Exception:
            # Reset may already have published new physical worlds. Without a
            # successful readback, old episode counters cannot describe them.
            self._poisoned = True
            raise
        self._length[selected] = 0
        self._episode_id[selected] += 1
        self._poisoned = False
        return observation

    def step(self, actions):
        if self._poisoned:
            raise RuntimeError("reset all worlds before continuing")
        action = np.asarray(actions)
        if (
            action.dtype.kind not in "iuf"
            or action.shape != (len(self.world_keys), self.action_size)
            or not np.isfinite(action).all()
        ):
            raise ValueError("actions must be a finite [world, action] matrix")
        action = _owned(action)
        before = self.batch.snapshot(self.world_keys)
        controls = self.task.controls(action, self.world_keys)
        if set(controls) != set(self.world_keys):
            raise ValueError("task must provide exactly one control per environment world")
        self.batch.set_controls(controls)
        accepted = self._step_physics()
        try:
            after = self.batch.snapshot(self.world_keys)
            final = self._observe(after)
            reward, terminated = self.task.evaluate(before, after, action, self.world_keys)
            reward, terminated = np.asarray(reward), np.asarray(terminated)
            if (
                reward.shape != self._length.shape
                or not np.isfinite(reward).all()
                or terminated.shape != self._length.shape
                or terminated.dtype != np.bool_
            ):
                raise ValueError(
                    "task must return finite rewards and boolean termination per world"
                )
            length = self._length + 1
            truncated = (length >= self._horizon) & ~terminated
            completed = terminated | truncated
            if np.any(completed):
                self.batch.reset(
                    tuple(key for key, done in zip(self.world_keys, completed, strict=True) if done)
                )
                observation = self._observe(self.batch.snapshot(self.world_keys))
            else:
                observation = final
            next_length = np.where(completed, 0, length)
            next_episode_id = self._episode_id + completed
            result = VectorStep(
                observation,
                reward,
                terminated,
                truncated,
                final,
                completed,
                next_episode_id,
                next_length,
            )
        except Exception as error:
            self._poisoned = True
            raise VectorEnvStepError(
                "physics accepted; task/readback failed; reset all worlds",
                accepted_worlds=accepted,
            ) from error
        self._length = next_length
        self._episode_id = next_episode_id
        return result

    def _step_physics(self):
        """Validate publication before reward, observation or episode transition."""
        try:
            returned = self._stepper(self.dt)
        except PhysicsStepFailure as error:
            if not error.acceptance_known:
                self._poisoned = True
                raise VectorEnvStepError(
                    "physical publication is uncertain; reset all worlds",
                    accepted_worlds=error.accepted_worlds,
                    acceptance_known=False,
                ) from error
            try:
                accepted = _exact_acceptance(error.accepted_worlds, self.world_keys)
            except Exception as invalid:
                self._poisoned = True
                raise VectorEnvStepError(
                    "physical failure has an incomplete/invalid acceptance map; reset all worlds",
                    accepted_worlds={},
                    acceptance_known=False,
                ) from invalid
            self._raise_rejected(accepted, cause=error)
            # A failure after every world accepted is still not an env transition.
            self._poisoned = True
            raise VectorEnvStepError(
                "physics accepted; backend failed afterward; reset all worlds",
                accepted_worlds=accepted,
            ) from error
        except Exception as error:
            self._poisoned = True
            raise VectorEnvStepError(
                "physical step failed with unknown publication state; reset all worlds",
                accepted_worlds={},
                acceptance_known=False,
            ) from error
        try:
            accepted = _exact_acceptance(returned, self.world_keys)
        except Exception as error:
            self._poisoned = True
            raise VectorEnvStepError(
                "physical step returned an invalid acceptance map; reset all worlds",
                accepted_worlds={},
                acceptance_known=False,
            ) from error
        self._raise_rejected(accepted)
        return accepted

    def _raise_rejected(self, accepted, *, cause=None):
        if all(accepted.values()):
            return
        self._poisoned = any(accepted.values())
        message = (
            "some physical worlds advanced; no env transition; reset all worlds"
            if self._poisoned
            else "all physical worlds rejected; no env transition; retry is permitted"
        )
        error = VectorEnvStepError(message, accepted_worlds=accepted)
        if cause is None:
            raise error
        raise error from cause
