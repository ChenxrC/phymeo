"""Stateless batched policies with explicit per-world physical-step clocks.

The borrowed environment supplies world_keys, action_size, reset(indices), and
step(actions) returning the VectorStep protocol. Construction never resets it.
Call reset() explicitly or provide its current observation. All subsequent
stepping/resetting must go through this runner so its clock stays authoritative.
Policy callbacks are stateless: policy(readonly_observation, readonly_due_mask)
returns finite actions for the complete batch. Only due rows replace held
actions; callback execution is not an RNN state/checkpoint transaction.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..training.resident_env import VectorStep


def _owned(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _numeric(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite real array with shape {shape}")
    return _owned(array)


@dataclass(frozen=True)
class PolicyRunnerState:
    """Read-only diagnostics, not a restorable physics or recurrent-policy checkpoint."""

    observation: np.ndarray | None
    held_actions: np.ndarray | None
    physical_ticks: np.ndarray
    refresh_mask: np.ndarray
    needs_reset: bool

    def __post_init__(self):
        for name in ("observation", "held_actions", "physical_ticks", "refresh_mask"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _owned(value))


@dataclass(frozen=True)
class PolicyRunnerStep:
    """Exactly one accepted physical step and the action actually sent to env.step.

    These are environment input actions, not the backend's quantized forces.
    physical_ticks counts runner-local accepted steps at entry, not an absolute
    backend tick. The snapshot describes its accepted exit. transition retains
    every standard VectorStep field.
    """

    actions: np.ndarray
    due_mask: np.ndarray
    physical_ticks: np.ndarray
    transition: VectorStep

    def __post_init__(self):
        for name in ("actions", "due_mask", "physical_ticks"):
            object.__setattr__(self, name, _owned(getattr(self, name)))


class PolicyRunnerProtocolError(RuntimeError):
    """Environment returned after physics, but its transition was unusable."""

    physics_accepted = True


class PolicyRunner:
    """Hold actions for integer physical substeps, independently for each world.

    A completed world (final_mask=True) refreshes on its next step. Resetting one
    world does not reschedule the others. Known atomic physics refusals may be
    retried with unchanged clocks and held actions. Partial/unknown acceptance,
    post-accept observer failure, or malformed transitions require full reset.
    Original environment exceptions propagate without hiding acceptance fields.
    """

    def __init__(self, env, policy, *, control_decimation=1, initial_observation=None):
        keys = tuple(env.world_keys)
        size = env.action_size
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("environment world_keys must be nonempty and unique")
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
            raise ValueError("environment action_size must be a positive integer")
        if (
            isinstance(control_decimation, bool)
            or not isinstance(control_decimation, Integral)
            or control_decimation < 1
        ):
            raise ValueError("control_decimation must be a positive integer")
        if not callable(policy) or not callable(env.reset) or not callable(env.step):
            raise TypeError("policy and environment reset/step must be callable")
        self.env, self.policy = env, policy
        self.control_decimation = int(control_decimation)
        self._count, self._size = len(keys), int(size)
        self._ticks = np.zeros(self._count, dtype=np.int64)
        self._refresh = np.ones(self._count, dtype=bool)
        self._held = self._observation = None
        self._poisoned = False
        if initial_observation is not None:
            self._observation = self._observe(initial_observation)

    @property
    def needs_reset(self):
        return self._poisoned or bool(getattr(self.env, "needs_reset", False))

    def snapshot(self):
        return PolicyRunnerState(
            self._observation, self._held, self._ticks, self._refresh, self.needs_reset
        )

    def _observe(self, value):
        array = np.asarray(value)
        if array.ndim != 2 or array.shape[0] != self._count or array.shape[1] < 1:
            raise ValueError("observation must have shape [world, nonempty features]")
        if self._observation is not None and array.shape != self._observation.shape:
            raise ValueError("environment observation shape changed")
        return _numeric(array, array.shape, "observation")

    def _indices(self, indices):
        if indices is None:
            return np.arange(self._count)
        selected = np.asarray(indices)
        if selected.dtype == np.bool_:
            if selected.shape != (self._count,):
                raise ValueError("reset mask must select the complete world batch")
            return np.flatnonzero(selected)
        if (
            selected.ndim != 1
            or selected.dtype.kind not in "iu"
            or len(set(selected.tolist())) != len(selected)
            or np.any(selected < 0)
            or np.any(selected >= self._count)
        ):
            raise ValueError("reset indices must be distinct in-range integers")
        return selected

    def reset(self, indices=None):
        """Explicitly reset selected worlds; a poisoned runner requires all worlds.

        The environment returns a full observation even for a partial reset.
        Held rows remain a record of the last accepted actions; refresh_mask
        guarantees the selected rows are replaced before their next physics step.
        """
        selected = self._indices(indices)
        if self.needs_reset and len(selected) != self._count:
            raise RuntimeError("reset all worlds after partial or unknown physics acceptance")
        try:
            observation = self._observe(self.env.reset(selected))
            if bool(getattr(self.env, "needs_reset", False)):
                raise RuntimeError("environment still needs reset")
        except Exception:
            # A reset or its observer may already have changed physical state.
            self._poisoned = True
            raise
        self._observation = observation
        self._ticks[selected] = 0
        self._refresh[selected] = True
        self._poisoned = False
        return self._observation

    def _transition(self, raw):
        observation = self._observe(raw.observation)
        final = _numeric(raw.final_observation, observation.shape, "final_observation")
        reward = _numeric(raw.reward, (self._count,), "reward")
        flags = {}
        for name in ("terminated", "truncated", "final_mask"):
            array = np.asarray(getattr(raw, name))
            if array.shape != (self._count,) or array.dtype != np.bool_:
                raise ValueError(f"{name} must contain one boolean per world")
            flags[name] = _owned(array)
        if np.any(flags["terminated"] & flags["truncated"]) or not np.array_equal(
            flags["final_mask"], flags["terminated"] | flags["truncated"]
        ):
            raise ValueError("final_mask must identify exactly completed episodes")
        counters = {}
        for name in ("episode_id", "episode_length"):
            array = np.asarray(getattr(raw, name))
            if (
                array.shape != (self._count,)
                or array.dtype.kind not in "iu"
                or np.any(array < 0)
            ):
                raise ValueError(f"{name} must contain nonnegative integer world counters")
            counters[name] = _owned(array)
        return VectorStep(
            observation,
            reward,
            flags["terminated"],
            flags["truncated"],
            final,
            flags["final_mask"],
            counters["episode_id"],
            counters["episode_length"],
        )

    def step(self):
        """Perform exactly one real environment step; never synthesize a transition."""
        if self.needs_reset:
            raise RuntimeError("reset all worlds before continuing")
        if self._observation is None:
            raise RuntimeError("provide initial_observation or call reset explicitly")
        if np.any(self._ticks == np.iinfo(np.int64).max):
            raise OverflowError("physical-step clock exhausted before physics")
        due = _owned(self._refresh | (self._ticks % self.control_decimation == 0))
        candidate = np.zeros((self._count, self._size)) if self._held is None else self._held.copy()
        if np.any(due):
            proposed = _numeric(
                self.policy(self._observation, due),
                (self._count, self._size),
                "policy actions",
            )
            candidate[due] = proposed[due]
        actions, entry_ticks = _owned(candidate), _owned(self._ticks)
        try:
            raw = self.env.step(actions)
        except Exception as error:
            # Only an explicit atomic refusal can be retried safely. N2's
            # partial/unknown acceptance metadata remains on the original error.
            if (
                getattr(error, "physics_accepted", None) is not False
                or bool(getattr(self.env, "needs_reset", False))
            ):
                self._poisoned = True
            raise
        try:
            transition = self._transition(raw)
            if bool(getattr(self.env, "needs_reset", False)):
                raise RuntimeError("environment returned a transition while requiring reset")
        except Exception as error:
            self._poisoned = True
            raise PolicyRunnerProtocolError(
                "physics returned; invalid transition; reset all worlds"
            ) from error
        # Commit scheduling only after a complete accepted environment transition.
        self._observation, self._held = transition.observation, actions
        self._ticks = np.where(transition.final_mask, 0, self._ticks + 1)
        self._refresh = np.array(transition.final_mask, copy=True)
        return PolicyRunnerStep(actions, due, entry_ticks, transition)
