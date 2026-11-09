"""Optional RSL-RL interface tests; no logical clock is a physics/training oracle.

AUTHORING_ONLY / UNEXECUTED. Install the isolated optional deps before parent
execution. Missing deps explicitly skip this optional file, not its acceptance.
"""

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("rsl_rl")

# These imports intentionally follow the optional-dependency admission above.
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from phymeo.training import ResidentVectorEnv, VectorEnvStepError  # noqa: E402
from phymeo.training.rsl_rl import RslRlVecEnv  # noqa: E402


@dataclass(frozen=True)
class Clock:
    tick: int


class Backend:
    def __init__(self, flags=None):
        self.ticks = {4: 0, 7: 0}
        self.flags = [] if flags is None else list(flags)
        self.calls = 0

    def snapshot(self, keys=None):
        return {
            key: Clock(value) for key, value in self.ticks.items() if keys is None or key in keys
        }

    def set_controls(self, controls):
        self.controls = {key: np.array(value, copy=True) for key, value in controls.items()}

    def step(self, h):
        self.calls += 1
        flags = self.flags.pop(0) if self.flags else {4: True, 7: True}
        for key, accepted in flags.items():
            self.ticks[key] += int(accepted)
        return flags

    def reset(self, keys):
        for key in keys:
            self.ticks[key] = 0


class Task:
    def __init__(self, bad_after=None):
        self.bad_after, self.evaluations = bad_after, 0

    def controls(self, actions, keys):
        return dict(zip(keys, actions, strict=True))

    def observe(self, states, keys):
        values = np.array([[states[key].tick] for key in keys], dtype=float)
        if np.any(values) and self.bad_after == "width":
            return np.hstack((values, values))
        if np.any(values) and self.bad_after == "overflow":
            return values + 1e100
        return values

    def evaluate(self, before, after, actions, keys):
        self.evaluations += 1
        return np.ones(len(keys)), np.zeros(len(keys), dtype=bool)


def make(*, horizon=3, flags=None, bad_after=None):
    backend, task = Backend(flags), Task(bad_after)
    env = ResidentVectorEnv(backend, task, (4, 7), action_size=1, dt=1 / 64, horizon=horizon)
    return backend, task, RslRlVecEnv(env)


def test_fresh_tensordicts_preserve_prior_frames_and_timeout_terminal_observation():
    backend, _, adapter = make()
    initial = adapter.get_observations()
    assert isinstance(initial, TensorDict) and list(initial.batch_size) == [2]
    action = torch.zeros(2, 1)
    first, reward, done, _ = adapter.step(action)
    first_saved = first["policy"].clone()
    second, _, _, _ = adapter.step(action)
    final, _, done, extras = adapter.step(action)
    torch.testing.assert_close(initial["policy"], torch.zeros(2, 1))
    torch.testing.assert_close(first["policy"], first_saved)
    torch.testing.assert_close(second["policy"], torch.full((2, 1), 2.0))
    torch.testing.assert_close(final["policy"], torch.zeros(2, 1))
    torch.testing.assert_close(extras["final_observation"]["policy"], torch.full((2, 1), 3.0))
    assert first["policy"].data_ptr() != second["policy"].data_ptr()
    assert reward.dtype == torch.float32 and done.dtype == torch.bool and bool(done.all())
    assert bool(extras["time_outs"].all()) and bool(extras["final_mask"].all())
    assert backend.ticks == {4: 0, 7: 0}
    action.fill_(100.0)
    np.testing.assert_array_equal(backend.controls[4], [0.0])


@pytest.mark.parametrize(
    "actions",
    (torch.ones(2, 1, dtype=torch.long), torch.full((2, 1), float("nan")), torch.zeros(1, 1)),
)
def test_invalid_tensor_action_never_enters_physics(actions):
    backend, _, adapter = make()
    with pytest.raises((TypeError, ValueError)):
        adapter.step(actions)
    assert backend.calls == 0 and not adapter.needs_reset


@pytest.mark.parametrize("flags", ({4: False, 7: False}, {4: True, 7: False}))
def test_original_rejection_mask_and_reset_requirement_propagate(flags):
    backend, task, adapter = make(flags=[flags])
    with pytest.raises(VectorEnvStepError) as caught:
        adapter.step(torch.zeros(2, 1))
    error = caught.value
    assert dict(error.accepted_worlds) == flags and error.acceptance_known
    assert error.physics_accepted == any(flags.values())
    assert adapter.needs_reset == any(flags.values()) and task.evaluations == 0
    with pytest.raises(TypeError):
        error.accepted_worlds[4] = True
    if adapter.needs_reset:
        adapter.reset()
    _, _, _, extras = adapter.step(torch.zeros(2, 1))
    assert backend.calls == 2 and extras["episode_length"].tolist() == [1, 1]


@pytest.mark.parametrize("kind", ("width", "overflow"))
def test_post_physics_tensor_failure_retains_transition_and_requires_full_reset(kind):
    backend, _, adapter = make(horizon=5, bad_after=kind)
    with pytest.raises(VectorEnvStepError) as caught:
        adapter.step(torch.zeros(2, 1))
    assert backend.ticks == {4: 1, 7: 1} and adapter.needs_reset
    assert dict(caught.value.accepted_worlds) == {4: True, 7: True}
    assert caught.value.transition.episode_length.tolist() == [1, 1]
    with pytest.raises(RuntimeError, match="reset all"):
        adapter.reset([0])
    adapter.env.task.bad_after = None
    adapter.reset()
    assert not adapter.needs_reset


def test_episode_clock_copy_and_unsupported_clock_only_randomization():
    _, _, adapter = make()
    adapter.episode_length_buf.fill_(99)
    assert adapter.episode_length_buf.tolist() == [0, 0]
    with pytest.raises(NotImplementedError, match="random"):
        adapter.episode_length_buf = torch.ones(2, dtype=torch.long)


def test_actual_runner_constructs_and_checkpoint_restores_inference(tmp_path):
    torch.manual_seed(20261110)
    _, _, adapter = make()
    cfg = {
        "num_steps_per_env": 2,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {"class_name": "PPO"},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [8],
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.4},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [8]},
    }
    runner = OnPolicyRunner(adapter, cfg, log_dir=None, device="cpu")
    observation = adapter.get_observations()
    with torch.no_grad():
        original = runner.get_inference_policy()(observation).clone()
    checkpoint = tmp_path / "runner.pt"
    runner.save(str(checkpoint), infos={"purpose": "interface-only", "physical_steps": 0})
    loaded = runner.load(str(checkpoint), map_location="cpu")
    with torch.no_grad():
        restored = runner.get_inference_policy()(observation)
    torch.testing.assert_close(restored, original, rtol=0, atol=0)
    assert loaded == {"purpose": "interface-only", "physical_steps": 0}
