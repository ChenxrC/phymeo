# MuJoCo-TYX N2: device VecEnv and upstream CUDA PPO

Status: AUTHORING_ONLY / UNEXECUTED. Parent base 63614090. No project import,
collection, numerical probe, tests, lint, formatter, installation or build is
authorized in this author lane. Parent owns the sole execution slot.

## Frozen API and ownership

- MuJoCoTyxActionSpec: immutable native ctrl indices, qfrc DOF indices and
  (body, wrench component) indices; FP32 scale/bias. CUDA mapping is
  bias + scale * tanh(policy action); unmapped native fields are zero.
- MuJoCoTyxVectorEnv borrows an already admitted CUDA MuJoCoRuntime and owns
  its task workspace, wrapper RLock and episode metadata. It does not close
  the borrowed runtime. All mutation goes through this wrapper during a run;
  arbitrary external writes/races are not prevented by this API.
- step(actions, world_keys=..., expected_frame=...) requires explicit frozen
  order and the current frame identity. Same-shaped unlabelled semantic row
  swapping is not detectable. Three native channels use public TYX Torch
  views/stream and copy-owned runtime CAS submission, with no private bank.
- Full task observation/reward are device exports. HOST episode lengths and
  termination-mask readback are counted. Existing complete HOST physics and
  input-bit audits remain; diagnostic public snapshots do not generate policy
  observations.
- Capture full final observation before any subset reset. terminated wins over
  timeout; time_outs = horizon_reached AND NOT terminated. Return fresh
  post-reset observation and immutable metadata/fresh tensor ownership.
- RslMuJoCoTyxVecEnv only wraps TensorDict and the existing device wrapper.
  RSL 5.5.2 owns PPO, storage, returns, Adam, backward and hidden reset.
  Timeout bonus uses saved values from act(input_observation), not the final
  observation critic. No copied PPO implementation.
- N2 does not expose training checkpoint/restore. Public runtime physics-only
  capsules exclude episode/learner/RNG state; N3 owns complete composition.

## Fixed qualification cases, before execution

N1 XML and initial-state factories are reused byte-for-byte from
examples/mujoco_tyx_cuda_policy.py. No fixture/mass/damping/limit changes.
hinge-slide XML SHA8596be527027c85f5c6f4f17d64ababf5820a75771943da71e18509347cb16e7;
support-slide XML SHAbf4eb00b8d0eb00ac26be3f5fa31c2972fcde895ce5290bc5bdde60d45d11e92.

| Task | W / sparse order | h | seed | horizons |
| --- | --- | --- | --- | --- |
| hinge-slide | 8 / 5+36*i | .002 | 11011 | 5,7,9,11,13,15,17,19 |
| finite box support-slide | 8 / same | .001 | 11012 | same |

Each task: 4 updates x 16 physical steps x 8 worlds = 512 transitions.
Actor/critic MLP [32,32], tanh, no observation normalization; actor independent
Gaussian std .1 (log parameterization). Adam lr .0003, one epoch and one mini
batch per update, fixed schedule, gamma .99, lambda .95, clip .2, value weight1,
entropy .01, max gradient norm1. No AMP/compile/RND/distributed/random episode
counter initialization. No learning-quality/reward-monotonicity claim.

Native action mapping retains N1: ctrl .01+.02*tanh(a), qfrc .005+.01*tanh(a),
last-body wrench component j .001*(j+1)+.01*tanh(a). All other body wrenches zero.
TaskSpec retains N1 affine observation and raw-coordinate negative quadratic
reward/bounds. Different fixed horizons cause partial endings without tuning
physics or rewards. Protocol negative tests use explicit alternate task bounds
solely to force terminated/timeout precedence, not the PPO qualification.

Additional recurrent qualification: same hinge task, seed11013, one-layer
GRU hidden16 then MLP[32,32], 8-step rollout; verify actual actor AND critic
hidden states, zero only done columns, preserve every non-done column exactly,
then one upstream recurrent PPO update. This is a short protocol test.

## Failure contract

Preflight or control CAS refusal leaves physical state/held controls unchanged
and publishes no transition. After action acceptance, physical failure leaves
new held controls accepted, no successful rollout transition, and requires
explicit recovery. After physical acceptance, task/export/snapshot/bookkeeping
or subset-reset failure reports that accepted advance (and whether reset
committed), poisons further rollout consumption, and never pretends rollback.
Adapter packaging failure after a complete wrapper step has the same learner
stop requirement. An explicit full reset starts fresh episodes; it is not
restoration of the failed learner trajectory. Previous successful transition
remains identifiable and is never reinserted.

## Execution evidence required, not yet produced

Two files cover real device continuous HOST-route comparisons, final observation
and subset reset, world-order/frame staleness, bad action and both post-submit
failure phases, runtime ownership/locks, RSL timeout target and GRU masks.
The example records every transition, full physical receipts, original controls,
episodes/frames, raw policy actions, and each update's loss, parameter hashes,
finite gradient checks and actual optimizer steps. CPU diagnostics are labelled
separately from device policy input. No N2 result is claimed by this document.

