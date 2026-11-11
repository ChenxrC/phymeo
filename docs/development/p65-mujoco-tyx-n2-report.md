# MuJoCo-TYX N2: device VecEnv and upstream CUDA PPO

Status: CODE_READY_TESTS_PENDING / UNEXECUTED. Parent base 63614090. No project import,
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

## Final author handoff (no execution)

Status: CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED.
Frozen product/test commit: be9a31261347e8abe59778d3fe1c297d244845d3.
Branch: codex/p65-gpu-training-n2-20261111.
Base remains 63614090b44e8e8762d9ef1f07c7b92a0ffb8143. Exactly six NEW paths;
no old product, N1 fixtures, exports, upstream code or dependency pins changed.

Public entrypoints are explicit module imports:
- phymeo.training.mujoco_tyx_env: MuJoCoTyxActionSpec, MuJoCoTyxVectorEnv,
  MuJoCoTyxTransition, MuJoCoTyxEnvError.
- phymeo.training.rsl_mujoco_tyx: RslMuJoCoTyxVecEnv, ppo_config, make_ppo,
  optimization_evidence.
- examples/train_mujoco_tyx_ppo.py: --case hinge-slide|support-slide|gru --output.

Manual source enumeration is **23 identities**, not a collection/test result:
17 wrapper cases (including two continuous models, six input refusals, two
identity refusals, terminal precedence, physical/task/reset/borrow cleanup
failures, ownership and external-revision recovery), plus six RSL cases
(two fixed PPO tasks, actual entry-value timeout, real GRU, prohibited
counter-only randomization and post-physics adapter packaging failure).
The continuous reference protocol uses horizons 3/5/9/11/13/15/17/19 over
eight steps to exercise distinct partial resets; PPO remains the frozen
5/7/9/11/13/15/17/19 table above.

Text-review fixes were made before any project execution:
1. Device native buffers explicitly use FP32 even if a caller changed Torch's
   global default dtype. Borrowed-view cleanup failure after control CAS keeps
   action_accepted=true and preserves the actual receipt.
2. Parent identified external runtime mutation followed directly by selected
   reset as a recovery bypass. Test-first 5a632b02 and fix be9a3126 require
   full-batch recovery on a changed runtime revision; normal wrapper-managed
   partial endings remain unchanged.
3. Terminal test checks all eight terminated/timeout flags explicitly. The
   first world times out, the other seven have actual v1 outside the declared
   .05 task bound; final observation is captured before their actual resets.

Report contract commit136c0410; initial tests3bf1433c; cleanup negative64924f88;
implementation91736f8e; final reset test/fix as above. A report trailing-empty-line
Git diff check was corrected before the next checkpoint. One read pointed at a
new child-only file in the parent tree and one guarded in-memory text replacement
detected LF/CRLF mismatch before writing; neither was project execution or a
product result. Final scope/diff checks are file/Git checks only.

| Frozen path | Working-byte SHA256 | Git blob |
| --- | --- | --- |
| src/phymeo/training/mujoco_tyx_env.py | 9ebb533bfd24b5862e8d10d00c404c30c6b86f4ef0c56cfbdb0f173e2cd7fad5 | c8a137873ebbf604e02c98f3aa2b7888027986bf |
| src/phymeo/training/rsl_mujoco_tyx.py | cc9862de1257d5bf29cf7892f0a3f305f3952fadd56779b68a4248ddf2756baf | a7297d4c8a90898f975e4ab405d4bc291ee9da0f |
| tests/training/test_mujoco_tyx_env.py | a2eee3614efa1583774f77e2bfc4ad96a36bd16123d601ad51b510f756265270 | ab1c486d78e49aa58e74e2bfd533f9ecfcc5b87e |
| tests/training/test_rsl_mujoco_tyx.py | 2b49698832fa61c99198e74abf23ffb7c17944d1276f8ddb4aade20f95e955f6 | 23044beda5a0c5cc9ee8432d5a0b57d5b9523840 |
| examples/train_mujoco_tyx_ppo.py | d59b978708bab32db1d184c23a19a7aae88f2c0866f55f107e103dce800b8b67 | 63e23530b3175322b4713f74aab30ddaa85cad5e |

The sixth file is this report; its final byte hash is handed off externally to
avoid a self-referential manifest. Each later style/bug fix needs its own actual
tested source binding; this author snapshot is not retroactively called tested.

Parent-only validation commands, **NOT RUN HERE**:
- Focused Ruff check and format check of the five NEW Python files.
- Qualified CUDA profile, no user-site contamination, sole fresh guarded slot:
  python -m pytest tests/training/test_mujoco_tyx_env.py
  tests/training/test_rsl_mujoco_tyx.py -q -x --tb=short -p no:cacheprovider
  with a fresh absolute basetemp.
- The two PPO and GRU tests already invoke the corresponding example functions;
  do not rerun the same green training just to obtain another summary.

Remaining limits: CUDA single-device synchronous VecEnv, decimation1, admitted
native models, fixed TaskSpec reward/bounds, HOST episode bookkeeping/readback
and complete physics audits. Public snapshots additionally run detached CPU
endpoint observations for evidence. There is no GPU PPO throughput claim,
general reward DSL, arbitrary external concurrency protection, training
checkpoint/restore, learning-quality gate, N3/N4 qualification or G1 claim.
Actual optional dependency imports, RSL behavior, CUDA kernels, complete
functional matrix and cleanup still require the parent's execution.
