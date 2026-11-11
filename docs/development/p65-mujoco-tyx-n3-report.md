# MuJoCo-TYX N3 update-boundary checkpoint

Status: AUTHORING_ONLY / UNEXECUTED. Base ed6f6d847c3d64abb13ab28f5f7042832c5b2f3e.
Parent alone owns execution; this author performs source/text/file/Git work only.

## Frozen public surface

MuJoCoTyxPpoSession(adapter, steps=16, recurrent=False, generators=None) owns an
actual upstream RSL5.5.2 PPO and next_iteration, borrowing the N2 device adapter.
Only the actual fixed N2 MLP or one-layer GRU configuration is admitted.
update() orchestrates upstream act/process_env_step/compute_returns/update.
save_checkpoint(path) requires at least one completed update, storage.step=0,
cleared upstream Transition, no failed or in-progress phase, and synchronized
CUDA producer work. restore_checkpoint(path, expected_revision=...) is the
versioned composition boundary. It does not deserialize arbitrary learners.

The N2 wrapper gains public composition lock/capture/restore/invalidation methods;
the checkpoint module never touches a private physical bank or runtime lock.
All candidate learner, optimizer, hidden and environment validation/allocation
occurs before public physics CAS. Candidate-construction RNG effects are restored
in try/finally. After CAS, fresh task refresh must reproduce saved observation
bits before publication; a late refresh or RNG-install failure marks the session,
adapter and env as needing recovery, reports physics_restored=true, and never
claims old-combination rollback.

## State ownership

| Owner | Saved/restored state |
| --- | --- |
| Public physical capsule | all original source/layout/capacity bindings, exact q/v/clock/tick, initial states, held ctrl/qfrc/xfrc, last physical receipts; original no-warmstart semantics |
| Env | task/action/world/h/horizon binding, episode IDs/lengths/returns, transition sequence, current CUDA observation, raw previous policy action (zeroed for reset worlds) |
| Upstream PPO | actor/critic state_dict including normalization buffers, complete Adam state and hyperparameters, actual module training flags, declared fixed algorithm configuration |
| Recurrent state | actor/critic real GRU hidden tensors outside state_dict, exact layer/world/hidden axes; MLP explicitly None |
| Randomness | Python global, NumPy legacy global, Torch CPU and selected CUDA global; each explicitly registered owned Torch/NumPy/Python generator |
| Session | exact next_iteration and empty-storage/cleared-transition proof, dependency/device/determinism settings and original configuration |

Registered generators are explicit; arbitrary caller RNGs are not discovered.
Operational resource counters, destination CAS revision and task frame epochs are
monotonic diagnostic identity, not rewindable physical time. Completed transition
objects are diagnostics already consumed by empty storage; restoration clears
their stale references while retaining all future-determining state above.

## Files and corruption boundary

Fresh sibling staging directory, bounded physics.bin and training.json, then
manifest.json with exact file sizes/SHA256 and complete binding. Read-back verify
before atomic same-volume rename. Existing destination is refused, never replaced.
No pickle/eval/import of a class named by serialized content. Fixed filenames,
JSON duplicate/nonfinite rejection, depth/array/total-size hard limits, source
binding before tensor allocation. Hashes detect corruption, not malicious signing.
Failed staging directories are retained as evidence; no recursive cleanup here.

## Predeclared tests and comparison

Two unchanged N2 tasks, W8, seed11011/11012, original .002/.001 timestep,
horizons5/7/9/11/13/15/17/19, 16-step rollout and fixed upstream PPO.
Perform updates0 and1; save next_iteration=2; A performs update2; restore the
same runtime at its current CAS revision; B performs the same update2.
Compare raw policy actions, current/final obs, rewards/done masks, episode
state/returns, every world physical state/clock/control/receipt, Adam state,
model parameters and RNG consumption. Only monotonic destination revision/
task owner/frame epoch and operational diagnostic counters are excluded, with
their monotonicity checked separately. No near-neighbor initialization/replay.

GRU seed11013: one original8-step update, save nonzero actor AND critic hidden;
continue another8-step update, restore, and repeat. Partial done masks must
preserve non-done hidden and zero only done columns. This is not a new learning
benchmark or a claim of cross-platform bitwise equivalence.

Negatives: component hash/truncation; wrong model/world/task/action/h/config;
wrong policy array size; missing RNG/RNN/registered generator; stale CAS;
nonempty storage, pending Transition, mid-update save; existing bundle target;
candidate-load failure after RNG consumption (old full state/RNG retained);
post-CAS task refresh failure and post-CAS RNG installation failure separately.
No old complete N2 group is rerun. Genuine nondeterminism must report its first
field difference; do not relax exact continuation or tune fixtures/budgets.
## Delivered authoring snapshot

Product/test checkpoint: c4e60434 (base ed6f6d84); final documentation follows.
The tests-first history is 17e407ce (22 new N3 nodes) and dd34e7db (one focused
previous-action ownership node in the existing env file). No test was collected,
imported or executed by this author; no numerical probe, lint, formatter, build,
installation or profile was run. Git whitespace checks only were used.

Exact seven-file scope:

- NEW src/phymeo/training/mujoco_tyx_checkpoint.py
- MOD src/phymeo/training/mujoco_tyx_env.py
- MOD src/phymeo/training/rsl_mujoco_tyx.py
- NEW tests/training/test_mujoco_tyx_training_checkpoint.py
- MOD tests/training/test_mujoco_tyx_env.py (one directly affected ownership test)
- NEW examples/restore_mujoco_tyx_ppo.py
- NEW docs/development/p65-mujoco-tyx-n3-report.md

The physical backend, original assets, N1/N2 task constants, public exports,
upstream RSL/PPO and environment/dependency pins are unchanged. The new module is
explicitly imported only in the optional CUDA training profile.

## Implemented public owner seams

- env.training_boundary() serializes wrapper step/reset and composition using
  the wrapper's own lock; no private runtime bank/lock is borrowed.
- env.training_binding() returns owner-free task/action/world/h/horizon/device
  identity, combined with the original physical capsule's complete binding.
- env.capture_training_state() makes owned HOST copies of episode metadata,
  observation and previous policy action.
- env.restore_training_state(checkpoint, state, expected_revision=...) validates
  native tensor layouts and allocates candidates before the original public CAS;
  a fresh device task export must reproduce saved observation bits.
- adapter.restore_training_state and invalidate_training_state preserve explicit
  learner-vs-physical failure stages and clear consumed last-transition references.
- session.update/save_checkpoint/restore_checkpoint/training_state are the new
  public orchestration surface. Runtime, env and adapter remain caller-owned.

Storage step zero is checked, and every field of the upstream pending Transition
must be cleared. Upstream rollout arrays remain allocated after clear(), but all
slots are overwritten by the next complete fixed-length rollout; they are not
future-determining update-boundary state. The previous step's gradients and
distribution-output cache are likewise overwritten before their next use.
Actual recurrent hidden tensors are different and are explicitly included.
No rollouts or model/Adam tensors are silently reset to manufacture equality.

Binding includes original fixed PPO config, exact package versions (RSL, Torch,
TensorDict, NumPy, MuJoCo, TYX), Python version, CUDA device/capability, Torch CUDA
version, deterministic/TF32/cuDNN flags and CUBLAS_WORKSPACE_CONFIG. This is a
same-profile reproducibility contract; it does not promise cross-GPU or version
migration. Named RNG objects must be registered and exclusively owned; unrelated
application RNGs and other CUDA devices are outside this one-device session.

The data-only codec uses fixed filenames, no pickle/import-by-string, duplicate
JSON-key/nonfinite rejection, depth32, 1,048,576 total items and per-array elements,
64MiB training data/decoded array bytes, 1MiB manifest and the existing128MiB
physical-capsule ceiling. Steps are1..64; saved completed iterations1..2^24.
All files are read/hash-checked before restoration; a new sibling staging
directory is verified before the Windows same-volume rename. Existing targets
are never replaced. Partial staging directories remain evidence.

A post-CAS failure is deliberately not recoverable by another update on the
poisoned session. The caller must discard its pending rollout/session and perform
explicit full recovery/reconstruction; this first slice does not auto-retry or
claim atomic rollback across CUDA, physical publication and RNG installation.

## Predeclared execution entry points for parent

Focused new CUDA group (22 nodes, uncollected here):

    python -m pytest tests/training/test_mujoco_tyx_training_checkpoint.py -q -x --tb=short -p no:cacheprovider

Directly affected env node, plus parent-selected old reset/export/ownership
neighbors rather than the complete already-accepted N2 PPO group:

    python -m pytest tests/training/test_mujoco_tyx_env.py::test_training_state_previous_action_and_episode_reset_ownership -q -x --tb=short -p no:cacheprovider

Manual example (the three positive test nodes call this exact path, so do not
repeat it merely to produce evidence):

    python examples/restore_mujoco_tyx_ppo.py --case hinge-slide --output FRESH_DIRECTORY
    python examples/restore_mujoco_tyx_ppo.py --case support-slide --output FRESH_DIRECTORY
    python examples/restore_mujoco_tyx_ppo.py --case gru --output FRESH_DIRECTORY

Each positive node writes exactly CASE.jsonl and CASE-bundle/{manifest.json,
physics.bin,training.json}. JSONL contains configuration, every warm update's
complete transition set, saved training/physical state, both complete continuation
records and RNG draws, restored state, exact comparison summary or failure.
The comparison excludes only destination revision/frame epoch/action epoch and
operational counters; full actual clocks, q/v/control/receipts, task/episode,
learner/Adam/RNG and RNN state remain compared. Selected accepted-step/control
counters and destination publications must remain monotonic.

Current status: CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED. Parent owns
all forthcoming static checks, focused CUDA validation and final integration.
