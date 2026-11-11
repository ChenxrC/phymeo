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
