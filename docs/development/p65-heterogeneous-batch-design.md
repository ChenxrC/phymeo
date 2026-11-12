# G1 heterogeneous MuJoCo-TYX small-batch design

Status: **S0_FROZEN / AUTHORING_ONLY**, 2026-11-12. Baseline `5d3b4c50`; research only. No import, collection, numerical probe, project check or physics execution was performed for this document.

This slice composes already admitted native robot domains into one public independent-world batch. It adds scheduling, complete identity binding and a common publication transaction. It does not change dynamics, combine cross-world contacts, turn different models into one kernel, or establish GPU scale/performance.

## 1. Fixed-source facts and the actual gap

- [Runtime](../../src/phymeo/backends/_mujoco_tyx_runtime.py), `step` at baseline lines 696–866: actual advance/download and every candidate state/full typed receipt are prepared before `device.publish`. [Device](../../src/phymeo/backends/mujoco_tyx/_device.py), lines 157–158: publish only swaps the active-bank index. This is the reusable ordinary-step seam; no per-step clone or CPU integration is necessary.
- The runtime constructor owns one independent native model and one plan for all its worlds. Its controls have model-specific nu/nv/nbody dimensions. Public integration observations and last-step receipts already retain actual q/v/act, applied forces, complete M/J/bias, contact/EFC data and solver/activation/transmission fields where applicable.
- [Checkpoint](../../src/phymeo/backends/mujoco_checkpoint.py), baseline lines 227–263 and 350 onward, binds full source/model/layout/world order and raw compensated clock components. Reset also carries unselected raw accepted-bank rows. These existing validation paths must be reused, not replaced by q/v-only reconstruction.
- [Compiler](../../src/phymeo/builders/mujoco_compiler.py), `MuJoCoSource` and `MuJoCoCompilation`: source and model bytes are owned; exposed native compiler objects remain mutable. Bucketing uses the owned source bytes and capacities, never mutable MjModel identity or merely equal dimensions.
- Local fixed primary source: parent `.venv-newton-1.7.0/Lib/site-packages/newton-1.7.0.dist-info/METADATA` identifies Newton 1.7.0, Apache-2.0. In `newton/_src/solvers/mujoco/solver_mujoco.py:10147–10253`, multiple worlds require matching entity counts, joint types/DOFs/mimic references and shape/constraint types. This solver is not an existing arbitrary-model bucket scheduler.
- Newton `newton/_src/sim/builder.py:14929–15025` groups equal local collision-template bytes and broadcasts world offsets. This is collision-template reuse, not heterogeneous dynamics. `newton/_src/solvers/featherstone/solver_featherstone.py:324–397` builds per-articulation matrix dimensions/offsets; its tile path explicitly assumes matching structure. Preserve that distinction.
- Corresponding official context: [Newton 1.7 worlds](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html). This proposal is Phymeo's explicit model-bucket transaction, not a claim of reproducing every Newton backend.

## 2. Public additive API

Proposed NEW `src/phymeo/backends/mujoco_heterogeneous.py`:

```python
MuJoCoTyxModelSpec(key, compilation, *, contact_capacity=64, constraint_capacity=256)
MuJoCoTyxWorldSpec(world, model_key, initial_state=None)
MuJoCoTyxHeterogeneousBatch(models, worlds, *, mode="cuda", cache_root=None)
```

The two specs are frozen and own immutable declaration metadata. Initial state/control arrays use the original owning public records. The native compilation record is not misrepresented as immutable.

Batch properties/methods:

- `world_keys`, `timestep`, `layout`, `capabilities`, `checkpoint_revision`, `last_step`, `counters`.
- `set_controls({world: MuJoCoControl, ...})` requires every logical world exactly once; model-specific shape and FP32 gates all run before publication.
- `step(h)` requires the same original h as every bucket and returns the original immutable world-to-accepted mapping. It publishes one aggregate receipt containing the complete original typed per-world receipts, model/bucket identities and actual revisions.
- `integration_snapshot(keys=None)` and `snapshot(keys=None)` retain native widths, original world keys and original observation semantics. No padded common q/v vector is invented; full snapshot still performs the explicit CPU endpoint observer.
- `checkpoint()`, `restore(capsule, *, expected_revision)`, `reset(keys=None)`, `close()`, context manager.
- No public child runtime/bank access, tensor-action adapter, training interface, facade clone, dynamic model insertion or multirate stepping in this slice.

All writers hold the facade lock and every affected child lock in canonical bucket order. A writer token rejects same-thread recursive writers rather than relying on RLock alone. Readers cannot observe partial publication. Child runtimes never escape.

### Identity and capacity

1. Require 1–8 declared models and 1–16 logical worlds; reject unused/duplicate model keys, duplicate worlds, unknown keys, invalid key types and empty batches. Existing per-model capacities and every model admission remain unchanged.
2. Group only identical tuples of original primary/archive/model bytes, MuJoCo version, mode and requested contact/constraint capacities. A hash is an index; compare the original bytes before merging. Equal nq/nv or similar geometry is insufficient.
3. Stable bucket order is first model occurrence; world order is the authored world order and per-bucket rows are its filtered subsequence. Layout records world, authored model key, bucket id, local row, source hashes and native nq/nv/nu/na/nbody.
4. Models with equal source but different declared keys may share a bucket; keep each world's declared model key in aggregate metadata. Same-coordinate worlds remain independent.
5. All compiled timesteps must match exactly. Do not rescale control, time or h. Each world has its own actual time/tick and partial reset may deliberately make those differ.
6. Supported models are exactly those individually admitted by the existing runtime. In particular, FREE+BALL contact/activation/limits/dry remain rejected. Combining models does not expand any individual model's domain.

## 3. Parent-owned ordinary-step transaction seam

Parent modifies only the shared runtime/export seam and focused seam tests. Suggested internal contract, acknowledged by parent:

```python
runtime._prepare_step_locked(h) -> frozen draft
runtime._validate_step_draft_locked(draft) -> None
runtime._commit_step_draft_locked(draft) -> None
```

Draft owns or binds owner, checkpoint revision, actual device/bank identity, active index, candidate data/receipts/ticks, accepted mapping and precomputed commit counters. Preparation runs original advance/download/status, input/control/clock/finite checks and full receipt construction. It never swaps accepted state. Validation rejects wrong owner, stale revision/bank, reused draft or mutated layout.

The facade prepares every bucket, creates the complete immutable aggregate, runs every draft validation and the bounded late-failure test seam, then commits. Commit contains only the prevalidated bank/metadata/counter assignments, without allocation, validation, observers, callbacks or device synchronization. All failures that can be injected are before the commit boundary. The original public single-model step delegates prepare–validate–commit with its original acceptance and failure semantics.

On failure, all logical worlds report unaccepted; their old bank, controls, ticks, exact clocks and published receipts remain. Actual launches, transfers and failed candidate work still count. Do not discard a failure or pretend the earlier prepared buckets never ran. Operational counters are not rewindable physics state.

Parent also supplies `_prepare_controls_locked(mapping)`, `_validate_controls_draft_locked(draft)`, `_commit_controls_draft_locked(draft)`, `_discard_controls_draft_locked(draft=None)` and post-commit `_finish_controls_cleanup_locked()`. Step discard is `_discard_step_draft_locked(draft=None)`. These form an analogous small control prepare/validate/commit seam, reusing `_validated_control`, original FP32 checks and device-control retirement/invalidation. This prevents sequential child `set_controls` from partially publishing a batch. No device-action entry is exposed by this facade yet.

## 4. Reset/restore boundary transactions without old decoder rewrites

Parent proposed and this design adopts replacements only at boundary operations:

- Capture the original whole-bucket checkpoint before preparing a replacement.
- For reset, clone only a bucket containing selected worlds, restore that original full checkpoint into the clone using the clone's actual CAS revision, then call the original public reset for that bucket's selected worlds. The checkpoint restore is necessary: clone packing alone is not claimed to preserve arbitrary original compensated-clock bit pairs. Unselected buckets keep the same owner; unselected worlds in replaced buckets retain all raw state/control/receipt fields.
- For restore, validate aggregate source/world-order/mode/capacity binding and destination facade CAS first; prepare replacement clones and run the original public checkpoint restore for each leaf capsule. Restore is whole-batch in this slice.
- Build every result/aggregate and perform the late-prepublication seam before the facade exchanges its complete bucket-owner tuple.
- Retire old owners only after publication. Cleanup errors are retained with strong ownership and reported separately; they do not turn an already accepted reset/restore into a false rejection. Unknown low-level release obeys the existing sticky resource ledger; do not blindly retry freed pointers.
- Candidate preparation failure closes every successfully created candidate and preserves the primary exception plus cleanup failures. Candidate cleanup owners that cannot close stay strongly held for the facade close path. Never drop them in a local list.
- Facade revisions are monotonic across controls, steps, reset and restore. Child revisions remain real child revisions; rebuilding an owner does not justify fabricating a prior child revision.

Proposed NEW `mujoco_heterogeneous_checkpoint.py` provides `MuJoCoTyxHeterogeneousCheckpoint`: owned checksum-protected bytes, explicit batch binding and a bounded ordered list of original `MuJoCoTyxCheckpoint` leaf bytes. No pickle, dynamic import or reinterpretation of leaf physics arrays. Aggregate maximum serialized size is 128 MiB, and original leaf limits still apply; limits are safety bounds, not altered numerical gates. Aggregate last-publication metadata is explicitly bound to the leaf identities/revisions, and leaf receipt payloads remain authoritative.

Counters report facade accepted calls/worldsteps/failed operations and each actual current/retired/candidate runtime's original counters without double counting inherited reset/restore counters. Allocation/cleanup work at boundaries is visible and is not called ordinary-step performance.

## 5. Three fixed qualified model families and independent reference

The primary 24-step matrix uses h=0.001, two worlds per model, authored sparse order `(41, 7, 305, 11, 203, 59)`, with model membership interleaved. Inputs are explicitly declared before execution, including nonzero q/v, controls and external generalized/body loads. Each family has forward, reverse and unloaded control phases; no input/threshold adjustment based on results.

| Family | Already-qualified source to copy explicitly | Distinct behavior and required evidence |
| --- | --- | --- |
| A | Original composed activation fixed R/P source, NONE plus FILTEREXACT, negative/shared scalar gear | nq=nv with nonzero act; actual scalar/generalized motor force, passive terms, full M/J/b and actearly/clamps by original fixture |
| B | Original fixed BALL activation `r4-pyramid` source and parameters | nq!=nv, native quaternion, nonzero activation, primitive condim3 contact, complete EFC and both-side COM force/moment; original combined receipt/schema |
| C | Accepted FREE+BALL `mixed` source from `tests/installed/check_free_ball.py` | nq13/nv11/nu3/na0, FREE root plus BALL/R/P, direct JOINT CSR, external wrench and exact source power identity; no contact |

Tests may reuse existing independent test oracle code; the installed consumer is self-contained and imports no checkout test/example helpers. Every family's numerical gate is copied from its original qualified test/consumer, field for field. No common wider tolerance replaces the original route-specific gates.

Each logical world has two native references: same-entry CPU calculations for full M/J/b/passive/applied/actuator/contact fields and one independent continuous CPU state initialized once. Never feed later TYX states into the continuous lane. Also compare grouped execution against separately constructed single-model TYX runtimes for the identical source/control trajectory; these references are verification only, not a fallback physics implementation.

## 6. Minimal complete case matrix

Both interpret and CUDA cover functional cases. HOST-only construction negatives run once. All attempted input, actual output, full typed receipts and first failures are saved before checks.

1. Three models, six interleaved sparse worlds, 24 steps: complete state/force/motion/native references and one advance per bucket per call.
2. Same models/worlds in a different authored declaration order, same-coordinate placement: identity/order mapping and independent-world isolation; compare by stable world key, not row index.
3. Two keys with identical source/caps genuinely share a bucket; changed source bytes or caps do not alias, even if dimensions match.
4. One model/one bucket equivalence to the existing public runtime, including launch count and stateless/stateful schema preservation.
5. Checkpoint bytes -> actual successor -> restore -> identical successor, including q/v/act/control, full receipts, root/child quaternion and raw compensated clock.
6. Partial reset selects one world in a shared bucket and one in another bucket; unselected bucket owner and all unselected world data remain, then a real successor follows.
7. Last bucket's last world returns a real/injected original nonzero status after earlier buckets prepared: no aggregate/child publication, complete failure prefix, then actual retry matches untouched independent references.
8. Late download/receipt/aggregate preparation failure after all kernels: no publication; original attempt/launch/transfer counters are retained.
9. Last bucket replacement/reset/restore preparation failure: original owner tuple and whole checkpoint remain; retry succeeds. Cleanup-before/after-publication outcomes are classified separately.
10. Invalid last-world control, wrong model/world mapping, stale aggregate CAS, swapped leaf checkpoint, wrong source/mode/capacity and cross-batch capsule fail before publication; pre-device construction negatives prove no earlier device owner is leaked.
11. Reentrant mutation through a test seam and close with outstanding failed-cleanup owner preserve ownership and reject ambiguous use; no recursive partial operation.
12. Source-external installed three-family consumer: bounded 16–24 steps plus checkpoint/reset successors, full provenance and saved evidence. This is functional qualification, not throughput.

No prior long G1 dance, learning run, old complete physics suite or accepted evidence audit is rerun to establish this wrapper.

## 7. Proposed ownership and stop rules

Author NEW only:

- `src/phymeo/backends/mujoco_heterogeneous.py`
- `src/phymeo/backends/mujoco_heterogeneous_checkpoint.py`
- `tests/test_mujoco_tyx_heterogeneous.py`
- `tests/test_mujoco_tyx_heterogeneous_transactions.py`
- `tests/installed/check_mujoco_tyx_heterogeneous.py`
- this design and `docs/development/p65-heterogeneous-batch-report.md`.

Parent owns `_mujoco_tyx_runtime.py`, public exports, any NEW internal transaction record/helper, focused single-runtime compatibility and all execution/build/install. No model/compiler/actuator/contact kernel or existing physics test modification belongs to this author.

Freeze concrete draft/control ABI before implementation. Stop and report a minimal blocker if common publication needs an unbounded private-bank rewrite, original checkpoint cannot preserve unselected worlds, or a model requires unsupported semantics. Two diagnoses without narrowing a failure cause a bounded saved reproducer, not parameter or precision-gate tuning. A successful slice requires all declared functional/failure cases and both installed modes; it closes only this small independent-world interface. Larger W, tensor training action, multiple physics timesteps, cross-world coupling, GPU throughput and Newton comparisons remain later work.
