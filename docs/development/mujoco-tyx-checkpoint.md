# A1 — exact MuJoCo-TYX accepted-state checkpoint

Status: **CODE_READY_TESTS_PENDING / AUTHORING_ONLY / UNEXECUTED**.
The author has not imported this product or its tests, collected tests, run a
numerical probe, linted, formatted, built, installed, or executed native code.
The parent owns the single execution slot and all acceptance decisions.

This implements lane A1 of
[the 2026-11-11 plan](../superpowers/plans/2026-11-11-newton-parity-until-0800.md).
It addresses the public mid-run recovery gap exposed by the T4 75/77 attempts:
an accepted FP32 free-joint quaternion is not a new FP64 unit-quaternion
declaration. It must retain its published bits. The original initial-state
validation and all physics/admission/solver thresholds remain unchanged.

## Public contract

For an admitted public `MuJoCoRuntime(compilation, mode="interpret"|"cuda")`:

```python
from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint

checkpoint = runtime.checkpoint()
saved_bytes = checkpoint.to_bytes()
loaded = MuJoCoTyxCheckpoint.from_bytes(saved_bytes)

# The caller deliberately authorizes replacing the current destination state.
revision = runtime.checkpoint_revision
runtime.restore(loaded, expected_revision=revision)
```

The whole batch is one transaction, including sparse world IDs and their order.
This version does not select, merge or remap worlds. It does not restore a CPU
MuJoCo or Warp runtime, nor migrate between interpret and CUDA.

`checkpoint_revision` increases after every successful control replacement,
accepted physical step, nonempty reset or checkpoint restore. It is independent
of the rewindable per-world physical tick. A stale destination revision rejects
without changing accepted q/v/time/ticks/controls/receipts/reset declarations.
A historical capsule is intentionally restorable using the current revision.

The source revision is available as `checkpoint.source_revision`; it identifies
the captured source transaction, not a requirement that the destination already
be at the same point. `checkpoint.sha256` is the digest of the complete serialized
capsule. `checkpoint.binding` is detached, immutable metadata.

## Complete state boundary

The capsule retains:

- The **accepted** device bank's full FP32 q/v buffers, including padding, and
  both compensated clock words. It never reads `download()`'s unaccepted
  candidate bank. No quaternion normalization or q/v conversion is performed
  during restore.
- Held original FP64 control packets, independently of the control applied in
  the most recent receipt. The next device step performs its normal FP32 packing.
- Every world's logical tick and original reset declaration.
- The original CPU mirror's `mjSTATE_INTEGRATION` state, including native
  observer warmstart/user state; qacc and published actuator/constraint/applied
  force channels are retained explicitly as well. The native state size comes
  from the already-bound model, not serialized dimensions.
- The complete immutable `last_step`: original/stored control, input q/v/time,
  dynamics, all device contact wrenches and mixed constraint rows, raw EFC/J/
  geometry fields, diagnostics and source/backend labels. Thus the first
  restored observation can inspect the same actual last-step force history.

TYX's admitted Euler dynamics has no persistent solver warmstart or contact
slip/work accumulator: the constraint solve starts with a feasible zero force
on every step, and all geometry/mechanics/solver work buffers are rebuilt.
The accepted bank and held controls therefore cover the engine state consumed
by the next physical step. This statement applies to the current admitted
stateless actuator/passive-force domain; a future history-dependent backend
must extend the schema before claiming this protocol.

Operational allocation, transfer, synchronization and launch counters are not
simulation state and do not rewind. Successful replacement adds the new
device's resource counters to the old owner's counters. Counter values are not
a trajectory equality assertion.

The CPU mirror remains a **detached observer**. Restore uses native get/set
state to retain observer inputs, and forward on prepared copies to check the
observer before publication. Neither CPU forward nor CPU step supplies TYX's
q/v, forces or integration. There is no CPU physical step in this path.

## Binding, bounds and failure behavior

Binding covers original XML/archive/MJB byte digests, native MuJoCo version,
TYX mode and model domain/kind, exact source timestep, native dimensions,
trusted native observer-state width, world IDs/order, contact/constraint
capacities, q/v/clock layouts, and schema/precision/history labels.

Format v1 is owned immutable bytes: a fixed header, SHA256 and canonical JSON.
Arrays use explicit whitelisted numeric dtype, shape and exact base64 bytes.
Records are reconstructed from an explicit local dataclass whitelist. There
is no pickle, eval, executable object or name-driven import.

The overall serialized cap is 128 MiB; checkpoint batches are capped at 256
worlds; each array has at most four dimensions and 1,048,576 items; recursive
decode depth is capped at 32. Binding is compared before decoding state arrays.
Shape products, element widths and encoded lengths are checked before array
allocation. Duplicate JSON keys, unknown record fields, nonfinite values,
wrong accepted-bank shape, incompatible mirror/clock and receipt world/tick/
backend bindings reject. These are artifact validation limits, not enlarged
physics or solver capacities.

A SHA256 detects accidental corruption; it is not authentication and does not
prove that a deliberately re-signed fabricated artifact came from a genuine
accepted execution. The portable format is for trusted application storage.
There is no cross-version/hardware determinism promise beyond the bound
backend and current schema contract.

Restore prepares all host records and a separate device owner, loads the exact
saved bank, reads it back byte-for-byte, and checks detached observers before
the publication under the runtime lock. A failed validation/upload/readback/
observer leaves the existing accepted owner in place. A cleanup error after
successful publication does not falsely report rollback; the old owner is
retained for public close to retry. Public close attempts every retired owner and the current owner independently,
retains only actual failures, and raises an ExceptionGroup containing all their
original errors. Physics access is then closed, while a subsequent public
close retries the retained resources; a successful owner is never retried.

## Controller and environment composition

The physics capsule intentionally does **not** clone an arbitrary policy,
ONNX session, RNG, Env episode, reward accumulator or PolicyRunner private
state. Current PolicyRunner snapshots are observations, not a generic restore
protocol. Controllers must own and restore their explicit history/clock/held
action, binding that application snapshot to `checkpoint.sha256`.

The authored composition test uses a nontrivial recurrent observation history,
two-step control decimation, a held action and a controller tick. It snapshots
these separately alongside the physics digest, then checks identical actions
and complete observations after both owners restore. This demonstrates the
composition boundary without pretending to recover opaque ONNX hidden state
or claiming a complete ResidentVectorEnv clone.

Existing `clone()` retains its prior contract. This change does not silently
reinterpret clone as exact cross-backend or full application-state recovery.

## Predeclared test matrix

`tests/test_mujoco_tyx_checkpoint.py` contains **47 manually enumerated nodes**:
24 interpret/non-CUDA nodes and 23 CUDA nodes. These counts are not collected
or executed results.

| Domain | Real initial state and continuation |
|---|---|
| Three-axis prismatic body | Nonzero q/v, offset COM, real held generalized and COM wrench |
| Fixed hinge/slide tree | Coupled articulated motion, damping/armature and actuator |
| Floating hinge/slide tree | Nonidentity original FP64 unit quaternion, nonzero world linear/local angular velocity, offset COM |
| Floating mixed constraints | Two primitive contacts with condim6, scalar limit/dry friction, nonzero motion and a real motor |

Each primary case uses two sparse world IDs, five accepted steps before capture,
a newly held control distinct from the previous receipt, and six subsequent
steps. It compares the complete state and dataclass field signatures, exact
array bytes, FP64 scalar bits, both original and restored future trajectories,
a separate compatible runtime, and reset semantics. No tolerance is used for
checkpoint continuity.

Additional cases cover:

- Published FP32 quaternion rejected as a new FP64 initial declaration but
  exactly accepted through checkpoint restore.
- Stale CAS after a control update; invalid control and wrong h rejection.
- Wrong source/world order/capacity/backend/layout; an actual distinct compiled
  mass/source as well as malformed binding artifacts.
- Huge declared array dimensions, nonfinite array payload, inconsistent clock,
  mismatched receipt tick, duplicate JSON keys and checksum corruption.
- A failure after replacement upload/readback and a real physical candidate
  rejected after its kernel but before accepted publication.
- Subset reset with distinct world ticks and exact two-word clock preservation.
- Post-publication old-owner cleanup failure without a false rollback; two
  retired owners plus the current owner all receive close attempts despite
  one failure, and only actual failed resources survive for a later retry.
- Explicit controller history/decimation/held-action/clock composition.
- Immutable owned serialized bytes.

The complete snapshots include detached CPU observations and the original
device last-step receipt; the future-path assertions are not just q/v or a
producer hash comparison. Saved capsules and SHA sidecars preserve the actual
mid-run input artifact for the main cases. The new test module does not import
or rerun old accepted numerical suites as an oracle.

Parent-only suggested commands after fresh ownership/guard and environment
binding (using the existing approved Python; paths relative to the active
integration checkout):

```text
python -m pytest tests/test_mujoco_tyx_checkpoint.py -m "not cuda" -q -x --tb=short -p no:cacheprovider --basetemp=<fresh-absolute-path>
python -m pytest tests/test_mujoco_tyx_checkpoint.py -m cuda -q -x --tb=short -p no:cacheprovider --basetemp=<different-fresh-absolute-path>
```

Parent determines adjacent regression and wheel checks under the shared serial
queue. No execution permission is inferred from these suggested commands.

## Delivery binding and remaining risk

Base: `76861760`. Product commit: `0be8df4c`.
Test commit: `7c10eac7ff0bb302b62bc54633ed4dccb804ba53`.
Parent text-review lifecycle correction: `fc90144639fdd12628f608b6c09a107761d01876`.
This correction was made before any A1 project execution.
No old numerical kernel, model admission, native asset, test, training file,
pin, NOTICE or solver budget was edited.

| File | Working SHA256 | Git blob |
|---|---|---|
| `src/phymeo/backends/_mujoco_tyx_runtime.py` | `23348b485db4f0d26586100c841c843a1b0643374ace719477f7126ea5c5505a` | `8b160fe39ff3784a27f367fd9b521092c5db5b22` |
| `src/phymeo/backends/mujoco_checkpoint.py` | `63cdca670b43f6bbc38d6e598cb7136c8b2ed6618bc25f52f2bb49ed6ad73069` | `02ceda81a60aa5713818880d3d18faa61a84365f` |
| `tests/test_mujoco_tyx_checkpoint.py` | `ebd935ca0d610dfb021b732dfef5b2ef1c1d83532553d485a5c7a1e9a8d8f8d5` | `83e091ee4243139ec6f277955a9b50345bc390a6` |

Only text review, exact authored test readback, explicit staged scope and
`git diff --check` have completed. There is no runtime success claim. First
execution may expose ordinary interface, serialization, lifecycle or test
fixture issues; parent must retain the first raw result and diagnose before
any fix. The original G1 full-dance run is a separate parent-owned experiment
and is not evidence for this new checkpoint API.
