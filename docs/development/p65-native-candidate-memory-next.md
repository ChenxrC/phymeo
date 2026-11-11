# TYX native candidate memory: bounded next slice

Status: SOURCE_REVIEW_ONLY / NOT_IMPLEMENTED / NOT_EXECUTED (2026-11-12).
Scope: the TYX runtime's HOST mirrors and detached CPU observations. No change
to native CPU stepping, TYX physics kernels, models, arena limits or tolerances.
Source reviewed from parent 8a9d64d21d11fad02523e2f99bb338b3bf4fdb5f.
This is a proposed follow-up, not an implementation authorization or speed claim.

## Evidence and API boundary

The already closed run64 memory-observation-004.json reports 24,576 intercepted
TYX runtime copy calls and zero copy failures. Both modes have 48 steps with
256 worlds; full snapshot copies were explicitly outside that interceptor.
Actual copied data had narena=13,631,488 B, nbuffer=4,384 B and
maxuse_arena=28,828 B. These are saved values, not fresh calculations or a basis
for shrinking the arena.

Selected saved process observations (private commit / RSS, bytes):

| Existing event | Private commit | RSS |
| --- | ---: | ---: |
| full construct-enter | 1115066368 | 736927744 |
| full construct-complete | 4886155264 | 902918144 |
| full first-step last copy | 9010900992 | 1253064704 |
| full first-step complete | 5559836672 | 1270046720 |
| integration construct-enter | 2596773888 | 487067648 |
| integration construct-complete | 6215569408 | 611557376 |
| integration step48 last copy | 9998934016 | 924213248 |
| integration close-after-owner-attempts | 2900836352 | 820289536 |
| integration mode-function-returned | 2701905920 | 581451776 |

This establishes material allocation pressure at the inspected copies. It does
not establish a leak, exclusive allocator cost, or that the previous OOM had a
single cause. Process values include unrelated live Python/Torch/TYX owners.
No old trajectory or compressed trace was read for this note.

Local qualified package METADATA is MuJoCo 3.3.7. Its installed
include/mujoco/mujoco.h:220-225 and introspect/functions.py describe the C
mj_copyData(dest,model,source), but introspection of C declarations is not proof
of an available Python function. The fixed official Python binding explicitly
skips mj_copyData and supplies MjData.__copy__ instead. Consequently, there is
no supported Python destination-copy replacement to call in this pin.
The fixed implementation allocates the model-sized arena for new MjData;
copying does not imply every arena byte is copied. Relevant primary sources:

- [3.3.7 Python function registration](https://github.com/google-deepmind/mujoco/blob/3.3.7/python/mujoco/functions.cc#L149-L156)
- [3.3.7 MjData copy binding](https://github.com/google-deepmind/mujoco/blob/3.3.7/python/mujoco/structs.cc#L701-L721)
- [3.3.7 data allocation and copy](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_io.c#L965-L1174)

The installed wheel contains headers/introspection and compiled bindings, not
these implementation .cc files; fixed-tag primary source was read to resolve
that exact API ambiguity. No binding was imported or called.

## Current source facts

1. `_mujoco_tyx_runtime.py:586-739`: after actual device advance/download and
   status checks, each world copies its entire accepted MjData, replaces ten
   published channels/time, verifies clocks/input bindings and builds complete
   receipts. All candidate data are retained until the last world passes. Then
   device.publish flips the accepted bank and the HOST dictionaries are swapped.
2. `integration_snapshot:741` needs only owned q/v/act/ctrl/time, applied forces,
   ticks, held controls and receipt. It never needs an arena.
3. `snapshot:769` currently delegates to the base snapshot. The base copies each
   accepted MjData, calls mj_forward on that detached endpoint and freezes all
   returned fields. Native forward is an observer, never the TYX step.
4. `mujoco_checkpoint.py:285-350` already separates MIRROR_FIELDS from the full
   mjSTATE_INTEGRATION vector. This signature includes acceleration warmstart,
   not merely q/v; source-bound complete state is part of the capsule contract.
5. `reset:782`, `clone:873`, checkpoint restore:354-529 and `_initial_data`
   construction all currently create/retain MjData. Optimizing step alone cannot
   remove the existing per-world accepted arenas or all restore/clone peaks.
6. Current TYX admission rejects plugins, mocap and userdata; constraints/activation
   have their own admitted plans. Do not use that fact to silently discard a
   state field or weaken future admission. Preserve the full signature layout.

## Recommendation: compact accepted mirrors plus one detached observer scratch

A minimal safe follow-up is a TYX-specific representation change, not a one-line
replacement for copy.copy. Keep the device's existing accepted/candidate banks.
Make each HOST accepted mirror an owned compact record containing every native
integration input plus existing MIRROR_FIELDS, exact time and model/layout
binding. Preserve qacc_warmstart even though TYX has a cold per-step solver;
CPU snapshot reconstruction and checkpoint continuation still consume it.
Keep the raw device two-component clock in its existing authoritative bank and
capsule. Keep held requested controls separate from actually stored controls.

Candidate mirror records and all receipts are fully allocated/validated before
publication. The step's input-state/activation/clock/finite/EFC/force checks stay
unchanged. At publication, flip the already validated device bank and replace
only prepared HOST references. No native allocation, mj_setState, array copying,
receipt construction or observer call may first occur after that boundary.
Do not optimize by writing accepted MjData in place before the last world passes;
that would lose the present all-world transaction on a late failure.

For full snapshots, own one model-bound MjData observer scratch per runtime,
protected by its existing lock. For each requested world: reset scratch, fill
all saved integration fields through public setters/mj_setState, restore any
explicit mirror fields outside that signature, call mj_forward, and freeze the
complete public state before using scratch for another world. Use the existing
FP64-to-FP32 relationship unchanged; never normalize checkpoint quaternions.
No scratch view, MjData object or native contact owner escapes in a snapshot.
On observer failure discard partial output, mark scratch dirty, and reset/refill
it on the next call. Accepted compact data, device banks and revision stay intact.
The extra scratch is not needed for integration_snapshot or ordinary TYX steps.

Use the pinned public state signature and mj_stateSize to define/validate a
complete packing adapter; do not infer state by a short q/v slice. Preserve
zero-sized supported fields, full warmstart and activation. Check both directions
against public mj_getState/mj_setState in tests. Preserve the existing capsule
schema/bytes where possible; do not silently reinterpret old checkpoint data.

## Exact seams and proposed files

- NEW `backends/_mujoco_tyx_mirror.py`: owned compact state, public native-state
  capture/materialization, shape/source validation. It is not a new solver.
- MOD `_mujoco_tyx_runtime.py`: initial mirror capture, quantization, candidate
  step construction, full/integration snapshot, reset, clone and close.
  `_pack_states` already consumes only mirror attributes; kernels need no edit.
- MOD `mujoco_checkpoint.py`: capture compact CPU integration state without
  requiring MjData; parse/validate old capsule into compact candidates; retain
  all bank verification and detached observer checks before the existing CAS.
- NEW focused mirror/transaction tests; one small addition to the existing
  allocation observer if required to count native constructors/step copies.
  Original generic CPU runtime stepping should remain unchanged.

Initial construction is an important seam: base `_initialize` calls the virtual
`self._initial_data` and uses only qpos/qvel/act/time when making reset records.
A TYX override can retain the existing base initial-state unit-quaternion and
CPU-forward validation, then immediately capture/release its one native object
into a compact record. This avoids first accumulating W native objects merely
to compact them later. A later optimization may reuse construction scratch;
it is not necessary for the first semantics-preserving step.

Reset prepares selected compact candidates and the replacement device before
publication; other worlds retain their exact accepted records. Clone copies
compact bytes/receipt ownership into the clone, whose scratch belongs only to
that clone. Restore validates every compact candidate and full observer output,
then retains the existing bank/CAS and old-resource retry semantics. Close drops
scratch and compact records after best-effort device/control cleanup; no retired
native scratch may retain model or world data inadvertently.

## Alternatives that do not satisfy this target

- Public Python mj_copyData: absent in this pin; no ctypes/private address bridge,
  binding patch or version change is proposed.
- Two preallocated MjData objects per world: viable to remove allocation churn
  using complete state transfer, but permanently retains two arenas per world.
  It does not solve the requested memory-reduction problem and still needs the
  same complete-state transfer proof; do not claim lower peak from this alone.
- One scratch with immediate in-place accepted-world writes: permits partial
  HOST publication if a later world/receipt fails, even if the device bank stayed
  old. Not acceptable.
- Compact q/v-only state or visualization copy: misses warmstart, activation,
  applied channels and full snapshot/checkpoint inputs. Not acceptable.
- Per-world retained snapshot scratch: quietly restores W arenas. Public results
  must own ordinary arrays and no native views.

## Bounded first verification, after separate authorization

1. Complete native-state roundtrip and owned-array nonalias tests on fixed tree,
   free tree with nonzero root/joint velocities, and stateful activation. Confirm
   checkpoint CPU vector, raw FP32 bank/time and all mirror bytes unchanged.
2. Sparse two-world simultaneous stepping with three control channels, contact
   support and activation: compare full/integration outputs and complete receipts
   to the pre-change route under existing gates. Keep snapshots across subsequent
   steps/resets to prove no scratch alias. No physics benchmark gate is weakened.
3. Inject last-world clock/EFC/receipt and mirror-allocation failures, plus full
   observer failure: all accepted worlds/bank/ticks/revision stay unchanged.
   Selected reset, clone, exact checkpoint rewind, stale CAS and restore late
   failure retain their existing contracts. Borrowed controls/task frames still
   invalidate only on actual committed revision changes.
4. Reuse the fixed S256 two-mode 48-step diagnostic once after correctness checks.
   Record real MjData construction/copy counts, exact narena, process commit/RSS
   and complete wall scopes. Require no per-world native allocation in step,
   bounded scratch ownership, identical physical/learning inputs and outputs.
   This is not an automatic repetition of the full nine-cell benchmark or G1.

If public full-state roundtrip or detached full snapshot cannot preserve a field,
stop and name that field before implementation expansion. If the new source
still retains W native owners, do not call the memory goal achieved. No measured
speedup, memory saving or native clone/checkpoint equivalence is claimed yet.
