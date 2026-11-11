# MuJoCo-TYX compact native mirrors

Status: CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED.
Authoring only: no project imports, collection, numerical probe, test, static
checker, formatter, build, installation or performance measurement was run.

## Change and ownership

Starting from parent cb316619, normally merged into child e0ea8636, this slice
replaces per-world persistent MjData and per-step native copies in the TYX HOST
mirror. The TYX accepted/candidate device banks and physical kernels are unchanged.
Original CPU runtime stepping is unchanged. This is a representation/ownership
change, not a different dynamics or observation accuracy mode.

Five changed paths:

- NEW src/phymeo/backends/_mujoco_tyx_mirror.py
- MOD src/phymeo/backends/_mujoco_tyx_runtime.py
- MOD src/phymeo/backends/mujoco_checkpoint.py
- NEW tests/test_mujoco_tyx_native_mirror.py
- NEW this report

Commits: 81d8baf8 tests first; 406878a9 owned codec; 3fa950b0 runtime/checkpoint
connection; feb30bcd contact and complete wire-field coverage. The earlier
p65-native-candidate-memory-next.md research remains separate and unchanged.

## Complete state and API

NativeStateLayout(mj, model) is a model-owner-bound codec for pinned MuJoCo3.3.7.
It obtains every mjSTATE_INTEGRATION component width with public mj_stateSize,
uses the fixed ascending mjtState order, and checks total width. It preserves
all components, including qacc_warmstart, activation, applied loads and zero-size
components. The exact native ordering was checked against the fixed public
[3.3.7 state implementation](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_support.c#L155-L227).
No unavailable Python mj_copyData, pointer access, ctypes or changed pin is used.

NativeMirror owns readonly byte-backed FP64 integration state and the same nine
MIRROR_FIELDS used by the existing checkpoint. capture(native),
from_parts(cpu_state, mirrors), changed(**updates), copy.copy(mirror) and
materialize(mirror, native) have explicit model/layout checks. Shared state and
mirror components must have identical bytes. Nonintegration qacc, actuator force
and constraint force remain separately stored. Source finite/layout errors are
rejected before publication; no quaternion repair or physical state correction.

Original reset declarations still pass the base FP64 unit-quaternion check and
initial native forward. The TYX override immediately captures that result and
releases the temporary native arena before the next world. Initial quantization
is prepared as owned compact records before constructing the device. Reset also
prepares every quantized record before replacement-device publication. It uses
identical FP32 packing; the existing raw device two-component clock is unchanged.

Ordinary step builds all compact candidates and complete receipts before the
existing device.publish and HOST dictionary swap. No MjData constructor/copy or
CPU forward is used there. Existing input/activation/time/finite/contact/EFC
checks remain. Failure in the last world leaves every old record, device active
bank, tick and revision accepted. The new records do not alias downloaded arrays.
Held requested controls remain separate from stored FP32 control channels.

## Observer, restore and lifecycle

Full snapshot owns one lazy runtime-local MjData scratch under the existing lock.
For every requested world it resets scratch, sets the complete saved native
integration state through public mj_setState, restores all explicit mirror fields,
and performs the same detached CPU forward. Every public state/contact/force
array is frozen before the scratch serves another world. integration_snapshot
continues to read accepted compact state directly. There is no new reduced full
snapshot and no CPU-forward result feeding device integration.

A dirty or failed observer is reset/refilled on the next use. Partial snapshot
results never escape. A scratch may be allocated during a failed snapshot or
restore; that owned resource is not accepted physics and is released on close.
No per-world scratch is retained. Original initial-state validation can still
allocate one temporary MjData at a time; this slice does not claim zero native
allocation in construction, reset, restore or observation.

Checkpoint schema, binding, raw bank/clock bits, complete cpu_state and explicit
mirror fields are unchanged. Capture copies the owned complete vector. Restore
validates its full mirror consistency, source/receipt/layout and raw bank binding,
then verifies the replacement device and every detached observer before the
existing CAS publication. Old-resource cleanup/retry semantics are untouched.
Clone retains the original copy.copy injection seam, but copies compact bytes and
rebinds them to the clone's model layout. It owns its own observer scratch. Close
drops scratch and compact records without modifying the existing resource ledger.

Private native derived caches, timers, arena pointers and native MjData identity
are not public accepted physics or a checkpoint payload. Full public observations
are still recomputed, and the tests compare their complete exposed fields.
This does not add support for previously rejected model/plugin/mocap semantics.

## Authored checks, not executed results

Manual enumeration: 29 nodes, consisting of nine HOST-only nodes, ten interpret
nodes and ten CUDA nodes. This is not an actual collection count.

- Four native roundtrips: fixed tree, floating tree, mixed floating contacts and
  stateful activation; complete public state and warmstart, readonly ownership.
- Four malformed state/mirror cases and wrong model-owner refusal.
- Four models per backend: three real accepted steps, nonzero state and all three
  load channels; forbid native allocation/copy during step; use an independent
  old native copy-and-patch shadow for full CPU snapshot and checkpoint field
  equality; complete contacts/wrenches, history ownership, clone, partial reset,
  exact checkpoint continuation and close. Saved JSON includes all states and
  the complete checkpoint, not only maxima.
- Four last-world failures per backend: finite, clock, receipt construction and
  mirror allocation; all accepted owners/bank/revision/checkpoint bytes remain.
- Snapshot and restore late detached-observer failure per backend; same accepted
  state and successful retry. Existing clone cleanup and A1 continuation tests
  remain applicable; their source was not changed or rerun by the author.

Parent may run focused static checks and the new module in its sole serial slot:

```text
<qualified-python> -m pytest tests/test_mujoco_tyx_native_mirror.py -m "not cuda" -q -x
<qualified-python> -m pytest tests/test_mujoco_tyx_native_mirror.py -m cuda -q -x
```

Parent selects directly affected existing checkpoint/clone checks. No old S256
trajectory or audit has been repeated. Allocation pressure, complete equivalence
and any timing benefit require actual future measurements. The original closed
run64 values are only the motivation; they are not results of this change.


## Parent qualification

The author-only status above is historical. Parent sole-owner runs114/115 passed 19 HOST/interpret and 10 CUDA cases (6.30s/4.61s). Directly affected original checkpoint/clone run116 passed55/21.16s; actual training checkpoint (MLP/GRU), post-CAS failure and full/integration continuation run117 passed9/42.53s. All RAW_EXIT=0. These qualifications compare complete old native copy-and-patch observations with the compact representation; they do not add new physical model domains. One import-order finding and four formatting changes were retained in runs108/109, repaired110/111, final112/113 succeeded. Native kernels and CPU runtime stepping stayed unchanged.

The S256 allocation/commit/RSS follow-up is separate and pending. No measured memory reduction or Newton throughput superiority follows from the correctness tests alone.
