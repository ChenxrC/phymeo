# G2b-A: owned CUDA payload allocation handoff

Status: **CODE_READY_TESTS_PENDING / UNEXECUTED**. Source-only authoring on
`codex/p65-g2ba-allocation-20261113`, base `096964e7`. The parent owns all project
execution and integration. This report does not accept W32/B32 or claim a
Newton performance result.

Product checkpoints: `cf51eff3` (descriptor, shared budget and cleanup ledger),
then `7e47bf16` (six device allocation paths). Only text/source comparisons,
file IO and Git diff checks were performed. No import, test, numerical probe,
formatter, compiler, build, installation or profiler was run by this author.

## API and ownership

- `MuJoCoTyxResourceBudget(device_payload_limit_bytes)` accepts a nonnegative
  builtin integer. `snapshot()` returns owned, nested immutable mappings and
  tuples. It records registered CUDA payload `current_bytes`, `peak_bytes`,
  `reserved_bytes` and `unknown_retained_bytes`; UNKNOWN is a subset of current.
- `admit(additional_bytes)` holds the budget RLock and checks
  `current + reserved + additional <= limit` for its full context. It does not
  reserve or double-charge bytes. Leaf reservation/release can reenter that lock.
- `allocation_manifest(plan, nworld, nbody, capacities)` creates metadata only.
  Its frozen `AllocationManifest` has `entries`, `aliases`, `schema` and
  `payload_bytes`. Every `AllocationEntry` has `name`, `shape`, NumPy dtype string,
  `role`, `source`, and derived `nbytes`. Source names bind the original plan
  arrays; `input:q/v/time/act` binds the actual packed input.
- Every leaf constructor adds optional keyword `resource_budget=None`. A CUDA
  leaf without a shared budget receives its own exact-manifest payload tracker.
  Interpret still validates all descriptor bindings but reserves no CUDA bytes.
- `device.allocation_snapshot()` returns the immutable manifest, explicit
  `manifest_payload_bytes`, declared allocations actually completed, owner state,
  shared budget snapshot, and the accounting scope. Closed budget metadata stays
  inspectable. No native handles are included in these snapshots.

The parent owns runtime/facade/checkpoint propagation, public exports, cap
changes, all-bucket constructor admission and boundary/clone overlap admission.
This author did not alter those shared files. Related operations must take the
budget lock before resource locks. `DeviceResources.add_buffer` and `close`
implement that order; allocation snapshots do not hold those locks in reverse.

## One descriptor drives the actual allocation

The six allocation paths now use named `_allocate` calls and the same manifest;
there is no second byte-only sizing formula. Each source/staging ndarray is
checked for exact shape, dtype and payload bytes. A returned CUDA array is also
checked against the descriptor, after its real owner has been registered for
cleanup. `_buffer(value)` and `DeviceResources.add_buffer(owner)` keep their
original signatures for existing fault-injection seams.

| Route | Descriptor preservation |
| --- | --- |
| T0 | Original model tables, both q/v/two-component-clock banks and all original control/scratch arrays. |
| Fixed tree | Original width8 mechanical scratch; independent packed q width `max(8,nq)`; native DOF/static tables. |
| Floating tree | Original width `max(8,nv)` and independent q/v widths, including FREE+BALL. |
| Normal | Base plus four original static tables, complete raw slots, full J, both S-by-S arrays and extended scratch, even when its kernel is disabled. |
| Mixed constraints | Base plus original static/float/int arrays; logical zero uses original `max(1, capacity)` padding. No EFC-by-EFC matrix is added. |
| Activation/CSR | Activation appends to the same owner. Enabled CSR has its real four allocations. Disabled CSR aliases parents/gravity/body-force and is not charged twice. |

The final normal/mixed/activation wrapper plan is described and reserved once,
before TYX Runtime creation and before leaf zero-filled staging arrays. Base
constructors defer manifest completion to their contact extension. Both state
banks are charged even if their initial contents are identical. Static plans and
existing packed HOST inputs are not relabeled as CUDA-owned storage.

All kernel call arguments, arithmetic, integrators, advance/download logic and
publication remain unchanged. Original allocation ordering is retained, including
normal working-set integer storage. Construction cleanup now preserves both the
primary exception and a cleanup exception in a `BaseExceptionGroup`.

## Lifecycle semantics and scope limits

A successful real buffer registration transfers its manifest bytes from reserved
to current; each successful free deducts only that allocation. A callback that
throws keeps its bytes as sticky UNKNOWN and retains the existing strong-owner
quarantine. Independent siblings are still attempted; the dependent runtime is
not closed when a buffer release is unknown. Repeated close never retries an
opaque UNKNOWN handle. Runtime-close failure is separately recorded even when
all known buffer payload has been released.

Construction exit cancels only unconsumed reservation. If an allocator throws
without returning an owner, this layer neither fabricates a handle/free event nor
claims knowledge of the allocator's hidden internal failure state. A failed
resource-ledger metadata construction cancels its reservation before propagating.

The fixed installed TYX source was read, not executed:
`.venv-training-cuda313/Lib/site-packages/tyx/cuda_array.py` defines public
`shape/nbytes` at lines33–34 and `dtype` at lines135–136. Its allocation calls
`max(nbytes,1)` at line39; payload remains the public logical `nbytes`, while
allocator minimum/rounding overhead is explicitly outside this ledger.

Excluded/UNKNOWN: CUDA context, allocator overhead, JIT/cache, unreturned native
allocation internals, HOST/Python/native MuJoCo memory, JSON/checkpoint temporary
peaks, and independently owned device-control/task-workspace buffers. Snapshots
therefore expose `total_device_live_bytes`, `total_device_peak_bytes`,
`host_live_bytes` and `host_peak_bytes` as `None`. Interpret close cannot be used
to claim HOST arrays were freed while an owner still retains them. Closed owner
metadata is retained without native handles; its HOST footprint is not budgeted.

## Exact authoring manifest and remaining qualification

New product files: `backends/mujoco_resource_budget.py` and
`backends/mujoco_tyx/_allocation_manifest.py`.
Modified product files under `backends/mujoco_tyx`:
`_resources.py`, `_device.py`, `_tree_device.py`, `_floating_device.py`,
`_normal_device.py`, `_constraint_device.py`, `_actuator_device.py`.
This report is the only additional authored document. All paths are below `src/phymeo`
except this report. The unrelated old T1 review remains untracked and unstaged.

Parent-owned pending qualification: independent descriptor-vs-real allocation
checks across all routes/padding/aliases; exact-fit and one-byte-short admission;
no new device allocation during ordinary steps; constructor and boundary failures;
successful cleanup and sticky UNKNOWN accounting; clone/reset/restore shared-budget
propagation; interpret scope; prior physical/transaction compatibility and a new
installed consumer. No result is inferred from this source handoff. W32/B32 and
its real GPU visualization are separate parent-owned work after these gates.
