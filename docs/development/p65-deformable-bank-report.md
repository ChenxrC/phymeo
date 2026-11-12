# D1 / M2-P persistent deformable node bank

Status: **CODE_READY_TESTS_PENDING / AUTHORING_ONLY**. No project import, collection,
numerical probe, test, lint, formatter, build, install or profile was run by this
author. This is a D1 producer implementation, not M2-Q or a contact-response result.

Baseline: `79e3e0892f364c3681d642f14df23f946c7617e8`, branch
`codex/p65-deformable-bank-20261112`. The parent owns public exports and the sole
execution slot. Product snapshot: `87da4db4`; later author commits change only the
NEW tests, standalone consumer and this report. A parent-owned loan entry-sync
poison fix is required by the final lifecycle test and is not authored here.

## Exact scope and ownership

The NEW files are:

- `src/phymeo/backends/tyx_deformable_batch.py`: immutable registration,
  complete owned HOST mirrors/receipts, source binding, budgets, CAS publication,
  checkpoint/reset/restore, scoped accepted-vertex loans.
- `src/phymeo/backends/_tyx_deformable_buffers.py`: one Runtime and persistent
  per-world native arrays, uploads/readbacks/counters and the existing resource ledger.
- `src/phymeo/backends/_tyx_deformable_state.py`: three small state helpers:
  candidate copy/clear, exact unselected-state copy, compensated clock/tick advance.
- `tests/test_tyx_deformable_batch.py` and
  `tests/test_tyx_deformable_lifecycle.py`: declared physics and transaction cases.
- `tests/installed/check_deformable_bank.py`: standalone installed public consumer.
- This report.

No existing cloth, StVK, R7, integrator, ordered projection, reconstruction,
collision, solver, precision threshold or dependency pin is changed. Public
exports are a parent integration action. Importing the public module imports no
TYX package and constructs no Runtime.

The supported first slice is one distance-constrained node world family, with
native FP32 initial position and velocity, nonnegative inverse masses, fixed
nodes with exactly zero initial velocity, explicit per-step force and gravity.
Constraints and reference data are frozen at construction. Each supplied surface
has stable point/facet IDs and fixed topology; its vertices are the actual
deforming node bank. This does not model triangle bending, continuum volume,
self collision, friction, contact energy, swept validity, or force response to a
query. Candidate surfaces are checked for finite, nondegenerate endpoint triangles;
that check is not an embedding, orientation-preservation or CCD certificate.

Hard domain ceilings are W=8, N=512 and E=1536 per world, **aggregate F=128 per
world across all surfaces**, and 200 position sweeps. Surfaces may be empty; a
registered surface itself is nonempty. Surface keys are unique within a world,
point IDs within a world, facet IDs within a surface. Facet identity is
`(world, surface, facet)`; no coordinate welding or row-index identity is used.
One explicit step has `StepSettings.substeps == 1`; callers control the number of
steps. These are admission ceilings, not an executed maximum-capacity qualification.

## Public API and complete state

The declaration API is `DeformableSurface` and `DeformableNodeWorld`.
`TyxDeformableBatch(worlds, mode, settings, trace, memory_limit_bytes, cache_root)`
owns the device storage. Its public operations are:

- `requirements(...)`: validate registrations and calculate the array-payload
  reservation without importing or constructing TYX.
- `step(inputs_by_world, h, expected_revision=...)`: every world must supply a
  `StepInputs` packet. Only force is uploaded during an ordinary step.
- `snapshot()`, `stats()`, `checkpoint()`, `reset(keys, expected_revision=...)`,
  and `restore(capsule, expected_revision=...)`.
- `borrow_vertices(expected_revision=...)`: scoped view of accepted native
  positions/velocities plus immutable point/facet registration, mode, source
  binding, revision and bank. Mutation, close and nested loans are prohibited.
- `close()` and context management.

Result types are `DeformableSurfaceBinding`, `DeformableMemoryRequirements`,
`DeformableWorldStep`, `DeformableAcceptedState`, `DeformableStepReceipt`,
and `DeformableCheckpoint`. Stable publication arrays are bytes-backed and
non-writeable. The scope checks do not physically prohibit a Python caller from
retaining or writing a raw device array obtained during a loan. The interface is
a protocol for trusted query consumers, not a security or hardware-read-only boundary.

Each current state stores x/v, tick, raw FP32 clock plus compensation, public time,
and its complete most recent step. A step contains original input x/v/clock/tick,
force, stored h, both predictor arrays, actual alpha/lambda, complete sweep count,
residual, geometric error and optional complete original 18-column trace per sweep.
The batch receipt binds all world states and one all-world publication.

The checkpoint is an **owner-bound in-memory** capsule; a strong object token
prevents confusion with another owner or a recycled Python object ID. It has the
full model/backend/world/layout/settings binding and a content hash over array
dtype/shape/bytes, all current world step payloads and aggregate receipt metadata.
It does not serialize native pointers, use pickle, or claim a portable import
format. Restore preserves historical receipts and physical clocks/ticks while the
owner CAS revision remains monotonic. Empty reset is a revision-preserving no-op.
Partial reset returns selected worlds to their original registered state and leaves
every unselected x/v/raw clock/tick/last-step value intact. The original registration
is never replaced by a reset or checkpoint.

## Native arithmetic and transaction

Per world the actual device sequence is:

1. Original `_tyx_kernels.integrate_particles`, accepted x/v to predictor.
2. New state-only copy to candidate x and lambda clear.
3. Original `_tyx_math_kernels.axpy_kernel` **twice** for
   compliance times stored (1/h) times stored (1/h), retaining the original
   two-operation FP32 ordering.
4. Original `_tyx_node_kernels.project_node_distances` for complete ordered
   sweeps, or its original traced variant. HOST residual observation decides
   stopping against the unchanged requested tolerance and 200-sweep ceiling.
5. Original `reconstruct_nodes`, then native compensated clock/tick helper.
6. Synchronize, read and validate the complete candidate; build all immutable
   receipts; perform one all-world bank swap and one CAS revision increment.

The HOST does not integrate, project, rebuild candidate vertices from two snapshots,
or upload x/v after each step. Residual, geometry and complete publication validation
remain HOST operations, with actual readback/synchronization counted. This is not
a device-only or zero-readback pipeline.

All work that can normally fail precedes publication. Candidate writes affect the
inactive bank; a late readback, last-world, receipt or replacement-verification
failure leaves the whole old publication unchanged. A known Python failure is
retryable only after recovery synchronization succeeds. A failed recovery sync
poisons the owner. The parent also applies poison on a failed loan-entry sync;
the loan is revoked and only cleanup remains usable. No opaque driver failure is
declared recoverable by the fault-injection tests.

Reset/restore use the existing inactive bank, upload only requested replacement
states and copy untouched worlds device-to-device. Full native readback must match
the requested bytes before publication. No second state allocation is made per
replacement. All mutating operations hold an owner lock plus an explicit reentry
guard; the loan holds the same lock for its scope.

The existing `mujoco_tyx._resources.DeviceResources` owns every CUDA array before
release and the Runtime last. Before-ledger cleanup failure is explicit
PENDING_CLOSE; a subsequent explicit close can enter the pending ledger. An
exception from an actual opaque free/close is sticky UNKNOWN_RELEASE, retained
strongly by the ledger, and is never blindly retried. Construction primary and
cleanup failures preserve both exceptions and a retained resource owner. This is
the existing release protocol, not a new resource-cleanup subsystem.

## Complete bounded array resource table

For each world let (p=\max(1,E)), I be the requested sweep ceiling and F the
**total** facets over all registered surfaces. E=0 is supported by a padded p=1
native allocation but emits empty logical alpha/lambda and performs zero sweeps.

| Persistent native arrays | Bytes |
| --- | ---: |
| Two x/v banks, four (N,3) FP32 arrays | 48N |
| Predictor x/v, two (N,3) FP32 arrays | 24N |
| Force (N,3) and inverse mass N, FP32 | 16N |
| Gravity (3), FP32 | 12 |
| Padded edge indices (p,2), int32 | 8p |
| Rest, compliance, alpha temporary, alpha, zero, lambda; six FP32 p arrays | 24p |
| Two raw clocks, each two FP32 words | 16 |
| Two int32 ticks | 8 |
| int32 status | 4 |
| Optional original trace (p,18), FP32 | 72p |

Thus native payload **D = 88N + 32p + 40 + (trace ? 72p : 0)**.
There are 21 persistent arrays, or 22 with trace. Interpret uses equivalent NumPy
compute arrays; the label native payload does not imply those bytes are GPU memory.

Compiled owner HOST payload is **C = 36N + 16p + 44F + 12**:
initial x/v (24N), inverse mass (4N), point IDs (8N), padded edges/rest/compliance
(16p), all facet IDs/stable triples/native triples (44F), gravity (12).

A maximum complete current publication payload is bounded by
**R = 84N + 8E + 16 + (trace ? 72EI : 0)**:
input x/v, force, predictor x/v, output x/v, alpha/lambda and input/output raw clocks,
plus every requested sweep's full logical trace.

The declared additional HOST reservation is **C + 3R + 2D**; total
`memory_limit_bytes` admission is **D + C + 3R + 2D** summed over all worlds.
The three R terms cover current and next publications with staging validation;
two D terms conservatively cover native-sized allocation/readback staging.
Immutable snapshots/checkpoints can share existing owned bytes. The owner does
not retain an unbounded history chain. The cap and cap-1 tests check this exact
declared reservation, including positive E and optional trace.

This is an explicitly conservative array-payload reservation, not measured RSS or
a total process-memory limit. Python objects, interpreter/driver/runtime/compiler
caches, caller declarations and caller-retained old receipts/checkpoints/tracebacks
are excluded and not falsely counted as zero. Native arrays are fixed at
construction; step statistics distinguish state versus force upload, allocation,
readback, kernels and synchronization. They do not claim timings or Newton speed.

## Predeclared cases and evidence

Manual enumeration: **42 logical pytest nodes**: 11 HOST registration/budget nodes
and 31 backend-selected nodes. This is not a collection result. Parent runs the
new files under its single serial execution slot; no prior accepted suite is rerun.

D1 is a three-node surface with one edge, nonzero opposed x velocity and compliance
(h^2), h=1/128. The first unit-mass edge's ideal ordered result is
(1+h/24); it must change length by more than (h/48), not merely undergo rigid FK.
D2 is a four-node two-triangle square, five ordered constraints, nonuniform native
velocity, with both free and fixed-node variants. D3 uses sparse worlds 7/31,
opposite force phases, stable point/facet IDs and reversed native/world storage;
keyed geometry/velocity must match without confusing identities.

The independent HOST position/lambda comparison uses 2e-5 relative/absolute;
the original separate TYX node math must match the native results exactly for the
same complete sweep count. Reconstruction is checked against actual predictor and
candidate position. The installed independent velocity audit uses 3e-3 (the 2e-5
position envelope divided by h is 2.56e-3); its uninterrupted reference uses
2e-4 position and 3e-3 velocity. These are fixed execution-before-output engineering
checks, not a claimed physical error theorem.

Other nodes cover E=0, trace arithmetic parity, no CPU projection fallback,
all explicit registration negatives, byte cap/cap-1 both without and with nonzero
trace payload, checkpoint/replay/partial reset and real successors, stale CAS,
wrong owner/world/topology/hash, post-construction NaN/Inf/wrong-shape force,
loan reentry/expiration, complete candidate readback and receipt late faults,
actual first-kernel-before and last-kernel-after known Python faults, pending close,
UNKNOWN no-retry, and the parent's entry-sync poison boundary.

The tests save complete native array payloads with dtype/shape/base64 in JSONL,
including real accepted receipts and states, complete available candidate
readback/receipt fault prefixes, full replay bindings and trace/plain comparisons.
Compact NPZ summaries remain supplementary. A failure does not invent a receipt
which had not yet been constructed.

The standalone consumer uses no checkout test/example or private physics helper.
It defines its own original inputs and independent FP64 ordered recurrence,
performs 12 continuous steps, save/A/restore/B and partial reset followed by two
real successors: **16 batch steps / 32 worldsteps per backend**. It captures
declarations, requests, full receipts/states, independent values, checkpoints,
loan identities, expected nonfinite rejection, actual counters and module origins.
Its HOST projection sentinel wraps only the product step; HOST validation and the
independent handwritten oracle are declared observations. Output is a fresh directory
with full gzip JSONL and summary including PASS/FAIL, mode, counts and consumer SHA.

Parent execution entrypoints (not executed here):

```text
pytest tests/test_tyx_deformable_batch.py tests/test_tyx_deformable_lifecycle.py --tyx-mode interpret
pytest tests/test_tyx_deformable_batch.py tests/test_tyx_deformable_lifecycle.py --tyx-mode cuda
python check_deformable_bank.py --mode interpret --expected-install-root ABS_SITE --output FRESH_DIR
python check_deformable_bank.py --mode cuda --expected-install-root ABS_SITE --output FRESH_DIR
```

## Delivery and remaining qualifications

Tests-first `6e16611b`; native owner `87da4db4`; standalone `2456f5dc`;
complete saved evidence `09a50a3d`; loan entry-sync test `e3792918`.
Only explicit-path Git staging and whitespace checks were used; they are not
project static validation. The author's complete-source readback matched authored
text exactly. One attempted chunked write was blocked by approval review as a
truncation risk; the existing source was retained, and a complete unified patch
with full readback safely replaced the author's own draft. No product execution
occurred during that file-I/O episode.

Execution, actual native frontend support, installed-wheel qualification and exact
runtime counts remain pending parent validation. The parent's entry-sync poison
change and public exports are integration-owned differences. No physical capacity,
maximum-scale throughput, rigid-soft contact loop, general fabric qualification
or complete P6.5 acceptance follows from this authoring handoff.
