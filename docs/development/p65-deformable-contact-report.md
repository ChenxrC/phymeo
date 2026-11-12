# D3 deformable / moving-rigid normal response

Status: **CODE_READY_TESTS_PENDING / UNEXECUTED**. This author has not imported,
collected, run, formatted, linted, compiled, profiled, or built the project.
The parent exclusively owns all execution and shared-seam changes.

Base: `5d3b4c50b67e9a3c082f86a6c5bc69390abab7be`.
Branch: `codex/p65-deformable-response-c-20261112`.
The implementation consumes the parent's separately authored shared transaction
interfaces at `9dea3ea1`; those shared files are not edited by this lane.

## Scope and public interface

New product modules:

- `backends/tyx_deformable_contact.py`: common owner, immutable accepted state,
  receipt/checkpoint, finite force buffers, normal-law invocation and audits.
- `backends/_deformable_response_source.py`: dual accepted-source adapter.
- `backends/_tyx_deformable_contact.py`: source-pose initialization and
  deterministic transpose of site forces to nodes and COM wrenches.
- This report is the only other file owned by this author. The original testing
  agent owns the independent response/protocol tests and installed consumer.

`TyxDeformableContactBatch(node_worlds, participants, model, bindings,
materials_by_pair, *, mode, settings, limits, contact_capacity,
memory_limit_bytes, node_memory_limit_bytes, cache_root)` owns its new node and
resident scene objects. It supports 1–8 sparse logical worlds, exactly one
ordinary floating participant with one physical body in each world, and the
original D1 node model/topology domain. It includes true six-coordinate motion,
COM offsets and principal-inertia orientation supported by that original
floating model. It does not support additional scalar children, affine/mimic
participants, multiple rigid participants, BALL, or new asset lowering here.
Those existing independent domains are not changed.

The explicit `DeformableBoundaryModel` contains original open
`DeformableBoundarySurface` declarations plus sphere/capsule shapes. Moving
rounds use direct `ResidentParticipantBinding` /
`ResidentFloatingTreeBinding` with group zero and complete physical topology.
An open surface is never labeled a closed mesh. The full eligible pair material
table is mandatory; `None` explicitly disables response, while masks/exclusions
remain authoritative. Every active facet is one declared discrete spring,
fraction one. This is not an area-pressure quadrature or a promise of identical
force after arbitrary remeshing.

Public methods:

- `step(inputs, h, expected_revision=...)`: complete world-to-`StepInputs`
  mapping; returns one `DeformableContactReceipt`.
- `set_controls(participant_inputs)`: original resident controls, with the
  common public revision advanced.
- `snapshot()`: world-to-`DeformableContactState(world, node, rigid)`.
- `checkpoint()`, `restore(checkpoint, expected_revision=..., worlds=None)`,
  `reset(worlds=None, expected_revision=...)`.
- `revision`, `world_keys`, `binding_sha256`, `last_step`,
  `last_attempt`, `stats()`, `close()`.

Checkpoint ownership is in-memory, strong-owner-bound and content hashed over
the complete original node capsule, rigid controls/history and accepted response.
It is not portable serialization or a generic clone. Partial replacement keeps
unselected physical banks/history; full restore preserves the original D1
batch receipt. Selected worlds are restored together through the parent common
transaction, never by two sequential public restore calls.

## Force law and exact sampling semantics

Sampling is **discrete entry**, explicitly not midpoint, backward Euler, CCD,
an implicit joint contact solve, or sustained-support qualification.

At a common accepted entry, actual node coordinates and actual resident FK
produce the complete finite-distance query. The original normal points from
the sphere/capsule axis toward the triangle. For original triangle barycentrics
`beta`, node velocity is `sum(beta_i v_i)`; rigid material velocity is
`v_COM + omega cross (p_rigid - COM)`. Relative velocity is node minus rigid.

The unchanged `normal_layer_rows` kernel evaluates:

`c=max(0,width-gap)`,
`rate=normal dot relative_velocity`,
`F=k*c*max(0,1-damping*rate)`.

Friction is zero. Node forces are `beta_i F normal`; rigid force is
`-F normal`; rigid torque is the true entry COM lever crossed with that force.
The original layer's unit-normal gate and native FP32 admission are retained.
No barycentric renormalization, fabricated normal or force clipping is added.

Only resolved rows with a defined legal normal and `gap<width` occupy force
slots. Inactive resolved rows remain in the complete query. Any unresolved
near candidate with a non-None law rejects the attempt before material/rigid
publication; CONDITIONING and DEGENERATE never become a force normal.
The two independent capacities are different: boundary candidate capacity
covers its complete query, while `contact_capacity` covers active force sites.

The original ordered node integration, two-stage alpha arithmetic, complete
distance sweeps and reconstruction receive actual uploaded total force.
The original resident mechanics receive actual uploaded COM wrench and project
it through current full Jacobians and mass response. There is no copied
physical integrator and no replacement of either solver.

## Device source and common publication

The D1 buffer owner borrows the resident scene's actual Runtime through
`ResidentDeformableRuntimeLease`. It never invents a matching runtime ID.
Every query holds both source loans, retaining original owner/topology,
world order, epoch, active bank and actual device ticks. Node ticks and rigid
ticks must agree world by world.

Registration reuses D2's bounded material registration for body-local constant
tables only; before any allocation/query these tables receive explicit moving
metadata. The two accepted gathers remain:

1. Original `gather_deformable_points` reads actual node x buffers.
2. Original `gather_boundary_points` reads actual floating meta/FK/ticks.
3. Original `translate_boundary_frame` puts the rigid native anchor in the
   same actual world frame as the node bank.
4. Existing bounds, BVH, pair packing and finite triangle distance run unchanged.

Separate node/rigid shape-selector metadata prevents one gather overwriting
the other. HOST readback verifies actual source arrays/material reconstruction;
it does not generate a current vertex/FK input.

A step holds the common owner and both physical locks. Entry query finishes its
loan/cleanup first; its private publication is only attempt evidence. The
resident sampler verifies the same entry epochs/ticks, computes contact loads,
and calls `nodes._stage_step` without publishing. Rigid candidates then use the
existing resident service. Every world must accept both domains. Candidate
mechanics, material impulse, force/work and immutable response/attempt records
are completed before the shared commit. The parent service publishes the two
banks through direct field assignments with no failing callback after the swap.
The public owner then assigns its already-prepared receipt, revision and epoch
tuple. No freeze/copy/allocation is intentionally left after physical commit.

Late failures preserve the previously accepted common state and receipt.
Attempt evidence keeps the complete query, actual node candidate and rigid
publication prefix when available. Failed query geometry is not advertised as
a newly accepted physical response. Unknown native cleanup retains dependent
owners and the actual Runtime lease; it is never blindly retried.

## Material reaction, work and energy interpretation

D3 enables the already accepted traced form of the original ordered distance
kernel. For each executed row and every executed sweep, its real
`direction[14:17]` and `delta_lambda[17]` produce the signed material impulse
`delta_lambda * direction / h` on the two endpoints. Fixed-point reaction
impulse cancels this accumulated material impulse plus the actual nodal
external/contact impulse. It does not substitute final-gradient times final
multiplier, or infer force from final displacement alone.

Fixed-node gravitational mass is not invented: the original D1 zero inverse
mass convention is retained. The reaction records the actual finite external,
contact and ordered-material loads in that model.

Receipts include both uploaded forces, source-addressed sites, complete node
receipt, original rigid entry/candidate/J/M/bias/projected force/diagnostics,
and fixed-point/material impulses. Work is evaluated at advected entry material
points on the actual candidate motion. Layer potentials use those same fixed
entry material addresses, not an uncomputed endpoint closest feature.
The recorded damping power and finite-step work residual are observations,
never an energy credit inserted into either solver. Endpoint finite geometry,
clock/tick and mechanics are audited; no continuous nonpenetration/CCD claim
is made. New contact discovery occurs at the next discrete entry.

## Complete native storage table

Let W be logical worlds, N all node counts, B=W physical bodies, C configured
force capacity, and P=max(1,C). New response storage is:

| Buffer | dtype/shape | bytes |
| --- | --- | ---: |
| original normal-law rows | f32[P,13] | 52P |
| law output | f32[P,5] | 20P |
| world/three-node/body identities | i32[P,5] | 20P |
| original barycentrics | f32[P,3] | 12P |
| actual rigid COM lever | f32[P,3] | 12P |
| node contact force | f32[N,3] | 12N |
| rigid COM wrench | f32[B,6] | 24B |
| world validity | i32[W] | 4W |
| node-world offsets | i32[W+1] | 4(W+1) |

Total: **116P + 12N + 24B + 4W + 4(W+1)**.
The padding for C=0 is physical allocation, not a reported active site.

The moving query adds `rigid_meta` i32[S,6] and anchor intervals f32[S,9]:
D2 persistent 32S+40V becomes **92S+40V**.
Its two f32[S,12] pose tables add 96S:
D2 prepass 36V+16S+4W becomes **36V+112S+4W**.
Original per-group BVH and **1060 bytes per narrow candidate** remain fully
charged by the generic plan. The query gets the native budget remaining after
the persistent response inventory, so its whole peak plus response storage is
bounded by `memory_limit_bytes`; capacity equality is allowed, overflow is not
silently truncated.

D1 separately retains its original **88N+32p+40+72p trace** device formula per
world and complete HOST reservation. Its per-sweep trace evidence and old/new
receipts are included in that reservation. Resident floating storage remains
the original bounded service allocation; it is separately reported, not hidden
inside a query cap. Borrowed source arrays have their explicit byte counts but
are not double-charged as new owned arrays.

HOST source readbacks, law rows/outputs, current and prior receipts, complete
attempts and trace copies can coexist. The limits are not an RSS/Python-object
hard cap or a bound on arbitrary historical capsules retained by callers.
This first slice openly downloads validated contact results/forces and uploads
the original physical load channels; it is not a zero-HOST-contact loop.

## Validation handoff, not results

The independent cases author freezes P/E/C pressure, eccentric fixed-node and
moving capsule cases; base/fine/mass variants, empty/no-force paths, sparse
world order, capacities, unresolved rows, restore/reset and late failures.
Its independent geometry, force and FP64 physical reference are not imported
by product code. The parent alone will run these new tests and the installed
consumer. Existing D1/D2/resident accepted suites are reused according to the
parent's affected-seam plan; this author has not rerun them.

D3 passing would cover this declared normal response slice only. D4 continuous
support/friction/facet changes/unload and D5 broader scene/asset composition
remain separate. There is no new Newton performance or learning-success claim.

## Parent source and installed qualification — 2026-11-13

The preceding authoring section remains historical. Parent product 3534f742
passed the complete new D3 files: interpret 27 (one CUDA-only deselected),
CUDA 28, plus the affected original staging seams. The installed consumer
passed both backends with 8 common steps / 13 worldsteps / 8 queries each,
including restore, partial reset and real successors. All 528 wheel products
match source and installed bytes. See [actual parent results](p65-three-track-batch-c-20261112.md).

The first sparse-world failure exposed a real construction ordering defect:
both owners now canonicalize by stable world ID. A later cleanup-test message
regex was corrected without changing retained-owner or release-count assertions.
Original failures, law, cases, budgets and raw output remain preserved.
D4 continuous support/friction/history and D5 broader scene composition are
still separate work; short-window entry/exit is not their acceptance.
