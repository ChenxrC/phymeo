2026-11-09 父最终状态：默认 K0 查询已限域验收，interpret/CUDA 各142、284完整记录、新wheel与440产品成员同源；见 [正式报告](p65-device-convex-query-acceptance.md)。下文保留作者历史检查点。可选 K1 patch 未验收。

# K0 device convex query implementation

**AUTHORING_ONLY / UNEXECUTED.** This report describes the candidate source, not
an accepted device result. No import, static checker, test, numerical probe,
benchmark or native execution was run by this author. The parent owns execution.
Parent execution after this checkpoint is recorded in plan §12: 142 identities
per backend have a pytest pass record, including the 10 touch identities. The
142 gzip files were not retained, and K0 is not accepted. This file stays the
author checkpoint and is not that record.

The frozen contract is `docs/superpowers/plans/2026-11-08-device-convex-query.md`.
The initial public RED is parent job 588: the independent SS separation and old
HOST finite patch passed; the call failed at the absent public query method.
That RED does not establish device query behavior.

## Source and scope

Four new files are owned by this slice:

- `backends/_device_convex_query.py`: immutable public records, bounded resources,
  original-source compilation, seven-array wire validation and atomic query publication.
- `backends/_tyx_convex_support.py`: single-expression TYX scalar dot/determinant/clamp helpers.
- `backends/_tyx_convex_query_kernels.py`: actual accepted FK gather, finite round
  projection, a shared core support scan/simplex query and bounded EPA.
- This implementation report.

The source bridge and old batch/broad/export hooks remain parent-owned. The
candidate supports only the frozen ordinary fixed-root tree/static source loan;
affine/floating source adapters, CCD, force response and complete contact patches
are not included. General hulls use their original vertices, never an AABB proxy.
A one-time HOST `HullShape` check enforces the existing full-dimensional material
registration even for filtered or distant shapes. No HOST geometric search fills
a device query result.

## Device path and material ownership

Dynamic points use the actual borrowed FK origin and rotation on the existing
runtime. Static FP64 world points are rebased against that world's actual stored
anchor before upload. The source packet retains original and uploaded coordinates,
radii, quantization differences, actual device pose/stamp, original bindings and
all registered shapes. Actual skin radii are distinct from padded broad bounds.

SS/SC/CC use closest points on finite segments, including a zero-axis capsule
with half/half endpoint ownership. SB/SH/CB/CH share core GJK then add the true
world-axis skin offsets. BB/BH/HH share GJK/EPA. Simplex faces are enumerated up
to four members. EPA starts from genuine support points, creates a containing
tetrahedron, extends only visible faces, and reconstructs an actual triangle's
material foot on convergence. Support ties use original integer IDs.

The scratch tile contains at most 32 pairs; the final seven wire arrays preserve
full capacity and deterministic padding. Iteration/vertex/face limits produce
non-success rows; unresolved normals, core containment, unsupported shapes and
source representation loss remain explicit. No unsuccessful row means clear.
The signed gap, surface closure and normal budgets are the plan's engineering
contract, not interval proofs or permission to weaken the old contact law.

## Transaction, resource and readback accounting

`ConvexQueryPlan._allocate(shape,dtype,initial=None)` registers every successful
allocation in its own ExitStack before returning. The complete allocation layout
is checked against the memory budget before any buffer allocation. Plan close
works even after its broad/batch has been marked closed; it never closes the
borrowed runtime. Empty models perform no device operations.

`_read_raw()` copies the seven real arrays before validation. The same borrowed
accepted tick buffer is independently read and saved as `source.accepted_ticks`,
with its original slot map; source stamps must match it. This copy is included in
actual readback accounting, including explicit-pair queries. A subnormal or a
nonzero original coordinate/radius rounded all the way to zero cannot silently
change the accepted source geometry.

`_build_result(view,raw,pose,stamp,pairs,attempt_delta,broadphase_delta)` validates
all raw identities, IDs, counts, finite values, padding and material reconstruction,
then owns the complete result. This is the late-record fault boundary. Only after
it returns does the default path publish the staged old broad result and the new
query result. Explicit pairs do not publish a broad result. Failures preserve the
last result and all physical state/history/epochs; retry uses a new private attempt
stamp, never a physical tick.

Cumulative plan counters separate actual CUDA transfer bytes from interpretation's
HOST-copy bytes. Each result stores the same-attempt delta and separate old broad
cost delta. Candidate discovery is still the existing HOST broad result followed
by a counted pair upload, labelled `host_broadphase`; explicit requests are labelled
`explicit`. Neither is described as a device-resident candidate handle. There is
no measured latency or speed claim.

## Pending parent validation

This section is the author checkpoint. Parent execution after it is plan §12:
142 identities per backend have a pytest pass record, including the 10 touch
identities. The 142 gzip files were not retained, and K0 is not accepted.

At this author stop, the frozen 142 identities/backend, including 20 mandatory
nonzero resolved family cases, were **UNEXECUTED for this candidate**. The
intended parent order was TYX compilation and the four initial SS/CH identities,
then the remaining finite geometry, transformed/ID cases, resource/error protocol
and W32 holdouts. K1 manifold weights and K2 actual load consumption require
later, separate contracts and are not implied by a valid K0 witness.

## Stopped authoring checkpoint

Authoring stopped at the parent's explicit stop instruction after the user's
long-running goal was cleared. This is **WIP / UNEXECUTED**, not CODE_READY.
Existing source is preserved; no incomplete source was removed or reset.

Known text-only follow-ups, not performed after the stop:

- The current `result.stats` includes the attempt counters directly, whereas the
  latest tests require an explicit `attempt_delta` mapping. This public schema
  mismatch remains unfixed in this checkpoint.
- The complete newly written TYX kernel has never been imported, compiled or
  interpreted. Its simplex/EPA convergence, branch lowering, scratch limits,
  finite-segment corner cases and all mandatory resolved cases remain unverified.
- EPA's normal-ambiguity classification currently inspects the constructed
  polytope's equal-depth faces; a text-review follow-up should distinguish an
  unexpanded equal-depth face from a genuinely equal supporting face.
- Parent's accepted-tick binding and nonzero-to-zero source-underflow fixes were
  authored, including the additional counted readback; neither was executed.
- Parent shared source/hooks are not merged into this isolated tree. Its HEAD
  must not be treated as a runnable integrated release.

No source-driven numerical result, measured speed, accepted capability, completed
matrix, native reference or wheel result is claimed by this checkpoint.
