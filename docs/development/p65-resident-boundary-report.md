# M1 resident rigid boundary query — authoring handoff

Status: **CODE_READY_TESTS_PENDING / UNEXECUTED**. Parent owns the only execution queue. This author used source reading, manual algebra, file IO and small Git commits only. No project import, collection, geometry probe, pytest, lint, formatter, compile, build or install ran here.

## Exact checkpoints and files

Base `48724ae1d8f2be527a62d4b86cf0a62a1e3d410a`, isolated branch `codex/p65-boundary-m1-20261112`:

1. `4397b61fd7fcb544e550e14531da7bd2ee6bf62f`: four NEW S0 files, fixtures/independent oracle/memory ABI before production.
2. `e563a51e`: one narrow protocol-test correction: construct the real empty plan before attaching synthetic native resource owners; direct plan.close is also rejected inside the existing readback reentry case. No changed geometry, matrix or gates.
3. `3a3dea979af8206e8d44d348e479a9884af48178`: five S1 product files. These bytes are frozen for parent review/validation. No runtime pass is attributed to this commit.
4. This report and the NEW installed consumer are the final authoring supplement, also unexecuted.

| Ownership | Path | Purpose |
| --- | --- | --- |
| NEW | `docs/development/p65-m1-buffer-layout.md` | S0 buffer ABI, cost derivation and node inventory |
| NEW | `tests/reference_newton/rr/resident_boundary_source_reference.py` | independent declared motion, finite candidates and Fraction geometry oracle |
| NEW | `tests/collision/test_resident_boundary_query.py` | four layouts, finite/mesh cases, real continuing source steps |
| NEW | `tests/collision/test_resident_boundary_query_protocol.py` | source/capacity/atomic/release failures and retry |
| NEW | `src/phymeo/backends/_resident_boundary_source.py` | full direct physical installation and accepted-bank loan validation |
| NEW | `src/phymeo/backends/_resident_boundary_query.py` | resource-owned orchestration, complete candidate prepass, immutable publication |
| NEW | `src/phymeo/backends/_tyx_boundary_source.py` | real FK gather, finite bounds, device pair-local packing |
| MOD | `src/phymeo/backends/_tyx_collision_bvh.py` | opt-in registered device-box entry, old HOST-packed path unchanged |
| MOD | `src/phymeo/backends/_device_triangle_distance.py` | extract the unchanged emitted-row validator for both input paths |
| NEW | `tests/installed/check_resident_boundary_query.py` | standalone installed public consumer, two sparse worlds, three real steps |
| NEW | `docs/development/p65-resident-boundary-report.md` | this handoff |

The parent alone owns `tyx_world_batch.py` lease/preclose/prepare hooks and public re-exports. Old four binding/storage implementations, accepted convex kernels, original COL2 distance kernel and old K1 WIP are untouched. No dependency or backend pin changes.

S1 SHA256: source `ab828aecc6d8af233b58f4f1c5d2a9f07918dc8c501f75aabdf198be7bd07d96`; query `21198e3155f0994b5ed9f05fa2c795c53e8677807abc4dae7c8a3e64c95abaff`; gather/pack kernel `07077e02bd1e31a9551f81e14a4ea3605b5e763c2206ee9190b4f47285238f07`; BVH seam `f917ec24488aa579127b6c2ca711c45783e8f496756936570c27e00f173386ca`; COL2 validator seam `03fbd83ec583850b5b45c5442df0892191c5951f019179dadeaef1226606625f`.

## Runtime/source contract

Constructor `ResidentBoundaryQueryPlan(batch, model, bindings, *, limits, memory_limit_bytes)` is the target for the parent's public `batch.prepare_boundary_query`. Public methods are `query(*, margin=0.0)`, `require_current(result)`, `stats()`, `close()` and properties `last_result`/`last_attempt`.

Only the four direct constructors with complete physical topology are accepted. Floating-affine also binds the owning graph token. `topology=None`/from_asset is explicitly outside this first slice. Source admission checks actual world/layout/physical row/group and body installation, not merely caller CollisionKey labels. Both F0 round shapes belong to follower body202, sphere shape0/capsule shape1. Mesh requires the original four-or-more faces; original CollisionModel triangle/round mixed same-body rejection is preserved. Eligible same-world, different installed-body round/surface pairs retain original masks/exclusions and full facet tables.

Dynamic geometry comes from the actual borrowed FK/meta/ticks arrays. Registration uploads authored constants once. Each query gathers on device, makes conservative finite bounds on device, refits/traverses the existing BVH and packs all discovered candidates on device. HOST reads complete evidence and validates already emitted data; it does not make current FK/bounds/closest-point results or upload repaired geometry. This is a device source path with explicit transfers, not a zero-readback claim.

Source records preserve all 20 frozen outer keys. `stamps` includes exact read rows/poses/accepted_ticks/point_bounds and `material_reconstruction_delta`. The last field compares emitted points with the same actual stored pose and authored local constants; it is only a readback validation, not an ideal FP64 FK oracle or an input to a device kernel. Pair-local material error uses the original `2e-4*L` witness gate. L uses triangle edges, finite axis and radius, never absolute world position or inter-object distance.

Every pair's frame is the actual gathered triangle first point; the original world anchor is retained separately. Packed geometry, frames, owner rows, generation/status, radii and scales are all read and bound. Surface and axis IDs/barycentric addresses remain original. New packet schema is `phymeo-resident-triangle-distance-v1`; new aggregate is `phymeo-resident-boundary-distances-v1`, explicitly `threshold-limited-boundary-distance` / `resident-rigid-device-v1`. No candidate is renamed a contact or a solid inside/outside result.

## Resource and transaction accounting

Use S shapes, N original material points, each group F original facets/R eligible rounds, T=2F-1, K=ceil(F/30), P_g full discovered pairs. The exact owned-array formulas (both execution modes) are:

```
B_persistent = 32*S + 36*N
B_global = 64*S + 36*N
B_group = 40*F + 48*T + (52+4*K)*R
B_prepass = B_persistent + B_global + sum(B_group)
B_peak = B_prepass + 1060*sum(P_g)
```

The 1060P is 104P pair pack + 956P original 18-candidate raw outputs. Source/bounds/BVH arrays remain live until every group completes. Constructor checks persistent bytes before allocations; query checks the prepass before source launches, computes **all** group candidates, then checks global/per-group capacities and peak bytes before **any** narrow allocation or launch. Empty has zero arrays, launches and runtime leases and requires positive public budget1; budget0 remains invalid. Actual allocations, uploads/downloads, bytes, launch labels, completed launches and syncs stay monotonic on failures. HOST constants/readback/packet storage are recorded separately and are not process/GPU peak memory.

Persistent/query ledgers each own a no-op parent-runtime lease, never the physical runtime itself. Normal free decrements live bytes only after the native operation returns. UNKNOWN_RELEASE retains the ledger/runtime lease/batch strongly, quarantines the parent and never retries opaque frees; sibling cleanup remains independent. PENDING from preclose is distinct and can retry close. The dedicated parent source scope prevents same-thread callback step/restore/prepare/close reentry. plan.close does not use parent public `_check_open` during parent preclose.

All raw downloads enter `last_attempt.raw_prefix` before structural/oracle checks. Complete immutable packet, counters and attempt are prepared before swapping last_result/sequence/layouts. Any gather/BVH/pack/raw/source/publication/cleanup failure preserves the previous accepted result and all already captured raw prefixes; failed generation/counters are not rolled back. `require_current` accepts only the exact latest committed result bound to this owner, registration, epoch and accepted buffer. This is a query transaction; no physical state is edited.

## Existing algorithms reused, exact changed seams

`TreeBuffers.hits_device` is opt-in. It verifies source lease and buffer ownership, uploads only integer layout/params, refits/traverses from device bounds, preserves the full **K+3** raw return (bit words, node visits, leaf tests, escape). Existing `hits` still returns K+2, as before. No old traversal or geometry kernel was edited.

The old COL2 `_build` validation body was moved into `validate_triangle_distance_rows`. Before the S1 commit, a pure string comparison verified the whole original body equals the new body after exactly a four-space dedent. Its candidate validity/ties/selection/material/normal checks and numerical gates are unchanged; original `_build` calls it and constructs the same old packet. This is text identity evidence, not a runtime test.

## Parent execution inventory, not collected here

S0 manual enumeration remains **63 shared-mode nodes +4 HOST-only release nodes +1 CUDA-only release node**:

- main27: four layouts × static/double-moving 8; mixed-layout1; finite features6; mesh3; permutation/large-anchor2; genuine rotation3; HOST-search denial1; conditioning/zero-area/through3.
- protocol36: source faults10; reset/restore1; capacity6; late faults4; empty/filtered/no-candidate1; reentry5; installation filter1; registration5; latest-owner/full-source1; allocation-prefix2.
- HOST4: real plan/ledger/parent hooks with fake native before/after free × query/batch-close, including weak ownership after the exercise exits.
- CUDA1: after actual successful native free inject an error, retain unknown ownership and verify no repeated free or parent runtime close.

Suggested first order: four HOST lifecycle nodes; one ordinary double-moving node; other new interpret nodes; complete new CUDA nodes including its finite release seam. Parent may adjust ordering while preserving the full fixed new matrix. Parent shared hook qualification (13 HOST) is separate evidence, not an M1 geometry pass.

Minimal affected old COL2/COL3 checks, each backend, for these exact two seams:

1. `test_device_triangle_distance.py::test_fixed_finite_geometry` identities `face_above`, `parallel_face`, `through_face`, `near_parallel_conditioning`, `zero_area`, `large_translation` (resolved/ties/undefined/conditioning/degenerate/frame outputs through the extracted validator).
2. `test_device_triangle_distance_protocol.py::test_late_failure_preserves_packet_and_real_retry` identities `ownership`, `barycentric`; `test_empty_candidate_packet_has_no_launch` (emitted-row rejection/empty compatibility).
3. `test_triangle_distance_dispatch.py::test_mesh_original_facets_and_boundary_semantics[shared_pruned]` (old HOST-packed device BVH route retains K+2 and pruning).

These ten logical identities/backend are the bounded changed-seam regression proposal, not a request to rerun old full matrices or resource suites. Mode-param full node names must follow the parent's installed pytest selection convention; no collection has occurred here.

## Installed consumer and remaining limits

`check_resident_boundary_query.py --mode interpret|cuda --installed-root ABS --output freshABS` embeds the original F0 URDF literals and uses only installed public constructors. Worlds7/4096 reuse original local IDs, full physical two-slider topology, nonzero opposite velocities and zero external gravity. Both triangle and sphere/capsule move for three real steps. Expected finite distance and material witnesses follow directly from the two accepted scalar coordinates; independent free-motion q/v recurrence is also checked. Per backend this is six worldsteps, three aggregates and twelve rows, with two stale-packet rejections after actual accepted steps.

The consumer writes every returned raw/source/counter/physical checkpoint **before** assertions, plus failure prefixes and independent owner-close outcomes. It checks all loaded Phymeo modules belong to the given installed directory, rejects checkout/source paths and requires consumer/cwd/output outside source. It does not import repository tests/examples or use HOST closest geometry. This consumer is authored, **not executed**; its counts are predeclared, not results.

No M2 deformable producer bank, CCD, contact/load response, closed-solid classification, curve patch or MuJoCo-TYX runtime interoperability is added. No guarantee of complete Newton parity or performance is made. Parent must perform the sole serial static/functional/installed qualification before any accepted-source claim.
## Parent integration amendment during author handoff

After importing the frozen author checkpoint, parent review identified a PENDING-cleanup ledger overwrite risk. Parent `2d9dddda` added the requirement that an earlier query ledger be CLOSED before replacement, and parent close-only handling when cleanup has not completed; two query-entry variants were added as new negative nodes. These changes belong to the parent shared validation source, not silently to author commit3a3dea97. Parent reported the four HOST release tests passed and queued the new full interpret/CUDA source qualification; at this report's authoring cut, those full results are not yet known. The author's original inventory remains63 shared+4HOST+1CUDA; the parent matrix adds two shared nodes (65 shared, plus4HOST/1CUDA). The installed consumer uses only public current APIs and does not depend on private ledger layout.
## Parent qualification — 2026-11-12 13:14

The parent completed the full declared finite source matrix: **4 HOST release nodes, 67 distinct interpret nodes, 68 distinct CUDA nodes (including one actual CUDA free-then-error quarantine)**. The original63 shared cases gained two pending-cleanup entries and two non-FP32-anchor floating families; no original case, gate, mass, capacity or unsupported-domain boundary was removed. The source matrix was completed by retained successful prefixes plus targeted unfinished/changed nodes, not by rerunning already-green full files.

Actual evidence in artifacts/p65-parity-m1-assembly-20261112-01: 032 HOST4/2.61s; 040 interpret23 passed before a counter assertion; 043 another41 passed before an allocation-prefix assertion; 046 four affected/pending nodes passed (includes the one tightened already-visited prepass check); 048 two anchor nodes rechecked after stable affine oracle reconstruction. CUDA034 passed27 before the new anchor oracle assertion; 049 remaining41 passed/61.62s. Thus the distinct accepted matrix is139 nodes, not the sum of every rerun/prefix. Run047 targeted old COL2/COL3 compatibility passed20/3.05s; the deliberate148 unselected old nodes were not run. No backend skipped or substituted.

Independent text review found and parent fixed three implementation issues: (1) pending query cleanup could lose its original ledger on reuse, now pending blocks query/physical use and keeps close-only ownership; (2) floating anchors were incorrectly rounded to FP32, now original physical FP64 anchors are retained for common static/dynamic frames; (3) persistent HOST registration/cache ndarray accounting omitted arrays, now identity-deduplicated registration and cached BVH topology are counted before publication. Device memory formula and original kernel/geometry gates are unchanged. Exact new physical anchor and cleanup regression cases cover these changes.

Original failures remain: run033 NEW independent oracle treated a tuple of vertex rows as multidimensional NumPy indices; run040 NEW test incorrectly required CUDA download counters in interpret; run043 NEW allocation fault fired after an8-byte selected-pair table but expected prepass-only peak. Each was corrected to the already-declared semantics. CUDA034 NEW witness oracle used bary@world_triangle, magnifying allowed FP32 barycentric sum error by a million-unit translation. Parent and independent reviewer derived the discrepancy (sum(b)-1)*triangle0 and changed only the NEW oracle to triangle0+bary@(triangle-triangle0), using original material vertices, retaining8u/2e-4L and no product frame or normalization. This is not a physical tolerance change. The first034 wrapper lookup before job JSON existed failed before guard/project execution; later actual034 has its own complete receipt.

Static first failures026/027 and036 remain; final030/031,038/039 and subsequent changed-file checks passed. Installed consumer now preserves a query's original exception even if post-quarantine snapshot capture fails, independently saving remaining raw diagnostics and collecting capture errors. Actual combined wheel/installed qualification is still pending at this checkpoint. No M2, CCD, force response, solid interior or Newton performance claim follows from M1.
