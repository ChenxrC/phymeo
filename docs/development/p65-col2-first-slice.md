# COL2 first slice: finite segment--triangle distance on TYX

Status: RESEARCH_PROPOSAL_ONLY / UNEXECUTED, 2026-11-12. Parent approval is required before implementation. This document authorizes no project execution. It is independent of the stopped B-point/C-point representation experiment; do not merge that experiment to obtain this slice.

## 1. The missing stage, and what already exists

The existing mesh path already has genuine TYX BVH refit/traversal. `_tyx_collision_bvh.py:10-66` returns candidate bitsets/original facet indices; it does not compute a triangle distance or material witness. `collision_pipeline.py:405-466` and `_mesh_round_candidates.py` preserve the mesh binding and select all relevant nearest/radius faces. Existing static, translated and rotating candidate qualifications remain reusable; no unchanged suite is rerun merely to create new numbers.

The next stage is still HOST: `MeshRoundQuery` in `_mesh_round.py:74-98` calls the Fraction-based `segment_minimum` in `_capsule_triangle_sweep.py:64-109`. That function enumerates finite endpoint/face, segment/edge and face-crossing candidates. `_capsule_triangle_sweep.py:110-158` retains the closest-axis interval and witnesses. Closed-mesh inside/contains logic is separate in `_mesh_round.py:186-255`. HOST mesh/deforming response is therefore not absent; the gap is this device finite-feature stage and its eventual source/consumer integration.

Fixed official Newton 1.7.0 source confirms a similar stage boundary, not the same algorithm: `newton/_src/geometry/narrow_phase.py:1554-1579` emits mesh/other-shape/triangle tuples; `1673-1794` gathers actual triangles and applies device GJK/MPR plus triangle-prism refinement; the later reducer around `3185` handles generated contacts. The inspected primary source is under the parent's `.venv-newton-1.7.0/Lib/site-packages/`. This proposal uses the simpler finite-segment special input shared by sphere and capsule, not a claim to port Newton's complete convex-mesh/SDF/hydroelastic pipeline.

## 2. Smallest independent deliverable

Implement one bounded **current-pose finite segment--triangle distance packet**. A sphere is a zero-length segment; a capsule has a finite axis and radius. The same kernel handles both without shape tags, pair handlers or GJK/EPA. It consumes explicit `(segment_id, original_facet_id)` candidates and immutable input snapshots. A real existing BVH producer supplies the candidate list in an integration test; the new packet binds the supplied list but does not independently certify its completeness.

First slice is a standalone backend API. Existing `CollisionWorkspace` HOST queries, contact patches, force consumers, resident plans and CCD remain unchanged. A topology-preserving deformation can supply a fresh vertex snapshot and run the same BVH refit plus packet; this is not borrowed device-particle state or a deformable response implementation. No dependency on B-point, a repaired curved normal, training or robot models.

Proposed import/API, to freeze before tests-first coding:

```python
from phymeo.backends._device_triangle_distance import DeviceTriangleDistanceBatch

batch = DeviceTriangleDistanceBatch(
    mode="interpret", vertex_capacity=..., face_capacity=...,
    segment_capacity=..., pair_capacity=..., cache_root=..., runtime=None,
)
packet = batch.query(
    mesh_key=..., revision=..., vertex_ids=..., facet_ids=..., faces=...,
    vertices=..., segment_ids=..., segments=..., radii=...,
    candidates=..., origin_m=..., snapshot_id=...,
)
```

`faces` contains vertex-row indices; separately retained stable vertex IDs and original facet IDs are never replaced by compact candidate ordinals. `segments` is `(S,2,3)`; `candidates` is `(P,2)` with unique explicit segment/facet pairs. Per-row world provenance is input metadata, not an invented resident-owner claim. No implicit complete-mesh search in the adapter. Validate all shapes, finite values, IDs, radii, capacities and binding before allocation/upload; copy immutable accepted inputs. Bind every returned row to the complete accepted snapshot and candidate membership. On a late readback/packet-construction error keep the previous packet, release temporaries, and allow a fresh real retry.

The output is an **engineering FP32 boundary-distance result**, not a certified interval, penetration depth, complete contact manifold or solid-inside classification. Per candidate row retain IDs, status/reason, distance and `distance-radius`, axis parameter, triangle barycentrics, relative witnesses, direction when defined, finite-feature selection, all valid tied candidate identities and actual resource counters. A zero boundary distance can mean a segment intersects a face; it has no unique normal and must not invent one from winding. A point inside a closed solid can have positive boundary distance; it must never be declared noncolliding by this packet.

## 3. Bounded generic algorithm and representation

For each supplied pair use all finite candidates, at most 18 before ties:

1. Project both segment endpoints onto the triangle plane; retain projections inside the closed triangle.
2. Intersect the finite axis with the triangle plane; retain a crossing inside both the segment and closed triangle.
3. For each of three triangle edges evaluate the four endpoint-to-opposite-segment clamped candidates and the interior stationary segment--segment candidate when its denominator is valid.

This includes the triangle vertices through finite edges, and avoids infinite-axis or infinite-plane false positives. Parallel segment/edge intervals are represented by their boundary candidates; retain ties and their distinct material addresses, including a shared mesh edge reported by two original facets. Do not remove points merely because world coordinates coincide. An approximate tie mask/axis span is not a proof of a unique exact continuum; preserve raw candidate values and mark any unresolved conditioning explicitly.

Triangle zero area is a rejected/unsupported geometry row, not silently an edge. Exact parallel branches use the finite boundary candidates. Near-parallel/ill-conditioned interior calculations require a declared conditioning failure if the complete minimum cannot be trusted; no epsilon denominator, arbitrary normal or HOST fallback. Freeze the arithmetic/conditioning policy with the first test commit, before seeing output. Bound work by the above enumeration, not an iterative convergence search.

All geometry arithmetic uses a shared local origin: subtract the supplied origin in the input representation before FP32 conversion, retain the original values/binding and conversion deltas, then compute edges, candidates and witness differences in that frame. Never add a small offset to a large absolute FP32 center and then subtract the origin. A large common translation must not enlarge the local error budget. Preserve actual uploaded/readback bytes and ownership of original face barycentrics. Reporting world points may add the origin at the HOST representation boundary; that is not a HOST geometry repair.

The snapshot constructor must reject unrepresentable local geometry under the frozen conversion policy rather than claim exact source provenance. A future resident/device-vertex bridge must bind its own device-produced frame; this standalone HOST-input snapshot does not implement that bridge.

## 4. Exact proposed ownership and file allowance

| Owner | Files | Change |
|---|---|---|
| COL author | NEW `src/phymeo/backends/_tyx_triangle_distance.py` | TYX finite enumeration/reduction; no existing GJK/patch edits |
| COL author | NEW `src/phymeo/backends/_device_triangle_distance.py` | Explicit runtime owner, bounded buffers, immutable packet/input binding, counters and atomic publication |
| COL author | NEW `tests/reference_newton/rr/device_triangle_distance_reference.py` | Fixed analytic/rational inputs and independent finite-feature oracle; no product algorithm import |
| COL author | NEW `tests/collision/test_device_triangle_distance.py` | Geometry/covariance/deformation packet cases and actual existing-BVH-to-packet test |
| COL author | NEW `tests/collision/test_device_triangle_distance_protocol.py` | Capacity, source/ID/membership, transfer and late-failure tests |
| COL author | NEW `docs/development/p65-device-triangle-distance-report.md` | Exact source, predeclared node map, results and remaining limits |
| Parent | No old seam edit in this slice | Review exact API and run serial qualification; later explicit public/resident integration is a separate decision |

Use established TYX Runtime/array ownership patterns, but do not modify `TyxCollisionBatch`, `CollisionWorkspace`, their callbacks or accepted kernels just to attach this first packet. No new package export is required. Keep a caller-supplied runtime borrowed and close only owned allocations; report actual allocations, kernel completions, transfers and bytes. This proposed six-file scope is sufficient; no broad collision refactor.

## 5. Fixed multi-case qualification before development

Predeclare exact literals and node count in the first test commit. Proposed compact matrix is 24 TYX geometry nodes per backend, plus a bounded protocol set (HOST preflight and real TYX late-readback cases); parameters must not be selected from product output.

| Group | Nodes | Required distinctions |
|---|---:|---|
| D01 endpoint/sphere | 6 | Face above/below, finite edge, vertex, beyond-edge miss, zero-axis equivalence; independent distance/bary oracles |
| D02 finite capsule | 6 | Parallel face interval, oblique endpoint, edge--edge, infinite-axis false positive, through-face normal-undefined, parallel edge with two tied extrema |
| D03 ownership/covariance | 6 | Rotation, large common translation, reversed winding/stable-ID storage order, two facets sharing an edge, noncoplanar crease, fresh nonrigid vertex snapshot with unchanged topology |
| D04 real pipeline/domain | 6 | Existing TYX BVH with genuine pruning and complete retained nearest ties; concave multi-face boundary packet; near-parallel well-conditioned case; declared ill-conditioned case; zero-area triangle; two segment IDs sharing coordinates without merging ownership |

Use dyadic and rational analytic geometry for independent expectations; an independent bounded Fraction reference may run only in the parent's serial test slot. It may not call production `segment_minimum` or the new kernel. For ordinary nonzero-distance positive controls, retain existing engineering scales: distance error `<=1e-4*L`, witness reconstruction `<=2e-4*L`, normal error `<=2e-3`, where `L` depends on local edge/axis/radius lengths, never world translation. These proposed gates require parent freeze; no existing threshold changes. Assert finite-axis membership, barycentric nonnegativity/sum, original-ID ownership, valid-feature/tie completeness and both witnesses, not distance alone. Intersections and conditioning failures have explicit status oracles, not arbitrary UNKNOWN acceptance.

Protocol checks: exact capacity succeeds and one-less rejects; malformed/duplicate candidate and vertex IDs reject; changed snapshot/topology cannot reuse an old packet as current; altered readback IDs/weights and late download failure publish nothing; retry performs real work. Verify both sides of the runtime ownership contract and complete saved input/raw/packet fields. Include one same-candidate-list HOST versus actual TYX geometry comparison and verify the device launch/counter path, with no HOST distance search during the query. Do not rerun old accepted mesh candidate matrices.

## 6. Acceptance, stop rule and next layer

Target a bounded module that can be authored and independently reviewed before 06:20, if the parent assigns it now; elapsed time is not evidence of completion. Parent serial order: first static checks, focused analytic geometry, complete new interpret set, complete CUDA set, then only directly affected source/protocol checks. Preserve the first failure and actual emitted packet. Stop after one evidence-based implementation correction; a missing generic case or conditioning failure remains an explicit gap. Do not chase green by increasing bounds, weakening feature checks, shrinking fixtures or silently substituting HOST output.

Success means the new explicit packet actually computes finite triangle distances on both backends with its stated cases/bindings/capacities. It does **not** complete COL2, mesh collision response, deformable contact, CCD, inside/outside classification or Newton parity. The next independently planned layer is device-vertex source ownership plus complete co-nearest reduction and closed-solid/shell semantics; only then connect qualified material sites to existing full-J/full-M loads. General convex--triangle, triangle--triangle deformation, persistent friction identities, SDF and hydroelastic remain separate gaps.

Research and this document used only text/file IO and Git. No implementation, imports, numerical probes, collection, tests, static tools or build were run.
