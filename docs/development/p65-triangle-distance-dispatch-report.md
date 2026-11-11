# COL3 boundary distance dispatch

Status: CODE_READY_TESTS_PENDING / UNEXECUTED / AUTHORING_SLOT_RELEASED.

This authoring slice connects original CollisionModel registrations and complete HOST motion snapshots to the accepted COL2 finite segment–triangle query. The parent owns the four CollisionWorkspace forwarding methods and sibling close integration. No old product, BVH, COL2 kernel, physics consumer or geometry threshold was changed in this author branch.

## Public and helper contract

`CollisionWorkspace.boundary_distances(motion, *, margin=0.0, limits=None)` delegates to `BoundaryDistanceDispatcher(workspace).query(...)`. `last_boundary_distances` reads `dispatcher.last`; `require_boundary_distances(packet, motion, margin=...)` delegates to `require_current`. `close()` closes the independently owned COL2 batch; it never borrows the old broad-phase runtime. The parent must attempt its independent old engine close even if this owner reports an unknown release.

The new module exports:

- `BoundaryDistanceLimits(groups=32, candidates=8192, vertices=512, faces=128, segments=64)`: frozen builtin integers in `[0, hard_ceiling]`. Zero deliberately supports exact/one-less capacity tests and an empty permitted budget. Booleans, expansion and untyped mappings are rejected.
- `BoundaryDistanceCapacityError`: fields `resource`, actual `required`, and supplied `limit`. No partial publication or truncation.
- `BoundaryDistances`: schema, semantics, owner, sequence, backend, full source hash, margin, limits, immutable complete source, complete pair records, all surface groups and total candidate count.
- `BoundaryDistancePair`: original canonical keys, filter/not-applicable/no-boundary-candidate/boundary-candidates status, reason, group/segment map, original facet ordinals and packet row indices.
- `BoundaryDistanceGroup`: original surface key; round keys and their original material vertex IDs; each original facet's material vertex triple; full candidate membership, outward leaf/query boxes, the actual same-call BVH bitset/counters, full COL2 packet and same-call narrow counters.

The aggregate's semantics are explicitly `threshold-limited-boundary-distance`. The pair `no_boundary_candidate` result is not a solid-separation assertion. Boundary rows retain `RESOLVED`, `CONDITIONING`, `DEGENERATE` and undefined directions unchanged. The candidate set can include rows outside the Euclidean band due to the conservative box test; it is not a reduced contact manifold.

Current input domain is sphere/capsule against registered zero-skin triangle/mesh facets, interpret/CUDA only. Same-category pairs remain explicit `not_applicable/same_family`. Existing world/body/mask/exclusion codes are retained in full model pair order. Unsupported registrations, nonconstant end paths, turns and independent orientation fields reject before upload. Constant end coordinates are admitted but their presence and bytes remain bound. Every registered shape/motion, original vertex order, revision, radius, filter/exclusion and pair table is hashed. This is a HOST snapshot binding, not a resident accepted-bank epoch.

## Candidate and transaction mechanics

For every eligible surface registration, one group contains all unfiltered round registrations. The original face table supplies dense facet ordinals with an explicit material-triple map; local segment ordinals have explicit CollisionKey and vertex-ID maps. Sparse world IDs and repeated local IDs never alias across groups.

Leaf boxes use original triangle coordinate extrema. A round query box encloses its entire finite axis and is expanded by the exact stored radius plus margin, then converted outward to FP32 using the existing bound converter. If Euclidean axis-to-facet distance is at most radius plus margin, the coordinate boxes must overlap. This is the completeness argument for the declared band; there is no HOST nearest-point search or inside test. The actual accepted BVH traversal produces the only candidate membership, and its original bitset is retained without another launch. Invalid facet bits reject.

Group/vertex/face/segment counts are complete before any device work. All groups finish actual BVH selection and the total candidate budget is checked before any narrow query. Per-group COL2 inputs retain the surface's original vertices and IDs, all registered round axes/radii, complete candidates and a frame derived only from the original first surface vertex. Each private packet is validated immediately with `require_current`, then owned by the aggregate; a later group's private publication does not erase earlier immutable packet evidence.

The dispatcher joins an existing workspace draft instead of opening a second nested draft that would hide staged publications. If there is no draft, it opens one itself. All work uses a copied pending-layout dictionary. Only a fully constructed aggregate assigns the pending layouts and stages `_publish`. The next sequence comes from this owner's already staged result in that draft, otherwise its last committed result. Thus two queries in one outer transaction have consecutive sequences, but only its final result publishes. Outer failure leaves `last` and original layouts unchanged. `require_current` intentionally rejects uncommitted returned packets, as well as wrong owners, stale results and changed complete inputs.

Attempted BVH evidence and completed earlier private packets remain available through the helper's immutable `last_attempt`. A later read failure adds actual partial COL2 readback and resource counters. Those diagnostics do not roll back real launches and do not pretend the failed aggregate was published.

COL2 retains its accepted DeviceResources ledger. Unknown query-buffer release leaves conservative live bytes and blocks further helper queries; unknown close also makes the helper sticky, including empty-group queries. Batch close retains the unknown owner and does not blindly retry free. This slice does not redesign the older broad-phase engine's resource implementation.

## Predeclared validation

Tests-first commits: `c9ccc6c4` (public matrix), `8ab7d18d` (explicit unresolved propagation and engineering permutation gate). No import or collection has occurred.

Manual node enumeration: **33 per backend + 1 HOST-only = 67 nodes**, pending actual collection.

| Family | Per backend | Independent behavior |
| --- | ---: | --- |
| Sphere/capsule finite face/front/back/edge/vertex/outside/through | 12 | Exact plane/edge/vertex distances, original distance `1e-4 L`, witness `2e-4 L`, normal `2e-3`; undefined through direction |
| Shared/pruned, closed inside, concave mesh | 3 | Original shared facets retained; distant facets actually absent; concave walls; positive inside boundary distance and empty small band do not classify a solid |
| Sparse worlds, vertex/shape storage and changed snapshot | 1 | Repeated IDs remain scoped; complete changed-coordinate binding; engineered storage covariance |
| Existing filters and same-category pairs | 1 | Full pair table retains excluded, same rigid body, mask, same-family |
| Exact/one-less groups/candidates/V/F/S | 5 | Actual required counts and unchanged old public packet |
| Final-group late read and retry | 1 | Earlier narrow query can finish, but whole aggregate remains old; same input retry succeeds |
| Outer transaction success/failure | 1 | Two staged sequences, no premature current, final-only publication and abort retention |
| Unsupported end/turn/skin/family/margin/membership | 6 | Explicit rejection; no reinterpretation as distance/solid result |
| CONDITIONING/DEGENERATE propagation | 2 | Original accepted COL2 fixed literals, no dropped status or fabricated normal |
| Wrong owner, margin and untyped limits | 1 | Complete source/owner publication guards |
| Limits bool/expansion | HOST-only 1 | True scalar-type validation before device allocation |

Expected complete saved fixtures: 28 gzip records per backend (the source/storage test saves two); domain and constructor rejections do not pretend to save a physical trajectory. Each saved record retains complete input/aggregate/BVH/COL2 raw packets plus actual workspace counters. The late read fixture retains its actual seven reads before failure.

Suggested parent order: read the three new files plus the parent shared forwarding delta; first static/format in the sole execution slot; finite features and transaction/late-read nodes; complete new interpret matrix; complete new CUDA matrix; the minimal changed CollisionWorkspace transaction/default-detect/close neighbors; finally a source-external public workspace consumer. The accepted COL2 full geometry suite is reused, not rerun wholesale.

Success requires actual public Workspace calls on both backends, all fixed cases and original error gates, complete source/candidate/raw identity, and transactional publication. No pass is claimed here. If the existing consumer needs signed solid penetration, swept events, resident vertex loans, curved surface force points or actual load response, stop at this explicit boundary instead of relabeling these packets.