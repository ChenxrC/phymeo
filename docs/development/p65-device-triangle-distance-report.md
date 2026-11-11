# COL2 finite triangle distance: authoring handoff

Status: **CODE_READY_TESTS_PENDING / UNEXECUTED / AUTHORING_SLOT_RELEASED**. Date: 2026-11-12. This report contains no test, collection, static-tool or native pass claim.

Base: parent `11243d46`, new isolated branch `codex/p65-col2-triangle-20261112`. The previous B-point experiment remains on its old branch and is not an ancestor of this new implementation. Parent owns every project execution; author used only source reading, manual algebra, file IO and Git. The [first-slice plan](p65-col2-first-slice.md) describes the accepted scope.

## Delivered API and scope

`DeviceTriangleDistanceBatch` is imported explicitly from `phymeo.backends._device_triangle_distance`. Constructor parameters are `mode`, four positive integer capacities, optional `cache_root` and optional borrowed `runtime`. `query` accepts the plan's mesh/revision/snapshot strings and arrays, including explicit `facet_ids`. Exact dtypes are float64 coordinates/radii/origin and int64 IDs/faces/candidate pairs. Modes are interpret/CUDA only. No fallback, root export or old consumer is changed.

The batch owns its lock; `require_current(packet, **inputs)` checks owner, current packet identity and the complete immutable input binding. Caller-supplied runtime lifetime remains with the caller. Each query allocates bounded buffers, computes one kernel, reads all four outputs, validates, frees temporaries and only then publishes `last_packet`. Failed download/construction retains the previous packet. `last_attempt_raw` contains the actual full or partial rejected readback for diagnosis; it is not an accepted query.

Cap ceilings: vertices512, faces128, segments64, pairs8192. Supplied topology and pair membership are preserved, not silently pruned. Capacity-bearing array lengths are checked before immutable copying. `L=max(original pair's three edge lengths, axis length, radius)`; the bounded arithmetic admission is `2^-30 <= L <= 2^30`. L contains neither world translation nor pair separation. Every original stable vertex, facet and segment ID remains bound to its address.

This is a **HOST-input packed device narrow-phase packet**, not device-native source borrowing or zero round trips. Existing HOST `CollisionWorkspace`, exact inside/contains/CCD, accepted BVH implementation, resident contacts, GJK/EPA and B/C-point code are unchanged. Supplied candidate completeness remains the caller's responsibility. The new BVH integration case captures the actual existing producer's bitset and counters in the same call; it does not call that producer twice to manufacture evidence.

## Finite-feature arithmetic and conditioning

Use one fixed pair frame at the original first triangle vertex. HOST performs only original-input subtraction, packing and checks; it never searches for a closest point. The caller origin and pair origins are retained. Kernel arithmetic is normalized by the original local L and output coordinates return to meters. Packet publication adds the immutable pair origin, records that addition's FP64 defect, and rejects lost local witnesses beyond the original `2e-4 L` gate.

There are exactly 18 potential candidates per supplied pair: endpoint-to-interior-face slots0/1, finite face crossing slot2, then three finite triangle edges with five candidates each. Edge candidates are axis endpoint0/1 projected and clamped onto the edge, edge endpoint0/1 projected and clamped onto the finite axis, and the interior stationary pair. Triangle corners are therefore included through edges. Intersections do not use an infinite axis. Ordinary parallel-face/parallel-edge cases retain all equal-distance boundary representations; they do not fabricate a single unique material point. Same-position candidates and co-nearest original facets remain separate.

The independent reference takes a different path: enumerate active faces of `[0,1] x triangle-simplex`, solve each small Fraction normal equation and retain all exact minimizers. It imports neither Phymeo nor TYX and does not call production `segment_minimum`.

All policies were authored before execution; `u=2^-24`, half FP32 machine epsilon:

- Original gates remain distance `1e-4 L`, witness `2e-4 L`, normal `2e-3`.
- FP32 coordinate/radius conversion error must be at most `8u L`; actual upload arrays and every conversion delta are stored.
- Normalized triangle cross-product norm squared must exceed `32u` for the full enumeration. A zero computed area returns `DEGENERATE` for the uploaded arithmetic; this is not an exact-real classification of the source mesh. A small nonzero area returns `CONDITIONING`.
- Segment-edge stationary denominator is the squared cross-product, compared to `32u * axis_squared * edge_squared`. Zero denominator alone is not an exact-parallel proof. A structural branch in which every cross-product term has a zero operand can use the four complete boundary candidates; otherwise the potentially missing interior candidate makes the row `CONDITIONING`.
- For face crossing compare denominator squared with `32u * normal_squared * axis_squared`. An ordinary structural parallel/coplanar axis does not need a unique crossing candidate. A robustly same-sided segment also does not cross. Only an undecidable potentially relevant crossing causes `CONDITIONING`; plane denominator zero does not unconditionally reject the finite query.
- A far pair for which `8u * max(abs(pair-local uploaded coordinates)) > 1e-4 L` returns `CONDITIONING` without claiming local-scale distance accuracy. This is an engineering resolution guard, not an IEEE-certified bound. It does not inflate L. The fixed protocol contains a representable distant example and a separate conversion-loss rejection.
- Approximate tied candidates use a squared-distance band `16u L^2`. All tied slot identities and min/max axis parameters are retained; this is not a certified exact continuum or a persistent material identity across time.
- A reliable distance at or below `32u L` has `normal=None`; its distance/gap/witness remain available. Exact zero also has no direction. The fixed no-HOST-search node includes an ordinary parallel positive, a finite coplanar positive, and a positive-distance example inside this direction band. No complete contact response is claimed from such a packet.

These are bounded engineering policies. `CONDITIONING`/`DEGENERATE` rows never authorize a normal or contact response. Their residual candidate values are diagnostics. A zero boundary distance is not a penetration depth; positive boundary distance does not establish that a point is outside a closed solid.

## Complete raw schema and ownership

The immutable packet stores full original input arrays and their SHA256 binding, uploaded geometry/radii/scales/owner rows, all conversion deltas, pair origins relative to caller origin, raw outputs, and typed rows. FP64 publication defects are `conversion_deltas.published_axis` and `.published_triangle`.

| Field | Shape/type | Columns |
|---|---|---|
| uploaded.geometry | P x 5 x 3 float32 | triangle vertices0..2; segment endpoints3..4, all pair-relative meters |
| uploaded.owners | P x 2 int32 | original segment-row and facet-row lookup indices; stable IDs remain in bound inputs/typed rows |
| raw.candidates | P x 18 x 11 float32 | s, bary0/1/2, squared distance, axis point xyz, triangle point xyz; meter frame |
| raw.valid | P x 18 int32 | 0/1 candidate validity; invalid slots fully zero |
| raw.summary | P x 16 float32 | distance, distance-radius, s, bary0/1/2, axis xyz, triangle xyz, direction xyz, L |
| raw.meta | P x 7 int32 | status(1 resolved/2 conditioning/3 computed-degenerate), selected slot, tie mask, direction-defined, generation, original segment/facet row indices |

The adapter checks original owner/generation, all slot validity, finite parameters, barycentric ownership, reconstruction, emitted distance and tie-mask consistency, same-selected witness publication and direction. It checks the already emitted finite list; it does not search new geometry, choose replacement candidates, normalize wrong weights or fill missing rows. All checks finish before publication.

`stats` reports actual kernel-buffer allocations/live/peak bytes, CUDA upload/download counts and bytes, synchronization, attempts/queries/failures and launches/completions. `packet_array_bytes` and `attempt_array_bytes` separately count retained NumPy payload bytes; they are not included in live kernel-buffer bytes and do not pretend to measure Python-object overhead. Kernel buffers use 1032 bytes per pair across four inputs/four outputs, at most 8,454,144 bytes for8192 pairs. Empty candidate packets launch/allocate no kernel buffers. Host input/immutable packet storage is additionally bounded by declared capacities.

## Authored node map, not collection

Original enumeration was 24 geometry +20 protocol identities per backend (88 total). Following the pre-execution resource finding, the current manual enumeration is **44 interpret +45 CUDA +4 HOST-only fake-owner identities =93 total**, UNCOLLECTED. Geometry contains declared conditioning/degenerate negatives, not 24 claims of resolved geometry.

- D01 six: face_above, face_below, edge, vertex, beyond_edge, zero_axis.
- D02 six: parallel_face, oblique_endpoint, edge_edge, extension_miss, through_face, parallel_edge.
- D03 six: rotation, large_translation with distant caller origin, reverse_winding, shared_edge, crease, deformed_snapshot.
- D04 six: bvh_pruned with two co-nearest original facets and six pruned faces, concave_faces, near_parallel_clear, near_parallel_conditioning, zero_area, distinct_segment_owners.
- Original protocol20 per backend: capacity4, malformed/preflight8, real late-failure/retry3, snapshot+owner1, conversion/resolution1, no-HOST-search+coplanar/direction-band1, empty1, borrowed-runtime1.
- Added resource protocol: HOST-only fake owner before/after free side effect crossed with owned/borrowed runtime (4); actual CUDA successful free followed by an injected exception (1). These are additional nodes, not claimed to fit the original44.

The ordinary positives independently check full finite memberships, original IDs, all candidate reconstructions and ties, distance, both witnesses and normal when defined. Geometry fixtures are not changed to obtain a status. Fixed late failures tamper ownership or barycentrics, or fail the final download; none use a fake physics result as a successful retry.

Original geometry/protocol inventory remains **33 gzip JSON files per backend**: geometry24, capacity4, late-failure3, conversion/resolution1, no-HOST-search1. The other11 original protocol identities have no promised trajectory artifact. The additional resource nodes save four HOST `fake-release.json` records and one CUDA `cuda_release_unknown.json.gz`: current totals are33 interpret gzip,34 CUDA gzip, plus4 HOST JSON. Every saved array uses exact dtype/shape/byte encoding, including actual rejected raw buffers; no nonfinite-to-finite substitution. The same BVH readback and its leaf/query arrays, IDs and device counters are embedded in the bvh_pruned record.

## Source-only checkpoints and next commands

Tests first: `f734d301` (three files). Pre-execution frame correction `2501c28f` separates common translation from a distant small shape's unrepresentable relative coordinate. `f4977681` adds explicit coplanar/direction-band semantics inside the existing protocol identity. `6361739c` adds the far-relative resolution example inside the same predeclared protocol identity. No production output or numerical probe informed these changes. Initial product checkpoint: `276517b0`; final source-only delta also checks/saves FP64 publication defects and captures the BVH raw return in the same existing call. `6250ba61` also verifies an actual changed vertex snapshot after a prior accepted packet and preserves stable IDs under vertex storage permutation; these were source-review coverage completions before any execution.

Parent first runs static tools on exactly these five NEW Python files, then the two test files for interpret and CUDA in the sole serial slot. Suggested command selection is `tests/collision/test_device_triangle_distance.py tests/collision/test_device_triangle_distance_protocol.py -q -x --tb=short -p no:cacheprovider -k interpret`, then the corresponding CUDA selection with a fresh basetemp. Keep original errors and do not turn an unresolved matrix case into a pass. No unchanged old suite, producer or counter audit is requested.

Source snapshot before final report commit:

| File | SHA256 |
|---|---|
| src/phymeo/backends/_device_triangle_distance.py | 68c1a086c863edcc9b65b035124880079ae361ee12d998857a8ac48793461a56 |
| src/phymeo/backends/_tyx_triangle_distance.py | 820ab1e3fe0e384a72d0117ba0affd7e749f677c9ee58086bb46552eb14b3c94 |
| tests/reference_newton/rr/device_triangle_distance_reference.py | af95f6e2019d25a5fd74a458e9758e61e94cc5c7ec6eff3192c00f8efebc5a32 |
| tests/collision/test_device_triangle_distance.py | 0510b9929daf1718ff888a3d343866328aa47f8b8608acee9d983db0d6e72fb2 |
| tests/collision/test_device_triangle_distance_protocol.py | 1dec223ff692a1c75a794701b62edb1b188b894218e576915d7a8593405310b9 |

Only the six approved NEW files differ from base. No old source/test/pin/export or rejected B-point implementation changed. Final validation and limited acceptance belong to the parent; no authoring result here means device qualification or complete COL2.

## Pre-execution resource review correction

Parent identified that the original ExitStack callback subtracted live bytes in a finally block even when opaque `array.free()` threw, losing the owner and allowing later runtime close. This was a source-review finding before any execution, not a failed numerical test. `2ad6c988` first authored the five additional resource nodes. The narrow product change imports the already accepted `mujoco_tyx._resources.DeviceResources` without modifying that module.

Each query now owns one DeviceResources ledger. Its runtime owner is a no-op lease retaining the actual batch runtime; normal query cleanup does not close that runtime. Buffer proxy `free()` calls the actual native free and decreases `live_bytes` only after normal return. If free throws before or after an unknown native side effect, DeviceResources retains the proxy/actual owner in its existing global quarantine, retains the runtime lease, attempts independent siblings, and reports sticky UNKNOWN_RELEASE. `live_bytes` conservatively includes those unknown owners; it is not a claim that an injected post-free fault physically leaked the same number of bytes.

Later query, require_current and close first consult the retained ledger, report its original unknown release, and do not retry free or close the dependent runtime. Batch runtime ownership is also recorded in DeviceResources; owned runtime-close ambiguity is sticky through the same accepted mechanism. Borrowed runtime close uses only a no-op lease. Successful normal cleanup and the original late-readback retry path retain their behavior.

The four HOST fake-owner nodes exercise before/after native-side-effect uncertainty with both ownership modes, three independent buffers, nonzero uncertain bytes, old-packet retention, no duplicate callback and strong quarantine ownership after dropping the batch. They use a clearly marked HOST packet sentinel, not fake accepted geometry. The CUDA node first produces an actual accepted packet, then injects an error only after the fourth buffer's real free has returned; later independent siblings must still free normally and the old packet must remain published. All future batch query/close attempts stay sticky. Only test-side knowledge that every real free completed permits explicitly retiring its held actual runtime after assertions; production does not recover or clear the unknown ledger.

Parent should run the4 HOST fake nodes once (name filter `host_fake_release`), then the mode selections. The original `-k interpret` selects44; `-k cuda` now selects45 including the additional resource seam. The kernel, geometry literals, independent oracle, capacities and original distance/witness/normal gates did not change in this correction. All additional checks remain UNEXECUTED.

## Parent qualification (2026-11-12)

Final tested092e20da: run1724 HOST resource nodes passed; run176 interpret44passed/49deselected,2.31s; run177 CUDA45passed/48deselected,4.42s. All93 declared nodes are covered and all accepted commands raw0. Original source snapshots and raw command exits are preserved under artifacts/p65-parity-until-0720-20261112-01; before/after and saved-source bytes match. Root integrated this seven-file branch including the new installed consumer; no legacy collision product was replaced.

run173 initially failed TYX frontend admission before geometric execution because pinned99f6 does not admit an AST bit-shift expression. The18-candidate tie-bit mask now uses an exact bounded doubling recurrence; candidates, witness/oracle values, capacities and engineering gates did not change. HOST fake-resource ownership test closures were scoped to release all incidental test references; real UNKNOWN_RELEASE ownership remains sticky. First static and admission failures remain recorded.

Acceptance covers current finite boundary distance for supplied sphere/capsule-to-triangle candidates, complete feature/tie/witness/source evidence and resource transactions. It is not solid-inside classification, CCD, triangle-triangle coverage or force response. check_triangle_distance.py is authored and statically checked, with final installed qualification still pending.
