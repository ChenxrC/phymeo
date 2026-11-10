# MuJoCo-TYX T3b primitive manifold source handoff

Status: **CODE_READY_TESTS_PENDING / UNEXECUTED**. This is pure authoring and source review, not native, interpreted, CUDA, lint, build, or integration success.

Base: accepted T3a parent 275872beca81767bdc2ee8a1d0ca52fb5ae9c100.
Branch: codex/mujoco-tyx-t3b-source.
Source checkpoint: c4ee03f806cddc52b1a02d784bf4a23c76856faa.

## Delivered scope

Only three files belong to this source lane:

- NEW src/phymeo/backends/mujoco_tyx/_normal_extended_geometry.py: bounded device kernel for capsule–capsule, capsule–box and box–box.
- MODIFIED src/phymeo/backends/mujoco_tyx/_normal_geometry.py: one raw-buffer argument and one extended-pair read branch. All six existing pair branches, complete current point Jacobian, frame, native impedance/regularizer/reference acceleration, and counting remain unchanged.
- NEW this report.

The parent owns admission, plan/schema, device dispatch, runtime, NOTICE and tests. The accepted coupled normal solver, tree/floating mechanics and integration are unchanged here. No helper agent or project execution was started by this lane.

## Fixed algorithms and version fidelity

The source follows fixed [MuJoCo 3.3.7 primitives](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_primitive.c), [boxes](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_box.c), and [driver](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_driver.c). MuJoCo-Warp 0.0.2 was read for comparison; differing simplified manifolds do not override CPU semantics.

| Pair | Slots | Native-order behavior |
| --- | ---: | --- |
| capsule–capsule | 2 | Nonparallel sequentially clamped segment solution; parallel endpoint attempts A+, A−, B+, B−, stopping after two actual contacts. Preserve native repeated points. |
| capsule–box | 2 | Endpoint/face tests, then edge-axis outer and corner inner traversal; strict closest-feature comparisons and native secondary point selection; native SphereBox rule at each selected point. |
| box–box | 24 | All 15 axes, native feature/tie selection, face or edge clipping, driver filtering and exact positional deduplication. No eight-point assumption or successful truncation. |

New feature numbers are the final compact native output ordinal within each pair, not persistent material addresses or cross-step identity. Original six-pair feature identities remain unchanged.

Specific fixed-version details:

1. Capsule–capsule preserves native parallel contacts instead of Warp 0.0.2's single-contact TODO.
2. Capsule–box preserves the CPU initial closest-distance bound and j-axis/corner loop order. Its diagnostic-only block has no live outputs and is omitted. Zero denominators in secondary-point min comparisons are skipped only where native infinite/NaN candidates cannot update that minimum.
3. SphereBox inside selection starts at twice the sum of box half sizes, compares absolute face distances in −x,+x,−y,+y,−z,+z order, and keeps the first strict-comparison tie.
4. The fixed CPU box-face branch publishes **half** the signed point depth after midpoint construction. This is preserved rather than replaced by Warp's full depth.
5. The fixed CPU box-edge face-corner loop reuses c1: a corner reaching its distance test overwrites the denominator used by later corners with its squared distance, even if subsequently rejected. This sequential behavior is preserved.
6. Driver filtering expands half sizes by margin and uses outsideBox ratio 1.01. Outside means any absolute local coordinate exceeds 1.01 times size; strictly inside means every coordinate is below size/1.01. Reject outside one box unless strictly inside the other.
7. Exact three-coordinate duplicates are compared only among surviving points. Remove the earlier duplicate, then compact stably. No epsilon deduplication.

All formulas run in TYX FP32. Branch identity and exact-equality outcomes can differ near floating-point degeneracies; no FP64 bitwise-equivalence claim or hidden correction is made.

## Kernel and buffers

The module publishes extended_contacts and extended_contacts_other wrappers:

~~~python
extended_contacts(
    count, ngeom, npair, nslot,
    geom_ids, geom_data, pair_ids, pair_law,
    motion, rotation, geom_world, raw_rows, extended_scratch, status,
)
~~~

The scalar _bit helper is a single expression. There are no shifts, XOR AST nodes, while loops, host callbacks, or configuration-specific precomputed geometry.

The existing contacts signature adds raw_rows after scratch and before diagnostics. Extended native-type pairs read raw manifolds and then enter the same complete Jacobian/normal-constraint path as the original six pairs.

| raw_rows[world, slot, 11] columns | Contents |
| --- | --- |
| 0 | Present |
| 1 | Native reported signed distance |
| 2:5 | World position |
| 5:8 | Native A-to-B normal |
| 8:11 | Initial tangent seed; zero for these pairs |

Every raw row is cleared each call. Each box pair temporarily uses its own 24 slots for local candidate positions/depths. No other pair's slots are used as scratch. Filtering and forward compaction precede consumption by contacts; inactive trailing slots are zeroed.

Extended scratch is [world,64,3], cleared for each extended pair. Capsule branches reuse rows 0–6. Box scratch:

| Rows | Use |
| --- | --- |
| 0–2 | Relative rotation R_A^T R_B |
| 3–4 | Relative centers in each local frame |
| 5 | Temporary cross-axis normal |
| 6–8 | Reference-face signed permutation |
| 9–11 | Incident rotation in reference-face frame |
| 12–14 | Transformed center, reference half sizes, incident half sizes |
| 15–16 | Selected cross normal and transformed normal |
| 17–19 | Base and two directions |
| 20–23 | Original incident vertices |
| 24–27 | Projected incident vertices |
| 28–31, 32–35 | Projected edge bases and directions |
| 36–39, 40–43 | Original edge bases and directions |
| 44–46 | Reference-face-to-world rotation |
| 47 | Reference world origin |
| 48 | Temporary candidate vector |
| 49–63 | Reserved and cleared |

Four box lines × two coordinates × two boundary sides give at most 16 intersection attempts; four reference corners and four incident corners follow. Therefore raw candidates are bounded by **24**, before filtering. Every append checks both actual pair slots and 24; excess sets status 1, never successful truncation. Capsule–capsule attempts at most four point pairs, stopping at two contacts; capsule–box attempts at most two. Every raw index is start+candidate in the parent's bounded slot interval.

G128/P16/total C32 remain. A box pair reserves 24 slots, leaving eight other slots. These are conservative reservations, not actual counts. Common actual present/active capacity checks remain strict greater-than, so equality with capacity is accepted.

## Failure behavior and parent wiring

Existing status 1 denotes slot/internal candidate capacity failure; status 2 denotes nonfinite/degenerate arithmetic failure. Capsule scaled-axis squares that become nonpositive/nonfinite, nonfinite SAT values, a required zero box-edge denominator, nonfinite scratch or candidate fields do not become successful empty manifolds. No host geometry or solver fallback is introduced.

The native compiler requires positive capsule radius/half length and box half sizes. Parent admission checks these before packing; device guards also cover squared-length underflow/overflow. Rejected degenerate arithmetic is a declared numeric-domain limitation, not proof of no collision.

Parent wiring adds one geometry launch only when plan.extended_contacts is true. Original six pairs remain five launches; extended pairs use six. This pass computes current geom transforms and raw manifolds. The common contact pass currently recomputes the transforms, preserving original flow. No host geometry upload intervenes, no per-contact independent force application occurs, and integration still runs once. Shared physical M and the coupled normal QP consume all rows together.

## Verification and handoff

Only source reads, fixed-source comparison, text/file IO, Git diff checks, and small commits were performed. The source checkpoint passed git diff --cached --check. Independent source audit read all algorithms, scratch lifetimes and candidate bounds. Its SphereBox initialization/absolute-distance finding was corrected before the source checkpoint. This is text review, not a test result.

No import, collection, numerical probe, pytest, lint, formatter, compile, build, install, profile, native execution, CI, or accepted group was run. Parent performs first serial validation and records actual output.

| Source file | Working SHA256 | Git blob |
| --- | --- | --- |
| _normal_extended_geometry.py | ed7da208d1af039903602964a0374552f45d878d2a4f695dafcb60907a9d1858 | 7b7b1967b28736482c9e37bd6fa4cb24c76f3765 |
| _normal_geometry.py | 101582d59a994d21719dd1aad726212f27b89910fdd18cec2aaabb881631ed39 | 53459fcea6ce50c73665ddc533caff5eb3ad7b4c |

Ready for parent integration and first execution. This handoff grants no execution to this source lane.
