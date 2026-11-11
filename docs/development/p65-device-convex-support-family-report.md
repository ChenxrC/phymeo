# COL1a device convex support-family authoring handoff

## Current parent refinement — 2026-11-11

**PARTIAL_VALIDATION / NOT_FULLY_ACCEPTED.** Final product `0ea0fee2` retains the generic support-ball overlap bound and restores the prior separation exit. It does not add a pair-specific solver, raise GJK/EPA capacities, change the original fixtures, relax the independent normal gate, or introduce HOST geometry fallback. Curved patch and required load remain explicitly unsupported.

| Final source execution | Actual result | Saved gzip payloads |
| --- | --- | ---: |
| run28 complete new TYX matrix, interpret | 127 passed, 4 failed, 6 HOST tests deselected; 69.50 s; RAW_EXIT=1 | 131 |
| run29 complete new TYX matrix, CUDA | 125 passed, 6 failed, 6 HOST tests deselected; 283.26 s; RAW_EXIT=1 | 131 |
| run30 affected old query/patch/source, interpret | 18 passed; 10.30 s; RAW_EXIT=0 | 18 |
| run31 affected old query/patch/source, CUDA | 18 passed; 59.04 s; RAW_EXIT=0 | 18 |

The original eight EPA-capacity failures now pass on both backends. The nine added rotated/contained/isotropic/degenerate/resource identities also pass, retaining the original independent depth, surface, witness, normal and actual resource assertions. The six unchanged HOST identities reuse their prior run; they were deliberately not rerun.

Four source-layout cylinder cases still publish normal error about 0.003119 against the original 0.002 gate. They are failures, not qualified normal results. CUDA additionally returns `RESOURCE_LIMIT` for `sep-HN` and `rotated_reversed_queries[HN]`; these pass interpret. This is the first complete new CUDA matrix, so these two failures are newly observed coverage, not a claim that a previously passing complete CUDA matrix regressed. Their arithmetic root cause is not yet established.

The experimental strict same-iteration Frank-Wolfe gate in `f4c66662` was withdrawn: run24 gave 21 passed / 2 failed; run25's disjoint remainder gave 98 passed / 10 failed. The latter introduced ten separation resource regressions. Final source removes that gate only, retains the support-ball work, and eliminates those interpret regressions. Both failed experimental runs and all first style-check failures remain saved. Final exact-scope Ruff and format checks pass. See [algorithm, engineering allowance and withdrawn gate](p65-curved-query-convergence.md).

Evidence root: `artifacts/p65-parity-n2-rob1b-col1a-20261111-01`. `col1a-refinement-evidence-index.json` binds six functional run receipts, 429 complete gzip files (including failed experimental records), original logs, before/after source snapshots and final current-source bytes. This is file-I/O/source provenance, not another geometry execution or a successful full-matrix certificate. No new wheel or full COL1a acceptance is claimed.

Next collision node is bounded saved-state instrumentation for the cylinder-normal and CUDA HN failures: preserve failed simplex vertices/weights, support direction/point and termination values, then choose a general repair from the actual discrepancy. Do not retry speculative thresholds. Complete query qualification precedes COL1b curved patches and COL1c loading. N3 training recovery and ROB1c multi-body cases continue independently.

## Previous parent validation: partial, not accepted

The parent integrated the eight-file handoff and its shared curved binding
prerequisite. This increment is **PARTIAL_VALIDATION / NOT_ACCEPTED**. The
authoring-only labels below remain historical; they do not override these results.
Evidence is under `artifacts/p65-env-refresh-20261111-01`.
`col1a-partial-validation-receipt.json` binds20 new command receipts, their
unchanged execution snapshots and187 saved gzip files. This is file/source
provenance and inventory, not a successful full-geometry saved audit.

- Shared finite-shape admission, outward broad radius and asset provenance:
  **11 HOST passed**, run21. New constructor/reference tests: **6 HOST passed**,
  run26. ROB1a's separate 33 registration tests are not curved-query evidence.
- Interpret's 122 predeclared identities have all been visited. The disjoint
  runs29/35/36, with the eight targeted harness corrections in run37, currently
  give **110 passing identities and 12 unresolved failures**. This is a union of
  recorded groups, not a claimed all-green final full-file execution.
- Eight failures exercise smooth EPA capacity: mandatory SE/NE/EE shallow
  overlap, E patch-boundary, two E broad-phase cases, exact-memory query and
  late-frame retry. The original SE counterexample uses GJK2/EPA61/support77,
  vertices64/faces124 and returns `RESOURCE_LIMIT/EPA_CAPACITY`. The radius,
  geometry, 64/128 capacities, 192 EPA iterations and numerical gates are unchanged.
- Four failures are the cylinder source case across ordinary/affine/floating/
  floating-affine layouts: published normal error is about0.003119 against the
  predeclared0.002 gate. No output-dependent tolerance relaxation was made.
- Bounded changed-source regression of the old ten families, affected patches
  and four source layouts passed **18 interpret and18 true CUDA** (run40/41).
  It preserves prior limited acceptance; it does not qualify new curved pairs.
- True CUDA's six analytic support cases, large-translation counterexample and
  repaired BE separation passed **8 tests in8.43s** (run42). This focused group
  verifies these repairs; the full122-case CUDA matrix has not been run.

The first new failures were preserved and diagnosed: TYX requires original
function names instead of aliases, and branch-assigned px/py/pz need lane-scope
initializers. The large common-translation counterexample then passed. BE
separation stalled because a common normal squared term hid the tangential
improvement in FP32; only equal-square full-surface candidates now compare the
complete `(q-c) dot (q+c)`, retaining original acceptance thresholds. The original
BE node passed after this repair and was included in the subsequent matrix.
run33 reads only the same failed fixed case's scratch; it is diagnostic evidence,
not another oracle or parameter search.

Three test-harness contracts were corrected without changing physical inputs:
empty logical pairs retain a padded raw lane; accepted FP32 root quaternions are
normalized when reconstructing their independent rotation; the required-load
negative checks the actual explicit finite-shape admission error. run37 then
passed6 identities and exposed the same cylinder-normal failure in the two
floating layouts. Every initial failure remains in its original directory.

Next collision work is a bounded generic smooth-support convergence/normal
refinement design, using these exact saved counterexamples and the existing
Newton/TYX source references. Choose an algorithm and independently check its
error/capacity contract before further implementation; do not raise capacity or
change fixtures to chase the matrix. Full new CUDA qualification, saved acceptance
and a new wheel are deferred until interpret's mandatory cases pass. Curved
patch/load/CCD remain unsupported. N2 training and ROB1b can proceed independently.

## Original author handoff

Status: **CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**. No project import, collection, numeric probe, test, lint, formatter, compile, build, installation or native execution was performed by this lane. The parent retains the sole execution slot. Git text diff checks are not functional validation.

Contract: [device convex support-family plan](../superpowers/plans/2026-11-11-device-convex-support-family.md), frozen parent `a33a09ef911ddd974ba1ff999b93a3b76c9b0cc7`. Branch `codex/p65-col1a-20261111`, isolated `p65-h1-actuator/phymeo`. Existing K1 untracked material was not used or changed.

## Checkpoints and exact ownership

1. `c4dc132af351d6fe54d2dffc6b33bc6e8b2c3834`: three new reference/test files, before production changes.
2. `cf234dda`: source frame/admission negatives and the planned rotated E broad case, before production commit.
3. `dcc1eed344663c898dfa7d816cf8d79a578e341b`: first production draft, preserved as history; not a validated version.
4. `6eed3f6d`: fixed exact large-translation counterexample before its repair; independent wire checker extends only the new GJK_SUPPORT phase, leaving K0 unchanged.
5. `df1ec03ce752fe0b9586c78ee4c232d893e98d31`: relative-support/frame repair, coherent frame binding negative and same-run saved-query coverage.

Final product/test scope is the seven paths below; this new report is the eighth path. `_resident_convex_source.py` was not changed. Shared `_resident_tree_binding.py`, `_resident_floating_binding.py`, capability registry and public batch forwarding remain parent-owned and are prerequisites for integration. No physical bridge, MuJoCo-TYX, training or ROB solver was modified.

## Implemented contract, still unexecuted

One analytic support family supplies Y/N/E to all three existing query call sites: GJK, EPA seed and EPA expansion. If either side is curved, S/C include their real radius in the same full-surface sample, H retains all original vertices, and the final stage adds no second radius. Original finite support IDs are preserved; repeated curved IDs coalesce after weighted surface coordinates are accumulated. Old ten-family core/skin decisions, budgets and patch math are retained.

Dimensions and asset XYZW/R are validated once and uploaded. Device gather composes the actual accepted body R with asset R for all four layouts. Source records bind declared/uploaded dimensions and matrices, quantization differences, original shape/material/world/layout/epoch and actual device effective R. Nonidentity asset R is supported. Independently specified static `CollisionMotion.body_orientation_xyzw` is deliberately rejected before query allocation for the new curved kinds only; parent approved this boundary. Old static S/C/H behavior is unchanged.

New curved patch rows are explicitly UNSUPPORTED, zero sites and false completeness flags, even when the distance query resolves. Required device-patch loading must reject atomically, without HOST fallback. No complete curved pressure/manifold or curved load capability is claimed.

## Parent-review repair: relative coordinate publication

The first source draft added a curved offset to a large FP32 coordinate before subtracting a local origin. The exact counterexample uses E axes `(1/4,1/4,1/4)` at `(2^24,0,0)`, S radius `1/8` at `(2^24+4,0,0)`, an original world anchor of zero, identity frames, and true gap `29/8`. All declared source words are exactly representable; rejecting precision or widening tolerance would conceal an implementation defect.

The corrected full-surface path subtracts its existing pair anchor, the gathered first original A point, before adding any curved/radius offset. Minkowski, simplex/EPA surface coordinates and weighted original core coordinates all stay in that frame. `skin_offset` is the difference of relative surface and relative core. The typed row publishes `origin_m = world_origin_m + pair_anchor`, so the original FP32 surface wire retains small relative coordinates. Old-only pairs retain zero pair shift and their original row origin.

A parent-approved additional immutable source field, `pair_origins_relative_m`, has shape `(logical_pairs,3)` and dtype FP32 in canonical query-pair order. It is taken from the actual existing device `world_points` readback, not HOST FK or geometry search. `_validate` and `require_current` verify it against the separately rounded gather expression and original uploaded coordinates, and bind every typed row origin. Empty query membership is `(0,3)`; failures precede publication. The added source field and its coherent-forgery rejection are saved by the new tests.

## Resource and numerical boundaries

For S shapes, V original material points, T existing query scratch lanes and P actual query pairs, four new device arrays add exactly `84*S + 24*T` bytes: dimensions `(S,3)`, asset R `(S,3,3)`, effective R `(S,3,3)`, and bilateral support scratch `(T,2,3)`, all FP32. The existing layout/byte ledger and memory preflight include all four arrays, even on old-only registrations. There are no new launches or per-query device allocations. Old-only source schema remains unchanged.

For a registration containing a new kind, two additional readbacks cost `36*S + 12*V` bytes per query; `_read` records real CUDA downloads/bytes and HOST-read bytes. The compact source pair-frame copy adds `12*P` host evidence bytes. No readback is used to generate a support point or replace lost arithmetic.

Each existing paired support sample increments the existing support counter once; the three analytic coordinate functions represent that same sample. The finite bound remains GJK 64, EPA 192, original 14 seed directions, vertices 64/faces 128, without retries or counter resets. A conservative maximum is 270 paired samples per iterative query. This is a source bound, not a measured runtime, FLOP count or benchmark.

Dimensions/nonzero matrix entries lost or subnormal during FP32 upload retain SOURCE_PRECISION behavior. A positive normalized curved dimension below `2^-63` also yields SOURCE_PRECISION, before squaring an axis can underflow in the support denominator. Overflow/nonfinite input is rejected. This arithmetic boundary is explicit; it does not change the fixed normal-domain fixtures or their gates. Deep/nonunique queries may honestly remain unknown/resource-limited. No compiler or TYX pin change is included.

## Independent oracle and manual node enumeration

The independent reference imports only stdlib and NumPy, and never a product support function or external producer. Seven kinds generate all 28 unordered combinations. The new 18 are SY/SN/SE/CY/CN/CE/BY/BN/BE/HY/HN/HE/YY/YN/YE/NN/NE/EE.

The G01 negative oracle is not a generic equation between placement and penetration depth. For each frozen identity, A contains its top-axis support point and B (always Y/N/E in this ordered matrix) contains an independently certified bottom-tangent ball of radius R greater than delta=1/64. The exact dyadic cone-side inequality, cylinder bounds and ellipsoid minor-axis bound are saved. Their Minkowski subset contains the delta-ball at the origin, while the +Z supporting plane gives the same upper bound; this proves depth delta and its normal for these declared cases. Arbitrary negative placement has no such assumption. Independent boundary/support, material reconstruction and bilateral closure checks still apply.

| New group | Manual TYX nodes per backend | Scope |
|---|---:|---|
| G01 | 54 | 18 families, separated/touch/shallow overlap |
| G02 | 18 | common cyclic rotation, translation and reversed request |
| G03 | 3 | oblique YE/NE/EH support-plane construction |
| G04 | 6 | Y/N/E, two asset frames, 13 directions plus scaled directions/ties |
| Curved patch boundary | 3 | full query with explicitly unsupported patch |
| GJK/EPA resources | 2 | original iteration/vertex budgets |
| No-HOST/empty, lost dimension, deep symmetry | 3 | one node each |
| Exact large relative translation | 1 | fixed parent-review counterexample |
| Four layouts | 12 | three kinds, four sparse worlds, three continuous real steps and selected reset/next step |
| Broad-phase extended surfaces | 6 | three kinds, static/dynamic or same-tree two dynamic bodies; E uses fixed TILT_Q |
| Source loan / late read | 5 | owner/world/epoch/layout/readback failures and retry |
| Exact memory, clone/next step, required-load rejection, static body-frame rejection | 4 | one node each |
| Immutable new metadata | 4 | dimensions, asset R, schema and pair-origin forgery |
| Last effective frame | 1 | late readback failure, old publication retained, retry |
| **Total** | **122** | **Uncollected; no pass claimed** |

Six additional HOST nodes cover five invalid registrations and the 28/18 identity/negative-certificate table. All 122 TYX identities author a compact gzip record using the fixed test identity and same-run raw/typed/source inputs or explicit pre-query rejection. Six HOST constructor/reference nodes have no fabricated device trajectory. Source rejection tests with no query have empty attempts by design. The four-layout records include actual q/v checkpoints and independent URDF-FK comparisons; a query's own pose is not the only oracle.

Original gates are unchanged: gap `1e-4 L`, surface/reconstruction/closure `2e-4 L`, normal `2e-3`, norm `2e-4`, barycentric `64*2^-23`, independent L without absolute world coordinate inflation. Touch/nonunique expectations remain bounded status contracts; mandatory separated/overlap cases cannot pass merely by returning unknown.

## Parent-only pending validation

No command below has been run. First integrate the parent-owned curved binding/broad/frame changes, review these files and preserve original source snapshots. The child did not collect tests or claim compilation success.

1. Static checks on the exact changed Python set, preserving first results; source-only style fixes must be separately bound.
2. Six new HOST nodes; then new direct support, exact translated relative frame, source and boundary nodes in interpret before the rest of the new matrix.
3. Complete new TYX matrix in interpret and true CUDA, original options and thresholds. Use `--require-tyx --tyx-mode=interpret` or `cuda`, `-q -x --tb=short -p no:cacheprovider`, distinct absolute basetemp, parent serial guard and environment.
4. Bounded changed-source regression: original `test_k0_family_gap_witness` at each of the ten `K0-Q01-<family>-sep` identities; `test_affected_k1_identity` for `cb_cycle_reverse_ids_segment` and `ch_cycle_reverse_ids_segment`; `test_partial_finite_support_polygons[cycle-partial_bh]`; `test_late_patch_failure_keeps_both_publications[late_read]`; and the four layout identities of `test_full_physical_last_body_supplies_patch` for SS. These cover the modified gather/support call plumbing without repeating all unrelated old accepted suites. Exact full node suffixes follow the existing decorators, not a new precollection.
5. Parent performs same-run saved evidence checks and the separately declared source-external wheel smoke. No saved result may be produced by rerunning geometry just to obtain counters.

The old ten-family accepted evidence, HOST curved response evidence, and all existing no-contact/native MuJoCo-TYX evidence are reused at their original scope. New device query acceptance, curved patch/load, CCD/mesh/deformation and performance claims remain pending or out of scope.

## Source manifest before this report

The manifest binds the final code/test checkpoint `df1ec03ce752fe0b9586c78ee4c232d893e98d31`. The report's own final SHA/blob is supplied separately at handoff to avoid a self-referential hash.
| Path | SHA256 | Git blob |
|---|---|---|
| `src/phymeo/backends/_device_convex_query.py` | `eeba5dc7c1a0fc4d45856f26dd635edd4c7449228165b97654c289a2800f6c7f` | `bf0f2fd1568707027c4e8db914964d96c9cd60b2` |
| `src/phymeo/backends/_tyx_convex_query_kernels.py` | `4547b54d973c79bb01491f270866786be500d2b215cbcfbdd958773538ac3bdb` | `7f02e2fc43a206a0123ef6e6053b7a2155050778` |
| `src/phymeo/backends/_tyx_convex_patch_kernels.py` | `bce8e8912796de38091e75f3be8f14d0a82bedc05c5be476679a0a7ed1943752` | `be5ac6d91e2dd79bf129d8b69c3ed23af372b4fc` |
| `src/phymeo/backends/_tyx_convex_support_family.py` | `8242606428a9fd08140c02f92efcb175678e119e9f9d53a3e4f3f4dfc4bc74c3` | `3629b678530ddd3c3de5ef02648b7896ade1a5a1` |
| `tests/reference_newton/rr/device_convex_family_reference.py` | `0587d13288f34660e6cf6e66f96ddb74352a8dcc51f536a52363fc5591bccd7a` | `15cf6d64914af09e043d0032feeb876a9adfcf81` |
| `tests/collision/test_device_convex_support_family.py` | `be7aaa1f4ad866aa2b2615d2c0546b729495850813d3408cb88c0b04fe5b8fd9` | `d7bb0acca8531df83f6f4a625235bbcf67bd6d0e` |
| `tests/collision/test_device_convex_family_sources.py` | `21405d49e5ca238df049c84229b9147daec348390bc87ffba6dea58bdc857961` | `84b29d04f82598c3c5c54207d547cac8573e89bf` |
