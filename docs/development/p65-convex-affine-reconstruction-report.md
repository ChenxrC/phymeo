# COL affine simplex reconstruction — implementation and validation report

Status: **VALIDATED_BOUNDED_REDUCER / COL1A_PARTIAL**. Authoring baseline
`ff478eaa6ee570562b229491cb25c95e17783740`. Parent alone owns project execution.
This is one bounded arithmetic hypothesis motivated by
[the six actual trace findings](p65-convex-query-trace-findings-20261111.md).
It does not reintroduce the withdrawn FW gate or change normal termination.

## Exact scope

Only `src/phymeo/backends/_tyx_convex_query_kernels.py` may change in product:
add a dimension-aware scalar `affine_simplex_coordinate` and use it for the
three GJK candidate q components. The helper selects count1/2/3/4 before
evaluating corresponding terms. No shape/world/axis/fixture specialization.
Existing e/f/g edge values and original normalized w1/w2/w3 are inputs.

The Gram calculations, degeneracy predicates, raw-weight gate, clamping,
normalization, selected/stored weights, ranking, support, iteration/capacity
gates, status, normal publication and trace schema remain as before. Trace
candidate q/square and post-q record the new actual arithmetic automatically.
The final material-owner coalescing and bilateral witness production retain
the original real stored-weight route. No source outside this kernel is needed.

## Manual identity and finite-precision boundary

For a simplex with n in 1..4, vertex v0 and exact normalized weights,
sum(w_i)=1 gives

`sum(w_i v_i) = v0 + sum_{i=1}^{n-1} w_i (v_i-v0)`.

The new expression uses the right side, in fixed edge order. If one stored
coordinate of all selected vertices equals the same finite FP32 value c, every
active difference in that coordinate is exactly zero. The helper therefore
returns c without summing weighted copies of c. This directly addresses the
captured HN alternative triangle's one-ULP departure from its constant plane.
An arbitrarily oblique plane need not be preserved bit-for-bit: its residual
is tested independently with the predeclared arithmetic error gate.

This is an exact-real identity, **not** a claim that stored FP32 normalized
weights sum exactly to 1. For the stored weights, the real-algebra difference
between the affine expression and the original weighted expression is
`(1-sum(w_i))*v0`; operation rounding adds its own error. Original q-to-witness
and material reconstruction gates remain required. No weight is silently
repaired, no requested owner is dropped, and no new physical tolerance replaces
the old gate.

The private helper's count precondition is 1..4. Count1 returns the pivot;
count2 evaluates only e1/w1; count3 only e1/e2/w1/w2; count4 all three edges.
Thus an unused term cannot inject `0*NaN`. In the existing GJK enumeration,
unused i1/i2/i3 are initialized to slot0, a valid initialized simplex slot;
all selected indices come from the current nonempty simplex. The helper adds
no array read and no scratch allocation. Tests poison all unused input slots
and helper terms, and exercise every count.

## Tests-first declaration (historical, before execution)

NEW `tests/collision/test_device_convex_simplex_reconstruction.py` declares
**6 TYX nodes per backend, UNCOLLECTED**:

- Three frames: identity, cyclic permutation, and an oblique 3/5–4/5 rotation.
  Each executes point, edge, triangle, tetrahedron and the captured positive HN
  triangle inputs in both vertex orders. Expected points use independent exact
  Fraction-weighted source vertices with exact rational normalization.
- Three complete queries: BB face tie, cyclic/reversed BB face tie, and
  independently placed oblique BE separation. Existing independent family
  geometry/material oracle remains authoritative; actual trace additionally
  checks q against both published witnesses and all visited usable constant
  affine-coordinate candidates.

Arithmetic comparison is fixed at 2e-6 absolute for these bounded unit inputs.
Exact constant-coordinate and poisoned-padding checks are stronger structural
checks, not relaxed physics gates. Material weights retain the original
nonnegative and `abs(sum-1)<=64*2^-23` gates. Actual query gap remains
`1e-4*scale`, normal `0.002`, bilateral/material witness `2e-4*scale`.
Full input/output/weights and query traces are saved in the existing compact
gzip format. No production helper is used as expected-value oracle.

The **six original affected targets remain unchanged in their original files**:
`test_four_layouts_four_sparse_worlds_continue_three_real_steps` with kind Y
and ordinary/affine/floating/floating_affine; plus HN/sep in
`test_eighteen_new_family_three_states` and HN in
`test_rotated_reversed_queries`. No new duplicate fixture replaces them.
Y normal failures remain an observation gate; this reducer-only hypothesis
does not promise to fix their separate distance-versus-normal stopping issue.

Parent validates the six new nodes and original affected targets first,
retaining any failure and using bounded diagnostics. Changed-algorithm
regressions are selected by parent from the existing matrices. Original
64 GJK /192 EPA /64 vertices /128 faces and all prior geometry gates remain.
No blind third arithmetic alternative follows if this one fails.

## Authoring handoff (historical, before parent validation)

Tests-first checkpoint: `5628e958`. Product arithmetic is now authored.
Pure source review confirmed that the only product delta is the helper plus
its three q-component calls. Existing selected index slots and trace writes,
all normalized stored weights, and final witness accumulation remain intact.
The new helper has no array parameter or memory access.

Manual review and `git diff --check` are complete. No project import,
collection, numerical execution, pytest, lint, formatter, compile or build
was performed in this lane. The declared six nodes per backend have not been
collected. Parent owns all validation and should retain original failures.

Suggested focused order under the parent's existing environment and guard:

1. The three arithmetic-frame nodes in the new test file.
2. The three full-query/material-witness nodes in the new test file.
3. Unchanged HN separation and reversed/cyclic HN nodes from
   `test_device_convex_support_family.py`.
4. Unchanged kind-Y ordinary/affine/floating/floating_affine nodes from
   `test_device_convex_family_sources.py`.

Run on interpret and real CUDA as assigned by parent, retaining the complete
same-run source and saved inputs/results/traces. A remaining Y normal failure
does not justify changing stopping rules in this patch. If the HN hypothesis
fails, preserve the new trace and report the observed limitation; do not add
another algorithm or change gates inside this grant.

## Actual parent validation — 2026-11-11

Current result: **the bounded affine reducer change is validated; COL1a remains
PARTIAL because all four Y source-layout normal failures remain**. The two
recorded HN budget failures are resolved on interpret and real CUDA. This
does not claim that every hull–cone placement, smooth normal, patch or load
operation is supported.

Tested source is `90741b39d128dec70a828d08d269ed4b07fb8df2`, after the
tests-first and arithmetic checkpoints. The parent changed only the new test's
import ordering and formatting in that commit. Run07 reports
`import_multiset_equal=true` and `nonimport_ast_equal=true`; this is the
qualified style proof, not a claim of identical raw import-order AST.
The kernel remains the arithmetic source with SHA256
`c86c4565aa8d2556e062f32c71d4d84b558a320abd171756b0ca3318296ccd26`.
The six functional run receipts bind that tested HEAD and source snapshot.

The parent evidence root is
`artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01`.
This report was updated by reading its actual `final.json`,
`RAW_EXIT.txt` and `stdout.log`; no query, geometry oracle or test was
re-executed to prepare the report.

| Run | Backend and scope | Actual result | Pytest time | Raw exit |
| --- | --- | --- | --- | --- |
| [08](../../artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01/08-col-affine-interpret/stdout.log) | interpret, all six new arithmetic/witness nodes | 6 passed | 3.31 s | 0 |
| [09](../../artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01/09-col-affine-cuda/stdout.log) | real CUDA, same six nodes | 6 passed | 9.43 s | 0 |
| [10](../../artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01/10-col-six-interpret/stdout.log) | interpret, the six original affected entries | 2 HN passed; 4 Y failed | 5.71 s | 1 |
| [11](../../artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01/11-col-six-cuda/stdout.log) | real CUDA, same six entries | 2 HN passed; 4 Y failed | 19.33 s | 1 |
| [12](../../artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01/12-col-matrix-interpret/stdout.log) | interpret, remaining affected matrix | 258 passed; 12 deselected | 119.05 s | 0 |
| [13](../../artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01/13-col-matrix-cuda/stdout.log) | real CUDA, remaining affected matrix | 258 passed; 12 deselected | 559.63 s | 0 |

The matrix's 12 deselections are explicit: the six original entries already
run in 10/11, plus six unchanged HOST-only identities under
`test_invalid_curved_registration_is_not_a_separation` and
`test_literal_matrix_and_all_negative_certificates`. They are not new passes,
skips, unvisited physical trajectories or grounds for claiming an all-green
COL1a suite.

The two repaired entries are the original
`test_eighteen_new_family_three_states[sep-HN]` and
`test_rotated_reversed_queries[HN]`. Their original fixtures and independent
geometry/material gates were retained. The passing six new nodes also retain
the count1–4 poisoned-padding checks, independent Fraction expectations,
nonnegative/sum constraints on stored weights, and actual q-to-bilateral
witness checks. The remaining matrix includes the selected old query,
patch and source regressions; it does not extend curved patch or force-consumer
admission.

The still-failing entries are kind Y with ordinary, affine, floating and
floating_affine source layouts. Both backend logs retain their normal-oracle
failures; the recorded ordinary value is
`0.0031187900921125196 > 0.002`. The affine reducer did not change the
distance-versus-normal termination policy. All four failures remain open
for a separately scoped stopping-rule decision, using their retained
actual trace evidence. No tolerance, GJK/EPA/support capacity, fixture,
budget, source representation or UNKNOWN/RESOLVED rule was changed to hide
these failures.

Static history is retained in the same evidence root: 01 first Ruff and 02
first format each exited1; 03 import repair and 04 formatting exited0; 05
final Ruff, 06 final format and 07 the qualified style proof exited0.
These were test-only style changes, with no numerical repair between the
six functional runs.

This lane is released after a documentation-only update. Parent remains the
integration owner; no further execution or follow-up algorithm is authorized
by this report.
