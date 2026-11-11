# COL affine simplex reconstruction — authoring report

Status: **UNEXECUTED / tests first**. Baseline
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

## Tests-first declaration

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
