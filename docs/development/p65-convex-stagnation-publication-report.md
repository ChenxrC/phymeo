# Convex retained-candidate stagnation publication

Status: **CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED / UNEXECUTED**.
This is one newly authorized bounded candidate, not acceptance of the rejected
same-candidate experiment and not a claim that any geometry regression is fixed.

## Evidence and scope

The original same-candidate experiment remains on this branch. Its first
affected matrix returned 273 passes and nine resource failures; its product was
not accepted into the parent. The parent then captured the same nine inputs on
the accepted baseline and the experiment and executed one saved-data analyzer:

- Evidence root: parent `artifacts/p65-parity-until-0720-20261112-01`.
- Actual report: `44-col-saved-analysis/analysis/analysis.json`, with
  `summary.md`; RAW_EXIT 0, COMPLETE, nine pairs / eighteen captures.
- These eighteen captures used interpret; they do not predeclare CUDA's branch.
- Every paired prefix has the same 416 recorded float fields. At the baseline
  exit, only decision/status integer fields differ.
- Eight experiment traces then repeat the complete candidate for 61 iterations;
  tilted CB repeats it for 62. All exhaust the unchanged 64 GJK limit. The
  baseline takes the original duplicate exit after four or three calls.
- CB separation has recorded normalized plane width
  8.009374141693115e-7, tilted CB 2.346932888031006e-6, and BH separation
  0.0001726113259792328. These are not all one-ULP misses. The analyzer's
  no-progress result does not establish exact closest points.

This authoring read the actual saved analysis and source. It did not execute
the analyzer again, rerun queries or numerically recompute a trajectory.

Exactly three files belong to this new slice:

| Change | Path |
| --- | --- |
| MOD | `src/phymeo/backends/_tyx_convex_query_kernels.py` |
| NEW | `tests/collision/test_device_convex_stagnation_publication.py` |
| NEW | this report |

All original six Y/HN inputs, nine regression inputs, source/frame bindings,
64 GJK / 192 EPA / 64 vertices / 128 faces limits, support counters, and
existing geometry/material/patch tolerances remain unchanged. No registration,
HOST completion, public schema, helper-library dependency or physical solver
changes.

## One generic predicate, two distinct stopping routes

The prior experiment's pre-reduction plane exit remains textually unchanged:
the current q queries its own support plane and, if the original 8e-7 condition
passes, publishes its retained simplex and weights. This is the route that
passed the four original Y inputs in the earlier experiment.

The new shared pure-scalar `retained_candidate_fixed` is called by the
actual GJK kernel and the new synthetic branch test. After a real unchanged
subset reduction, it requires all of:

1. The existing support-duplicate flag is one and no vertex was appended.
2. Entry count is 1..4 and equals the selected count.
3. Entry q and squared norm are finite, the squared norm is positive, and every
   outgoing component and squared norm equals the entry value.
4. Each valid selected index is its original index: no reordering or removal.
5. Every valid weight is finite/nonnegative and exactly equals its entry value.

Equality means FP32 value equality, with no ULP/epsilon tolerance. Positive and
negative zero compare equal; no bitwise-equality claim is made. Original
clamping/normalization remains upstream. Unused snapshot weights start at zero
and are not loaded; unused helper weight/index arguments cannot affect a valid
lower-dimensional comparison.

The pre-existing near-origin/round-core unsupported/EPA branch executes first.
Only its non-near-origin alternative can accept this fixed candidate. The new
exit sets the existing RESOLVED/valid state and decision 3 (`duplicate_exit`),
publishes normal from the entry q, and retains the selected weights and
bilateral material vertices. It does not accept a duplicate if q, count,
ordering or weights changed.

The underlying argument is finite and local: without append, neither the
retained Minkowski vertices nor their two original material slots change.
Identity selection and unchanged normalized weights therefore preserve the
same bilateral weighted material points. Equality of q/square makes the
direction used for the current support and the published direction refer to
that same candidate. Affine reducer q and the weighted witness arithmetic are
still different arithmetic routes; the original witness error gates must pass,
and this report does not assert bitwise equality between those routes.

**Stagnation is the original engineering acceptance alternative, restricted
to a complete fixed candidate; it is not a strict plane certificate.**
The current support bound, gap lower/upper endpoints and actual width remain
unmodified. A width larger than 8e-7 is never clamped or relabelled as having
passed the plane gate. Real subset mask/Gram data remain in the existing trace.

If the predicate fails, the original reducer, compaction, search direction,
resource accounting and eventual unresolved/resource result continue unchanged.
No shape kind, world/owner ID, axis, fixture or iteration-budget value selects
the new behavior. No new support call, loop, buffer or launch is added: only
scalar snapshots, a bounded scalar predicate and the accept branch.

## Tests first and predeclared coverage

Tests were committed before the helper or kernel change:

- Tests-first: `0a6ee4ab729cfb4c26acf5a587a534800ea8d920`.
- Product checkpoint: `99e3d233760830df9bdd50fa1f779c92fd23d25f`.
- Product SHA256:
  `eaf1de24fdf71fa9529a52c904724e22150da0157f21d3d872e80996ae58cde5`.
- Pre-execution branch review correction:
  `aa83b93957a28daa8ec6d6d50446ab85a6b8e0b9`.
- Final new test SHA256:
  `3e0bc5cbee2734da42970ec5a6397914bafaf1764a0bc3e53fc130ca9c97abb9`.

Manual enumeration: **10 TYX nodes per backend, UNCOLLECTED**.

One node invokes the same production predicate in a real small TYX kernel:
five fixed positive vectors (counts 1..4, plus a clamped zero weight) and
fourteen negative vectors. The negatives change duplicate/append, count,
q with unchanged norm, squared norm, normalized weights, selection order, or
finite/nonnegative validity. Unused slots contain NaN/-777 to catch accidental
dependence. Expected 0/1 outputs are predeclared literals, not a mirror
implementation. These are synthetic branch checks, not physical fixtures.

Nine nodes use the frozen diagnostic helper's declarations and preparation,
without invoking its historical kernel-hash-bound capture entrypoint:

- K0-Q01: CB separation, CB overlap, BH separation.
- K0-Q03 CB: reverse, cycle, tilt, permute, cycle_reverse_ids.
- K1: cb_cycle_reverse_ids_segment, preserving its eight-site allocation and
  complete original patch oracle.

Each preserves the original independent geometry/material checks, runs a
same-state plain/trace comparison, and saves both full results. Interpret must
retain decision 3 for all nine fixed captures. Before any new execution, the
parent's review clarified that CUDA rounding may legitimately satisfy the
existing earlier plane gate: CUDA therefore permits only decision 2 or 3.
Decision 2 requires its actual current width <=8e-7, mask zero, no subset
enumeration and unchanged complete entry candidate. Decision 3 requires the
full fixed-point predicate, actual identity subset and honest width >8e-7.
Both branches require no append, unchanged q/vertices/weights/material slots,
and the same bilateral material owner/weight/core/skin/surface checks. No other
exit or relaxed geometry oracle is permitted. The FP32 bracket wire check has an
explicit operand-scaled rounding allowance; it does not change any geometry
acceptance threshold. Complete trace, original inputs, before/after state,
original geometry verdict, witness checks and accounting are saved even when
an assertion fails.

The original candidate-publication plane/budget tests are not weakened.
The original four Y plus two HN entries remain in their original test files.

## Parent's pending serial validation and stop rule

No import, collection, probe, numerical script, pytest, Ruff, formatter, compile,
build or native execution occurred in this authoring lane. Only text/source
inspection, file IO, SHA hashing, scoped Git diff checks and commits occurred.
The test-first commit was not a claimed executed RED result.

Suggested first execution order, at the parent's sole slot:

1. The new actual-predicate node on interpret/CUDA.
2. The nine frozen regression nodes with full trace/publication checks.
3. The original six Y/HN entries and existing plane/budget nodes, unchanged.
4. The bounded affected query/source/patch/trace matrix, without repeating an
   identical source group already covered in the same validation sequence.

Only actual complete results can accept this candidate. If it introduces a
new regression or does not reduce the obstacle, preserve the failure and stop
this candidate; do not add capacities, relax geometry gates, change fixtures,
or proceed to a third speculative algorithm.
