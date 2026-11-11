# COL fixed stagnation captures and accepted-baseline comparison

Status: **AUTHORING_ONLY / UNEXECUTED**. Parent execution owner is `/root`.
Scope is exactly this document and
`tests/reference_newton/rr/device_convex_stagnation_trace.py`.
No product, geometry fixture, tolerance, capacity or old test changes.

The [same-candidate experiment](p65-convex-candidate-publication-report.md)
remains **NOT_ACCEPTED**. Its branch and original failed run28 stay intact.
This diagnostic batch does not merge that product into the accepted parent.

## Fixed identities and original construction

The runner's nine identities are declared before reading new output; selection
does not depend on a returned status, chosen support or current candidate.

| Ordinal | CLI identity | Original construction |
| --- | --- | --- |
| 0 | K0-Q01-CB-sep | K0 `family_case("CB", GAPS["sep"])` |
| 1 | K0-Q01-CB-overlap | K0 `family_case("CB", GAPS["overlap"])` |
| 2 | K0-Q01-BH-sep | K0 `family_case("BH", GAPS["sep"])` |
| 3 | K0-Q03-CB-reverse | K0 `transformed(family_case("CB", -0.01), "reverse")` |
| 4 | K0-Q03-CB-cycle | same original case, `"cycle"` |
| 5 | K0-Q03-CB-tilt | same original case, `"tilt"` |
| 6 | K0-Q03-CB-permute | same original case, `"permute"` |
| 7 | K0-Q03-CB-cycle_reverse_ids | same original case, `"cycle_reverse_ids"` |
| 8 | cb_cycle_reverse_ids_segment | K1 `cb_overlap()` literal, original case plus `cycle_reverse_ids` |

The first eight use `test_device_convex_query.py`'s original ordinary tree,
body quaternion, model/bindings and requested-pair orientation. They omit
`site_capacity`, preserving the original disabled-patch query.
The ninth reuses `test_device_convex_patch_qualification.py::prepared` with
`site_capacity=8` and `requested_reverse=True`. Its reversed stable keys make
the actual canonical `request_reversed` flag false, exactly as the old test.
Its original full patch geometry/weights/material/moment checks and normal
gate remain, with the original independent qualification literals.

No physical step or reset is inserted; all nine original entries query their
accepted tick-zero geometry. A cycle is a coordinate rotation, not merely a
vertex permutation. Original body frame and static points are preserved.
The old three untracked K1 files are neither imported nor modified.

## Capture contract

For each fixed identity and backend, construct one original public scene,
then call ordinary `query()` and opt-in `query_with_diagnostics(pairs=(keys,))`
at the same accepted state. Preserve both complete outputs. The existing
`same_state_geometry` helper checks every typed geometry/source/frame field
and exact raw bytes; only the documented attempt tag is normalized. The
distinct stats are retained rather than asserted equal; owner/registration
identity must still match. Any other off/on difference remains a capture
failure, with both actual outputs saved.

The saved schema is `COL-stop-stagnation-capture-v1` and contains:

- exact original node ID, fixed ordinal, role, mode, original declaration and
  independent literal/patch expectations; canonical input SHA256;
- selected-checkout package origin, all relevant helper/reference/product
  file hashes and sizes before/after; fixed kernel binding for the role;
- original complete physical snapshot before/after, unchanged-state check,
  complete ordinary query and complete opt-in trace/query, owned source/frame,
  raw query arrays and raw 416-float/96-int GJK history for the original bound;
- original diagnostics policy/actual counters and real raw terminal status,
  duplicate flag, support direction and both supports, incoming/outgoing q,
  simplex/material slots and weights, subset/Gram/denominator decisions and
  reported bounds/termination; no CPU reconstruction replaces these fields;
- original geometry oracle verdict for both ordinary and traced outputs,
  separate capture status, failure type/text, live/closed resource stats.

All nonfinite scratch keeps its original dtype/shape/raw bytes plus the
existing explicit nonfinite value tags. No NaN is replaced with a finite
number to make JSON succeed. Gzip output is exclusive and preserved on errors.

`capture_status=COMPLETE` means only that capture, source binding, unchanged
state, full trace protocol and off/on equality succeeded. It does **not**
mean the geometry passed. `geometry_status=FAILED` retains the original
assertion and actual row status/reason. The standalone CLI exits **2** after
a complete capture with failed geometry, and **0** only when both original
geometry oracles pass. A transport/source/oracle implementation exception
propagates with `capture_status=FAILED`; it is not absorbed as an expected
geometry failure. This is a collection-free runner, not nine pytest passes.

The runner retains the original 64 GJK /192 EPA /64 vertices /128 faces and
all current diagnostic allocation/transfer accounting. It adds no custom
scratch, product trace fields or alternate geometry algorithm.

## Two-source comparison, parent executes once

The experiment role binds tested `b4f632e9` kernel SHA256
`4d75b7c68f86090bae4fbdd248beeccc4b9e290716124002e29def5545de5492`.
The accepted-baseline role binds parent `a18305cf` kernel SHA256
`c86c4565aa8d2556e062f32c71d4d84b558a320abd171756b0ca3318296ccd26`.
These are explicit admission guards, not output-dependent role selection.

1. Commit these two NEW files on the unchanged experimental product branch.
2. Parent cherry-picks **only that diagnostic commit** into the accepted
   checkout. Do not merge the experiment's product commits.
3. Parent's unique serial queue captures all nine fixed identities once from
   each role, with separate fresh output/cache roots and source snapshots.
   Interpret is the first matched backend because run28 failed there. A
   different backend is a separate declared comparison, never mixed with it.
4. Compare matching identity/mode pairs. Require identical canonical input
   hashes, fixture/helper/reference hashes and input geometry/source/frame
   arrays. Product kernel hashes must differ only as declared; no baseline
   product is inferred from a branch name alone.
5. Read the actual bounded histories: first divergent candidate, duplicate
   step, repeated simplex/weights, stable or changing q, support-plane width,
   raw terminal decision and actual budget use. Keep prefix alignment by
   iteration/meaning; do not delete rows or select a favorable subset.
6. Retain the original four Y/two HN inputs for any later repair qualification,
   but do not recapture them in this nine-regression diagnosis.

Example for the parent, from the selected checkout, using its existing
qualified interpreter and fresh evidence directory:

~~~text
python tests/reference_newton/rr/device_convex_stagnation_trace.py \
  --case K0-Q01-CB-sep --mode interpret --role candidate-experiment \
  --output <fresh-run>/complete.json.gz --cache-root <fresh-run>/cache
~~~

The parent must replace the example interpreter with the actual qualified
absolute executable and its normal no-autoload/thread/source guards. The
same CLI with `--role accepted-baseline` runs in the accepted checkout after
only this diagnostic commit is imported. This author has not executed it.

## Primary-source research and bounded repair decision

The pinned local Newton 1.7.0 distribution was read as source only:
`<parent>/.venv-newton-1.7.0/Lib/site-packages/newton/_src/geometry/simplex_solver.py`,
SHA256 `114eff2dbe7bea8b160079a0abd97639112c1ccc9e85818e711cf8b2f073b28b`.
Its metadata declares version 1.7.0. At lines445-477 it checks the current
support's relative Frank-Wolfe gap and independently stops on duplicate
support before adding/reducing. At lines543-579 it reconstructs the retained
simplex's bilateral witnesses; ordinary output direction/distance normally
comes from their displacement, with a separate near-contact path.
This is a useful generic pattern, not proof that its numerical thresholds or
fallback result satisfy Phymeo's existing independent gates.
Upstream location: [Newton simplex solver](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/geometry/simplex_solver.py).

The fixed [MuJoCo 3.3.7 native GJK](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_gjk.c)
was read at its official source. Lines165-187 use a Frank-Wolfe condition;
222-250 compute/reduce barycentric coordinates and stop when the next closest
point equals the prior one under its native scalar comparison. Lines261-270
reconstruct both witnesses from the retained simplex. Its distance-only and
contact-only early exits differ (189-220), so those exits cannot be copied
as a complete Phymeo gap/normal/material acceptance rule. This is source
research, not a new native MuJoCo execution or comparative benchmark.

The present experiment deleted the old duplicate-only exit. Source control
flow therefore permits a duplicate to add no point, reproduce the retained
simplex and miss the same plane-width gate until the original budget ends.
That is a hypothesis until the paired real traces establish whether it
happens in each failure. A plausible generic next candidate, **not authorized
here**, is to preserve the same retained simplex/weights on a documented
stagnation exit, and make direction, distance and witnesses share one
candidate. It must retain honest unknown/resource outcomes where geometry or
material checks do not justify success. No pair-name, axis, coordinate,
world ID, fixture label or capacity value may choose the path.

Only one evidence-backed repair may be proposed after all fixed comparisons
are available. It must pass the unchanged original six, these nine and the
affected matrix within the original budgets. If that one attempt does not
shrink the obstruction, retain the failure and move to an independent module;
do not tune thresholds or attempt a third speculative stopping rule.

No project imports, numerical probes, collection, tests, formatter, lint,
build, installation or new simulation were run during this authoring.
