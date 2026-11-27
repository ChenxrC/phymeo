# Moving-state contact sign and force-lever precision

The saved original M16 state at 1 ms exposed two different defects. Both are
repaired in shared TYX device code, with the original model, physical criteria,
solver tolerances and support geometry unchanged.

## Signed feature refinement

For geom 350/1736, a rounded query reported a negative gap of
8.610279156e-11 m. Independent projection of the original vertices under the
saved native binary64 transforms instead proves separation along a support
axis: 1.842428367e-11 m. That sign was checked with exact rational dot products
along the selected axis; this is not a claim that every SAT axis or device
operation was evaluated exactly.

Feature refinement discarded positive support gaps, preserving an earlier
false overlap. It now retains the signed interval and includes queries whose
initial gap rounds to zero. A rotation regression caught the latter omission.
The existing axis-agreement and 5 micrometre correction envelopes remain. No
epsilon contact margin, geometry-ID special case or UNKNOWN suppression is
introduced. Tests cover the actual case, order changes, rotation, translation,
and a genuinely overlapping perturbation.

## Precision through the contact lever arm

The rigid producer already emits COM and body-Jacobian high/low lanes, but
contact assembly consumed only the high lanes. The saved equal-contact world1
had Jacobian errors up to 1.33e-7. Independent row decomposition attributes most
of its generalized-force discrepancy to those Jacobians rather than changed
friction or regularization.

`MeshContactWorkspace` now accepts either the original three motion arrays or
those arrays plus **both** `com_low` and `body_jac_low`. It validates shape,
dtype, device and finiteness, preserves borrowing/generation rules, and ignores
old low buffers when subsequent calls omit them. Point-minus-COM, angular
cross products, body differences and frame projection retain high/low FP32
precision before publishing the existing FP32 contact Jacobian. The velocity
dot product uses that published Jacobian consistently with the solver.

This adds two bounded reusable input buffers; no per-contact host copy, native
forward fallback or state-dependent NumPy computation enters production.
Explicit host diagnostics and static model/fixture construction still use
NumPy. This is compensated FP32, not hardware FP64 or a general throughput
qualification.

Ten new checks plus 120 directly affected regressions pass with CUDA required
(130 total, including host admission checks). Earlier overlapping runs are not
additional tests. At the saved eight-world state, all 51,650 geometry queries
resolve, all contact sets match the native reference, and all original
force-balance / acceleration checks pass. Three saved lever-arm regressions
have independent Jacobian errors below 1e-9.

## Bounded movement and the next reference-contract issue

A new cap64 probe passes one 1 ms batch in eight worlds: position error
3.79e-9, velocity error 1.22e-5 (generalized coordinate units), normalized
force-balance error 1.28e-6 < 2e-6. The next batch rejects 1,044 contacts against
1,024 slots; native reference also has 1,044. Its ledger closes charged32,
knownGPU8/native8, remaining32, no unsettled reservation.

A separately registered cap48 suffix starts from the newly retained 2 ms state
without replaying the successful step. Query admission8000 plus eight box-box
pairs at at most24 raw contacts each gives a conservative contact bound8192.
Contact and solver buffers use8192, with their original byte-budget checks;
query overflow still rejects. This is a resource bound, not a contact filter
or a guarantee that future queries will fit8000.

That suffix passes capacity and geometry, then stops before integration because
world0 fails the original native-contact-point force comparison:
2.646932909e-4 > 2e-6. Its ledger closes charged16, known0, remaining32. All
eight acceleration checks pass. Contact sets, normals and gaps agree closely,
but geom1107/1473 and1133/1473 have points about0.297 mm from the native points.

Independent intersection of the original retained edges reproduces the GPU
points within 1.72e-12 and 4.32e-12 m. Their squared sine is about0.5898, so this
is not an ill-conditioned edge intersection. To isolate the consequence, the
native reference rebuilt its constraints using only these two independently
reconstructed midpoints; native mass, gap, normal, material and constitutive
law stayed unchanged. The residual drops to5.740063354e-7, passing the same
2e-6 criterion. No GPU Jacobian, reference acceleration or force was supplied
to that oracle, and no oracle result was uploaded to production.

This establishes a reference contact-point selection mismatch for this saved
state. It does not make the original comparison pass retroactively, prove
global manifold correctness, or qualify further movement. The next validation
contract must separately verify support/material/closure geometry and then
compare forces for the same independently verified contact geometry. Keep the
native point-selection comparison as a separate diagnostic. Do not silently
replace its gate or force the GPU to an approximate native point.

All evidence is under
`D:/cxr/alphanist-online-20261026/phymeo_tyx_moving_response_20261127`.
The current probes remain closed; unused budgets cannot be borrowed. This
component contributes eight GPU and eight native integration steps, zero
policy samples and zero PPO updates. The original continuous four-step gate
remains incomplete. No training release, 32/256-world qualification or old
access-violation repair is claimed.
