# Original support-feature points and GPU moment arms

This extends the experimental split-pose mesh query and contact-row boundary.
It does not admit the complete original model to integration or training.

## What the saved cases established

All 192 previous point discrepancies over 5 micrometres have **unique** points
in the original geometry: 103 edge/edge and 89 vertex/face intersections.
An independent FP64 reconstruction agrees with native midpoints within
2.007e-16 m. Nonunique manifolds exist in general, but do not explain these
192 cases. The earlier tentative explanation is superseded for this dataset.

Selecting a precise normal/gap while retaining the old EPA witness was
insufficient. The broad 5 micrometre closure certificate does not establish
the correct tangential position on nearly coplanar features.

## Device implementation and contract

The existing static original hull directions remain host-compiled. Per-state
support classification and point reconstruction run in TYX CUDA; no NumPy
narrow phase, native contact lookup or per-geometry exception is introduced.

Pair-local transformed vertices and the selected support axis retain FP32 high
and low lanes. For small hulls of at most 16 vertices, compensated projections
select support features. Vertex/face points use the original face's material
weights; edge/edge points use the translated edge intersection. The two sides
retain original vertex IDs and barycentric weights. Ill-conditioned detected
unique intersections or invalid closure cannot enter force response.
Unresolved/nonunique feature combinations retain the prior certified witness;
the receipt does not claim these have a uniquely reconstructed point.

`MeshQueryResult.read()` adds `witness_receipt` (status, support count A, support
count B) and `surface_low`. Status 0 means inapplicable, 1 means retained without
a unique reconstruction, 2 means reconstructed, and -1 means reconstruction
failed and response eligibility was removed. The scoped device view includes
`surface_low`. The existing generation, ownership, resource and input checks
remain; nonfinite or overlapping point low lanes are rejected before response.

`MeshContactResult` adds `point_low`. Use
`geometry[:, 1:4].astype(float64) + point_low` for diagnostic world midpoints.
Device consumers must form the lever as `(point_hi - body_com) + point_low`,
not round `point_hi + point_low` to FP32 first. The box/box manifold convention
is unchanged and its existing single-lane points have zero low lanes.
All additional buffers are bounded and included in workspace memory admission.

This precision propagation matters: reconstructing points but discarding the
world-coordinate low lane removed all 192 large point errors, yet still failed
five original force-balance checks. Fixed offline interventions showed that
retaining original material-point precision, even with the existing rounded
motion inputs, passes all eight. Those interventions are attribution only;
the production point comes from GPU source geometry throughout.

## Validation and limits

The eight original saved M16 poses retain exactly 4,154 native contact pairs.
All 51,666 query gaps and normals are bitwise unchanged from the split-pose
parent when matched by world and geometry IDs. Material identities, weights,
support membership and closure are checked independently from original source
geometry. No native points are copied into production and no margin, friction,
solver or success/failure threshold is changed.

The new geometry/points/rows feed the existing resident elliptic dynamics
solver. Independent evaluation against the original FP64 mass, native rows and
constitutive operator passes all eight normalized force/moment balances under
the unchanged 2e-6 criterion, and all componentwise acceleration comparisons
under `.005 + 5e-4 * abs(native)`. The final archived report gives the exact
measurements. 22 new tests cover unique points, original material weights,
pair reversal, rigid transforms, workspace reuse, genuinely nonunique faces
and invalid low lanes; 63 directly affected regression checks are also run
with CUDA required (including host preflight tests).

Final fixed-state maxima: midpoint error 3.148e-10 m (high plus low), original
force/moment balance 1.865e-7, acceleration absolute difference 3.692e-5 across
mixed generalized-coordinate units. All 85 checks pass. These figures describe
this exact eight-state dataset and do not establish trajectory-level accuracy.

These are fixed-state component checks, not integration, trajectory equivalence,
learning, throughput at 256 worlds or a resolution of prior access violations.
Original COM/Jacobians/velocity and mass/smooth/scalar inputs still come from
saved native forward captures. Next implement and verify original device
kinematics and mass/bias, then mocap/equality, total-drive clamp, IMPLICITFAST
and PD/observation/reward/reset/rollout before any learning admission.

MuJoCo's pinned 3.3.7 collision implementation was inspected as a reference:
[convex collision](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_convex.c),
[GJK/EPA witnesses](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_gjk.c).
The unique-feature reconstruction here uses original geometric intersections;
it is not an implementation of the full native collision algorithm.
