# M16 split-pose GPU contact precision

Historical stage report. The subsequent [contact-point and moment-arm
implementation](m16-tyx-contact-points-2026-11-27.md) resolves the remaining
fixed-state balance failures and establishes that all 192 discrepant points
below are unique in the original geometry. The earlier tentative nonuniqueness
interpretation does not explain those cases.

This experimental component extends `feat/tyx-mesh-manifold`. It does not enable
the full original-model dynamics compiler, integration, or a learning runner.

`MeshQueryWorkspace.query()` splits original host poses into FP32 high/low
components. Device producers can pass `positions_low` and `rotations_low` to
`query_device()` together; omission uses separate zero buffers, never residuals
from a preceding upload. The original producer must retain those bits: this API
cannot reconstruct precision already discarded upstream. Borrowing, generation,
shape, finite-value and overlapping-lane checks remain enforced.

Original support vertices transform directly into a centered pair-local frame
with compensated products, sums and division. World-space vertices are rounded
only for conservative broad-phase bounds. The narrow phase no longer subtracts
two already-rounded world vertices.

For intersecting small hulls (at most 16 original vertices per side), static
original face/edge directions are compiled in FP64 into two FP32 lanes. GPU
selection checks every supporting face and every vertex-pair cross axis near
the existing certified direction. Diagonal axes are redundant SAT bounds; no
vertex is removed. Using only an EPA-retained triangle had selected a different
feature in saved thin-mesh cases. The axis low lane is retained through support
projection; rounding the normalized normal first also lost penetration bits.
The output normal is still FP32. No pose-dependent NumPy narrow phase, native
contact substitution, geometric margin, or physics-threshold change is used.

On the eight fixed original M16 poses:

- All 22 former contact-activation disagreements are resolved. Native and GPU
  active pair sets match exactly: 4,154 contacts.
- All 51,666 saved candidates pass the unchanged independent 5 micrometre
  geometry/witness criterion, with zero unresolved cases.
- Active penetration depths differ from original-geometry FP64 SAT by at most
  4.986e-13 m; normal difference from native contacts is at most 1.199e-7.
- Geometry-generated contact rows feed the existing GPU elliptic solver.
  All eight acceleration comparisons pass the original componentwise criterion
  `.005 + 5e-4*abs(native)`. Maximum absolute difference is 0.0010821, versus
  7.2624 before split poses; generalized coordinates have mixed units.
- An independent native constitutive operator evaluated on the GPU's rows
  gives normalized balance at most 1.882e-7, below the unchanged 2e-6 criterion.

The stronger comparison against the native engine's *own* contact points is
not fully passed: five worlds exceed the same 2e-6 balance criterion (maximum
4.960e-6). There are 192 contact midpoints differing by over 5 micrometres,
maximum 0.7561 mm, despite independently valid material witnesses. Point choice
can be nonunique on nearly parallel features and changes moment arms.
A diagnostic-only replacement of common-pair midpoints, keeping the new GPU
gaps/normals/solver, makes all eight original balances pass (maximum 2.481e-7).
Those native points are never used in production. This isolates the remaining
fixed-state comparison issue; it does not prove trajectory equivalence.

Final validation: 34 new regression cases plus 17 directly affected interface
checks, 51 total, run with CUDA required; host preflight checks are included.
The complete changed geometry and actual contact rows were independently
checked. Saved COM/Jacobians/velocities, M/smooth and scalar constraints still
come from original native forward captures. There were zero integration steps,
policy samples, or PPO updates, and no 32/256-world training qualification.

One development process exited with 0xc0000005 after saving geometry, before
the unchanged box-manifold kernel completed first compilation. Its dead-PID
lock and evidence remain. Subsequent checks reused an exact hash-verified
previous binary for that unchanged kernel in a new cache. The exception root
cause is unproven; neither this workaround nor the contact precision fix is
claimed to fix the earlier native access violations.

Next: define and verify contact-point/moment-arm selection from original GPU
geometry, then complete original kinematics, mass/bias, mocap/equality, total
drive clamp and IMPLICITFAST integration before PD/rollout/PPO admission.
Do not hardcode world/geometry IDs or import native points into the device path.
