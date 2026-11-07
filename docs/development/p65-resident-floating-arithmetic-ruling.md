# C9 floating FP32 arithmetic and first execution findings

Status: implementation boundary recorded; complete C9 validation remains in progress.
This note applies only to the new resident ordinary floating-tree computation.
It does not change the already accepted fixed/affine solvers or their arithmetic
helpers, the physical cases, public input gates, or residual tolerances.

## Actual failure and minimal diagnosis

The first P03 public recoil test passed in interpret run445 and CUDA run443;
run453 then passed all 15 HOST literal/input-contract cases. Run454 started the
new interpreted matrix and rejected P01 free fall on its second physical step,
retaining the complete accepted first-step state.

Run456 reissued that original failed P01 window with scratch readback only.
The second-step root, FK and twist were finite, but kinetic energy was NaN.
Its root and child x speeds were respectively
-3.308722450212111e-24 and +3.308722450212111e-24 m/s. These speeds are
normal FP32 values. The inherited strict multiplication helper replaced their
nonzero, below-normal-minimum square with NaN, making an unrepresentable
negligible kinetic contribution reject the entire otherwise finite candidate.

A weight-first native FP32 kinetic-energy calculation removed that second-step
failure. Run457 accepted two steps and then rejected the third with NONFINITE
in the remaining dynamics calculation. This demonstrates why an energy-only
special case is insufficient for a floating tree with small rotational and
cross-coordinate rounding terms. All original failures, full prefixes and
retained state remain saved; none is rewritten as a pass.

## Floating-only arithmetic model

The new private module `_tyx_floating_numeric.py` supplies the floating
kernels' add/multiply/divide. A finite operation result with magnitude below
2^-126 is flushed to zero. NaN and infinity remain nonfinite: neither is
converted to zero or accepted. This is an explicit FP32 engineering model,
not an exact-real intermediate-arithmetic certificate.

The existing public normal-FP32-or-zero input conversion, positive masses and
inertias, quaternion/domain checks, complete mass SPD check, finite bounded
PD solve, full equation residual and final finite candidate gates remain.
There is no host rescue, automatic smaller time step, mass adjustment,
clipping of a physical trajectory, or increase of an acceptance tolerance.
Stored root/joint values produced through these helpers remain compatible
with the original reversible checkpoint representation. Restore recomputes
geometry/energy with the same implementation.

Kinetic energy keeps the physical nonnegative quadratic form. Each term
computes weight times speed, then times speed and one half, using native
FP32 products. This avoids first discarding speed squared before a larger
mass/inertia can scale it. The total must still be finite and nonnegative.
Motion is never changed merely to make a diagnostic energy finite.

This boundary makes no uniform exact-real error claim for every extreme
input scale. Actual dynamics, state, force, geometry and trajectories are
checked by the originally declared engineering gates. The old
`_tyx_numeric.py` is unchanged. Current-pose floating bounds retain their
existing outward FP32 padding implementation.

## Independent holdouts and execution status

Five fixed actual-device arithmetic inputs cover a tiny product, tiny
quotient, cancellation below normal minimum, overflow, and invalid division.
The first three must return the declared finite results; the last two must
still trigger the existing nonfinite-result rejection. The kernel is a
module-level test helper because TYX does not resolve the locally imported
function symbols in the original nested test kernel: run458 records that
frontend failure before any arithmetic vector executed.

Run459 passed those five interpret cases and the original full 40-step P01
window, with unchanged h, mass, gravity and tolerances. Overflow/invalid
warnings belong to intentional negative controls. Run460 continues the
remaining interpreted matrix, reusing only these same-source six cases;
P03 is exercised again because the product arithmetic has changed. CUDA,
native Newton reference, saved-record audit and new installed-wheel checks
remain required. No whole-C9 or whole-P6.5 acceptance is claimed here.

The independent source review also found a default-domain declaration
omission in the folded root/tool bounds test. The test now explicitly
excludes its same-world dynamic material pairs, as the existing default
static-only query contract requires. This does not change the model,
materials, integration, collision filter semantics or contact support.

Sources: [C9 plan](../superpowers/plans/2026-11-08-resident-floating-tree.md),
[source integration review](../benchmarks/p65-resident-floating-integration-review.md),
[test authoring report](../benchmarks/p65-resident-floating-tree-tests.md).
Execution records are under `artifacts/p65-generic-collision-20261107-01`.

## Principal-frame gyroscopic term and the original pi holdout

Run460 finished with 48 passed, one failed and 21 deselected in 3692.83 s.
It includes the complete W1/W4/W32 physical windows and replay. The exact-pi
root-only case failed on its second step: an isotropic inertia with a real
stored z angular velocity of 3216.990966796875 rad/s acquired x/y velocity
about -0.001204785 rad/s. Its original tolerance remains 5e-6.

The old test saved only after its assertions, so run460 has no complete
payload for this failed pi case. Run468 retained the identical two-step
failure after moving observation before assertions and saving in finally.
The first accepted quaternion produced an FP32 rotation with an orthogonality
error about 1.1920929e-7. The second step's old world-frame cross product
formed artificial angular bias (0.6168503165, 0.6168503761, 0), despite zero
external torque and isotropic principal inertia.

The general correction uses the full body rotation times principal frame R.
With u = transpose(R) omega, the local gyro components are
((Iz-Iy) uy uz, (Ix-Iz) uz ux, (Iy-Ix) ux uy), then multiplied by R.
The I alpha_bias term stays in the same assembly. The old world cross is
removed, so the gyro is added exactly once. This is the ordinary principal
inertia formula, not an object-name/isotropic-input special case.

The two expressions are mathematically equivalent for an orthogonal R;
their FP32 operations are not bitwise equivalent. The new expression makes
the isotropic gyro exactly zero without orthogonalizing or repairing the
stored pose. M, J, energy, physical states, domain gates and original
residual tolerances are unchanged by this correction. It makes no claim to
remove every rounding error in the rotated mass or energy calculations.

Run469 passed both original pi-neighbourhood nodes (2 passed, 1.22 s).
Run470 then passed the original nontrivial principal-frame, eccentric-force,
rotated eccentric and 16-joint branch cases (4 passed, 184.06 s).
Runs473/474 passed the exact eight changed Python files' lint/format checks;
475 confirmed their raw ASTs match the pre-style source exactly. The product,
holdouts and failure observation are committed as ac082e8.

Run476 continues only the 31 original interpret cases not yet completed.
The passed run460 prefix remains bound to its actual earlier gyro source;
it is not relabelled as a final-source rerun. Focused correction qualification
and the forthcoming complete final-source CUDA matrix are separate evidence.
Native Newton, outside-source wheel and complete saved-record verification
remain required before limited C9 acceptance.