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
