# S2 response: parent derivation of the ideal-map enclosure

Status: HAND_DERIVATION_AND_SOURCE_INSPECTION_ONLY. This note closes the pending
algebraic interpretation check for the existing S2 R2 enclosure. It neither adds
a solver path nor reports an executable test. The original S2 owner retains the
serial validation slot; CUDA run32 is active when this note is written.

Read source snapshot: child bcf4c1d343df2ee0a15be9f62f1bd1e50ca087d4,
`validation/cloth_self_mapping.py` functions `capacity_decomposition`,
`capacity_witness`, `_ideal_capacity_error`, and
`validation/cloth_self_response.py` functions `_residual_global`,
`_residual_speed`, `_extreme_velocity` and `_exact_phase`.
The authority remains the frozen [self-response design](../superpowers/specs/2026-10-30-p65-self-response-design.md),
including R1/R2 and all original budgets. No threshold, fixture, phase law or
supported physical domain changes here.

## Actual stored-map decomposition

Let B be the complete stored difference map, W the positive original diagonal
inverse mass, A = B W B.T, and L = A^-1 B W. The existing exact positive-LDL and
strict condition-number gates remain mandatory. In this full-row-rank domain,

    L B.T = I,
    p = L ell,
    Q = I - B.T L,
    r = Q ell = ell - B.T p,
    L r = 0.

Here B and L are dimensionless, W has units 1/kg, and ell, p and r have units N*s.
All shared-node cross terms are already present in A. These values are a
certification decomposition of an observed delivery, never a replacement
request, physical state or uniquely measured owner pressure.

## What the contraction actually proves

At the selected material addresses, independently evaluate the exact geometric
Jacobian B_star, with E = B_star.T - B.T and componentwise |E| <= Gamma.
The code accumulates shared-node contributions before enclosing each map entry.
For any native prediction uncertainty delta with |delta| <= e, define the
ideal-map certification representative by the STORED-gauge equation

    p_star = p + L delta - L E p_star.

Set G = |L| Gamma and gamma = ||G||_infinity. If gamma < 1, this affine map is a
contraction in the infinity norm for every admitted E and delta. Equivalently,
I + L E is invertible. Therefore there is one representative in this declared
gauge for each E and delta; this is not uniqueness of physical contact pressure.

Writing z = p_star - p gives

    |z| <= |L| e + G |p| + G |z|.

With b = |L| e + G |p| and u = ||b||_infinity / (1 - gamma),

    ||z||_infinity <= u,
    |E p_star| <= Gamma (|p| + u*1).

Consequently t = e + Gamma (|p| + u*1) encloses the modified nodal uncertainty,
and the sharper coordinate bounds are

    |p_star - p| <= |L| t,
    r_star = Q (ell + delta - E p_star),
    |r_star - r| <= |Q| t,
    ell + delta = B_star.T p_star + r_star.

These are the expressions produced by the initial decomposition, feedback
matrix, uniform radius, ideal nodal error and second decomposition in the
current source. The center p and r are unchanged. The code rejects gamma >= 1;
this proof does not permit selecting a favorable E, suppressing an error term
or increasing a proof/physical budget.

The statement is deliberately narrower than computing

    (B_star W B_star.T)^-1 B_star W (ell + delta).

The implementation does not compute that ideal-map mass-weighted pseudoinverse,
and this proof does not identify the two representatives. The ideal remainder
is orthogonal under the stored gauge L, not automatically under B_star W.
Separate remainder response, resultant, moment and work checks remain necessary.

## Limits and independent physical checks

Gamma concerns exact-vs-stored Jacobians at the plan's selected material
addresses. It alone does not enclose all differences between those addresses and
every complete ideal clipping extremum. The latter are handled separately by
the exact material addresses in _extreme_velocity, actual/ideal velocity
enclosures, and _residual_speed evaluated over both maps.

The capacity reader uses lower normal and upper tangent bounds over its complete
pressure enclosure. The extra phase-uncertainty test is a sufficient rejection
gate inside the original ceiling; it supplies neither new capacity nor a new
allowed error. The same enclosure cannot be treated as an independent random
error and averaged away.

The global reader checks the retained remainder's resultant, moment, virtual
work and held/terminal work bounds. The actual-state joint energy uses ORIGINAL
external inputs and never subtracts internal contact work. Actual terminal
momentum and complete selected-pre binding remain independently checked. These
checks are necessary because Q orthogonality says nothing by itself about all
ideal extrema, moments or work.

No part of the argument turns a frozen velocity response into a nonlinear held
position bound. Held support and positive area require the separate continuous
path proof, complete two-normal CCD and actual final-state gates already frozen.
The proof does not cover incident folds, composite material/R7 self-response,
rank-deficient B, general robots or the final P6.5 demonstration.

## Evidence status

This derivation was performed by reading source and algebra only while the S2
owner ran CUDA. Existing host136 and interpret80 results remain tied to their
recorded source snapshot. Host saved audit28 and interpret saved audit31 verify
their own complete authored rebuild pairs. No numerical probe, project import,
new physical sample, additional test suite or fresh reviewer was launched for
this note. Final S2 delivery still requires the remaining backend results,
same-run evidence audit, original reports and formal clean/process-zero release.
