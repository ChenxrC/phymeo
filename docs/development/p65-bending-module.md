# P6.5 B1 signed-dihedral bending code checkpoint

Status: **CODE_READY_TESTS_PENDING**. Source and independent tests are authored;
no test, numerical probe, solver import, compilation, lint/format execution,
build, dependency installation or CI run was performed in this lane. The human
approved parallel coding and a single serial execution queue. This is not a
historical TDD RED/GREEN claim, native capability or physics-stage acceptance.

Worktree: `C:/Users/38347/.codex/worktrees/p65-bending-module/phymeo`.
Branch: `codex/p65-bending-module`.
Frozen contract base: `1ccab18635a3b89c09a0c2e3f735bca2f823e65d`.
Ownership is limited to `constraints/bending.py`, `constraints/bending_trace.py`,
`tests/test_bending.py`, `tests/test_bending_trace.py` and this document. No shared
exports, distance law, runtime, native backend or material ledger are modified.

## Angle, derivative and units

For the ordered hinge `(a,b,c,d)`, the faces are `(a,b,c)` and `(b,a,d)`:

- `e=b-a`, `u=c-a`, `v=d-b`;
- `nL=e cross u`, `nR=(-e) cross v`;
- `theta=atan2(e_hat dot(nL_hat cross nR_hat), nL_hat dot nR_hat)`;
- `C=theta-rest_angle`, with no wrapping or rest-angle replacement.

The opposite-vertex gradients of this convention are
`gc=-|e|*nL/|nL|^2` and `gd=-|e|*nR/|nR|^2`.
Let `tc=(c-a) dot e/|e|^2`, `td=(d-a) dot e/|e|^2`.
The endpoint gradients are `ga=(tc-1)*gc+(td-1)*gd` and
`gb=-tc*gc-td*gd`. Rotation about the edge gives the negative signs at both
opposite vertices. The endpoint coefficients are obtained by differentiating
the oriented edge and the vertex projections; they enforce the translation
identity `sum(g)=0` and rotational identity `sum(x cross g)=0` in exact arithmetic.
Production uses these analytic expressions, with each face scaled before
cross products and divisions. Finite differences exist only in the tests.

A flat unit hinge has `ga=(0,0,2)`, `gb=0`, `gc=gd=(0,0,-1)`.
With `d=(0,0,-1)` its angle is `+pi/2` and compliance 2 gives energy `pi^2/16`.
Swapping `c,d` alone or `a,b` alone negates the angle. The complete consistent
face exchange `(a,b,c,d)->(b,a,d,c)` preserves it: both edge axis and normal
order reverse. The parent explicitly confirmed this convention.

Coordinates are metres; angle is radians (dimensionless for dimensional
analysis). Compliance has units `1/(N*m)=s^2/(kg*m^2)`; therefore
`E=C^2/(2*compliance)` is Joules. The caller supplies
`alpha=compliance/h^2`, with units `1/(kg*m^2)`. The gradient has units `1/m`,
so `sum(w*|g|^2)` has the same units as alpha. Lambda and delta-lambda have
units `kg*m^2`; `dx=w*delta_lambda*g` is metres. This multiplier is not a
world impulse and is not automatically an angular momentum.

## Supported discrete domain and explicit limits

Every configuration owns read-only FP64 rest/compliance arrays and canonical
int64 indices. Each row contains four distinct nonnegative indices. Duplicate
hinges are rejected using the unordered edge and unordered pair of opposite
vertices, independent of either orientation. Positive compliance is mandatory;
hard zero-compliance bending is outside B1. Mutable working arrays must be
finite FP64 NumPy arrays. Masses are nonnegative; alpha is strictly positive.
The implementation accepts disjoint strided publication targets and rejects
read-only, self-overlapping or input-aliasing mutable storage.

For each face, scale is the maximum absolute component of the two edges from
its first oriented vertex. The norm of their scaled cross product must exceed
`1e-10`, matching the explicit dimensionless offset-port admissibility rule.
There is no example-specific tolerance, clamping, normal fallback or normal
repair. Overflow, invalid arithmetic, division by zero and underflow in the
actual evaluation/update are rejected as nonrepresentable. Scaled geometry
avoids unscaled area overflow, but B1 does not promise every extreme input with
a mathematically finite result is evaluable through its FP64 intermediates.

**Parent branch ruling:** B1 certifies only each discrete projection's recorded
pre/post domain: both theta values are strictly in `(-pi,pi)` and both angular
errors satisfy `abs(C)<pi`. Additionally `abs(theta_after-theta_before)>=pi`
rejects an explicit atan2 jump. A returned trace's `branch_status="smooth"`
means exactly these checks. It does not prove the full affine correction segment
avoids degeneration or crosses the branch cut and returns. Such continuous-path
proof, if required by B2 or accepted step/CCD integration, remains a separate gate.

`bending_energy` and `bending_residual` apply the same evaluation domain. The
residual returns `max(abs(C+alpha*lambda)/max(abs(rest_angle),reference_angle))`
and `max(abs(C))` in radians, with positive finite `reference_angle` (default 1).
The existing normalized-error helper preserves positive residual underflow.
An empty constraint set has zero energy/residual and an empty trace tuple.
All-fixed soft hinges still update lambda and record a complete projection;
their positions remain unchanged. No fixed-node or hard-law fallback is added.

Invalid shapes/types/configuration are `ValueError`/`TypeError`. Unsupported
geometry/branch or numerical evaluation is `SolverFailure`. Input validation
and all local numerical work precede caller-state publication.

## Sweep and trace ownership

The sweep evaluates hinges in declared row order on private candidate copies.
It computes `dlambda=(-C-alpha*lambda)/(sum(w*|g|^2)+alpha)` and then applies
`dx=w*dlambda*g`. Every local pre/post state and gradient is owned by an
immutable `BendingProjectionTrace`. The record includes element/node identity,
angles, errors, old/new multiplier, delta, supplied alpha, actual denominator
and branch status. Trace order is complete element order; a future integrator
adds the sweep index. A later hinge failure discards all private updates and
publishes neither positions nor multipliers nor a partial returned trace tuple.
The final writes target validated disjoint mutable storage; concurrent external
mutation of these arrays is not supported.

Records describe FP64 operations and actual stored endpoints. They are not
exact-real-angle certificates, energy-acceptance flags, force/torque proofs,
continuous geometry proofs or a second time integrator. In particular,
`Fraction(float(atan2(...)))` would not certify an ideal exact dihedral.

## Authored checks awaiting the serial queue

`tests/test_bending.py` covers literal flat/right-angle values, all three
orientation permutations, rigid translation/proper rotation/scaling, independent
centered angle derivatives, translation and rotational identities, a two-hinge
shared-node energy derivative, constitutive residual normalization, all-fixed
soft behavior, owned configuration, duplicate/invalid inputs, dimensionless
face conditioning, antipodal/degenerate/extreme geometry and empty sets.

`tests/test_bending_trace.py` covers an independent centered-gradient two-hinge
reference sweep, full ordered trace replay, late second-hinge geometry/arithmetic
rollback, explicit branch jump, pre/post angular-error rejection, trace ownership
across later sweeps, malformed records, invalid input transactionality, supported
strided targets and rejected overlapping targets. These are independent local
hinge/sheet fixtures, not handoff-specific cloth/contact scenarios.

Every authored test is **UNEXECUTED**. Parent must select an immutable source,
grant the sole execution queue and record commands/backend/pins/raw exit and
fresh basetemp before any verification. No acceptance threshold was adjusted
from observed numerical results because no such result exists in this lane.

Native B2 bending, a composite distance/bending material path ledger, full
force/moment/power and NUM005 accounting, backend parity, coupled contact/grasp,
mesh/time-step/duration/repeat/perturbation/failure acceptance and folded-cloth
integration remain open. The existing distance-only ledger is unchanged and
must not accept these records until the separate composite protocol is proved.
