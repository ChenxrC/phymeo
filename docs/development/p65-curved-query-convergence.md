# COL1a curved-query convergence

Status: **PARTIAL_VALIDATION / NOT_FULLY_ACCEPTED**. Final retained product `0ea0fee2` keeps the interior-ball overlap refinement and withdraws the experimental strict GJK normal gate. Interpret: 127 passed / 4 failed; CUDA: 125 passed / 6 failed. The affected old query/patch/source regression passes 18 tests per backend. Full results and limitations are in the [parent report](p65-device-convex-support-family-report.md). Baseline `63614090` and the original authoring proposal below remain historical evidence.

The original eight smooth-EPA capacity cases and all nine added identities pass on both backends without larger capacities or relaxed geometric gates. Four cylinder-source normal failures remain on both backends; two HN separation cases additionally exhaust resources on CUDA. Curved patch/load remain unsupported. This is a bounded retained improvement, not completion of COL1a.

## Evidence and choice

The saved SE overlap reaches GJK 2 / EPA 61 / support 77 / vertices 64 / faces 124 and returns EPA_CAPACITY. The stored ordinary-Y source case reaches GJK 7 with distance 0.0312520601, lower 0.0312518217 and normal z=-0.0031187863: its depth gate passes while the independent 0.002 normal gate fails. The other three layouts share that algorithmic issue. Existing BE closest-face and large-translation fixes remain intact.

Fixed local Newton 1.7.0 primary source, under the parent's `.venv-newton-1.7.0/Lib/site-packages/newton/_src/geometry`:

- `collision_convex.py:150-254`: shared relative-frame support with MPR overlap and GJK separation.
- `mpr.py:162-411`: constant-size portal refinement; its inflate and iteration-limit hit policies are not this query contract.
- `simplex_solver.py:380-583`: support-plane lower bound and relative Frank-Wolfe gap.

Copying an MPR hit does not establish this API's globally least penetration or unique normal. The selected change keeps the existing hull EPA fallback and adds a generic interior-ball bound at already charged support calls. No pair-name, test identity, world, fixed axis or capacity value selects a shortcut.

## Withdrawn experiment: separated distance and normal

This proposed gate is **not in the final retained source**. Targeted run24 passed 21 cases and left two floating-cylinder resource failures. The disjoint remaining matrix in run25 exposed ten new separation resource regressions. Parent removed only this gate and restored the prior distance/duplicate-support termination, leaving the interior-ball code unchanged. The mathematical relationship below does not establish that the FP32 simplex iteration can reach its stricter gate within the existing budget. No output-dependent guard adjustment or extra iteration budget was used. A subsequent attempt must first capture and diagnose failed simplex/support/weight/FW state; the present saved payload does not establish an arithmetic root cause.

For a nonempty current Minkowski simplex, let q be its closest point, U=|q|, and w the support in -q. Both belong to the same current support problem. L=q.w/U is a global distance lower bound when positive. Let q* be the true nearest point. Projection optimality gives q*.q >= |q*|^2, hence cos(angle(q,q*)) >= L/U and

`normal_error <= sqrt(2*(1-L/U))`.

Evaluate the stable Frank-Wolfe numerator `q.(q-w)`, not subtraction of rounded squared lengths. Test the existing absolute 8e-7 depth gate and a stricter relative numerator gate of 1e-6 U^2, with a local FP32 arithmetic guard. This leaves room inside the unchanged independent 2e-3 normal gate. Publish that same q, its prior compact simplex weights, and its bilateral material witness; do not mix a new closest point with the previous direction's evidence. The first support call has no existing simplex and cannot claim this certificate.

For new full-surface queries a repeated support cannot bypass these gates. Failure remains unresolved/resource-limited; old ten-family core paths are unchanged. The existing touch/overlap routing still enters EPA.

## Interior-ball penetration bound

For unit n and exact support s=sA(n)-sB(-n), suppose the Minkowski body contains the tangent ball B(s-r*n,r). S and C contribute their declared round radius. For E with local semiaxes a, contribute `min(a)^2/max(a)`; Y, N and finite H contribute zero. Radii add under Minkowski sum; zero-radius shapes still contribute their actual support point. This is a shape property, not an SE/NE/EE pair dispatch.

For the ellipsoid support h(n)=sqrt(n^T Q n), tangent t perpendicular to n satisfies

`t^T Hess(h) t = min_alpha (t+alpha*n)^T Q (t+alpha*n) / h(n) >= min(a)^2/max(a) * |t|^2`.

Thus h-r|.| is convex homogeneous and the tangent ball lies inside the ellipsoid. This applies at every support direction, including nonprincipal directions. Equivalently the Minkowski body is L plus a radius-r ball for a convex L. A checked d<r gives h_L(n)<0, so the origin lies outside L. The closest point of convex L is unique; its direction is the unique penetration normal. The separate angular bound controls approximation to that direction, while d>=r provides no such uniqueness argument.

Set d=n.s, tangent v=s-d*n and m=r-d. For m>0:

`lower = r-sqrt(m^2+|v|^2) = d - |v|^2/(sqrt(m^2+|v|^2)+m)`,
`upper = d`.

A strictly positive lower bound proves actual interior overlap. A tight interval alone does not certify a normal. The inner ball also implies a bound on every minimum-depth normal:

`normal_error <= 2*|v|/m`.

Use this separate angular gate, the unchanged 8e-7 depth-width gate and complete bilateral surface/closure gates. When m is small, zero or invalid, or the lower bound is nonpositive, there is no shortcut. Concentric equal-depth directions therefore cannot become a fabricated unique normal. A tilted support on a face/edge is accepted only if these global bounds and its real bilateral support points suffice; otherwise original EPA tie handling remains authoritative.

## Stored frames and FP32 engineering boundary

The uploaded/effective R is not assumed exactly orthogonal. Bound the symmetric defect R^T R-I by its maximum absolute row sum eta, padded by 2^-18. If eta<1/4, an ellipsoid rolling-radius lower estimate is

`min(a)^2*(1-eta)/(max(a)*sqrt(1+eta))`.

A final factor 1-2^-20 rounds the radius inward. Invalid or excessive defect disables this acceleration; it does not broaden source admission.

Use pair-relative support coordinates throughout. The ball depth arithmetic uses the rationalized width above, avoiding cancellation of r and |s-r*n|. A local FP32 guard is charged to both depth endpoints and tangent residual: 4 unit-roundoffs times the sum of the two normalized pair-relative support Euclidean norms and r. This is a fixed engineering arithmetic allowance, not an interval-arithmetic or exact-native-rounding theorem. It is never a replacement for the original independent support/surface/normal gates. With guard e, require m>e, use tangent upper |v|+2e, depth lower d-e-(|v|+2e)^2/(2*(m-e)), upper d+e, and normal bound 2*(|v|+2e)/(m-e). Reserve part of the normal gate by requiring this bound <=1e-3. The original 2e-3 independent gate is unchanged.

All support calls count toward the existing loops. No additional support/refinement loop, CPU geometry, allocation, vertex/face default, source precision relaxation or device-to-HOST fallback is introduced. The new phase SUPPORT_BALL uses the existing bracket, witness and actual resource fields. Curved patch/load remain unsupported.

## Fixed tests and original authoring scope

First rerun the original twelve failing identities on their unchanged fixtures and gates; parent decides the final affected groups. Add only:

1. Rotated shallow NE and EE overlap using the existing independent tangent-ball oracle.
2. Isotropic E plus S with oblique eccentric and contained centers: analytically two spheres, exact least depth and unique normal. The contained positive also has a uniform 4x scale variant; a 4x eccentric one-support case has guard width strictly above 8e-7 and must remain resource-limited.
3. Same-center isotropic E plus S: all directions tie, no SUPPORT_BALL or invented unique normal.
4. Near-concentric eccentric E plus S with a one-support budget: the ball direction is too weak to certify; retain resource failure.
5. NN shallow overlap with four EPA vertices: no positive rolling radius, genuine EPA capacity negative.

Keep the existing NE/cap4 input. Resource limits constrain actual usage: it may now succeed by SUPPORT_BALL only if the original independent depth, normal, both support witnesses and <=4 actual vertices pass; otherwise retain the original RESOURCE_LIMIT outcome. Do not disable the algorithm when cap=4 just to reproduce an obsolete exhaustion expectation. GJK1/YE separation remains a mandatory negative.

Exact source scope: `_tyx_convex_query_kernels.py`, existing `_tyx_convex_support_family.py` only if needed, `_device_convex_query.py` for appended phase. Tests remain in the three COL1a new files; this note is the sole additional document. No patch, binding, source registration, original K0 fixture, schema or training/robotics change. All authoring checkpoints are UNEXECUTED until the parent's sole serial slot validates them.


Before any execution, parent approved replacing the initially proposed 8u/L1 engineering allowance with fixed 4u/Euclidean norms. The 8u proposal made independently exact normalized spheres fail only its bookkeeping margin. This is explicitly an engineering choice, not a directed-rounding proof. No later test-output-dependent guard reduction is allowed. A false success or independent normal-gate failure stops this acceleration's expansion for this batch. Nine new TYX identities are authored; not collected or executed.

## Historical authoring handoff

Tests-first ec8f56b0 and 44bf7c48 fix nine extra identities before product execution. Implementation checks the new ball condition at both existing GJK and EPA support sites. A ball success uses one real bilateral support pair; existing EPA consumption is retained if refinement already happened. GJK publishes its old compact simplex only after the same iteration's new support establishes both bounds. Old ten-family exits and EPA fallback equations are unchanged. Radius work is skipped before an existing source-precision rejection, so it cannot divide by an already lost uploaded dimension.

No project commands have run in this lane. Parent should first run static checks on the three changed product files and changed test file, then the original twelve failures plus the nine new identities, followed by the parent-selected affected interpret/CUDA groups. Existing source fixtures, previous failures and raw artifacts remain intact. If actual false success or independent normal error appears, keep that evidence and stop expansion of this acceleration for this batch; do not reduce the fixed guard or alter geometry/caps to turn it green.
