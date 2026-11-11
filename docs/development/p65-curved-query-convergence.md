# COL1a curved-query convergence

Status: AUTHORING / UNEXECUTED. Baseline `63614090`; parent owns all execution. This is one bounded successor to the saved 110/122 interpret result, not new geometry acceptance.

## Evidence and choice

The saved SE overlap reaches GJK 2 / EPA 61 / support 77 / vertices 64 / faces 124 and returns EPA_CAPACITY. The stored ordinary-Y source case reaches GJK 7 with distance 0.0312520601, lower 0.0312518217 and normal z=-0.0031187863: its depth gate passes while the independent 0.002 normal gate fails. The other three layouts share that algorithmic issue. Existing BE closest-face and large-translation fixes remain intact.

Fixed local Newton 1.7.0 primary source, under the parent's `.venv-newton-1.7.0/Lib/site-packages/newton/_src/geometry`:

- `collision_convex.py:150-254`: shared relative-frame support with MPR overlap and GJK separation.
- `mpr.py:162-411`: constant-size portal refinement; its inflate and iteration-limit hit policies are not this query contract.
- `simplex_solver.py:380-583`: support-plane lower bound and relative Frank-Wolfe gap.

Copying an MPR hit does not establish this API's globally least penetration or unique normal. The selected change keeps the existing hull EPA fallback and adds a generic interior-ball bound at already charged support calls. No pair-name, test identity, world, fixed axis or capacity value selects a shortcut.

## Separated distance and normal

For a nonempty current Minkowski simplex, let q be its closest point, U=|q|, and w the support in -q. Both belong to the same current support problem. L=q.w/U is a global distance lower bound when positive. Let q* be the true nearest point. Projection optimality gives q*.q >= |q*|^2, hence cos(angle(q,q*)) >= L/U and

`normal_error <= sqrt(2*(1-L/U))`.

Evaluate the stable Frank-Wolfe numerator `q.(q-w)`, not subtraction of rounded squared lengths. Test the existing absolute 8e-7 depth gate and a stricter relative numerator gate of 1e-6 U^2, with a local FP32 arithmetic guard. This leaves room inside the unchanged independent 2e-3 normal gate. Publish that same q, its prior compact simplex weights, and its bilateral material witness; do not mix a new closest point with the previous direction's evidence. The first support call has no existing simplex and cannot claim this certificate.

For new full-surface queries a repeated support cannot bypass these gates. Failure remains unresolved/resource-limited; old ten-family core paths are unchanged. The existing touch/overlap routing still enters EPA.

## Interior-ball penetration bound

For unit n and exact support s=sA(n)-sB(-n), suppose the Minkowski body contains the tangent ball B(s-r*n,r). S and C contribute their declared round radius. For E with local semiaxes a, contribute `min(a)^2/max(a)`; Y, N and finite H contribute zero. Radii add under Minkowski sum; zero-radius shapes still contribute their actual support point. This is a shape property, not an SE/NE/EE pair dispatch.

For the ellipsoid support h(n)=sqrt(n^T Q n), tangent t perpendicular to n satisfies

`t^T Hess(h) t = min_alpha (t+alpha*n)^T Q (t+alpha*n) / h(n) >= min(a)^2/max(a) * |t|^2`.

Thus h-r|.| is convex homogeneous and the tangent ball lies inside the ellipsoid. This applies at every support direction, including nonprincipal directions.

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

Use pair-relative support coordinates throughout. The ball depth arithmetic uses the rationalized width above, avoiding cancellation of r and |s-r*n|. A local FP32 guard is charged to both depth endpoints and tangent residual: 8 unit-roundoffs times the two normalized pair-relative support L1 magnitudes plus r. This is a fixed engineering arithmetic allowance, not an interval-arithmetic or exact-native-rounding theorem. It is never a replacement for the original independent support/surface/normal gates. With guard e, require m>e, use tangent upper |v|+2e, depth lower d-e-(|v|+2e)^2/(2*(m-e)), upper d+e, and normal bound 2*(|v|+2e)/(m-e). Reserve part of the normal gate by requiring this bound <=1e-3. The original 2e-3 independent gate is unchanged.

All support calls count toward the existing loops. No additional support/refinement loop, CPU geometry, allocation, vertex/face default, source precision relaxation or device-to-HOST fallback is introduced. The new phase SUPPORT_BALL uses the existing bracket, witness and actual resource fields. Curved patch/load remain unsupported.

## Fixed tests and scope

First rerun the original twelve failing identities on their unchanged fixtures and gates; parent decides the final affected groups. Add only:

1. Rotated shallow NE and EE overlap using the existing independent tangent-ball oracle.
2. Isotropic E plus S with an oblique eccentric center: analytically two spheres, exact least depth and unique normal.
3. Same-center isotropic E plus S: all directions tie, no SUPPORT_BALL or invented unique normal.
4. Near-concentric eccentric E plus S with a one-support budget: the ball direction is too weak to certify; retain resource failure.
5. NN shallow overlap with four EPA vertices: no positive rolling radius, genuine EPA capacity negative.

Keep the existing NE/cap4 input. Resource limits constrain actual usage: it may now succeed by SUPPORT_BALL only if the original independent depth, normal, both support witnesses and <=4 actual vertices pass; otherwise retain the original RESOURCE_LIMIT outcome. Do not disable the algorithm when cap=4 just to reproduce an obsolete exhaustion expectation. GJK1/YE separation remains a mandatory negative.

Exact source scope: `_tyx_convex_query_kernels.py`, existing `_tyx_convex_support_family.py` only if needed, `_device_convex_query.py` for appended phase. Tests remain in the three COL1a new files; this note is the sole additional document. No patch, binding, source registration, original K0 fixture, schema or training/robotics change. All authoring checkpoints are UNEXECUTED until the parent's sole serial slot validates them.
