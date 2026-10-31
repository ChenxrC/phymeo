# SC2 first contact under the prescribed hinge-rate perturbation

This hand derivation bounds first contact in the zero-pressure material replay used to detect impact. It covers the three original time grids and all three prescribed velocity perturbations. A second result bounds the terminal impulse in the special case where that free replay reaches contact exactly at a grid endpoint. The bounds use the ordinary target with particle mass 1, robot mass 4, unit compliances and zero rest angle. They also hold with fixed unit compliances for uniform particle mass m in [1/2,2] and robot mass 4m. [Control and acceptance obligations](p65-b2-contact-integration-proposal.md).

Under the declared homogeneous scaling from unit mass, multiplying every mass by s and dividing every compliance by s divides both terms in each row denominator by s. Starting at zero multipliers, induction multiplies each multiplier and its increment by s and gives identical position increments. The terminal impulses below then scale by s as well. A common translation and a proper rotation applied to the entire setup preserve the estimates in patch-normal coordinates.

## Free material motion through the candidate impact step

Write u=1/8, g0=3/1024 and eta in {63,64,65}/4096 for the upward velocity of node d relative to the three other nodes. Use the original length, time and mass units. The robot is at rest in the free replay. Its square patch occupies the plane z=-rho, with rho=1/64. The initial four nodes share the translation g0 e_z. Define the ballistic comparison by translating every node through (g0-u t)e_z and adding eta t e_z to node d only. At a flat state the angle gradient contracted with the initial velocity is -eta.

Set T=1/32 for the free comparison, including the continuation beyond its first touch. Each of h=1/64, 1/128 and 1/256 divides T. In step k the predictor is y_(k-1)+h v_(k-1). Let d_k denote the complete finite material displacement relative to that predictor. Reconstructing velocity from the actual ideal displacement gives

    v_k - v_ballistic = sum_(l=1..k) d_l/h,
    y_k - y_ballistic(kh) = sum_(l=1..k) (k-l+1)d_l.

Use the four-node Euclidean product norm. For each step the multipliers start at zero. Any complete sweep count between 1 and 512 is allowed. Fix P=1/1800. If the predictor is within P of the translated flat reference, its initial constitutive residual is at most 3P. All four constraints vanish at that reference and their gradients have norm at most 3 throughout the intervening segment. Since P<1/128, the finite ordered-row bound then gives ||d_k||<=26h^2(3P). [Local geometry and sweep bounds, sections 1–6](p65-b2-local-row-derivative-bounds.md).

To prove that premise at every step, induct from the original velocities. The predictor error relative to its ballistic value contains only terms with l<k in the position sum. Bound these using preceding steps. Including the new displacement after its sweep gives the same conservative bound for the endpoint. For kh<=T the total error bound and the translated-reference check are

    E = 78P T(T+h_max)/2 = (117/2048)P < 1/30720,
    eta T + E < P,       h_max=1/64.

The second inequality follows from 65*1800 + 117*64 < 131072. This also starts the induction at k=1, when the predictor error is zero. The affine path between endpoints retains both position bounds because its ballistic comparison and reference translation are affine in time. Each physical segment velocity differs from its ballistic velocity by at most V=78P T=13/9600.

## The first-touch interval on all three grids

For clarity let b_i be the height of a middle node above z=0. Its lower prism vertex lies at height g_fi=b_i+c_f relative to the patch plane, with c_f=rho(1-n_f,z). On the smaller domain just proved the raw oriented area differs from e_z by at most 4P+4P^2<1/400. Both face normals are upward with ||n_xy||/n_z<1/399. Hence

    0<=c_f<1/(128*399^2),       E+1/(128*399^2)<1/30000.

Let gamma be the minimum across both faces and all six lower addresses. Each node has ballistic height at least g0-u t. Each face contains a node descending with the full speed u. The position and tilt bounds give

    |gamma(t)-(g0-u t)| < 1/30000     for 0<=t<=T.

This minimum equals the exact signed full-prism/patch gap minimized over the two pairs. Indeed each face has lower heights of absolute value less than 1/256, since |g0-u t|<=3/1024 and its middle-node deformation and tilt correction together are bounded by P+1/(128*399^2). Upper heights exceed 3/256 by n_z>1/2. All projected vertices lie strictly inside the square with coordinate magnitudes less than 33/32. The support-function calculation for the complete square now applies with |ell|<1/256, H=3/256 and M=31/32. It gives the unique maximizing direction +e_z for each prism. All tied lower addresses on that face belong to its original lower vertices. [Finite support calculation, section 12](p65-b2-sc1-active-material-reference.md).

There is a strict closing-speed margin along every open segment. Subtracting the common downward velocity leaves product norm at most eta+V<1/56. A face normal is invariant under common translation and its derivative norm is less than 4 in the outer domain. Thus each lower-vertex vertical speed, including the director derivative, is at most

    -u+(1+4rho)(eta+V) < -1/8+17/896 < -1/10.

Continuity and the finite piecewise-affine middle path extend strict descent across step boundaries. Therefore the minimum crosses zero at a unique first-touch time. Simultaneous owners may share that time. For L=47/2048 and U=49/2048 the common ballistic heights are +1/16384 and -1/16384 respectively. Their magnitudes exceed the gap uncertainty. Even the earlier gap exceeds the largest permitted h*tau because 1/16384-1/30000>2e-5/64. Both first entry into the contact budget and first zero gap are strictly inside (L,U).

The coarse grid therefore brackets free impact inside its second step. The finer grids have 3/128 as a grid boundary inside the bracket. They retain both adjacent step possibilities and exact endpoint arrival. This establishes a first-touch bracket for the perturbed, finite-sweep free replay independently of which support vertices attain the minimum.

The bend is active there. On the ballistic path its signed angle is -atan(eta t) with the original winding. For 0<=eta t<=65/131072 the derivative of atan is greater than 3/4. The angle error caused by the material displacement relative to ballistic is at most 3E, using the full gradient bound along the segment between the two configurations. For any t in [L,U],

    -theta(t) >= (3/4)(63/4096)(47/2048) - 3/30720 > 1/8192.

With unit compliance, at a free grid endpoint in this bracket whose exact final constitutive gate |theta+h^(-2)lambda_B|<=tau holds, the final bending multiplier exceeds h^2/10000. The statement about angle applies throughout the time interval, while the multiplier statement uses only such a grid endpoint.

## Terminal impulse for the endpoint-only branch

Suppose first zero gap occurs at a grid endpoint reached by the free replay with zero held transfer. Let I contain every touching owner. Use the current full triangle-offset Jacobian and the prismatic robot coordinate. With normal +e_z the relative normal velocity at owner i is B_i v. Here B_i includes the current director derivative on the face and robot coefficient -1. At this endpoint the reconstructed free velocity is the incoming segment velocity, so B_i v_minus < -1/10 for every i in I.

For the ideal perfectly inelastic, frictionless terminal law, project v_minus in the full kinetic metric M onto the closed cone B_i v>=0 for i in I. This cone contains zero. A common upward cloth velocity with stationary robot makes every inequality strict. Positive definite M yields a unique velocity minimizer. The linear constraints give nonnegative multipliers L_i supported on I, with

    v_plus = v_minus + M^(-1) B_I^T L,
    L_i (B_i v_plus)=0 for every i in I.

Keep separate material owners even when rows coincide. The summed impulse J=sum_i L_i admits bounds independent of its split. The cloth part of each row has norm at most 1+4rho=17/16, so every entry of B_I M^(-1) B_I^T has absolute value at most (289/256+1/4)/m=353/(256m)<3/(2m). Choosing any touching row gives J>m/15 from its required increase in speed. Even the weaker condition B_i v_plus>=-tau gives J>(2m/3)(1/10-tau)>m/16 for this pure-normal response.

Complementarity gives v_plus^T M(v_plus-v_minus)=0, so the projection decreases kinetic energy by ||v_plus-v_minus||_M^2/2. Before impact the cloth product velocity norm is bounded by 2u+eta+V<9/32 and the robot is stationary. Afterwards the robot speed has magnitude J/(4m). Consequently J<9m/16. These are impulse bounds for the ideal endpoint-only branch. They are independent of any selected solver output.

A step whose free endpoint penetrates before the terminal phase also includes a held replay from the original step entry. Its endpoint, active set and incoming terminal velocities need separate inclusion. Stored R7 motion, actual phase delivery, selected-pre binding, later contact iterations, the complete physical window and the original energy, angular and transaction checks remain open.


## Held pressure in the impact step

Let t0 be the left endpoint of the first grid interval whose free path reaches zero gap, and let t1=t0+h. The bracket gives 1/64<=t0<=3/128 and t1<=1/32 on each original grid. The step begins at the free entry. All six lower gaps are positive there. The horizontal square contains all six projections in its interior, so the open anchors have initial normal +e_z. Use all six candidate addresses for the existence comparison. The implemented workspace builds its held load from the entry Jacobians, and its terminal load from the Jacobians at the selected held endpoint. [Held and terminal loads](../../src/phymeo/solvers/_thick_contact_replay.py).

Fix any count N between 1 and 512 for this step. Let p_i>=0 denote the held normal impulse on owner i, J its sum and P_j its aggregate on physical node j. Work first on the compact simplex J<=m/3. In the Euclidean cloth coordinate space the initial normal row at a lower vertex of node j is the unit nodal vertical row plus a correction of norm less than 1/6000. Indeed the correction has norm rho ||D n_z||. Projection of e_z onto the plane perpendicular to n has norm ||n_xy||, and ||D(raw area)||/area<4. The entry slope bound 1/399 therefore gives rho ||D n_z||<1/6384. The robot coefficient is -1.

Relative to the zero-pressure predictor, the held cloth predictor moves by h B_cloth,0^T p/m, and the robot moves by -hJ/(4m). The product norm of the added cloth motion is at most h(1+1/6000)J/m. The free predictor lies within 1/1800 of the common translated flat reference by the preceding induction. Thus the held predictor's product deviation is below 1/160 throughout the simplex. Its initial constitutive residual is below 3/160, and all ordered sweeps together move the cloth by at most (78/160)h^2 relative to this predictor. In particular the material velocity correction has norm less than 1/128. These bounds cover every trial pressure in the simplex before any existence argument.

Let g_i(p) denote the exact endpoint lower gap after these N sweeps. The endpoint and its straight physical path from entry have cloth product deviation below 1/150 relative to the translating flat reference. In this domain the raw oriented area differs from e_z by at most 4/150+4/150^2<1/36, giving ||n_xy||/n_z<1/35 and n_z>9/10. All lower heights relative to the moving patch have magnitude less than 1/80, using

    3/1024 + 1/150 + h/12 + 1/(128*35^2) < 1/80.

All upper heights exceed 1/64, and projected coordinate magnitudes are less than 33/32. The square-support argument therefore gives the exact +e_z maximizing direction throughout the simplex and path. Bottom support subsets and tied owners can change.

For owner i on node j, write its entry lower gap as g_i^0. Its entry middle height is at least -1/(128*399^2), since g_i^0>=0. On the three grids, c_f^0/h is below 1/65536. The entry nodal vertical velocity is at least -1/8-V, the normalized predictor error from the initial normal-row correction is less than 1/18000, and the endpoint tilt correction is nonnegative. Combining these with the material velocity bound yields

    g_i(p)/h >= P_j/m + J/(4m) - 9/64.

Here 1/8+13/9600+1/65536+1/18000+1/128 < 9/64. Summing with weights p_i and using the four physical nodes gives

    p.g(p)/h >= sum_j P_j^2/m + J^2/(4m) - (9/64)J
              >= J [J/(2m)-9/64].

For fixed N the complete finite material map, current normals and gaps are continuous on this simplex. The auxiliary map projecting p-a g(p)/h onto the simplex is continuous for any a>0, so Brouwer supplies a fixed point. Its variational inequality is g(p).(q-p)>=0 for every q in the simplex. Testing q=0 gives p.g(p)<=0. At J=m/3 the lower bound is strictly positive since 1/6-9/64=5/192. The fixed point is therefore interior to the *sum* constraint. Increasing one entry at a time, and decreasing it wherever p_i>0, gives all the original unilateral conditions and the aggregate enclosure

    g_i(p)>=0,     p_i g_i(p)=0,     0<=J<=9m/32.

The simplex only establishes existence of an ideal finite-material held solution. The implemented solver still uses its own orthant projection and full-score backtracking.

For the h=1/64 impact step, t0=1/64 and t1=1/32. At zero held pressure the free comparison gives min_i g_i(0)/h < -3/50 for every N, since 1/16-64/30000>3/50. On the segment from 0 to p the finite material response changes by at most (1+95h^2) times the change in its predictor. The lower-point normal row has cloth norm at most 17/16 throughout. Including the robot translation, each endpoint gap divided by h changes by at most

    [(17/16)(1+95h^2)(1+1/6000)+1/4] J/m < 3J/(2m).

Applying this to a most penetrated free owner gives J>m/25 on the coarse grid. On each finer grid the same proof gives J >= (2m/3) max(0,-min_i g_i(0)/h), retaining the possibility of an endpoint-only event. [Finite material response, section 8](p65-b2-local-row-derivative-bounds.md).

## Coarse-grid nonpenetration and terminal response after held transfer

At the coarse entry every lower gap satisfies g_i^0/h>3/50 by the free bound at t0. Write c(s)=rho(1-n_z(s)) on one face of the affine path to a held solution, with s in [0,1] the path fraction. Subtracting the common downward velocity from the four-node path velocity leaves product norm less than

    eta + V + (1+1/6000)/3 + 1/128 < 3/8.

The norm of the derivative of c with respect to cloth position is below 1/560, using rho*4/35 and translation invariance. Hence its physical time derivative has magnitude less than 3/4480 along the segment. Its departure from the endpoint chord is bounded by comparing both terms to c(1),

    |c(s)-(1-s)c(0)-s c(1)| <= (1-s)h/700.

The nodal middle height and the robot plane follow their endpoint chords exactly. Therefore the lower gap along this physical path is bounded below by

    g_i(s) >= s g_i(p) + (1-s)(g_i^0-h/700).

Every lower vertex stays strictly above the patch until the endpoint. At the endpoint at least one owner touches because J>m/25. This establishes nonpenetration on the complete ideal coarse held segment.

At every touching owner, the incoming relative normal velocity uses the *endpoint* lower-point Jacobian applied to the reconstructed displacement velocity. Its difference from the gap secant is c_dot(1)-(c(1)-c(0))/h, with magnitude less than 1/700. Thus B_i v_minus < -3/50+1/700 < -1/18. This estimate uses the actual tilted endpoint map of the ideal held path.

Apply the terminal kinetic projection from the endpoint-only branch to all owners touching this held solution. Its summed impulse L is positive. Each normal response entry has absolute value below 3/(2m), hence L>m/27. Even a post-impact lower-velocity tolerance tau gives L>(2m/3)(1/18-tau)>m/28 for pure-normal delivery. An upper bound follows from the robot velocity after both phases, -(J+L)/(4m). The pre-terminal cloth speed has product norm less than 2u+3/8=5/8, and the robot speed is at most 1/12. The pre-terminal kinetic norm is therefore less than (2/3)sqrt(m). Terminal complementarity bounds the post-terminal kinetic norm by that same value and gives J+L<4m/3.

For the original exact controls, the entry bend is already active on every grid. The ballistic angle magnitude is atan(eta t0)>=(7/8)eta t0, while the configuration error changes it by at most 3E. Since t0>=1/64, their difference exceeds 1/10000 rad. This entry bound is independent of the terminal pressure split.

## Effect of stopping a trial at a different sweep count

Take a 512-sweep held solution from the existence argument. At its pressure an ideal trial stopped after M complete sweeps with R_M<=tau has endpoint displacement relative to sweep 512 of at most 26h^2 tau by the geometric tail of the ordered rows. The robot coordinate is identical at this fixed pressure. Each endpoint gap divided by h therefore changes by at most 28h tau. This supplies an approximate held comparison for other counts, subject to the residual gate. Contact membership and one-sided endpoint feasibility must be verified for the selected replay. The exact existence result permits multiple pressure splits.

The fine-grid held-to-terminal lower bound still depends on the true entry gaps near 3/128. Native replay and delivery, finite backtracking progress, switching sweep counts, continuation through the remaining window and original energy/angular and transaction gates remain open.


## Which fine-grid step contains the ideal impact?

Both fine grids contain t_b=3/128. The symmetric position envelope straddles zero at that time. The signed contribution of the bending row to the upper face's outer node c resolves the ideal event. Keep the three original eta values, uniform m in [1/2,2] with unit compliances and any choice of 1 to 512 complete sweeps per free step.

For h=1/128 or 1/256 and t_k=kh<=t_b, the free-motion induction bounds the distance between *every row stage* of step k and the ballistic endpoint by

    E_k = 39P t_k(t_k+h) <= (13/19200)t_k,     P=1/1800.

Indeed the weighted sum for previous completed steps is unchanged, and any partial sweep has the same cumulative motion bound 78P h^2 used for the whole step. In particular 3E_k < eta t_k/7. Every stage is within P of the common translated reference. The signed ballistic angle is -atan(eta t_k), and the largest ballistic constraint magnitude is at most eta t_k, including the a-d distance. Each ideal gradient norm is at most 3. Thus the entrance residual with fresh zero multipliers is below (6/5)eta t_k. At the last stage, -theta is at least (7/8)eta t_k-3E_k. Even one complete sweep gives |theta+h^(-2)lambda_B| <= (3/160)eta t_k. Combining these estimates gives

    lambda_B > (2/3)h^2 eta t_k,
    sum_s |Delta lambda_(j,s)| < (44/35)h^2 eta t_k for each row j.

The second bound uses the complete-history factor 22/21 from the local material analysis. It controls possible changes of sign of individual multiplier increments. [Row history and residual bounds](p65-b2-local-row-derivative-bounds.md).

Exactly two material rows act on c, the bend and the a-c distance. At the flat translated reference their c-vertical gradient components are -1 and 0. At every stage their deviations from those values are bounded by 33P and 3P respectively, by the full Hessian bounds. With d_k the material displacement from the free predictor, summing both ordered rows gives

    m (d_k)_(c,z) <= -lambda_B + (33+3)P (44/35)h^2 eta t_k
                  < -(3/5)h^2 eta t_k.

This signed displacement accumulates even when each physical step stops at a different sweep count. At t_b=nh the ballistic c-height is zero. Using sum_(l=1..n) l(n-l+1)=n(n+1)(n+2)/6 in the endpoint position identity gives

    y_(c,z)(t_b) < -eta h^3 n(n+1)(n+2)/(10m)
                 <= -eta t_b^3/(10m).

All three nodes of the upper face have the *same* ballistic height. Their actual height differences are generated solely by their deviations from that common value. The horizontal edge bound in SC1 section10 gives an upper-face slope at most 2E_n, so its lower-offset tilt correction is at most 2rho E_n^2 <= 1/(32*30720^2). Retaining this positive correction yields

    g_(U,c)(t_b) < -2^(-27).

For example eta t_b^3/(10m) >= 1701/(20*2^33), and after subtracting 2^(-27) the remaining 421/(20*2^33) exceeds the tilt bound. The free gap was strictly positive at L=47/2048 and decreases continuously. Thus on both fine grids the exact zero lies strictly in (L,t_b), in the step with entry t0=t_b-h and endpoint t_b. The endpoint-only analysis above remains applicable wherever the actual selected replay reaches that branch. In particular the guaranteed ideal depth 2^(-27) is smaller than h*tau for the native tolerance on both fine grids. Identifying the actual branch requires its stored geometry and phase evidence.

## Fine-grid held and terminal impulse bounds

At this entry all six lower gaps satisfy g_i^0/h > 1/8-1/(30000h)>1/9. At zero pressure the free endpoint penetration divided by h, D_0=max_i max(0,-g_i(0)/h), satisfies

    2^(-27)/h < D_0 < 1/(30000h).

The lower bound uses the U,c address for the same fixed-count replay. The upper bound uses the full free position enclosure at t_b. These statements hold for each choice of fixed N in the held existence argument, since the preceding signed proof covered every complete count.

There is also an aggregate upper bound in terms of D_0. On the established simplex, the initial cloth row differs from its vertical nodal row by less than 1/6000. Integrate the finite displacement correction derivative 95h^2 along the predictor segment. The endpoint tilt derivative has norm less than 1/560 throughout this domain. With J the summed held impulse and P_j the aggregate on physical node j, these three errors give

    |(g_i(p)-g_i(0))/h - (P_j+J/4)/m| <= beta J/m,
    beta = 1/6000 + (1+1/6000)[95h^2 + (1+95h^2)/560] < 1/32.

The 95h^2 term acts on the entire cloth predictor change, including the offset contribution. The robot translation is exact in this formula. At any unilateral solution with J>0, weight the lower estimates by p_i and use sum_j P_j^2>=J^2/4. Complementarity gives 0>=-J D_0+(1/2-beta)J^2/m. Combining with the earlier single-owner lower estimate yields, on the fine grids,

    (2m/3) 2^(-27)/h < J <= (32m/15)D_0 < m/54.

The chord and current-Jacobian estimates of the coarse-grid proof apply on the same pressure simplex. For each owner and path fraction s the lower gap satisfies g_i(s)>=s g_i(p)+(1-s)(g_i^0-h/700). Thus the full ideal held path stays strictly clear up to its endpoint. Since J>0 at least one owner touches there. At each touching owner the incoming current-map normal speed is below -1/9+1/700. With the same response bound 3/(2m), the ideal terminal projection transfers total impulse greater than m/14. The lower bound still exceeds m/14 for a pure-normal postcondition allowing speed down to -tau. This gives positive held and terminal impulse bounds and one-sided path feasibility for the ideal first impact on all three grids.

A computational pressure iteration still needs to reach an admissible candidate within the replay budget. Actual early stopping, R7 rounding, terminal membership and the selected-pre binding require their own error enclosures. The analysis here ends at the first impact and its ideal terminal velocity.


## Original-input energy up to the first terminal impact

All original forces, gravity and robot effort vanish in SC2. The joint energy is therefore cloth plus robot kinetic energy K and the elastic potential U from the three distance rows and the hinge. In the units above each unit-compliance potential is half the squared constraint. Use the actual endpoint displacement for the ideal reconstructed velocity, including every finite material update. [Composite energy definitions](../../src/phymeo/validation/composite_material_energy.py).

### Each preceding free step

Consider any free physical step ending at t_k<=t0. The established impact entries give t_k<=5/256 and t_k+h<=1/32 on every grid, including the single coarse free step. The stage bound E_k<=39P t_k(t_k+h) from the signed-bend calculation therefore applies. Its entrance residual satisfies R_0<(6/5)eta t_k<1/2600. Any complete ideal sweep count N>=1 gives R_N<57 h^2 R_0 from the complete-sweep factor 56h^2+80h^4. The local ordered-gradient drift bound then controls the full backward-Euler force residual F,

    F = (m/h)(v_1-v_0) + grad U(x_1),
    ||F|| <= (15/2)R_N+4576h^2 R_0^2 < h^2/6.

Both its terms have zero resultant. Each material row conserves total linear momentum, and U is invariant under translation. Thus F does work only on the velocity relative to the common translation. [Ordered material residual, section 6](p65-b2-local-row-derivative-bounds.md).

Along the physical segment x_0+s v_1, 0<=s<=h, subtract the common velocity -u e_z from every node and denote the resulting product velocity by w. The ballistic velocity error at this step has norm at most 78P t_k<1/1152, so ||w||<=eta+1/1152<1/59. The flat angle gradient G_0 has norm sqrt(6) and gives -eta on the ballistic velocity. The full angle Hessian bound 33 shows everywhere on the segment that

    |grad theta . v_1| >= eta - (5/2)/1152 - 33P/59 > 1/80.

All segment positions are within P of the translated reference. Along w, the potentially negative contributions C_j D^2 C_j to the elastic Hessian have combined magnitude at most (99+9 sqrt(2))P ||w||^2 < ||w||^2/16. This uses |C_bend|<=3P and ||D^2 theta||<33, together with |C_dist|<=sqrt(2)P and ||D^2 C_dist||<3 for each of the three lengths. Keeping the positive squared bend-rate term alone gives

    v_1^T D^2 U v_1 > 1/6400 - 1/(16*59^2) > 1/8000.

The endpoint Taylor identity for the potential and the kinetic difference now give

    Delta(K+U) = h F.v_1 - (m/2)||v_1-v_0||^2
                 - integral_(0..h) s v_1^T D^2 U(x_0+s v_1) v_1 ds
               < h^3/354 - h^2/16000 < 0.

The final sign uses h<=1/64. It holds for each free step independently and for every allowed finite sweep count.

### Energy across the ideal held and terminal phases

Take any of the fixed-count unilateral solutions proved above and its ideal inelastic terminal projection. Let A_0=B_0 M^(-1)B_0^T use the complete entry rows, including the robot. In the auxiliary velocity v_P=v_0+M^(-1)B_0^T p keep only the held update for this algebraic decomposition. Each entry normal speed is below -1/10. The direct kinetic difference for that auxiliary velocity is

    K(v_P)-K(v_0) = p.B_0 v_0 + (1/2)p^T A_0 p.

On all three grids D_0<1/15. On the coarse grid use D_0<1/16+64/30000 and on the finer grids D_0<256/30000. The earlier uniform response bound then gives J<32m/225<m/7. To estimate the quadratic term, write A_flat for the repeated nodal vertical matrix with the 4m robot. Complementarity p.g(p)=0 and the beta<1/32 response estimate give

    p^T A_flat p <= D_0 J + beta J^2/m.

For epsilon=1/6000 the entry cloth row differs from its nodal row by at most epsilon in product norm. The aggregate nodal load has norm at most J, and the change in that aggregate cloth load has norm at most epsilon J. Squaring their kinetic norms bounds |p^T(A_0-A_flat)p| by (2epsilon+epsilon^2)J^2/m. Since beta+2epsilon+epsilon^2<1/30,

    K(v_P)-K(v_0) < [-1/10+1/30+1/420]J < -J/16.

The finite material step adds d/h to the cloth velocity and leaves the robot velocity fixed. The stronger bound J<m/7 puts the predictor within 1/1800+(1+epsilon)h/7<1/350 of its translated flat reference. The ordered-motion bound gives ||d||<=78h^2/350, so ||d/h||<1/280 and the material endpoint is within 1/330. The cloth part of v_P has norm at most 2u+1/56+(1+epsilon)/7<5/12. Consequently the increase in kinetic energy from adding d/h is at most m[5/(12*280)+1/(2*280^2)]. Also U_1 <= 15/(2*330^2) by the three length gradient bounds and the bend gradient bound. Since m>=1/2, the sum of these two contributions is less than m/600. The entry elastic energy U_0 is nonnegative.

At any touching owner let b be the magnitude of its incoming negative normal velocity. The terminal projection increases this velocity by at least b. Weighted Cauchy-Schwarz and B_i M^(-1)B_i^T<3/(2m) give

    K(v_minus)-K(v_plus) = ||v_plus-v_minus||_M^2/2 > m b^2/3.

On the coarse grid J>m/25 and b>1/18, giving a complete step defect below -m/400+m/600-m/972 < -m/600. On the fine grids b>1/9-1/700>1/10, so the terminal loss exceeds m/300 and the complete step defect is again below -m/600. Positions, hence U_1, are fixed during the terminal projection. This is a joint original-input calculation in which the held and terminal transfers are internal.

For this exact prefix, the sum of positive per-step energy defects is zero on every grid because each step has its own strict bound. Homogeneous scaling from unit mass scales these energies by the same mass factor. Stored-state perturbations, interval widths, angular defects and steps after the first terminal response still require separate bounds within their original budgets.

### Transfer to stored endpoints

For the same masses and original zero loads, suppose independent evidence encloses a stored endpoint within velocity error e_v in the joint mass norm and cloth position error e_x in the product norm of its ideal comparator. Let D_x be the ideal cloth deviation from a common translation of the flat reference. The segment joining the two cloth states must lie in the original outer geometry domain. Since ||grad U||<=15D_x at the ideal endpoint and ||D^2 U||<19 there and along the segment, the absolute difference of original-input energies is at most

    T_j = ||v_j||_M e_v + e_v^2/2 + 15D_x e_x + (19/2)e_x^2.

Here v_j includes the robot coordinate. This follows by expanding the kinetic quadratic exactly and integrating the elastic Hessian. If a stored step's independent energy enclosure overestimates its exact defect by at most w_j, its positive upper defect is at most

    max(0, T_(j-1)+T_j+w_j-margin_j),

where margin_j=h^2(1/16000-h/354) on a free step and margin_j=m/600 on the impact step. Summing these positive parts keeps each endpoint and enclosure charge within its own step. The stored trajectory errors and enclosure widths still need to be established on the original inputs.

## 原 R7 的细网格自由分支在首碰前遇到范围拒绝

这是独立于 SC1 压力候选的第二个反例。取原 SC2 的 m=1、η=1/64、无重力、零压力，h=1/128 或 1/256。记 t=hη、L=th²。以下按 R7 已声明的最近偶数和正常数或零运算，并要求每条实际通过的材料行满足既有 τ=2×10⁻⁵ 角度审计。没有执行设备。[输入及分支](../benchmarks/p65-sc2-r7-free-range.json) · [有序材料行](../../src/phymeo/backends/_tyx_represented_material_kernels.py) · [独立角度检查](../../src/phymeo/validation/material_representation.py)

第一次自由预测在共同平移外只有 d 高 t。三条距离行的 FP32 长度均为 1，弯曲行的左/右法向原向量为 (0,0,1) 与 (0,t,1)，两长度都舍入为 1。t_c=t_d=0，四个梯度为 (0,t,2)、0、(0,0,−1)、(0,−t,−1)，分母 h⁻²+6。真实角为 −atan(t)；如果首行通过，独立账本给观测角与它的误差 ≤τ。连同实际分母和 Δ 的 FP32 舍入，首个弯曲增量 Δ/L 在 h=1/128 时属于 (0.83,1.17)，h=1/256 时属于 (0.66,1.34)。不需要给 atan2 编造逐位结果。

令 s=tΔ（乘二的幂，正常），u/v/w 分别为 a 的上移、c/d 的下移，均相对首个自由预测。它们距 2Δ、Δ、Δ 的单次位置写回误差分别不超过 L/128（h=1/128），或 L/32（h=1/256）。水平 y 修正为 (s,0,0,−s)，全部 x 不动。若同一物理步再扫一次取 ℓ=1；若首步扫一次就被接受，则下个自由预测取 ℓ=2。后者使用真实重建顺序：竖向 d−d_pred 为相近 FP32 数的精确差，除以二的幂 h，v_pred 加回得到可精确表示的 d/h，水平亦然；竖向参考加位移在本例 FP64 精确。两种情况下，第二次弯曲运算前的边向量都是 e=(1,−ℓs,−ℓu)、u_edge=(0,1,−ℓ(u+v))、v_edge=(−1,−1,ℓ(t−w))，其中接近 ±1 的水平分量按实际 FP32 舍入。距离行在这次仍无修正：最长竖向差不超过 2t，平方最多 2⁻²⁴，水平主轴为单位长度。到这个待评估材料行为止，两种分支的中面最低高度减去偏置 ρ 后，相对平台仍有大于 1/2048 m 的间隙。若过程先拒绝，也不能形成正控轨迹。

第二次弯曲前，e²、左法向长度和 er² 均舍入为 1。t_c 是 −ℓs+ℓ²u(u+v) 的 FP32 结果，负且非零；正项相对 ℓs 不到 0.003。累加 ev 的初值 −1，后两项分别过小，故 ev=−1、t_d=0。左法向 x 的幅值为 ℓu 加 ℓ²s(u+v) 的舍入，后项远小于前项。由此 b 的水平梯度 g_bx 正常且非零，幅值 <1.01ℓ²su。两档分别有 s<1.17tL、u<2.35L，或 s<1.34tL、u<2.72L；下界可取 tL²/4。因此 |g_bx| 分别 <12tL² 及 <15tL²，均严格小于 2⁻⁶³，但大于 2⁻¹²⁶。原分母要求物化 g_bx²，非零平方不能以正常 FP32 表示。按代码，若到达这一运算就无法返回完整有限的行记录。这里不预测运行时错误码。[带检查的乘法](../../src/phymeo/backends/_tyx_numeric.py) · [有限返回检查](../../src/phymeo/backends/tyx_math.py)

**决定。** 该障碍在布遇到平台以前出现，无需选择步内或终端接触分支。只更换压力迭代、只增加回溯或只接通材料分派，均不足以使原 R7 通过这两个必需细网格。理想首碰和能量结果仍是原有条件证据，本节没有把源码负控写成运行失败。
