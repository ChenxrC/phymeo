# B2 local geometry and exact row-derivative bounds

Status: HAND_DERIVED_LOCAL_BOUNDS_ONLY. No project execution, measured convergence, new test result or physical implementation grant. The unchanged ordinary SC1/SC2 targets in [the proposal](p65-b2-contact-integration-proposal.md) still need their complete trajectory/pressure inclusion and original energy/angular/native gates. This note supplies explicit local derivatives for that analysis; it does not assert that either target stays in the neighborhood.

## 1. Declared neighborhood and units

Use the original metre-scale hinge reference a=(0,0,0), b=(1,0,0), c=(0,1,0), d=(0,-1,0). For an arbitrary fixed common translation z, define the convex neighborhood

    x_i = X_i + z + d_i,  ||d_i||_2 <= delta = 1/64 m, for every i.

This is an independently selected ANALYSIS domain, not a new fixture, solver clamp or accepted trajectory box. A missing inclusion cannot be replaced by this definition. Constants below are numerical in the original 1 m length unit. The four-node norm is the Euclidean product norm; a unit directional perturbation means sum_i ||v_i||^2=1. Gradients of theta have units 1/m, Hessians 1/m^2.

Write e=x_b-x_a, u=x_c-x_a and w=x_d-x_a. Each differs from its reference vector by at most 2 delta. The right oriented raw area is (-e) cross (x_d-x_b) = -e cross w. Therefore both face areas can be bounded with length-one reference factors; the longer vector x_d-x_b is not incorrectly assumed to have length one.

For e,u,w,

    31/32 <= ||e||, ||u||, ||w|| <= 33/32.
    ||e cross u - e_z|| <= 4 delta + 4 delta^2 = 65/1024.
    ||-e cross w - e_z|| <= 65/1024.

Each raw area A is in [959/1024,1089/1024], its z component is >=959/1024, and its unit normal has positive z component. Edge length and area stay uniformly away from zero. A face quality bound is

    A / max(||edge||^2) >= (959/1024)/( (sqrt(2)+1/32)^2 ) > 1/4,

where the possible triangle edge c-b or d-b is included. This is far stronger than the original 1e-10 face threshold but does not change that threshold. For example sqrt(2)<3/2 gives denominator<(49/32)^2 and the ratio>959/2401>1/4.

Each unit normal has horizontal/z ratio at most 65/959 < 1/14. The two normals thus remain within a common small hemisphere, with positive mutual dot product. The stated oriented atan2 angle and its gradient stay on the original smooth branch throughout the convex neighborhood. This establishes a branch for differentiation, not an offset-prism contact/support certificate.

## 2. Bending gradient and Hessian

Use exactly the analytic law in [bending.py](../../src/phymeo/constraints/bending.py), interpreted as an ideal smooth function before storage rounding:

    t_c = (u dot e)/(e dot e),  t_d = (w dot e)/(e dot e),
    g_c = -||e|| n_L/A_L,  g_d = -||e|| n_R/A_R,
    g_a = (t_c-1)g_c + (t_d-1)g_d,
    g_b = -t_c g_c - t_d g_d.

The reference dot products u_0 dot e_0 and w_0 dot e_0 are zero. Hence

    |t_c|,|t_d| <= (65/1024)/(961/1024) = 65/961 < 1/14,
    ||g_c||,||g_d|| <= 1056/959 < 9/8,
    ||g_a|| < 135/56 < 5/2,
    ||g_b|| < 9/56 < 1/6,
    ||G_theta||^2 < 25/4 + 1/36 + 2(9/8)^2
                    = 2537/288 < 9.

In particular ||G_theta||<3. This is the full four-node gradient, including shared-node terms.

For any product-norm unit direction, ||e'||,||u'||,||w'||<=sqrt(2), so

    ||(e cross u)'||, ||(-e cross w)'|| <= 2 sqrt(2)(33/32) < 3,
    |A'| < 3,  |(||e||)'| <= sqrt(2) < 3/2.

Differentiate -||e|| n/A. The normal derivative satisfies ||n'||<=||c'||/A. Consequently

    ||g_c'||,||g_d'|| <= (3/2)(1024/959)
                         + 2(33/32)3(1024/959)^2
                       = 7961088/919681 < 9.

The quotient derivative for either t is bounded by

    |t'| < 3/(31/32)^2 + 2(1/14)(3/2)/(31/32)
          = 3072/961 + 48/217
          = 22992/6727 < 7/2.

It follows that

    ||g_a'|| < 7(9/8) + 2(15/14)9 = 1521/56 < 28,
    ||g_b'|| < 7(9/8) + 2(1/14)9 = 513/56 < 10,
    ||H_theta v|| < sqrt(28^2+10^2+9^2+9^2)
                   = sqrt(1046) < 33.

Thus ||H_theta||_2<33 m^-2 on the complete neighborhood. These are conservative analytic inequalities, not finite differences or interval runs.

Integrating the gradient along the segment from X+z gives

    |theta(x)| <= 3 ||d|| <= 3(2 delta) = 3/32 rad.

For SC1 and BOTH declared rest perturbations, |r|<=2^-10+2^-12=5/4096; SC2 has r=0. Therefore

    |C_bending| = |theta-r| <= 389/4096 < 1/10 rad.

This includes the original rest values and perturbations without replacing the actual strained trajectory by its unstrained reference.

## 3. Distance rows and material energy

For each original distance edge (a,b),(a,c),(a,d), the ideal C=ell-1 satisfies

    |C| <= 1/32 m,  ||G||^2=2,
    ||H_C|| = 2/ell <= 64/31 m^-1.

At the unit-mass targets, each original distance compliance is 1 m/N and the bending compliance is 1/J. For U=sum_j C_j^2/(2 alpha_j),

    Hess U = sum_j (G_j G_j^T + C_j H_j)/alpha_j.

The full shared-node assembly is retained. The triangle inequality yields the following operator bound in N/m:

    ||Hess U|| <= 3(2 + (1/32)(64/31)) + 9 + 33/10
                 = 192/31 + 123/10
                 = 5733/310 < 19.

This is an absolute Hessian bound; no convexity of the elastic energy alone is assumed.

For a FIXED effective-force input and a FIXED predictor, the backward-Euler material function

    Phi(x) = (x-x_hat)^T M (x-x_hat)/(2 h^2) + U(x)

has Hessian >=(h^-2-19)I at unit particle masses. On all original h<=1/64, this lower bound is >=4077 N/m. Therefore, for two points x,y whose connecting segment stays in this neighborhood,

    (grad Phi(x)-grad Phi(y)) dot (x-y) >=4077 ||x-y||^2,
    ||x-y|| <= ||grad Phi(x)-grad Phi(y)||/4077.

IF a root y of the same fixed-input material residual is independently proved to exist in the neighborhood, the residual at x gives this conditional error-to-root bound. It proves uniqueness in the convex neighborhood. It does NOT prove existence or invariance of that neighborhood.

Under the explicitly homogeneous scaling M -> s M, alpha_j -> alpha_j/s, the lower bound becomes 4077 s. This sentence is conditional on that declared scaling; it does not silently assign a compliance change to any independently specified mass-only variant. A constant prismatic coordinate with mass4 and no elastic term can be appended to this fixed-input Hessian statement; configuration-dependent/general robot mass is outside it.

The actual contact/material solve does not hold its pressure input fixed. Changes in pressure, the held map, final gap derivatives, redundant owners and branch/birth conditions require their own joint residual/Jacobian inclusion. The fixed-input inequality cannot be substituted for that KKT problem.

## 4. Derivative of one ideal ordered XPBD row

Retain the COMPLETE current configuration x and current row multiplier lambda. For constant W and physical compliance alpha>0, set

    a=alpha/h^2, g=grad C(x), H=Hess C(x),
    s=g^T W g, d=a+s, r=C(x)+a lambda,
    Delta=-r/d, x_plus=x+W g Delta, lambda_plus=lambda+Delta.

At fixed lambda,

    grad_x s = 2 H W g,
    grad_x Delta = -g/d + (2r/d^2) H W g,
    partial_lambda Delta = -a/d.

Consequently the exact block derivative is

    D_x x_plus = I + Delta W H + W g (grad_x Delta)^T,
    partial_lambda x_plus = -(a/d) W g,
    D_x lambda_plus = (grad_x Delta)^T,
    partial_lambda lambda_plus = s/d.

These are ideal-arithmetic derivatives of the precise row update. All other row multipliers are unchanged in this one-row map. The next row sees the changed x; the full finite ordered sweep derivative must compose these blocks, including every original distance and bending row and its multiplier history.

At unit W=I and h<=1/64, a>=4096. For the bending row,

    s<9,  partial_lambda lambda_plus <9/4105,
    ||grad_x Delta|| <= 3/d + 198 |r|/d^2,
    ||D_x x_plus-I|| <= 33 |Delta| + 3 ||grad_x Delta||.

For each distance row, s=2, so partial_lambda lambda_plus=2/(a+2)<=2/4098; its remaining blocks use ||g||=sqrt(2), ||H||<=64/31 in the exact formulas above. Units of each block follow its row's distance/angle and multiplier units; these blocks must not be combined into a dimensionless norm without explicitly scaling those coordinates.

The small scalar multiplier derivative is NOT a contraction of the joint x/lambda sweep. The x block contains I, mixed blocks couple shared nodes and every row, and r/Delta still need independent bounds. Neither these formulas nor the positive-compliance denominator prove that512 iterations converge, that the outer contact replay converges, or that native rounding follows the ideal map.

## 5. Remaining connection to actual observations

Use the [ordered residual identities](p65-b2-contact-discrete-residual-identities.md) to retain epsilon transitions, observed-gradient errors, final-gradient drift, lambda discrepancies, predictor error and velocity reconstruction. The bounds here can constrain independently enclosed ideal gradients/Hessians only after all actual trace configurations are proved to lie in the declared neighborhood. Native R7 raw/decode/parameter/rounding charges stay separate and noncancelling.

Still unresolved for BOTH complete unchanged controls:
- actual full-window trajectory/impulse inclusion, all perturbations/mass variants and h grids;
- nonlinear offset geometry, whole-path support/birth signs and physical pressure/direction uncertainty;
- complete finite ordered-map and joint contact derivative/error bounds, including multiplier bounds and proof of domain invariance;
- original per-step and whole-window energy/angular ceilings, NUM005 and paired robot moments;
- actual native finite-iteration evidence and typed shared integration.

No full SC1/SC2 success or physical grant follows from this local lemma. No test, import, arithmetic probe, formatter, lint, compiler or build was executed to obtain it.

## 6. Conditional ideal material-sweep convergence

The earlier bounds also yield a concrete result for the EXACT-ARITHMETIC material replay under an additional predictor hypothesis. This is not a claim about any stored/native trajectory or the outer pressure/gap solve.

Use length L0=1 m, mass m0=1 kg and time unit1 s. Normalize each distance constraint by Cref=1 m and bending by Cref=1 rad. Write xi=x/L0, c_j=C_j/Cref_j, g_j=grad_xi c_j, H_j=Hess_xi c_j, Wbar=m0 W and dimensionless multiplier

    ell_j = lambda_j Cref_j/(m0 L0^2).
    a_j = alpha_j m0 L0^2/(Cref_j^2 h^2).

For the unchanged unit physical compliances, all a_j=A=h^-2>=4096 numerically. These definitions justify comparing distance and angular residuals in one dimensionless norm; their physical units are not identified.

Assume constant particle masses between1/2 and2 kg, keeping these unit compliances for this conditional statement, so eigenvalues of Wbar and Mbar are in[1/2,2]. Unit base masses satisfy this hypothesis. This does not change the interpretation of any already declared mass/compliance variant. A separate explicitly homogeneous mass/compliance scaling can instead be reduced to unit mass by its original homogeneity law.

Assume the ideal predicted configuration xi_hat is within1/128 of every translated reference node, and all four multipliers start at zero. Apply exactly the original three distance rows and one bending row once per sweep, in any fixed order, with the exact row equations of section4. All h, rows and rest perturbations are unchanged.

Define the dimensionless maximum row residual R=max_j |c_j+A ell_j|. At the predictor,

    R0 <= 3/64 + 5/4096 = 197/4096 < 1/20 < 1/8.

The distance residuals are <=1/64; the displayed larger bound covers the bending row and every declared rest perturbation.

### One row and one complete sweep

Let R be the residual maximum at a sweep entrance, R<=1/8. For one ideal row i, d_i=A+g_i^T Wbar g_i>=A and

    |Delta ell_i| <= R_stage/A,
    ||Delta xi|| <= 6 R_stage/A.

For any other row j, its multiplier is unchanged. Provided the row segment is in the neighborhood, the mean-value theorem gives

    |Delta(c_j+A ell_j)| <= 18 R_stage/A.

For the updated row itself the linear term cancels exactly:

    c_i(xi+Wbar g_i Delta ell_i)+A(ell_i+Delta ell_i)
      = c_i(xi+Wbar g_i Delta ell_i)-c_i(xi)
        -g_i^T Wbar g_i Delta ell_i.

Its absolute value is <=594 R_stage^2/A^2, using ||H_i||<=33 and ||Wbar g_i||<=6. There is no assertion that its nonlinear remainder is zero.

For all four row entrances and exits,

    R_stage <= (1+18/A)^4 R < (33/32)R.

One convenient proof is (1+x)^4<=1/(1-4x) for 0<=4x<1; at x=18/4096 this is512/503<33/32. The updated-row quadratic bound is smaller than R_stage in the stated range, so it also obeys this stage bound.

Each row has at most three later rows in the sweep. With u=33/32,

    R_next <= [54u/A + 594u^2 R/A^2] R
            <= [56/A + 80/A^2] R
            < R/64.

The final inequality follows already at A=4096:
56/4096+80/4096^2=14341/1048576<1/64.
This is a contraction of the COMPLETE four-row constitutive residual after a sweep, not a claim that the full unconstrained x/ell Jacobian has norm<1.

### Neighborhood invariance and finite history

The total product-norm configuration motion in one sweep is bounded by

    4*6*u R/A = (99/4)R/A.

Summing the geometric sweep bound gives

    sum_all_rows ||Delta xi|| <= (99/4)(64/63) R0/A
                              = (176/7) R0/A <26 R0/A.

Even the weaker R0<=1/8 gives26/(8*4096)=13/16384<1/128. Thus a predictor in the inner per-node1/128 neighborhood cannot leave the outer per-node1/64 neighborhood during these row segments.

This can be proved by induction without assuming the desired invariance: previous complete sweeps obey the geometric residual bounds; the partial current sweep has at most the four stage bounds above. Their cumulative displacement is below the available1/128 margin. The next entire affine row segment is therefore inside the convex neighborhood where its gradient/Hessian bounds apply. No physical-time path is inferred from this computational sequence.

For every row separately,

    sum_all_sweeps |Delta ell_j| <= u R0/[A(1-1/64)]
                                 = (22/21) R0/A.

The sequence of ideal configurations and multipliers is Cauchy. For a finite N sweeps,

    R_N <=64^-N R0.

In particular N=512 gives the exact ideal upper bound2^-3072 R0. No executable result, finite-precision residual or underflow behavior is asserted by that expression. Rounded native updates can have a nonzero error floor and still require their original independent evidence.

### Ordered-gradient drift remains nonzero

At any finite end configuration xi_N, define eta_N from the full ideal row history as in the earlier residual identity. With zero initial multipliers and exact arithmetic there are no conversion/predictor/lambda discrepancies in this particular ideal theorem, but changing gradients remain:

    xi_N-xi_hat = Wbar sum_j g_j(xi_N) ell_j,N + eta_N.

The three distance Hessian norms are each <3 and bending is <33. Combining their sum<42 with the displacement and per-row absolute multiplier bounds yields

    ||eta_N|| <=2 *42 *(26 R0/A)*((22/21)R0/A)
               =2288 R0^2/A^2.

Thus residual contraction does NOT identify the ordered-XPBD limit with the backward-Euler material root. The nonzero drift term is retained explicitly; it is not discarded because lambda updates have converged.

For the dimensionless fixed-predictor material force residual,

    R_force,N = Mbar A(xi_N-xi_hat) + grad Ubar(xi_N),

the complete identity gives

    ||R_force,N|| <= (15/2) R_N +4576 R0^2/A,

because the sum of the four gradient norms is <3 sqrt(2)+3<15/2 and ||Mbar||<=2. This is an absolute upper bound, not measured error, external work or an energy credit.

### A nearby fixed-input material root

The same predictor hypothesis also proves existence of a root for this FIXED-predictor material problem. On the product neighborhood, Phi has Hessian >=mu I with

    mu=A/2-19>=2029.

The Euclidean product ball of radius1/128 about xi_hat is entirely within the outer per-node1/64 neighborhood. At its center,

    ||grad Phi(xi_hat)||=||grad Ubar(xi_hat)||<=(15/2)R0.

At a boundary displacement d of length1/128, strong monotonicity gives

    grad Phi(xi_hat+d) dot d
      >= mu ||d||^2 - (15/2)R0 ||d|| >0.

The last strict inequality follows from mu>=2029 and R0<1/20. The continuous Phi attains a minimum on that compact ball; the outward-gradient inequality excludes every boundary point, so an interior critical point exists. Strong monotonicity makes it unique in the convex outer neighborhood.

Calling that root xi_BE,

    ||xi_N-xi_BE||
      <= [(15/2)64^-N R0 +4576 R0^2/A]/(A/2-19).

For all stated A>=4096 and N=512 this is <2^-19 in the original metre length scale; already replacing64^-N by64^-2 and R0 by1/20 gives a strictly smaller bound. This is a conditional ideal finite-replay fidelity enclosure around a separately defined fixed-input material root. It does not prescribe or correct a solver state.

### Scope still missing from SC1 and SC2

The theorem starts with an ideal predictor in the stated inner neighborhood and a fixed effective input; it does not prove that either full physical target, every outer contact trial or every later physical step has that predictor. The outer held pressure and offset geometry change the input and gap derivatives, and SC2 birth/terminal branches remain to be enclosed. The fixed-input root is not the joint contact KKT solution or a continuum trajectory.

All actual R7/native gradient, parameter, update, conversion, predictor and reconstruction errors must be propagated through this finite history. The direct original-input energy/angular gates, support and whole physical path, paired robot moments, nonlinear phase capacity and finite shared resource budget are still separate. No contact tolerance is replaced by2^-19 m, and the older1e-8/2e-5 normalized or energy/angular limits are unchanged.

The new result narrows the earlier open derivative/iteration question only for this explicit ideal fixed-predictor subproblem. It supplies no new execution authorization or product/native pass.

## 7. Conditional material compliance for a fixed normal map

This section adds a hand-derived pressure-response statement for the fixed-input material root of section6. It is not a proof for the actual changing offset gap, the finite native solver or either complete SC1/SC2 window.

Use the same dimensionless length, mass, constraint and multiplier conventions as section6, and explicitly introduce the unchanged time unit t0=1 s. Write hbar=h/t0, so A=hbar^-2. Normalize an impulse by m0 L0/t0. The original four material rows, unit physical compliances, rest perturbations and cloth mass interval[1/2,2] kg are unchanged.

Let B_n be a FIXED, nonzero, predeclared normal-direction map containing every contact owner row. It is selected from the mathematical input before observing any delivery, only for the already declared frictionless candidate. It does not depend on the pressure variable in this section. No unrestricted three-dimensional quotient representative is substituted for a normal pressure.

One may append the original constant-mass prismatic translation coordinate: the material potential is independent of that coordinate, its mass is positive, and its root equals its predictor exactly. For the uniform bounds below require that additional dimensionless mass to be at least1/2; the original4m prismatic mass with m>=1/2 satisfies this. The lower mass bound and ||Wbar||<=2 then remain valid, although the enlarged Mbar need not have norm<=2. Section6's finite material-history bound is applied only to the cloth coordinates. This argument does not apply to a general robot with configuration-dependent mass.

### Predictor family and derivative of its material root

For a dimensionless pressure impulse p, define the predictor family by the held-force bookkeeping identity

    xi_hat(p) = xi_hat_0 + hbar Wbar B_n^T p.

Assume a convex set P of pressures for which EVERY cloth predictor belongs to the inner per-node1/128 neighborhood of section6, using one fixed common translation. This is a hypothesis to be proved for the actual contact trials, not a consequence of declaring P. Fresh zero material multipliers are required when applying the finite-replay bound to each trial.

For every p in P, section6 gives a unique cloth material root y(p) in the convex outer neighborhood. Any appended prismatic coordinate is its own predictor. The combined ideal equation is

    Mbar A [y(p)-xi_hat(p)] + grad Ubar(y(p)) = 0.

On the cloth neighborhood ||H_U||<19, where H_U=Hess Ubar; the appended block is zero. The positive lower Hessian bound A/2-19 permits implicit differentiation. The cloth roots lie strictly inside the outer neighborhood, so the same local branch is available at endpoints of pressure segments as well. Differentiating the complete equation yields

    D_p y = hbar [Mbar + H_U(y)/A]^-1 B_n^T.

This is the derivative of the ideal material root, not the derivative of one XPBD row or of the finite ordered replay.

Put Wbar=Mbar^-1 and

    epsilon = 38/A <= 19/2048 < 1.
    A_n = B_n Wbar B_n^T.
    C(p) = B_n [Mbar + H_U(y(p))/A]^-1 B_n^T.

Indeed ||Mbar^-1/2 H_U Mbar^-1/2/A||<=epsilon. Factoring Mbar+H_U/A through Mbar^1/2 and bounding the eigenvalues of the middle factor gives the Loewner inequalities

    Wbar/(1+epsilon)
      <= [Mbar+H_U/A]^-1
      <= Wbar/(1-epsilon),

and therefore

    A_n/(1+epsilon) <= C(p) <= A_n/(1-epsilon).

All these inequalities are for the full matrices; no owner is removed. Since the middle inverse is positive definite,

    ker C(p) = ker A_n = ker B_n^T.

A pressure shift in this fixed kernel leaves the predictor and material root unchanged. This fact concerns the fixed map only; it neither discards a requested null stress from ideal-map error accounting nor certifies a nonnegative physical pressure.

### Frozen linear gap and a preconditioned contraction

Define the explicitly FROZEN linear observation

    g_H(p) = B_n y(p) - b,

where b is fixed and independently specified. Its derivative is hbar C(p). In particular, for p,q whose segment lies in P,

    (p-q)^T [g_H(p)-g_H(q)]
      >= hbar/(1+epsilon) * (p-q)^T A_n (p-q).

This proves strong monotonicity on S=range(B_n)=ker(B_n^T)^perp. It does not prove monotonicity for a changing nonlinear contact gap.

Fix a pressure center p_c and vary pressures in the affine space p_c+S. The kernel component of p_c is then fixed; this does not claim it is the only possible physical choice. On S let A_n,S denote the positive-definite restriction of A_n. Let a proposed frozen-gap target g_target obey the necessary compatibility condition

    g_target - g_H(p_c) belongs to S.

Since every difference g_H(p)-g_H(p_c) belongs to S, a target failing this condition cannot be solved by pressures in this model. It must not be silently projected away.

For compatible targets, consider the mathematical map

    T(p) = p - hbar^-1 A_n^+ [g_H(p)-g_target],

where + denotes the Moore-Penrose inverse acting as A_n,S^-1 on S and zero on its orthogonal complement. This formula is used for analysis only; no new implementation, API or execution is authorized.

Use the pressure norm ||d||_A=(d^T A_n d)^1/2 for d in S. In this norm the derivative of T is similar to the symmetric matrix

    I_S - A_n,S^-1/2 C(p)|_S A_n,S^-1/2.

The Loewner bounds put every eigenvalue in
[-epsilon/(1-epsilon), epsilon/(1+epsilon)]. Integrating the derivative along a pressure segment therefore gives

    ||T(p)-T(q)||_A <= kappa ||p-q||_A,
    kappa = epsilon/(1-epsilon)
           <= 19/2029 < 1/100.

Unlike a conclusion based on one row's multiplier partial derivative, this is a contraction for the entire ideal material-root response under the stated FIXED normal map. Its proof retains the nonlinear material Hessian and the full normal-row nullspace.

### A sufficient pressure-ball inclusion, still to be instantiated

Let K be the closed ball {p_c+d: d in S, ||d||_A<=R}, R>0, and require K to lie in P. Define the independently bounded center residual

    delta_c = hbar^-1
              ||A_n,S^-1/2 [g_H(p_c)-g_target]||_2.

Here the vector is in S by the compatibility condition. If

    delta_c <= (1-kappa) R,

then ||T(p)-p_c||_A<=kappa R+delta_c<=R for every p in K. Thus T maps the complete ball into itself. Banach's theorem gives one unique fixed point in K, and compatibility makes its FULL frozen-gap residual zero. For iterations starting in K,

    ||p_j-p_star||_A <= kappa^j ||p_0-p_star||_A.

The theorem supplies a checkable sufficient condition; no actual SC1/SC2 pressure ball or center residual has yet been bounded by this note. Strong monotonicity or the displayed small kappa alone does not supply that missing inclusion.

Nonnegative pressures are also not automatic. A sufficient independent condition for the whole ball is, for every original owner i,

    (p_c)_i >= R sqrt((A_n^+)_(i,i)).

To see this, write d=A_n,S^-1/2 u, ||u||<=R, in an orthonormal basis of S and apply Cauchy-Schwarz to coordinate i. Strict inequalities provide a strictly positive margin. This condition retains duplicate owners and tests the chosen kernel component rather than changing it after observing a result. Whether the frozen target, pressure cone and actual geometric contact support are compatible remains a separate obligation.

### Finite replay error is an additional term

Let y_N(p) be the ideal finite material replay at a particular trial and e_N=y_N-y. Section6 bounds its cloth Euclidean norm under that trial's predictor hypothesis. An appended ideal prismatic coordinate contributes zero to e_N. Replacing y by y_N in the frozen observation changes T by

    -hbar^-1 A_n^+ B_n e_N.

Its A-norm is

    hbar^-1 ||(A_n^+)^1/2 B_n e_N||_2
      <= hbar^-1 ||e_N||_M,

where ||e||_M=(e^T Mbar e)^1/2. The inequality follows because
Wbar^1/2 B_n^T A_n^+ B_n Wbar^1/2 is an orthogonal projector. It uses all rows and does not assume full row rank.

For this ideal cloth error, ||e_N||_M<=sqrt(2)||e_N||_2, even if an appended prismatic mass exceeds2, because its error component is zero. Define delta_N as the corresponding conservative upper bound on this pressure-map perturbation. It is an analysis quantity, not a new tolerance, physical residual allowance or state correction.

If every approximate pressure trial remains in the declared predictor domain and its perturbation is bounded by delta_N, the error recurrence is

    E_(j+1) <= kappa E_j + delta_N,

so E_j<=kappa^j E_0+delta_N(1-kappa^j)/(1-kappa).
A sufficient invariant-ball condition for these approximate trials is the stronger
delta_c+delta_N<=(1-kappa)R. This accounts for the finite ordered-gradient drift rather than treating the XPBD limit as the material root. Actual R7/native errors require their own additional propagation and cannot be set to this ideal delta_N.

### What is still absent for the physical contact problem

For an actual differentiable offset-gap function g(x), its derivative along this ideal predictor family is

    D_p g(y(p))
      = hbar B_gap(y(p))
        [Mbar+H_U(y(p))/A]^-1 B_n^T,

where B_gap is its full current Jacobian. The difference B_gap(y)-B_n must be independently enclosed, including director derivatives, material address, normal changes and all support extrema. The displayed matrix need not be symmetric and is not covered by the frozen-gap Loewner argument. Nondifferentiable support switches and SC2 contact birth need their full path and branch proofs.

Consequently the frozen-map contraction does not justify multiplying a stored C W r velocity bound by h to obtain a nonlinear held-position enclosure. Nor does it resolve full requested-pressure ideal-map error, actual delivery uncertainty, original-input energy/angular balance, paired robot moments, NUM005, finite native history or global step/window inclusion. The original controls and all numerical/resource budgets remain unchanged. The complete ordinary-mass typed physical contract remains NOT_FROZEN.
