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
