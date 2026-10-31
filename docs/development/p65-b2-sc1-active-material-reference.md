# SC1 active-material reference and conditional normal-pressure enclosure

Status: HAND_DERIVED_IDEAL_REFERENCE_AND_FROZEN_MAP_ENCLOSURE. This note instantiates part of the [local material analysis](p65-b2-local-row-derivative-bounds.md) for SC1 in the unchanged [B2 proposal](p65-b2-contact-integration-proposal.md). It is not a result from the finite native/runtime solver, an accepted physical test or a replacement of the complete ordinary-mass contract.

## 1. Data and units

Use the ordinary analysis target's original hinge a=(0,0,0), b=(1,0,0), c=(0,1,0), d=(0,-1,0), three distance rest lengths1, unit distance and bending compliances, half-thickness rho=1/64, robot patch[-2,2]^2 and prismatic mass4m. Particle gravity is g=-1/8 in the z direction, all other original forces/effort are zero, and initial velocities vanish. Rest angle r is any of the three already declared values3/4096,4/4096,5/4096.

Write numerical formulas in the dimensionless units L0=1 m, m0=1 kg, t0=1 s used in sections6–7 of the local note; hence h means hbar, A=h^-2, and p is an impulse in units m0 L0/t0. The primary ordinary reference is m=1. The formulas also hold for uniform cloth masses m in[1/2,2] with the compliances HELD at their unit values. This last mathematical statement does not reinterpret any separately declared homogeneous mass/compliance variant or silently change a mass-only variant's compliance.

The physical grids are the original h in{1/64,1/128,1/256}, T=1/8. No value is selected from an executed trajectory.

## 2. An active backward-Euler endpoint branch

Define the exact reference sequence

    v_0=z_0=0,
    v_(k+1)=v_k-h/16,
    z_(k+1)=z_k+h v_(k+1),
    x_i,k=X_i+z_k e_z,     q_k=z_k.

All four cloth nodes and the robot share the same vertical velocity. Therefore

    v_k=-k h/16,       z_k=-h^2 k(k+1)/32.

At every reference state the three distance residuals are zero, theta=0 and the bending residual is C_B=-r, which is strictly nonzero. The ideal angle gradient is

    G=(2e_z,0,-e_z,-e_z),

so grad U=-r G and the bending force is r G. This active force is retained below; the zero-stiffness comparison is not substituted for the strained material.

Define physical-node normal impulse aggregates

    F_a=h(m/16-2r),     F_b=h m/16,
    F_c=F_d=h(m/16+r).

For the six original lower-face owners ordered(Ua,Ub,Uc,Db,Da,Dd), use the explicit reference pressure

    p_c=(F_a/2,F_b/2,F_c,F_b/2,F_a/2,F_d).

It retains the duplicated a/b owners and chooses their equal split before any solver output. Every owner is positive; uniformly over the stated masses and rests,

    min_i (p_c)_i >= 59h/4096 >0.

The minimum is the a split at m=1/2,r=5/4096:
h(1/64-5/4096)=59h/4096. No tangent pressure is introduced.

At a flat state, projecting the COMPLETE lower-offset point derivative onto e_z gives just the nodal vertical derivative: the derivative of the unit director's vertical component is zero there. Thus the six-row normal map B_n has one cloth-node z entry+1 and robot q entry-1 per owner. This conclusion uses the flat normal projection of the full director law; it does not replace the full three-dimensional Jacobian elsewhere.

The effective cloth predictor at this reference input is exactly

    xi_hat_i(p_c)=X_i+z_(k+1)e_z-h^2(r/m)G_i.

At the proposed endpoint, the material backward-Euler equation is satisfied:

    m A [x_i,k+1-xi_hat_i(p_c)] + (grad U)_i
       = r G_i-r G_i =0.

The algebraic endpoint bending multiplier is h^2 r, and all distance multipliers are zero. These are multipliers of this ideal endpoint equation; they are not claimed to equal the accumulated finite ordered-XPBD multipliers.

The sum of normal aggregates is m h/4. Its robot reaction produces
Delta v_R=-(m h/4)/(4m)=-h/16, exactly the reference robot update. Directly, each cloth force is

    m g e_z + (B_n^T p_c)_cloth/h + r G
       = -m e_z/16,

and the robot force is -m e_z/4. Hence both masses' discrete acceleration equations, the full frozen normal-map equation and zero endpoint gaps hold simultaneously.

The reference lower offsets and robot patch coincide at z=z_k-rho. The entire affine segment between consecutive reference endpoints is a common translation: face shapes, directors and thickness remain unchanged and each cloth footprint stays at least1 m inside the patch's outer square. This is a geometric statement about this reference sequence only.

At the last step, |v_N|=1/128 and

    |z_N|=(T^2+T h)/32 <=9/16384.

The reference thus gives an explicit full-window path with nonzero material strain and positive held transfer. It is not yet an inclusion of the runtime trajectory.

## 3. Original-input energy on the reference

The joint kinetic energy is K=4m v^2, the material energy is the constant r^2/2, and the cloth gravitational energy change is -4m g Delta z. Robot original work and all other original force work are zero.

Using v'=v+h g/2 and Delta z=h v',

    Delta(K+U+G)
       =4m[(v+h g/2)^2-v^2]-4m g h(v+h g/2)
       =-m h^2 g^2
       =-m h^2/64 <0.

Every reference step separately has zero positive energy defect; no temporal cancellation or internal-contact work credit is used. This exact identity does not certify the rounded/finite replay energy interval or its width. It supplies no angular-budget or general-robot theorem; NUM005, actual paired moments and the original physical audit remain required.

## 4. A concrete frozen-map pressure ball

Fix ONE step whose input equals the ideal reference entry above. This restriction is essential: an inclusion for subsequent steps starting from previously perturbed runtime states has not been proved.

For the fixed full normal map, the already derived matrix is

    (A_n)_ij=(delta_same_physical_node+1/4)/m,
    A_n=B_n Wbar B_n^T.

It has rank4, the two duplicate-owner differences form its kernel, and its positive eigenvalues lie in[1/m,(7/2)/m]. The equal-split p_c lies in S=range(B_n).

Choose, independently of any solver output,

    R=h/128,
    K={p_c+d: d in S, ||d||_A<=R},
    ||d||_A^2=d^T A_n d.

Because ||A_n^+||<=m<=2, every owner perturbation satisfies

    |d_i|<=sqrt(2)R<3h/256=48h/4096.

Thus every point of K has all original owner pressures greater than11h/4096. This is a normal-pressure cone margin for this ball, not a proof of actual contact-feature persistence.

For zero-start analysis use the convex set P=conv({0},K). Write its points as
p=s(p_c+d), 0<=s<=1, d in S, ||d||_A<=R.
The cloth predictor is a common translation plus

    -s h^2(r/m)G + s h (Wbar B_n^T d)_cloth.

The second term's full mass norm is at most h R, so its cloth Euclidean norm is at most sqrt(2)h R. Relative to the common translation appropriate to s, the per-node displacement is bounded by

    4h^2 r + sqrt(2)h^2/128
       < (5+12)h^2/1024
       =17h^2/1024.

Relative to the SINGLE reference translation z_(k+1), the additional common displacement is at most h^2/16. Hence all predictors in P are within81h^2/1024<1/128 of that translated reference. This establishes the predictor-domain premise for this ONE reference-entry step, rather than merely assuming a pressure set has that property.

Translation invariance also gives a sharper initial material-residual bound. The deformation above has four-node product norm at most twice the per-node bound. With ||G_theta||<3 and the three unit distance rows,

    R0 <= r+6(17h^2/1024)
        <=5/4096+51/(512A)
        <=2611/2097152
        <1/800.

The distance residuals are smaller than this common bound. All partial-row invariance and complete-sweep conclusions of section6 therefore apply throughout P with this improved R0.

## 5. Conditional finite material replay and an abstract zero-start map

For each pressure trial, let the exact-arithmetic material replay start with fresh zero multipliers and perform the original512 complete ordered sweeps. Let y_N(p) denote that finite result and y(p) the independently defined material root.

All three original A values are at most65536, so64^-512<A^-2. Section6 then gives

    ||y_N-y||_2
      <=[15/(1600A^2)+4576/(640000A)]/(A/2-19)
      <1/(64A^2)=h^4/64.

The last inequality follows by multiplying positive denominators:
3/5+(286/625)A<A/2-19 whenever53A>24500, already true at A=4096.
The norm here is the cloth Euclidean norm; the ideal prismatic coordinate equals its predictor and has zero finite material-replay error.

For the FROZEN linear gap g_H=B_n y-b, set its target to the zero gap attained by the reference. The range compatibility of section7 holds, and T(p_c)=p_c. Its ideal root-response map contracts on P in the A_n norm with kappa<=19/2029<1/100.

Replacing y by the finite material replay defines the ANALYSIS map T_N. No current solver is asserted to implement this map. Its one-trial perturbation obeys

    ||T_N(p)-T(p)||_A
       <=sqrt(2)||y_N-y||_2/h
       <3h^3/128 = delta_N.

The full reference pressure has norm

    ||p_c||_A^2=h^2(m/32+6r^2/m)<h^2/9.

Therefore the abstract first update from ZERO pressure satisfies

    ||T_N(0)-p_c||_A
       <h/300+3h^3/128
       <h/256<R.

For every later trial in K,

    ||T_N(p)-p_c||_A <=kappa R+delta_N
       <R(1/100+3h^2)<R.

Thus this abstract zero-start iteration enters the independently chosen positive-pressure ball in one update and stays there. The general finite error recurrence is

    E_j <=kappa^(j-1) h/256
          +delta_N(1-kappa^(j-1))/(1-kappa),   j>=1.

In particular, the persistent term is less than h^3/32. This is an inclusion of FINITE ideal material responses inside a specified frozen-map analysis, not convergence of the actual contact coordinator, not a claim that T_N has a unique limit, and not permission to feed p_c or any oracle state into runtime code.

The material-root sensitivity also gives, within P,

    ||y(p)-y(p_c)||_M
       <= h ||p-p_c||_A/(1-38/A).

Together with the explicit E_j and finite material-replay error above, this supplies a non-vacuous one-step position enclosure about the active reference, conditional on using this frozen-map analysis and a reference entry state. It is not silently extended to a whole runtime window.

## 6. Remaining physical obligations

An actual finite endpoint need not be exactly flat. Its offset gap and current Jacobian B_gap are then different from the frozen linear observation; an error bound on positions alone does not prove the same positive-area support feature persists. In particular, tilting a triangle can change a face contact into an edge or point contact even when the position change is small. The current loaded-support gates cannot be bypassed by the pressure cone margin in this note.

The complete proof still needs the true offset-gap/director/normal extrema over every physical segment, any support switches, full actual material/contact iteration and selected replay history, original held/terminal delivery and ideal-map uncertainty, propagation from perturbed entry states through all physical steps, all native R7 errors and the original energy/angular audits. SC2's open approach, perturbed birth time and terminal transfer are untouched.

The reference's all-step active C and positive pressures are exact ideal facts, whereas the ball and replay statements above concern ONE step started from that ideal reference. None is a new measured test result. All fixtures, material variants, meshes, h grids, rebuilds, failure/transaction requirements and budgets remain unchanged; the ordinary-mass coupled physical contract is still NOT_FROZEN.

## 7. Existence of one finite-material contact endpoint, without an outer-loop claim

There is a further ONE-STEP conclusion for an input exactly on the reference branch. It concerns at least one mathematical fixed point of T_N, not the state obtained after512 actual contact iterations.

Every ideal material row is smooth on the admitted neighborhood and has denominator at least A. Its finite512-sweep composition is therefore continuous in p. The pressure ball K is compact and convex in the four-dimensional space S, and section5 proves T_N(K) is contained in K. Brouwer's theorem gives a fixed point p_star in K.

The full frozen-gap residual of a finite replay belongs to S, since it is
B_n[y_N(p)-y(p_c)]. At a fixed point A_n^+ times that residual is zero. The restriction of A_n^+ to S is invertible, so the FULL residual is zero, not merely its projection. All six owner pressures remain greater than11h/4096. No uniqueness of this finite-material fixed point or convergence of a particular outer iteration follows from Brouwer's theorem.

### Size and physical gap of this endpoint

At p_star the previous error recurrence gives

    ||p_star-p_c||_A <=delta_N/(1-kappa)<h^3/32.

Combining material-root sensitivity, the cloth lower mass bound and finite-replay error yields

    ||y_N(p_star)_cloth - (X+z_(k+1)e_z)||_2
      <[sqrt(2)/(32(1-38/A))+1/64]h^4
      <h^4/16.

For the last inequality, sqrt(2)<10/7 and
1/(1-38/A)<=2048/2029<64/63 give coefficient
20/441+1/64=1721/28224<1/16.

Because every row of the frozen-gap residual is zero, all four cloth z coordinates equal the robot q at this endpoint. The local face-area bounds preserve the original positive orientations. Consequently both oriented face normals and every raw-area-star director are EXACTLY+e_z, including the shared hinge nodes. Thus the actual lower offsets equal x_i-rho e_z and have zero actual normal gap to the robot patch. This is a consequence of a full zero residual at this mathematical endpoint; it is not an implication from a merely small residual.

Each ideal material row also conserves total cloth position weighted by mass: its increment obeys sum_i m Delta x_i=sum_i G_i Delta lambda=0, by translation invariance of the distance and angle laws. The held normal contributions of cloth and robot cancel in the corresponding combined mass-weighted position. Therefore

    sum_i m (x_i,star)_z +4m q_star
      =8m(z_k+h v_k)+4m h^2 g.

All five vertical coordinates equal q_star, so
q_star=z_k+h v_k+h^2 g/2=z_(k+1).
The endpoint deviation from the reference is consequently horizontal only.

Both the starting reference and this endpoint have a shared vertical coordinate. Their entire physical affine segment stays planar with positive oriented face areas; its full directors remain+e_z. The two triangles stay on opposite sides of their shared oriented edge, so their closed intersection is exactly that shared edge, and their vertical material extrusions share only the corresponding material side. Their footprint remains inside the original robot patch. This proves the elementary flat geometry for this one exact endpoint/segment, without asserting that a rounded outer-loop iterate has the same support feature.

The endpoint bending residual is exactly-r. Since the finite material residual satisfies |C_B+A ell_B|<=64^-512/800<r/2, its actual ideal finite-replay multiplier obeys ell_B>r/(2A)>0. Thus neither active material nor held pressure has disappeared from this finite-material existence statement.

### One-step endpoint energy for this ideal root

Let d be the horizontal cloth displacement from the reference endpoint, with
||d||<h^4/16. If the ideal step reconstructs velocity as its actual displacement divided by h, the vertical velocities equal the reference velocities. The only additional kinetic term is m||d||^2/(2h^2), and the three distance energies together are at most6||d||^2, since each original unit edge changes length by at most2||d||. Bending energy remains r^2/2 and gravitational energy remains the reference value.

Its original-input endpoint defect therefore obeys

    D_E <= -m h^2/64 +(m/(2h^2)+6)||d||^2
         < -m h^2/64 +h^6/256+3h^8/128
         <0.

For an elementary uniform sign check, m>=1/2 and even h<=1/2 make the last expression at most(-64+5)h^2/8192<0; the original three h values are smaller. This bound includes the finite material drift's in-plane kinetic and strain energy. It does not subtract contact work or cancel positive defects between steps.

This remains an ideal one-step existence and endpoint-energy theorem. It does not provide the current coordinator's finite-iteration convergence, actual pressure/delivery certificate, native reconstruction error, energy enclosure width, angular/NUM005 bound, or propagation from a perturbed entry into a full physical window. In particular, an approximately flat numerical endpoint still needs the original exact support/path checks. The ordinary-mass physical contract remains NOT_FROZEN.

## 8. Uniqueness and convergence of the abstract finite frozen-gap iteration

Section8 of the [local derivative note](p65-b2-local-row-derivative-bounds.md) now supplies a complete finite-history derivative bound that strengthens the preceding Brouwer existence result. It proves ||D d_N||<95/A for the ideal material displacement correction, including all four rows, multipliers and changing-gradient history. For the frozen all-owner normal map, the resulting abstract T_N has Lipschitz factor

    kappa_N=190/A<1/20

in the A_n norm on the pressure range. This is a derivative of the finite ordered replay; it is not obtained by declaring its endpoint equal to the backward-Euler root or differentiating only a small endpoint-error estimate.

Apply it here with N=512 material sweeps. Section4 already places every predictor from P=conv({0},K) strictly inside the required inner neighborhood. Sections5 and7 already show that the abstract zero-pressure first update enters K and that T_N(K) is contained in K. The four-dimensional pressure ball K is complete in its A_n norm. Banach's contraction theorem therefore gives a UNIQUE p_* in K in the prescribed range/equal-duplicate-owner gauge, and the abstract sequence p^(0)=0, p^(j+1)=T_N(p^(j)) converges to it.

This does not assert uniqueness among physical pressures with different nullspace components. All six original owners remain present, and no computed output is used to choose the pressure space.

The independently retained material-drift comparison from sections5 and7 remains valid:

    ||p_*-p_c||_A_n<h^3/32,
    ||p^(1)-p_c||_A_n<h/256.

For every j>=1, combining these with the new finite-map contraction gives

    ||p^(j)-p_*||_A_n
      <20^(-(j-1)) [h/256+h^3/32]
      <(h/128)20^(-(j-1)).

In particular the j=512 ideal ABSTRACT pressure iterate has distance less than h/(128*20^511). This is an exact analytical bound for that map and count, not an observed floating-point residual or an authorization to spend512 full replays in any existing coordinator.

The full frozen gap lies in S. Writing e_j=||p^(j)-p_*||_A_n, its dual-norm residual satisfies

    ||B_n y_N(xi_hat(p^(j)))-b||_(A_n^+)
      =h||p^(j)-T_N(p^(j))||_A_n
      <=h(1+kappa_N)e_j.

Even this small bound has no prescribed componentwise sign. It does not establish exact flatness or nonpenetration at any finite iterate. Exact flatness, the unchanged common vertical coordinate, active bending and the negative one-step energy result in section7 apply to the unique fixed point p_* under the original exact-reference-entry hypothesis. They are not automatically inherited by a rounded iterate or a later physical step.

The relevant existing thick-contact workspace uses synchronous projected updates with a scalar full-world response step size, recomputed geometry and backtracking; the separate finite point-patch coordinator uses serial diagonal coordinate updates. Neither is the complete range-quotient T_N. Equivalence, invariance and finite convergence for the actual thick-contact path remain unproved. The full native/delivery/geometry and resource accounting is also required. No old source, iteration cap, fixture, physical budget or execution grant changes.

The remaining SC1 physical-window propagation, SC2 birth/terminal control, current full offset map, rounded support and paired energy/angular/NUM005/R7 obligations are still open. This addition is manual mathematics only; no import, test, numerical or derivative probe, lint, formatter, build or native execution was performed.

## 9. A frozen-gap synchronous projected iteration

The existing thick-contact proposal retains synchronous contact updates. Its actual [workspace](../../src/phymeo/solvers/_thick_contact_workspace.py) computes alpha0 from full-world response blocks, uses newp=max(0,oldp-a*normal) simultaneously, and tries a=alpha0/2^power with recomputed full metrics. The following manual statement isolates the ideal frozen-gap, fixed-step version of that update. It does not change the algorithm or assert that its actual line search accepts the full step.

Continue to use N=512 exact material sweeps, all six original normal owners, the original SC1 reference entry, masses and unit compliances. Let

    g_N(p)=B_n y_N(xi_hat(p))-b,
    G_a(p)=projection_orthant[p-a*g_N(p)/h].

The root p_* of sections7-8 satisfies g_N(p_*)=0 and is strictly positive. The SC1 range S consists exactly of vectors with equal duplicated a-owner entries and equal duplicated b-owner entries; c and d entries are free. Orthant projection preserves this particular S. This fact is specific to the stated normal map, not a property of arbitrary pressure ranges.

### A predictor domain containing the zero-pressure start

Define the analytical set

    Q={p in S: p>=0 and ||p-p_*||_2<=||p_*||_2}.

It is convex, contains zero, and does not prescribe p_* to an implementation. From section7 and lambda_min(A_n|S)>=1/m,

    ||p_*||_2<=sqrt(m)||p_*||_A_n
              <sqrt(m)(h/3+h^3/32)<h sqrt(m)/2.

Every p in Q therefore has ||p||_2<h sqrt(m). At a shared cloth node its normal aggregate has at most two owner entries, hence absolute magnitude <=sqrt(2)||p||_2. Relative to the one fixed zero-pressure free-translation predictor, each cloth predictor's displacement is consequently

    <(h/m)*h sqrt(2m)<=2h^2<=1/2048<1/128.

Thus all of Q satisfies the section6 inner-predictor hypothesis, with fresh zero material multipliers for every mathematical evaluation. Section8's finite sensitivity bound holds on every segment in Q, independently of whether Q is invariant.

### Full-world step size at the unchanged flat reference

Here use the six original nodal bottom-offset owners and their complete initial triangle-offset Jacobians. For either oriented reference triangle the derivative D_f of its unit normal has Frobenius norm2: in the original coordinates it maps vertical nodal increments to the two differences delta z_a-delta z_b and plus or minus(delta z_a-delta z_outer). Other components are zero.

For a nodal bottom owner, J_i=E_i-rho D_f with rho=1/64, where E_i selects its physical node. Therefore

    ||J_i||_F<=sqrt(3)+2rho<2,
    ||J_i||_2<=1+2rho=33/32.

The full-world initial response block, including the dynamic4m prismatic reaction, is

    W_ij=(J_i J_j^T)/m+(e_z e_z^T)/(4m).

Define the exact ideal counterpart of the workspace's scalar response bound,

    L=max_i sum_j ||W_ij||_F,   a=1/L.

The normal projection n^T W_ij n equals the unchanged A_n,ij=(delta_same_node+1/4)/m. A duplicated-node row has normal row sum7/(2m). Frobenius domination and the complete six-block upper bound give

    7/(2m)<=L
      <6[(2)(33/32)+1/4]/m=111/(8m)<14/m,
    m/14<a<=2m/7.

This is an exact-arithmetic comparison with the relaxation=1 formula, not a bound on its stored rounded evaluation. Every cross block, duplicate owner and robot contribution is retained.

### Contractivity and finite ideal iteration

Section8 gives on S

    D(g_N/h)=A_n+E_N,
    ||A_n^(-1/2) E_N A_n^(-1/2)||_2<=kappa_N,
    kappa_N<=95/2048.

Inverses here mean inverses restricted to S. Symmetry of E_N is not assumed. With lambda_min>=1/m and lambda_max<=7/(2m), and any a<=2m/7,

    ||I-a*D(g_N/h)||_(2,S)
      <=1-a/m+a*kappa_N*7/(2m)
      =1-(a/m)[1-(7/2)kappa_N].

The bracket is >=3431/4096>5/6. For the exact full step a=1/L above, the factor is therefore

    q_sync<1-5/84=79/84<16/17.

Orthant projection is Euclidean nonexpansive and fixes p_*. It also preserves S. Integration along segments in Q now proves

    ||G_a(p)-p_*||_2<=q_sync||p-p_*||_2.

Thus Q is invariant, and the zero-start ideal synchronous frozen-gap sequence converges to the same p_*. After j updates,

    ||p^(j)-p_*||_2<(16/17)^j||p_*||_2.

Since (17/16)^16>2 by its binomial expansion,512 such updates give a bound <2^-32||p_*||_2. This is analytical convergence for the specified exact frozen-gap iteration and the full scalar step, not an executed512-update trace.

### Remaining differences from the real thick-contact path

The real _improve uses midpoints of actual current offset-gap bounds, full normal/friction/native/mapping/moment metrics for its score, an early-return threshold, up to13 backtracking attempts, and possible anchor rebuilds. Decrease of Euclidean distance to p_* does not prove decrease of that actual maximum score or acceptance of power0. A reduced step still has the displayed conditional contraction with its own a, but the512-update full-step rate cannot be transferred to a backtracked sequence.

Actual changing offset gaps, stored step-size rounding, native material/predictor/delivery error and the complete replay resource account remain separate. A nonzero frozen-gap bound also has no one-sided sign, so exact positive-area contact and zero penetration are not inferred at finite iterates. Section7's energy statement is still about the exact fixed point entered from the exact reference. Full physical-window propagation, SC2 birth/terminal, angular/NUM005 and transaction requirements remain open, with no new source or execution grant.


## 10. Changing triangle-offset gaps and the normal-kernel component

This section keeps the same ONE-step exact SC1 reference entry, all six nodal bottom owners, unit compliances, masses, N=512 ideal material sweeps and fixed initial normal delivery map. It analyzes the current per-triangle offset law used by [held_feature_gap](../../src/phymeo/solvers/_thick_contact_geometry.py). It does not replace that law with the newer conforming full-star shell geometry. All numerical constants use the normalization already declared in the local material note.

Let P_S be the Euclidean projection onto S=range(B_n), Pi=I-P_S, z=P_S p and k=Pi p. Here z denotes a pressure-range vector; write z_ref for the common vertical reference translation. The ideal predictor and finite material endpoint depend on z alone, since B_n^T k=0. This exact statement about the fixed initial normal map does not remove requested kernel pressure from later actual-versus-ideal map-error accounting.

Use the larger convex pressure-range ball

    Q_bar={z in S: ||z-p_*||_2<=||p_*||_2}.

It is an analysis set, not a solver projection or prescribed pressure oracle. Section9's bound ||z||<h sqrt(m) holds without requiring z>=0. Thus each cloth predictor is within2h^2<=1/2048 of the one fixed free common-translation predictor. The existing inner-predictor theorem applies throughout Q_bar, including segments between its points; all ideal material row states and endpoints stay in the outer per-node1/64 geometry domain.

### Finite support and an exact gap formula

The reference patch projects to [-2,2]^2. Throughout the outer domain, each middle node's horizontal coordinate has magnitude at most1+1/64. The previous face-normal bound gives ||n_f,xy||<1/14. Both lifted face vertices, and hence their convex hull, therefore have projected coordinates bounded by

    1+1/64+rho/14 <33/32,   rho=1/64.

The ideal nodal bottom points project strictly inside the finite patch, with coordinate margin greater than31/32. The same statement holds along the ideal affine middle-position segment from the exact entry to one such endpoint: subtract its affine common vertical translation and use convexity of the outer neighborhood. This proves finite projected-feature membership for these ideal points. It does not prove absence of penetration, unchanged contact dimension, absence of other SAT ties, self-contact validity or stored-point membership.

Write q_R for the robot translation and b_N(z)=B_n y_N(z)-b for the full six-owner frozen gap. For either current face f, let

    c_f(z)=rho(1-n_f,z(z)) >=0.

The upward held support plane has height q_R-rho. The exact current nodal bottom point is y_i-rho n_f, so inside that finite support feature its signed ideal gap is

    g_fi(z)=y_i,z-q_R+rho(1-n_f,z)
           =b_N,fi(z)+c_f(z).

Thus g(z)=b_N(z)+c(z), with
c=(c_U,c_U,c_U,c_D,c_D,c_D) in owner order(Ua,Ub,Uc,Db,Da,Dd).
This is a changing-director identity, not an endpoint frozen-normal approximation. The implementation's stored point and rational enclosure midpoint still require their separate error bounds; held_feature_gap returns a hull containing both ideal and actual gap bounds.

If all six exact g_fi vanish, subtracting gaps within each face shows that its three nodal z coordinates coincide. The shared nodes connect the faces, hence all four nodal z coordinates coincide. Positive orientation in the outer domain then forces both normals to equal e_z, c=0 and b_N=0. Conversely b_N=0 forces the same flatness and g=0. The exact zero sets agree within this domain; a small nonzero gap does not imply flatness or a one-sided gap sign.

### Quadratic tilt correction and its derivative

For f=U or D use horizontal edge columns

    E_U=[(y_b-y_a)_xy,(y_c-y_a)_xy],
    E_D=[(y_b-y_a)_xy,(y_d-y_a)_xy].

Their reference matrices are diag(1,1) and diag(1,-1). Each column perturbation has norm<=1/32, so ||E_f-E_f,ref||_2<=sqrt(2)/32<3/64 and

    ||E_f^(-1)||_2<64/61<16/15.

For d_f=(y_b,z-y_a,z,y_outer,z-y_a,z), the face slope is
t_f=E_f^(-T)d_f and n_f,z=(1+||t_f||^2)^(-1/2).
Writing b_f for the three entries of b_N on that face, the two-difference operator has norm sqrt(3). Therefore

    ||t_f||<=(16/15)sqrt(3)||b_f||<=2||b_f||,
    0<=c_f<=rho||t_f||^2/2<=2rho||b_f||^2.

Each correction is repeated three times. Summing both faces without removing duplicate owners gives

    ||c||_2<=sqrt(3)(c_U+c_D)<=4rho||b_N||_2^2.

A directional derivative, with dots denoting an arbitrary pressure-range direction, satisfies

    ||dot t_f||<=2(||dot b_f||+||t_f|| ||dot y_f,xy||),
    |dot c_f|<=rho||t_f|| ||dot t_f||
             <=4rho||b_f|| ||dot b_f||
               +8rho||b_f||^2 ||dot y_f,xy||.

The first inequality differentiates the complete inverse edge matrix as well as its vertical right-hand side. No horizontal shape derivative is omitted. Cauchy-Schwarz over the two faces, and the bound on each face's norm by the full cloth norm, yield

    ||dot c||<=8rho||b_N|| ||dot b_N||
               +16rho||b_N||^2 ||dot y_cloth||.

The finite material sensitivity and section9 response bounds imply on Q_bar, with e=||z-p_*||,

    ||b_N(z)|| <= (4h/m)e,
    ||D_z b_N|| <4h/m,
    ||D_z y_cloth|| <3h/(2m).

The last inequality uses ||B_cloth^T||=sqrt(2) and
sqrt(2)(1+95/A)<3/2. Combining these estimates with rho=1/64 gives

    ||c(z)|| <= (h^2/m^2)e^2,
    ||D_z(c/h)|| <= (2h/m^2)e+(6h^2/m^3)e^2
                 < 2h^2/m.

For the last step, e<=||p_*||<h sqrt(m)/2, m>=1/2 and h<=1/64 imply
m||D_z(c/h)||<h^2/sqrt(m)+(3/2)h^4/m<2h^2.
In particular c=0 and D_z c=0 at e=0.

### The range does not contain every changing gap

In the declared owner order the kernel part is exactly

    Pi c=((c_U-c_D)/2,(c_U-c_D)/2,0,
          (c_D-c_U)/2,(c_D-c_U)/2,0).

It vanishes when the two tilt corrections coincide. They need not coincide for a general endpoint in the domain. Consequently the actual six-owner gap need not lie in S even though its frozen part does. Discarding Pi g in a quotient solve would omit a real, boundable residual.

Consider the full exact UNCLIPPED update p^+=p-a*g(P_S p)/h only as the candidate before the real workspace's orthant projection. Its decomposition is an identity:

    z^+=z-a*P_S g(z)/h,
    k^+=k-a*Pi c(z)/h.

There is no added projection in the implementation. P_S is used only to analyze this comparison sequence. Since endpoint geometry is constant along k, a full-space strict contraction is impossible: different kernel splits can deliver the same ideal endpoint.

For the range component, projection cannot increase the derivative bound of c/h. Combining it with section9 gives

    ||D_z[z-a*P_S g(z)/h]||
      <=1-(a/m)[1-(7/2)kappa_N-2h^2],
    1-(7/2)kappa_N-2h^2
      >=3429/4096>5/6.

Thus for the same exact full step m/14<a<=2m/7 this UNCLIPPED comparison has range contraction factor<16/17 on Q_bar and fixes p_*. The ball is invariant. From p^(0)=0 its range error satisfies e_j<(16/17)^j||p_*|| for j>=1.

The kernel drift is absolutely summable. With e_0=||p_*||,

    sum_j ||k^(j+1)-k^(j)||
      < (a*h/m^2)*e_0^2/[1-(16/17)^2]
      < (289/462)h^3 <(2/3)h^3.

Its limit k_infinity lies in ker(B_n^T), and the tail from j is bounded by
(2/3)h^3(16/17)^(2j). In particular the comparison converges to p_*+k_infinity, with the same ideal endpoint. Because p_*,i>11h/4096,

    p_*,i+k_infinity,i
       >11h/4096-(2/3)h^3
       >=31h/12288>h/512.

This establishes a positive limiting split for this analytical sequence. It does not establish that every intermediate candidate is nonnegative. Therefore it does not prove that the real orthant projection is inactive, or transfer this rate to the actual projected/backtracked iteration. The residual kernel component is retained throughout the argument, not declared harmless to all physical certificates.

### Outstanding physical obligations

The comparison now includes the exact nonlinear tilt correction and a finite-support margin, but still begins from the exact reference and uses the fixed initial normal delivery map with exact finite material arithmetic. Orthant activity from the zero start, actual full-metric line-search acceptance, stored gap-midpoint and step-size errors, R7/native delivery, anchor discovery/rebuilds, finite-iterate one-sided penetration and contact-sign gates remain unresolved. The exact ideal limiting endpoint has the earlier one-step energy property; no finite or rounded iterate inherits it by assertion.

The SC1 perturbed-entry/full-window induction, SC2 birth/terminal case, paired angular/NUM005 and transaction gates also remain open. No fixture, oracle, tolerance, iteration/resource ceiling, physical API or execution grant changes. This addition is manual mathematics and source-text/file-IO/Git work only.


## 11. Nonnegative candidates with changing gaps and variable ideal steps

The comparison in section10 can be strengthened to include the original coordinatewise pressure projection. Continue to assume its exact SC1 reference entry, uniform cloth mass m in[1/2,2], robot mass4m, unit compliances, r in{3,4,5}/4096, six fixed initial normal owners and N=512 ideal material sweeps. No conclusion here applies to stored/native errors or a perturbed physical-step entry without further proof.

Allow an arbitrary sequence of step sizes

    0<=a_j<=2m/7,   t_j=a_j/m,   T_j=sum_(l<j) t_l.

For example this contains every exact backtracking candidate a_j=a_0/2^power with the section9 exact full step, and a_j=0 represents no accepted update. No positive progress is assumed. First construct the UNCLIPPED comparison of section10 with these steps. For each chosen step its pressure-range map contracts by at most1-(5/6)t_j, so Q_bar remains invariant. This argument applies pointwise to each chosen step and does not differentiate an adaptive step-selection rule.

### A componentwise bound on the nonlinear forcing

For the fixed initial cloth normal map B_c, every row has norm1 and ||B_c^T||=sqrt(2). Uniform cloth mass is used explicitly here. The complete finite material sensitivity gives the ROW bound

    ||D_z[b_N,i(z)/h-(A_n z)_i]||
       <=(95 sqrt(2)h^2)/m.

The robot's ideal material displacement correction is zero. Define
R_i(z)=b_N,i(z)/h-[A_n(z-p_*)]_i. At p_* it is zero. Integration on Q_bar and ||z-p_*||<=||p_*||<h sqrt(m)/2 give

    |R_i(z)| < (95/2)sqrt(2/m)h^3 <=95h^3.

This uses the full finite replay, not a final-equilibrium substitution. The earlier bound ||p_*-p_c||_A_n<h^3/32 and the diagonal A_n,ii=5/(4m) imply

    |[A_n(p_*-p_c)]_i|
       <=sqrt(5/(4m)) ||p_*-p_c||_A_n <h^3/20.

Section10 also gives 0<=c_i/h<=||c||/h<=h^3/(4m)<=h^3/2. Consequently the full changing ideal gap can be written

    g(z)/h=A_n z-f+delta(z),   f=A_n p_c,
    |delta_i(z)|<96h^3.

Every owner is retained. In the declared order,

    f=h(1/8-2r/m, 1/8, 1/8+r/m,
        1/8, 1/8-2r/m, 1/8+r/m).

For the original h<=1/64 and all stated r,m,

    (492/4096)h<=f_i<=(522/4096)h,
    |delta_i|<=(96/4096)h.

P_S averages each duplicated owner pair and leaves the two outer owners unchanged, so it cannot increase the maximum component bound. Define F=f-P_S delta. Uniformly on Q_bar,

    f_minus=(99/1024)h <=F_i<=(309/2048)h=f_plus,
    d_f=f_minus-(3/5)f_plus=(63/10240)h>0.

Here d_f is a positive forcing margin, not a face director, displacement or a changed physical load. The notation F is the algebraic right-hand side of this comparison update; it is not a replacement for the original external force F0 or a new runtime force channel.

### An invariant aggregate bound and positive range pressures

Let S_j=sum_i z_i^(j), and let d_i=2 for the four duplicated a/b owner entries and d_i=1 for the two outer owner entries. Since z belongs to S,

    (A_n z)_i=d_i z_i/m+S_j/(4m),
    z_i^(j+1)=(1-d_i t_j)z_i^(j)
                 +a_j[F_i-S_j/(4m)].

Starting at z^(0)=0, induction proves

    z_i^(j)>=0,   0<=S_j<=(12/5)m f_plus.

Indeed 1-d_i t_j>=3/7. Under the proposed aggregate bound,
F_i-S_j/(4m)>=d_f>0, so every new component is nonnegative. Also

    S_(j+1)
      =S_j-t_j sum_i d_i z_i^(j)-(3/2)t_j S_j+a_j sum_i F_i
      <=[1-(5/2)t_j]S_j+6a_j f_plus.

The coefficient is >=2/7, and substitution of S_j<=(12/5)m f_plus proves the same upper bound at j+1. This induction uses both the sum and the repeated-owner structure; a Euclidean contraction alone would not establish the componentwise statement.

For every component, the stronger scalar recurrence is

    z_i^(j+1)>=(1-2t_j)z_i^(j)+m t_j d_f.

It yields

    z_i^(j)>=(m d_f/2)[1-product_(l<j)(1-2t_l)]
            >=m d_f T_j/(1+2T_j).

The last step follows from
product(1-2t_l)^(-1)>=product(1+2t_l)>=1+2T_j.
All factors are positive under t_l<=2/7.

### Kernel drift relative to elapsed step size

The fixed-kernel decomposition remains exact:
k^(j+1)-k^(j)=-a_j Pi c(z^(j))/h. Let
e_j=||z^(j)-p_*|| and e_0=||p_*||<h sqrt(m)/2. Then

    ||k^(j+1)-k^(j)|| <=(h/m)t_j e_j^2.

The range contraction gives
e_(j+1)<= [1-(5/6)t_j]e_j. Since this factor is in[0,1],

    e_j^2-e_(j+1)^2 >=(5/6)t_j e_j^2,
    sum_(l<j) t_l e_l^2 <=(6/5)e_0^2.

Together with the simpler bound e_l<=e_0, this proves BOTH estimates

    ||k^(j)|| < (h^3/4)T_j,
    ||k^(j)|| < (3/10)h^3

when T_j>0. The zero case is exactly k=0. These bounds include every off-range tilt residual; no kernel pressure has been discarded or assigned zero map error.

Combine p_i=z_i+k_i with the componentwise lower bound. If 0<T_j<=1, then

    p_i^(j) > (21/20480)h T_j - (h^3/4)T_j
             >= (79/81920)h T_j > (h/2048)T_j.

If T_j>=1, then

    p_i^(j) > (21/20480)h - (3/10)h^3
             >= (39/40960)h >h/2048.

Here m>=1/2 and h^2<=1/4096 were used. The rational deductions use (1/4)/4096=1/16384 and (3/10)/4096=3/40960. Therefore, for every j with T_j>0,

    p_i^(j) > (h/2048) min(T_j,1) >0

for all six owners. At T_j=0 all pressures are exactly zero. This proves that coordinatewise projection onto p>=0 leaves every candidate of this IDEAL comparison unchanged. Each tentative backtracking candidate is a one-step extension of the accepted comparison history, so the same argument covers it without assuming that its score is accepted.

Thus the changing-gap comparison now includes the original orthant projection for arbitrary exact step sequences in the stated interval. With the full step on every update, section10's range convergence rate<16/17 and summable kernel limit apply to that projected ideal comparison. For varying steps the justified bound is instead

    e_j<=e_0 product_(l<j)[1-(5/6)t_l].

The proof does not show that the actual full-metric line search selects any particular positive step, or selects one at all. Even if every selected step is among powers0..12 of the full ideal step, the worst allowed step size is much smaller; the512-update full-step rate cannot simply be reused. No extra replay or iteration budget is granted.

Stored point/gap-midpoint/normal/step rounding, native material and actual delivery errors, discovery/rebuild, finite-iterate one-sided penetration/contact signs, original energy/angular/NUM005, SC1 whole-window propagation and SC2 remain separate. This is a manual ideal one-step comparison, not an executed contact result or a frozen physical integration contract.
