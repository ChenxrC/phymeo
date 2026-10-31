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
