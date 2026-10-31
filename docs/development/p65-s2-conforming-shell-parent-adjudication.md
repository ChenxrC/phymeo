# S2-F parent ruling: conforming material geometry

Current status: STRICT_HOST_GEOMETRY_ACCEPTED; FULL_CONTACT_PHYSICS_NOT_FROZEN. See [parent acceptance](p65-conforming-shell-geometry-parent-acceptance.md) for tested be650fda/report df904777,99 host tests and full saved evidence, normal parent merge83c933e7 and released execution ownership. The original authoring ruling below is historical; its no-execution statements describe that earlier checkpoint.
This ruling follows a complete text reading of the362-line [proposal](p65-s2-conforming-shell-proposal.md), authored at c35a13a645bc68e912d390779ec29732ef07b644 and imported as cf44431bdc8b8e988ea4cfe2cae4e72d2613fc09. Working SHA256 a408663d056ff9883df8c3b3ce1f7230ad5c80bb251a7c72be6eedabcf2f7eff and Git blob33a9f375406dbbfc711f41a2a3bdad429a1a2b7a matched. The import receipt is artifacts/p65-parallel-parent-audits/s2-conforming-proposal-import-01.json. No source, test, numerical probe or executable checker ran for this ruling.

## Parent decisions and alternatives

Select the derived raw-area-star director and declared middle-particle point inertia model as the next finite conforming material model. It admits a changing nonflat material body while providing a complete force pullback. Independent directors would require separate angular inertia, state, energy and equations; the current proposal supplies none of those as a replacement. Keeping independent old face prisms would not provide a shared reference material body. Neither alternative is silently implemented.

The exact topology/director law and F1/F2 handwritten algebra are sufficient to start a concrete geometry prerequisite. They do not justify immediate wiring into the old finite-contact coordinator. That wiring still lacks changing-director held work, remote boundary self-touch admission, full cone ownership and active B2/general-robot energy proofs.

Use a bounded rational-interval GLOBAL proof route first, with independently certified planar convex constant-thickness reference components. This is the parent's finite-domain choice, not a claim that all valid shell references have that form. General topology and variable thickness remain representable; geometry admission explicitly reports an unsupported reference chart where the sufficient theorem is unavailable. Do not implement the proposal's incompletely specified multiquadratic/CAD route or infer its successful resource bounds. Every required later capability remains open.

The new [S2-FG contract](../superpowers/specs/2026-10-31-p65-conforming-shell-geometry-design.md) freezes six source files, six tests and two reports for the original S2 author. It is code/test authoring only. There is no executable owner. Original B2 remains released, and no duplicate builder/reviewer/implementer is started.

## Independent F1 derivation

The flat reference diamond times[-rho,rho] is convex and its two middle triangles partition it. The globally ordered staircase gives exactly the six oriented tetrahedra listed in proposal section8; every reference tetra has volume2*rho/3 and their total is4*rho. The side diagonal is shared, not independently selected per face.

On each triangle b=abs(eta) is affine, ||grad b||_2=1. In T0 the material signed outer weight is w=1+zeta/rho-b; in T1 andT2 it is w=b. Check these values at all four actual lifted vertices of each tetrahedron: affine interpolation then establishes the entire simplex formula. It is not a smooth prism or a bilinear interpolation formula.

For outer height q, a_outer=rho*(0,plus_or_minus q,1)/sqrt(1+q^2), while both hinge directors stay rho*e_z. Differentiation gives ||d n_outer/dq||=1/(1+q^2)<=1 and therefore ||a_outer-rho*e_z||<=rho*q for q>=0. Thus every PL tetra has
    ||D(F-Id)||_2
      <= q + rho*q*sqrt(1+rho^-2)
      = q*(1+sqrt(1+rho^2)).
For rho1/16 and q in[1/8,1/4],
    k <= (1/4)*(1+sqrt(257)/16) <33/64,
because sqrt(257)<17. Integrating the derivative along a segment in the convex reference body proves lower Lipschitz constant greater than31/64. The homotopy I+s*D(F-Id) never becomes singular, so reference orientation is retained. This is an all-simplex global injectivity proof, covering incident pairs without deleting them from contact tests. It implies each pair intersection is exactly its common material simplex.

The current raw areas are2*(0,-q,1) and2*(0,q,1); hinge S=4*e_z. All face/star/domain gates have nonzero margins: projected ratio and hinge-star ratio at least4/sqrt(17), and the invariant face-shape ratio at least1/2. These checks remain independent admission fields in code.

With unit middle masses, no elasticity/forces/gravity and only the two outer velocities e_z, free point equations give q(t)=1/8+t, K=1, P=2*e_z and L=0 through T1/8. This is a derived continuum-in-time free-particle trajectory of the declared point model, not a prescribed pin. The new geometry code will receive its analytic endpoints. It will NOT yet establish that an actual runtime/history/native path generated it; that later physical test remains necessary. Nor does this free fold stand in for active strained B2.

## Independent F2 derivative and impulse algebra

For the four-face fan at the flat reference, S0=sum_ring x_j cross x_{j+1}=4*e_z and is independent of x0. Its differential gives
    delta n0.x=(-delta x1.z+delta x3.z)/2;
    delta n0.y=(-delta x2.z+delta x4.z)/2.
For rho1/4 the top-center map is J0=I and the four ring blocks are
    J1=-(1/8)e_x e_z.T; J3=+(1/8)e_x e_z.T;
    J2=-(1/8)e_y e_z.T; J4=+(1/8)e_y e_z.T.
Consequently a tangential force at the top center pulls opposite vertical loads onto opposite ring vertices. Their moment is rho*f_x*e_y; replacing them with a center force preserves some resultants but loses this moment.

The shell contribution J*J.T is diag(33/32,33/32,1). Adding the genuine independent unit external point gives W=diag(65/32,65/32,2). There is no duplicate boundary-facet owner or extra mass. The normal impulse solving incoming speed-1 is1/2. With mu1/2, tangent capacity is1/4:
- incoming u65/256 sticks with t=-1/8;
- incoming u65/64 slips with t=-1/4 and remaining speed65/128.
Using p dot u + (1/2)*p.T*W*p gives respectively-1089/4096 J and-451/1024 J. Shell angular impulse is-rho*t*e_y and point angular impulse+rho*t*e_y. Their sum is zero. The point's whole free approach has constant angular momentum rho*u*e_y about the stated origin. The postimpact director rate-rho*(1/8)*t is nonzero.

These independent numbers support the complete-map algebra controls. They do not demonstrate actual applied impulse, original/selected-pre/native binding, late rollback, a postimpact trajectory or rotating loaded continuation. Those are separate future gates. The present source grant contains no contact coordinator.

## Proof boundaries the implementation must preserve

The reference constructor establishes combinatorial material identity, not physical embedding. Geometry certificates derive the ideal body from the exact values of stored FP64 middle coordinates and explicitly bound the proposed FP64 lifts/J. They are not claims of native FP64, R7 admission or exact stored force balance.

The sufficient global proof requires independently checked convex reference coverage, full lifted topology, each tetrahedron's reference gradients and all current/path domain gates. Reference projected triangle intersections include zero-area contacts, not just positive-area overlaps. Other components require complete support separation. Failed sufficient proof means unknown/unsupported; it must not be relabeled a witnessed collision.

A fixed complete Jacobian satisfies exact covariance/work identities by its analytic derivation. An interval residual containing zero is only a necessary numerical observation, never proof of an exact identity. The independent verifier reconstructs the formula and demands claimed intervals contain independent outward enclosures; overlap-only comparisons are forbidden.

The normalized path is nonlinear even though middle positions are affine. Exact polynomial edge/star coefficients, outward normalization and complete rational time coverage are mandatory. A common translation is factored symbolically before bounds. There is no frozen-velocity-times-h substitute. The strict certificate rejects remote boundary touching because the distinct material addresses would violate injectivity; no all-self-contact accepted label may be attached.

## Required follow-through, still open

1. A weak boundary-contact admission rule allowing legitimate distinct remote boundary contacts while proving interior disjointness and complete boundary fan/material/time occupancy.
2. Changing-director loaded continuation: actual held force replay must prove work, normal/capacity, couple and nonlinear path consistency. Endpoint J or a path-averaged heuristic does not supply that proof.
3. A physical noncollinear cone/owner law, including rank deficiency without spending nullspace self-stress.
4. External material-boundary robot manifold/CCD, complete public full-star port, actual one-replay/one-terminal coordinator, original-input joint energy and full R1/R2/ideal-extrema accounting.
5. Actual host/interpret/CUDA complete F1 and F2 windows/rebuilds/transforms, physical conservation and transaction faults. Current F1 geometry paths/F2 map equations cannot be recorded as those physical passes.
6. Active ordinary-mass B2 contact/fold controls, explicit R7 dispatch and general robot inertia/predictor accounting, then full fold/grasp/handoff and stage review/build/wheel.

No accepted S2-L, B2 R7, quotient, G2-P or adapter tests are repeated. No P6.5 completion or overall percentage follows from this ruling.
