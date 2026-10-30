# B2 pressure direction: an exact frictionless counterexample

Status: MATHEMATICAL_FINDING_AND_RESTRICTED_CANDIDATE_ONLY. This is parent hand algebra against the existing TriangleOffsetPointPort map, not a numerical probe, test failure, change to accepted quotient algebra or implementation grant. Full B2 contact physics remains NOT_FROZEN. It refines the known cone limitation in [the parent adjudication](p65-b2-contact-adjudication.md).

## Existing map and declared domain

Use the already proposed SC1/SC2 flat middle geometry:
a=(0,0,0), b=(1,0,0), c=(0,1,0), d=(0,-1,0).
The oriented triangles are U=(a,b,c), D=(b,a,d), both normal+e_z. All six vertex anchors use bottom offset-rho, rho>0. Retain both owners at each shared vertex a,b. There is one physical particle base and one prismatic z robot whose point velocity is v_R*e_z. Nothing is dropped or merged by coordinate equality.

Let B have18 owner-world rows and13 physical coordinate columns (12 particle translations and one prismatic coordinate). It maps particle-minus-robot relative point velocities. Let l_edge=1 m and r_offset=rho/l_edge, which is dimensionless. At either triangle's vertex i, direct differentiation of the normalized face normal and bottom offset gives

    upper:
      u_i.x = v_i.x + r_offset*(v_b.z-v_a.z),
      u_i.y = v_i.y + r_offset*(v_c.z-v_a.z),
      u_i.z = v_i.z-v_R;

    lower:
      u_i.x = v_i.x + r_offset*(v_b.z-v_a.z),
      u_i.y = v_i.y - r_offset*(v_d.z-v_a.z),
      u_i.z = v_i.z-v_R.

These are the complete offset map, not a barycentric-only normal map. They follow from upper normal rate(-v_b.z+v_a.z,-v_c.z+v_a.z,0)/l_edge and lower normal rate(-v_b.z+v_a.z,v_d.z-v_a.z,0)/l_edge, with y_i=x_i-rho*n. The source formulas in coupling/offset_port.py and their assembly in solvers/_thick_contact_geometry.py match these signs. No project import or call was used to establish them.

## Exact null vector

The following DIMENSIONLESS owner-world null-basis vector k has only four nonzero blocks. Multiplying it by an impulse scale in N*s would give a null impulse:

    k_Ua=(0,+1,+r_offset),
    k_Da=(0,-1,+r_offset),
    k_Uc=(0,0,-r_offset),
    k_Dd=(0,0,-r_offset);
    k_Ub=k_Db=0.

For any physical velocities,

    k dot (B v)
      = u_Ua.y-u_Da.y
        +r_offset*(u_Ua.z+u_Da.z-u_Uc.z-u_Dd.z)
      = r_offset*(v_c.z+v_d.z-2*v_a.z)
        +r_offset*(2*v_a.z-v_c.z-v_d.z)
      =0.

The robot terms also cancel exactly. Hence B.T*k=0. The tangent entries have opposite signs on coincident-in-world but distinct material owners. This is an algebraic null vector, not a physically admissible frictionless load.

Now take a purely normal owner impulse p, with physical-node aggregates

    F_a=c0-2*b0, F_b=c0, F_c=c0+b0, F_d=c0+b0,

where c0>2*b0>=0, in N*s. Both shared-node splits can be arbitrary as long as their aggregate is preserved. Every aggregate is strictly compressive, and a split with positive members exists. Nevertheless,

    p dot k = r_offset*(F_a-F_c-F_d)
            = -r_offset*(c0+4*b0) !=0

with units N*s. This value is independent of how either shared vertex is split among its two normal owners. The simpler b0=0 case already suffices. The nonzero b0 family has the aggregate pattern of the initial SC1 support target, but no particular finite-XPBD trajectory or parameter-dependent magnitude is assumed by this counterexample.

## Consequence for the unrestricted world gauge

For exact on-range delivery ell=B.T*p, the accepted quotient identities give

    p_eff=Pi_range(B)*p, r=0,
    p_eff perpendicular to every k in ker(B.T).

Suppose that the zero-nullspace representative p_eff were purely normal. The x/y physical forces would vanish and its z-node/robot delivery would have the same aggregates F. The previous dot-product calculation would then force p_eff dot k=-r_offset*(c0+4*b0), contradicting orthogonality. Therefore EVERY zero-nullspace representative of this delivery has a nonzero tangential owner component.

With mu=0 the exact frictionless law requires each tangent impulse to vanish. Treating the unrestricted three-world-coordinate p_eff as the required physical pressure would reject this legal pure-normal delivery, even with no conversion error, no remainder and perfect arithmetic. This is stronger than a loose error bound: it is an explicit incompatibility for this witness selection. It does not prove SC1/SC2 dynamics fail, and it does not prove a numerical backend is incapable of them.

The standalone quotient implementation remains mathematically correct. Its contract explicitly promises an unconstrained Euclidean representative, not a pressure law. Do not modify that accepted implementation, rerun its83 tests, add the proposal's null self-stress to its output, relax friction, or spend an epsilon tangent allowance to hide this result. S2's already frozen full-rank R1/R2 contract also remains unchanged.

## Candidate direction restriction for a separately declared mu=0 law

A future frictionless-only physical contract may declare its admissible pressure directions BEFORE seeing a delivery. Let N0 be the18-by6 block embedding with each column the corresponding fixed normal n_i. It preserves every owner; the physical law is p_world=N0*lambda with scalar lambda_i>=0. Define

    B_n=N0.T*B, ell=B_n.T*lambda+r_n.

Applying the unchanged quotient algebra to B_n certifies a zero-nullspace representative within this declared normal-only law. It does not select a favorable three-dimensional null self-stress AFTER examining the outcome. Retain the full B, N0, requested world loads and original owner identities in the enclosing physical receipt. All tangential physical-coordinate delivery/uncertainty remains in r_n; it is not discarded.

For this flat example each row of B_n is exactly v_i.z-v_R. Its six rows have rank4; the two null directions are the differences of duplicated a and b normal owners. The on-range minimum scalar-owner norm split is F_a/2 at both a owners, F_b/2 at both b owners, F_c at c, F_d at d. It is strictly positive when c0>2*b0 and has r_n=0, without manufacturing additional normal capacity.

At particle masses m and robot mass4m,
    (A_n)_{ij}=(delta_same_physical_node(i,j)+1/4)/m.
For m=1, ||A_n||_infinity=7/2: duplicated-node rows sum to2+6/4, other rows to1+6/4. On the four-dimensional range, the duplicated-incidence Gram matrix has eigenvalues at least1, and the robot term is positive semidefinite and annihilates that nullspace. Their sum therefore has exactly the same nullspace as the duplicated-incidence Gram matrix. Thus A_n's positive eigenvalues lie in[1,7/2]. H_n=A_n/(7/2)+Pi_null has eigenvalues in[2/7,1], including unit nullspace eigenvalues. Consequently kappa_2(H_n)<=7/2 and kappa_infinity(H_n)<=6*kappa_2<=21<10^12. Uniform positive mass scaling leaves H_n and the representative unchanged. This is a hand proof for the fixed flat map, not a curved/native window condition result.

This candidate MUST remain a new, explicitly declared mu=0 physical domain:
- No use when either static or dynamic friction is positive, no velocity-dependent basis selection, no post-failure fallback from a world witness and no hidden row deletion.
- N0 and its complete original/actual/ideal normal bindings need independent checks; changing geometry or uncertain normals are not solved by this flat algebra.
- Requests, actual predictor/terminal delivery and all original R2-style observed uncertainty remain distinct. Full ideal-extrema response, resultant, couple, virtual work and original-input joint energy must bound r_n and all normal/map errors.
- Scalar lower-normal bounds must remain genuinely positive where loaded, both phases have separate once-only capacity, and an open birth cannot borrow another phase's pressure.
- Full requested loads, including any null component and their ideal-map discrepancies, remain charged. No internal work is external credit.
- Noncollinear/positive-friction physical laws, changing held geometry, ordinary-mass nonlinear success controls and actual R7 bounds remain unresolved. No shared source is granted by this note.

This resolves why the unrestricted world representative cannot by itself supply the proposed frictionless success controls. It does not yet supply the full replacement physical contract.
