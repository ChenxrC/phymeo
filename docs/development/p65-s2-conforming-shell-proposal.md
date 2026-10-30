# S2-F conforming simplicial shell and lawful-fold proposal

Status: **PROPOSAL_NOT_FROZEN**. Documentation, source reads and handwritten derivation only. No import, numerical probe, collection, test, formatter, build, installation, profile, helper or reviewer was run. This is a NEW material/body contract; accepted S1/LG/S2-L and their evidence remain unchanged.

Starting child: f504f53e77407426c24cae769cc80b7c08c395dd, branch codex/p65-self-contact-module. Authority read by absolute parent path: task-2-conforming-shell-proposal-brief.md at parent 1fe63de1e97f8b1907a1ceeb5c4dabf6c0012316; the integration design including R4/R7; self-response design R1/R2; parent response math audit; B2 contact adjudication and exact contact-quotient design. No parent merge occurred. Only this document is writable. The final commit/hash are reported outside the document to avoid a self-referential hash.

## 1. Recommendation and precise physical claim

Recommend a derived-director, point-lumped-inertia, conforming piecewise-affine material body, with an explicitly empty elastic model admitted as well as future independently audited middle-surface elasticity. Its configuration is the existing middle-node position vector. Directors are differentiable functions of that configuration, not independently advanced kinematic state. Every load on the geometric body is pulled back through the COMPLETE differential. This gives a coherent finite mechanical model and a nonempty successful changing-fold domain below. It does not claim a continuum fabric law, volumetric inertia or active strained B2/contact acceptance.

An independent-director alternative requires genuine orientations R_i in SO(3), angular velocities, positive inertia I_i [kg*m^2], kinetic energy 1/2 sum(omega_i.T I_i omega_i), director/shear energy and rotational equations I_i*omega_dot + omega_i cross (I_i*omega_i) = applied torque - rotational energy gradient. For a_i=rho_i R_i e_z, delta a_i=delta theta_i cross a_i and a material force contributes a_i cross f to the rotational equations. That changes state families, mixed-unit mass/response metrics, native integration and transaction schemas. Neither copying directors nor assigning zero inertia supplies those dynamics. The accepted all-translational quotient algebra does not authorize this mixed-coordinate extension. This larger alternative is not recommended for the first S2-F contract.

Legacy TrianglePrismSurface remains exactly its present three-node face-normal prism. The new body is neither that prism, the warped bilinear map sum b_i(x_i+s a_i), nor the convex hull of six endpoints. Its material interpolation and subdivisions below are the definition of the body. An averaged normal is not asserted physically valid without these declared law, mass and admission choices.

## 2. Owned reference complex and conforming product tetrahedra

Input K is a finite pure oriented simplicial two-manifold, optionally with boundary and multiple connected components. Own nonnegative distinct int64 node IDs, distinct int64 face IDs, oriented triples with three distinct nodes, immutable reference positions X [N,3] in metres and positive per-vertex rho [N] in metres. Row indices are storage only. Reject duplicate unoriented faces, unused nodes, nonmanifold edges, inconsistent winding, a disconnected vertex link, a bow-tie link, or a link other than a cycle (interior vertex) or a single interval (boundary vertex). Check each edge has one or two incident faces; the two induced edge orientations must oppose. K's boundary is its one-face edge set with the induced orientation. No ghost faces or boundary-normal extension are inserted.

The reference material space is |K| times [-1,+1], with globally shared lifted vertices (node_id,sigma), sigma=-1/+1. A material top/bottom label is global, never recomputed by world height. Let i<j<k be the immutable global node-ID order of a face. Subdivide its prism product into these UNORIENTED four-vertex sets, with the displayed preliminary orders:

    T0=(i-,j-,k-,k+)
    T1=(i-,j-,j+,k+)
    T2=(i-,i+,j+,k+).

Let epsilon be +1 if the sorted triple (i,j,k) has the same orientation as the caller's face triple, otherwise -1. Preliminary product orientations are epsilon*(+1,-1,+1) for T0,T1,T2. If negative, swap the last two entries of that tetrahedron. This rule follows by taking determinants in an oriented triangle chart with its positive thickness coordinate; it does not inspect rounded world coordinates. On every edge i<j the side diagonal is i- to j+, independent of the neighboring third vertex or face winding. Neighboring cells consequently share identical triangular side facets with opposite induced orientation. The oriented simplex boundary formula sum_a (-1)^a [v0,...,omit va,...,v3] supplies all facets; two-incident facets cancel. Top triangles inherit K's orientation, bottom triangles reverse it, and the remaining external side triangles lie over boundary edges.

A tetrahedron ID is (face_id, staircase_index); its oriented vertex tuple and caller/canonical permutation remain bound. A boundary facet ID binds its sorted lifted IDs, its unique incident tetrahedron and induced orientation. Never infer identity from coincident coordinates. Permuting node/face storage with stable IDs preserves this subdivision. Arbitrarily changing the ID order changes the material triangulation unless the ordering keys are transported explicitly; do not claim invariance under a different subdivision.

Reference lift Z_i^sigma=X_i+sigma*a_i(X) uses the SAME director law as current geometry. The reference itself must pass topology, determinant and complete embedding verification. Invalid reference geometry cannot be blessed as rest overlap. The abstract incidence relation is the intersection of lifted vertex sets in this validated simplicial complex, recomputed by the verifier. The dimension of a shared simplex may be -1,0,1,2 or 3. Internal tetrahedra/facets are material interpolation, never friction surfaces or separate capacity owners.

## 3. Director law, support domain and complete sparse differential

The proposed version raw-area-star-director-v1 has FIXED weights w_if=1 for every incident oriented face and zero otherwise. For f=(a,b,c),

    c_f=(x_b-x_a) cross (x_c-x_a) [m^2],
    S_i=sum_(f incident to i) c_f [m^2],
    n_i=S_i/||S_i||,    a_i=rho_i*n_i [m].

Every face appears once in its entire vertex star. Boundary vertices use exactly their incident faces; no face count normalization, current unit-normal weighting, angle weighting, history or current-area-dependent scalar weights are hidden. The current raw areas already carry area magnitude, and their derivatives below are mandatory. A future nonconstant w_if requires the additional c_f*D w_if term and a new law version.

Proposed admission requires every c_f and S_i nonzero over the complete path, all tetrahedral determinants strictly positive in the fixed reference orientation, and complete material embedding. Additionally propose the explicit geometric support domain n_i dot (c_f/||c_f||) >= 1/4 for every incident (i,f). Thus each corner director has projected half thickness at least rho_i/4 (not a claimed global minimum body thickness); this is a declared restriction, not a distance/angle exemption for collision or an epsilon-expanded body. Check the sign and squared inequality exactly/outward. A proposed invariant conditioning gate is ||c_f||/max_edges ||edge||^2 > q(1e-10), and ||S_i||/sum_incident ||c_f|| > q(1e-10). These are NEW sufficient admission gates for this port, not claims of equality with legacy scaled-area admission; legacy guards remain untouched. They are parent decisions, fixed before execution, not values fitted to examples. Vanishing sums, unresolved inequalities, rounding losses or exhausted proof resources reject; never retain a previous director.

Write [u]_x v=u cross v. The exact raw-area differential blocks are

    H_f,a=[x_c-x_b]_x,
    H_f,b=-[x_c-x_a]_x,
    H_f,c=[x_b-x_a]_x,
    H_i,j=sum_(f incident to i and j) H_f,j,
    D a_i/D x_j=(rho_i/||S_i||)*(I-n_i*n_i.T)*H_i,j.

For an owned tetrahedral material point with fixed nonnegative barycentrics lambda_a, sum lambda=1,

    y=sum_a lambda_a*(x_i(a)+sigma_a*a_i(a))
     =sum_i beta_i*x_i + sum_i gamma_i*a_i,
    beta_i=sum_(a:i(a)=i) lambda_a,
    gamma_i=sum_(a:i(a)=i) sigma_a*lambda_a,
    J_j=beta_j*I + sum_i gamma_i*(rho_i/||S_i||)
                                  *(I-n_i*n_i.T)*H_i,j.

J_j is dimensionless and has a block for EVERY node in the union of the relevant incident stars, including nodes outside the point's parent face. Gather overlapping blocks by stable physical node before response/scatter. This is an analytic map, not a finite difference, frozen-normal map or three-node attachment.

Translation gives sum_j H_i,j=0 and therefore sum_j J_j=I. For any omega, rotation covariance of the cross product gives D S_i[omega cross x]=omega cross S_i, D a_i[omega cross x]=omega cross a_i, and

    sum_j J_j*(omega cross x_j)=omega cross y.

Consequently delta y=sum_j J_j delta x_j and, for f [N] or p [N*s],

    sum_j (J_j.T f) dot delta x_j=f dot delta y,
    sum_j J_j.T f=f,
    sum_j x_j cross (J_j.T f)=y cross f.

These identities follow from the actual law and hold for proper rotations and translations. Reflection requires an explicit orientation/top-bottom transformation; it is not silently treated as a proper rotation. Independent checks use source positions/topology/rho and exact/outward arithmetic, not producer J. Actual stored blocks and actual stored world points also need representability/resultant/moment checks inside the original budgets.

## 4. Mass, energy and one physical response

Declare fixed positive ordinary point masses m_i [kg] at the middle nodes. K=1/2 sum_i m_i |v_i|^2 and L=sum_i x_i cross m_i v_i. The geometric offsets introduce NO independent kinetic state, hidden rotary inertia or free director update. A reference-volume-based lumping rule could set these fixed masses, but it must be explicit; user masses are not automatically a density/thickness continuum law. This point-lumped inertia omits through-thickness volumetric inertia. If that inertia is required, the consistent kinetic metric is M(x)=integral rho_ref*J_Y(x).T J_Y(x) dV_ref, with its configuration-dependent inertial terms. Keeping constant diagonal masses while claiming that integral is invalid.

The minimal complete model admits U=0 explicitly. For middle-surface elasticity, distance energy (length-rest)^2/(2*alpha_e) and hinge energy (theta-theta0)^2/(2*alpha_h) require their complete original gradients and independently certified branch/energy/path evidence. Existing B2 supplies a separately scoped middle-surface law and v1/v2 path contracts, not a volumetric shell law or automatic shell-contact dispatch. If director shear, volume or thickness energies are added, declare their coefficients, reference measure and chain-rule terms through every D a_i/D x_j. For example a tetrahedral density Psi(F_T) needs derivatives of every lifted corner and hence the same full stars. No unstated elasticity or continuum modulus is attributed to raw-area averaging.

For stacked self/external material rows, D contains the COMPLETE endpoint maps, with signs and any other participant's actual point map. ell=D.T p; W=D M^-1 D.T, including all shared-node, star-overlap and shared-robot cross terms. Aggregate before mass response and before kinetic squares. At a fixed terminal configuration,

    u_post=u_pre+W p,
    DeltaK=p.T u_pre + 1/2 p.T W p.

The full-row-rank R1/R2 sufficient domain and strict condition ceiling remain available. The accepted quotient module is only fixed-map algebra when rank is deficient: it neither supplies a cone-admissible physical pressure nor lets proposal null self-stress create capacity. A new physical use requires its own proof and parent adjudication.
## 5. Complete material incidence and a finite embedding certificate

For every unordered tetrahedron pair T,U, including subdivisions of the SAME cell, recompute S=T intersect U in the abstract reference complex. Require nondegenerate positive affine maps and

    F(T) intersect F(U) = F(S),

with empty right side when S is empty. For T=U the identity case is checked as the same tetrahedron, not an exemption for distinct duplicate cells. ID sharing is only the proposed relation, never its geometric proof.

Sufficiency is direct: if two material points have the same geometric image, choose containing tetrahedra. The intersection condition puts that image in the common simplex. Injectivity of each affine tetrahedron then places both preimages at the same common reference point, so the assembled continuous map is injective. The finite compact reference body has a continuous inverse on its image: images of closed reference subsets are compact and therefore closed. The validated product topology supplies the three-manifold/boundary structure, and positive oriented tetrahedra make the singly incident facets the true oriented external boundary. Positive determinants alone do not supply the pair-intersection premise.

An implementable sufficient pair witness uses a separator vanishing on S and STRICT inequalities on all off-S vertices. For a common facet choose its oriented cross-product normal; for an edge e choose d=e cross w with a supplied fixed rational w; for a vertex use a supplied rational d through that vertex. Independently verify the zero identities on S and d dot(v-s0)>0 on every off-S A vertex and <0 on every off-S B vertex, or the reversed signs. Every convex combination in the intersection must have all off-S coefficients zero, hence belongs exactly to F(S). For disjoint tetrahedra, a strict support gap on a supplied rational axis proves emptiness. A failed separator is unknown, not a demonstrated collision or a hidden omission.

At a static algebraic snapshot, complete convex intersection is an additional route: use barycentric feasibility lambda,mu>=0, sum lambda=sum mu=1, Z_T lambda=Z_U mu. Prove all feasible vertices have zero total off-S barycentric mass (and equal shared-simplex coordinates), or provide a verified point outside S. Equivalently, a complete checked LP dual can bound that nonnegative objective by zero; a disjoint pair uses a checked infeasibility dual. Enumerating all active sets of this fixed small polytope gives a finite candidate procedure; signs/rank/equalities must be exact, not float topology. The verifier establishes completeness/binding independently. Unsupported degeneracy returns unknown.

A second sufficient route proves the entire body at once. If the reference body is a certified convex polytope D and the continuous PL map obeys ||D F-I||_2 <= k<1 on EVERY reference tetrahedron, then integration along the segment joining any two reference points gives |F(p)-F(q)| >= (1-k)|p-q|. It is globally injective. The homotopy I+s*(D F-I) is nonsingular for all s in [0,1], so every orientation remains positive. This route covers all tetra pairs, not just neighbors. Reference convexity/coverage must itself be proved, for example by a nonoverlapping planar triangulation of a convex polygon whose exact triangle areas sum to the polygon area, followed by its conforming constant-thickness product. A nonconvex or disconnected reference cannot use this shortcut without a separate domain proof.

IMPORTANT support distinction: this STRONG embedding certificate excludes coincidence of distinct remote material boundary addresses. Thus it cannot simultaneously be called acceptance of remote zero-gap self-contact. A future contact-admissible boundary relation would allow only explicitly paired EXTERNAL-boundary coincidences (top, bottom or true boundary-side facets and their incident edges/vertices), while excluding every material-interior intersection and every undeclared overlap. Internal staircase facets are never eligible. Shared-reference material points remain the SAME material point, without friction; two remote touching addresses remain DISTINCT material points/endpoints and are never merged by coordinate equality. A capacity owner binds that declared material pair/generation, with duplicate facet/tetra representations removed by reference identity, not by dropping distinct physical owners. The extension must verify the complete paired boundary set, one-sided material occupancy at every participating fan, all surrounding tetrahedra, time coverage and contact/phase laws. Its sufficiency is still an unresolved obligation, not a reinterpretation of the strong certificate. External robot contact involves a different body and does not violate the shell's internal embedding. The first nonempty folded domain is useful without claiming that the remaining self-touch problem is finished.

## 6. Continuous curved geometry, exact identities and finite resources

The declared physical path is x_i(t)=x_i^0+t*(x_i^1-x_i^0), t in [0,1], for each actual accepted substep h [s]. Directors are recomputed by the law THROUGHOUT that path. Therefore S_i(t) is quadratic, Q_i(t)=S_i dot S_i is quartic, and z_i^sigma(t)=x_i(t)+sigma*rho_i*S_i(t)/sqrt(Q_i(t)) is generally nonlinear. Neither endpoint tetrahedra nor a frozen velocity times h proves this path.

Use exact rational polynomial coefficients from the stored endpoint values. Prove Q_i bounded strictly above zero on each time cell using polynomial/Bernstein bounds or a positive projection S_i(t) dot r_i onto a fixed rational r_i. For the latter the polynomial is quadratic, so its minimum includes its interior critical point; divide by an outward upper norm of r_i. Root bounds are rational and outward. Check the face/domain/projected-thickness and tetrahedral orientation conditions on the complete time cover independently of candidate culling.

Permanent material sharing is factored symbolically BEFORE interval evaluation: one lifted ID has one x_i/rho_i/S_i expression; equal shared vertices cancel exactly, shared edges are the same affine material map, and d=e cross w or d=e1 cross e2 has the required exact zero dot products. Then interval-bound only off-shared-vertex inequalities. No positive ordinary separation is demanded at a permanently shared vertex/edge/facet. Every pair is accounted for by a checked interval separator, a checked global embedding bound, or a remaining obligation; a broad phase may omit a pair only with its own whole-time disjoint-box certificate.

For the global route, use the exact reference tetrahedron barycentric gradients and collect coefficients by middle-node ID:

    D(F-Id)=sum_i (x_i-X_i) tensor grad(beta_i)
              +sum_i (a_i-a_i_ref) tensor grad(gamma_i).

Common translation cancels exactly. Sum bounds on the rank-one terms (or another independently checked matrix-norm bound) may certify k<1. A supplied proper rigid gauge of the reference is permitted only with exact orthogonality/orientation and binding checks. This is a generic certificate from the law and reference data, not a branch for a named fixture. Section8 proves a coarse one-cell successful instance, so success does not rest on an unbounded subdivision promise.

For a fixed material point, y''=sum_i gamma_i*a_i'' because the middle path is affine. With r_i<=||S_i||, the exact normalized-vector differentiation gives the conservative bound

    ||n_i''|| <= ||S_i''||/r_i + 3*max||S_i'||^2/r_i^2.

It can bound curved support/chord errors together with all relevant directors, but cannot replace the incidence or complete external-boundary proof. Evaluate J(t) directly with bounded S,H,n denominators for phase-work/moment bounds; preserve dependencies at permanent identities.

At a fixed rational time, a pair of cell tetrahedra uses at most six distinct vertex-director square roots. Propose a NEW bounded multiquadratic kernel, not the legacy two-root type. Build an independent square-class basis: for a new positive rational radicand C, test C divided by every product of existing basis radicands for an exact rational square; at most64 subset tests with six roots. Pairwise ratio tests alone miss higher-order dependencies. A reduced basis has at most64 coefficients. Coefficient zero is exact only after this reduction; nonzero sign uses outward refinement, and exhausted refinement is unknown. Static normals/clipping use unnormalized algebraic directions and sign-aware squared comparisons rather than inventing additional unit-normal roots. Unrelated global rows are accumulated as rational outward bounds, not forcibly embedded in one small field.

No full algebraic-function solver is claimed for time-varying root dependencies. The first continuous producer uses interval inequalities plus independently checkable polynomial/shared-feature identities. An equality outside that supported identity language is unresolved. A later exact elimination/Sturm/CAD extension would need its own bounded proof contract. The two controls below use identities and strict bounds already supplied here.

Proposed NEW proof ceilings:64 nodes,40 middle faces,128 tetrahedra (the40-face product has120),16 faces per vertex star,8192 unordered tetra pairs,6 local roots/64 field coefficients,64 initial sqrt bits doubling to4096,4096 time cells, depth32,4096 numerator/denominator bits and2,000,000 checked rational scalar operations per producer or independent verifier invocation. Smaller caller limits are allowed; no automatic expansion. These delimit the first implementation and are not a theorem for arbitrary meshes. A separate exact cover receipt binds every cell/pair and all pending siblings, with a gap-free closed [0,1] partition. Producer-reported counters or a constructed receipt do not authenticate themselves.

Existing response limits128 contacts,64 anchors/pair,8 rebuilds,512 outer/normal iterations,96 tangent iterations,16 line halvings and512 material iterations stay unchanged. Preserve penetration0, anchor error2^-40m, Lref=1m, Vref=1m/s, Jref=1N*s; host tau1e-8, actual tau2e-5; host energy/angular1e-8 and actual1e-5 per step AND noncancelling full window. Enclosure width remains inside1/64 of the corresponding original budget. No B2 standalone1e-3 allowance is borrowed.

## 7. True boundary, external contact and public delivery

Enumerate only singly incident oriented tetra facets as material boundary. Do not convex-hull-fill a concave shell or create contacts on internal staircase faces. Boundary edges/vertices carry the full incident boundary-facet fan and minimal reference-simplex identity. Exact geometry retains every supported maximal tie, complete clip extrema and both material endpoints. Multiple coplanar facets at one material point do not multiply a capacity owner.

For a dynamic point, a finite oriented robot patch, or a separately declared convex robot primitive, new shell geometry must cover all relevant tetra/boundary candidates, finite patch edges/vertices and one-sided material occupancy. Individual tetra SAT may prove an omission, but an internal tetra face cannot become a physical friction normal. A boundary-triangle-only crossing test is insufficient to establish that a robot is outside the material union. Robot penetration into any material tetrahedron, external-boundary crossing and complete swept coverage must be accounted for. Bind actual robot geometry/FK/path/identity; the first exact path domain may be coordinate-prismatic/affine only. General robot rotation/tree energy is not supplied by this proposal.

All exact active normal directions are evidence. The first PHYSICAL loaded domain may require one oriented exposed ray after complete boundary-fan checks, including a coplanar multi-face vertex as in F2. Geometry still covers edge/vertex ties; noncollinear loaded normal cones remain unresolved pending a genuine cone/friction/capacity law. Neither a lexicographic first face nor the quotient algebra chooses that law. Full self-touch and general cone support remain mandatory later work, not hidden omissions.

Own material addresses as tetra/minimal-simplex IDs and four nonnegative int64 counts summing2^52, plus ideal algebraic barycentrics and exact caller/canonical correspondence. A point on a shared simplex has one canonical lifted-ID address, not one per incident tetrahedron. Quantization must retain the actual selected finite support and independently bound error at all ideal extrema; rounded world equality is not ownership. rho and +/- labels belong to geometry, not a fabricated legacy signed face-normal offset.

A NEW ShellMaterialPointPort over one public all-node base exposes positions, velocities, response, snapshot and candidate audit with the full sparse J. It binds the original base snapshot and is rebuilt whenever configuration, material address or law changes. A graph response/scatter is a proposal only. Held requests aggregate ell_H once and enter SceneLoadReplay.evaluate through the registered force channel; terminal requests aggregate ell_T once into PointVelocityImpulses and TerminalImpulseProgram. Never apply a graph impulse and then apply the terminal batch again. PointParticipantProgram already supports stateless mathematical particle domains and exactly-once validated terminal batches; it is not a robot-domain adapter. Parent owns the separate articulated/robot composition and all changed shared seams.

Changing directors make held loading a new obligation. The existing replay applies a CONSTANT nodal addition ell_H/h. If ell_H=J(x0).T p, it is generally not the continuously varying material-point force J(x(t)).T p/h. For fixed p,

    p dot Delta y/h - ell_H dot Delta x/h
      = p dot integral_0^1 (J(x(t))-J(x0))*Delta x dt / h.

Likewise frozen-map nodal moment differs from the actual moving material-point moment. Bound these complete path defects and phase-normal/capacity/slip obligations inside the original budgets, or reject. A path-averaged/discrete-gradient force construction would change the held scheme and requires a separately derived work/moment/native contract; it is not silently supplied here. Existing constant-normal S2-L verification cannot certify a new rotating loaded held step. F1 has zero held contact; F2 has only a terminal birth. Nonzero rotating loaded continuation is therefore a precise remaining physical obligation.

Terminal geometry stays fixed during the velocity operation. Independently reconstruct actual impulse from native pre/post momentum, keep requested batches separate, and extend R2 to the complete stored full-star map. Exact stored-map capacity decomposition and ideal-map bounds must retain all shared-star terms and the residual. Gamma at selected addresses remains a stored-gauge certification enclosure, not an ideal pseudoinverse, unique physical pressure, all-extrema coverage or nonlinear position bound. Independent maps at every ideal boundary extreme remain necessary.

Original force/gravity work, explicitly external actuator work and internal self/cloth-robot transfer are distinct. Joint energy is actual Delta(K+U)-W_original_external; internal held/terminal work and residual work are never external credits. Actual separated-endpoint couple is (y_A-y_B) cross p, not automatically zero. Exact coincidence plus complete pullbacks gives joint impulse moment cancellation; approximation/native discrepancies remain charged. Late geometry/material/path/phase/diagnostics/close failures must roll back the whole frame, original inputs and history, including earlier accepted substeps.
## 8. F1: an actual free-particle changing fold, with global proof

This is a new four-node connected two-face material body, not the earlier nonincident contact handoff. All lengths are metres, times seconds, masses kilograms. Stable IDs and reference positions are

    X0=(-1,0,0), X1=(1,0,0), X2=(0,1,0), X3=(0,-1,0),
    faces=(0,1,2),(1,0,3), rho_i=1/16, m_i=1.

The reference planar diamond has two nonoverlapping oriented triangles of area1 each and its product is the EXACT convex domain

    D={(xi,eta,zeta): |xi|+|eta|<=1, -rho<=zeta<=rho}.

Reference directors are e_z. There are exactly these six positively oriented tetrahedra, in the fixed order of section2:

    upper T0=(0-,1-,2-,2+)
    upper T1=(0-,1-,2+,1+)
    upper T2=(0-,0+,1+,2+)
    lower T0=(0-,1-,3+,3-)
    lower T1=(0-,1-,1+,3+)
    lower T2=(0-,0+,3+,1+).

Both face cells share the side diagonal0- to1+. Each reference tetrahedron has volume2*rho/3>0. Their volumes sum to4*rho, the diamond area2 times thickness2*rho; the staircase inequalities give disjoint interiors and exact coverage. No convex-hull replacement is used.

Set the actual initial positions x0=X0, x1=X1, x2=X2+(0,0,1/8), x3=X3+(0,0,1/8). Velocities v0=v1=0, v2=v3=e_z. Use the explicitly EMPTY elastic model U=0, zero original force, zero gravity, no pins, no welds and no contact requests. The free-particle equations m_i*x_ddot_i=0 uniquely give

    x0(t)=X0, x1(t)=X1,
    x2(t)=(0,1,q(t)), x3(t)=(0,-1,q(t)), q(t)=1/8+t,
    0<=t<=T=1/8.

Thus this is an actual physical solution of the declared mass/energy model, not a geometry path imposed on a dynamically different system. It is deliberately not an active strained-B2 trajectory. Use three complete windows h=1/64,1/128,1/256, respectively8,16,32 steps, with TWO independent factory reconstructions per h from the same owned original inputs. Reconstruct every next step from the actual published state/history, not a retained private geometry cache.

Throughout the window q is in[1/8,1/4]. Raw areas and directors are

    c_upper=2*(0,-q,1), c_lower=2*(0,q,1),
    S0=S1=4*e_z,
    a0=a1=rho*e_z,
    a2=rho*(0,-q,1)/s, a3=rho*(0,q,1)/s, s=sqrt(1+q*q).

The projected-thickness ratio is at least1/s>=4/sqrt(17)>1/4. The face shape ratio is at least1/2 (the longest edge has squared length4); the hinge sum ratio is1/s and the outer sum ratio1, so the stated conditioning gates hold with a large exact margin. The two oriented face normals differ by angle2*atan(q). Its change over the window is between4/17 and1/4 radians, obtained by integrating2/(1+q*q) over[1/8,1/4]. This is a nonzero fold interval, with an ordinary shared edge and valid full surrounding stars.

Here is a proof for EVERY PL simplex, rather than just the smooth/bilinear wedge. On either wing let b=|eta| and delta_a=a_outer-rho*e_z. For fixed reference tetrahedral barycentrics let w=lambda_outer_plus-lambda_outer_minus. Directly from the lifted coordinates,

    on T0: w=1+zeta/rho-b;
    on T1,T2: w=b;
    F_t-Id=q*b*e_z+w*delta_a.

Changing the display orientation of the lower tetrahedra only permutes barycentrics; it does not change these formulas. In the Euclidean norm ||grad b||_2=1 and on EVERY tetrahedron ||grad w||_2<=sqrt(1+rho^(-2)). Since ||d n_outer/dq||_2=1/(1+q*q)<=1, ||delta_a||_2<=rho*q. The induced matrix2-norm of a rank-one tensor is the product of the two vector2-norms. Therefore on all six tetrahedra and at every time,

    ||D F_t-I||_2
      <=q*(1+sqrt(1+rho*rho))
      <=(1/4)*(1+sqrt(257)/16)
      <33/64<1.

The straight segment between ANY two points of the convex D can be partitioned into its finite tetrahedral pieces (and evaluated by continuity on facets). Integrating their derivative bounds yields

    ||F_t(p)-F_t(q)||_2 >= (31/64)*||p-q||_2.

Thus the whole body is injective for the entire interval, including every same-cell, side-face, edge and vertex incident pair. The homotopy to identity proves every determinant positive; every current tetrahedron volume is at least(31/64)^3 times its reference volume. This proves the complete strict material-intersection relation without asking permanent shared features for a positive separation. It also establishes a finite, nonempty success domain for the proposed global verifier: one convex reference witness and the displayed one-cell bound suffice, without time subdivision or an infinite precision promise.

The exact physical balances for the full window are K=1J, P=2*e_z kg*m/s, L=0 kg*m^2/s and Delta(K+U)=W_external=0. No internal facet produces an impulse, contact history, capacity or external-work credit. Each complete run must preserve those values within the original per-step and noncancelling-window budgets, complete geometry certificates and owned rebuild equality. The equations above are hand oracles; no host or native pass is claimed.

## 9. F2: four-face full-star response and one complete terminal birth

This independent connected body has an INTERIOR four-face vertex star, not F1's two-face edge. Use stable IDs

    x0=X0=(0,0,0), x1=X1=(1,0,0), x2=X2=(0,1,0),
    x3=X3=(-1,0,0), x4=X4=(0,-1,0),
    faces=(0,1,2),(0,2,3),(0,3,4),(0,4,1),
    rho_i=1/4, m_i=1, U=0, original forces=gravity=0.

The middle reference is the planar diamond with center vertex0. Its four triangle interiors are disjoint and cover the diamond. Its twelve product tetrahedra use section2's order; its interior link at0 is the four-cycle and each outer vertex link is an interval. All directors are e_z and the reference/current shell is the convex diamond product. Shell initial velocities are zero.

The other participant is a genuine free dynamic POINT of mass1kg with three translational coordinates and zero elastic energy. It is a supported external point contact control, not a pinned proxy, fabricated robot-domain participant, finite gripper or general articulated-energy claim. A later finite-patch/articulated adapter must supply its own complete finite geometry, mass and energy proof. Set T=1/16, rho=1/4 and

    r(0)=(-u*T,0,rho+T), v_R=(u,0,-1),
    r(t)=(u*(t-T),0,rho+T-t), 0<=t<=T.

At every t<T, its z gap to the ENTIRE shell is T-t>0; at T it first reaches y=(0,0,rho). This fixed-axis inequality is a complete preimpact external-point sweep proof. The shell stays fixed and embedded until impact. The four coplanar top facets meeting at0 represent one material boundary point and one exposed normal ray, not four independent capacities. Their complete tie/fan evidence must be retained.

Use h=1/16,1/32,1/64 (one, two and four steps over the SAME window) and TWO independent rebuilds for each h and each speed below. Set restitution0, mu_static=mu_dynamic=1/2. Before terminal application the ORIGINAL selected-pre state is exactly: the shell at X with all velocities0; the dynamic point at(0,0,rho) with velocity(u,0,-1); all original force/gravity inputs0; unchanged owned topology, rho and reference ID; and the actual selected trial's history. No held impulse occurs. Bind the native trial receipts to this state rather than writing it into a candidate. All original external work is0.

For this star, writing ring indices cyclically,

    S0=sum_(j=1..4) (x_j-x0) cross (x_(j+1)-x0)
      =sum_(j=1..4) x_j cross x_(j+1)=4*e_z.

The x0 terms telescope for ALL configurations, so D S0/D x0=0 exactly. At the stated flat configuration,

    delta n0.x=(-delta x1.z+delta x3.z)/2,
    delta n0.y=(-delta x2.z+delta x4.z)/2,
    delta n0.z=0.

For y=x0+a0 and r_coeff=rho/2=1/8 the COMPLETE Jacobian is

    J0=I,
    J1=-r_coeff*e_x*e_z.T, J3=+r_coeff*e_x*e_z.T,
    J2=-r_coeff*e_y*e_z.T, J4=+r_coeff*e_y*e_z.T.

All other blocks are zero. In particular node3, outside the arbitrary parent face(0,1,2), must be present. A shell force f=(f_x,0,f_z) pulls back to ell0=f, ell1.z=-r_coeff*f_x, ell3.z=+r_coeff*f_x. The nodal resultant is f and its moment is2*r_coeff*f_x*e_y=rho*f_x*e_y=y cross f. This directly verifies virtual work, resultant and torque without referring to a producer's own Jacobian result.

Take relative velocity u_rel=v_R-J*v_shell, with p=(t_impulse,0,lambda) applied to the point and -J.T*p to the shell. Independent unit-mass kinetic squares give the shell tangent response1+2*r_coeff^2, plus the point response1. Hence

    W=diag(kappa,kappa,2), kappa=2+rho*rho/2=65/32.

This 65/32 derives from all five middle nodes and the other participant. All three exact LDL pivots are positive and kappa_inf=65/64<10^12 for this complete diagonal W. No diagonal regularization, reduced three-node port or copied previous fixture is involved. Incoming relative velocity is(u,0,-1). Restitution0 gives lambda=1/2 and the friction capacity mu*lambda=1/4. Two exact speed choices exercise the transition:

| Mode | Incoming u [m/s] | Tangent impulse t_impulse [N*s] | Final tangent slip [m/s] | Joint DeltaK [J] |
|---|---:|---:|---:|---:|
| Hold | kappa/8=65/256 | -1/8 | 0 | -1/4-kappa/128=-1089/4096 |
| Slip | kappa/2=65/64 | -1/4 | kappa/4=65/128 | -1/4-3*kappa/32=-451/1024 |

For either row the equation DeltaK=p dot(u,0,-1)+(1/2)*p.T*W*p proves the stated value. Hold lies strictly inside capacity; slip uses its boundary and opposes the nonzero final slip. Thus these are distinct Coulomb/KKT modes, not a pressure-uniqueness claim. The exact outgoing physical velocities are

    v_R+=(u+t_impulse,0,-1/2),
    v0+=(-t_impulse,0,-1/2),
    v1+.z=r_coeff*t_impulse, v3+.z=-r_coeff*t_impulse,
    v2+=v4+=0, with all unlisted ring components zero.

Joint momentum remains(u,0,-1). Shell angular impulse about the origin is-rho*t_impulse*e_y, and point angular impulse is+rho*t_impulse*e_y, so total L_y=rho*u remains unchanged. Along the entire original free-point approach, r(t) cross v_R=rho*u*e_y; the preimpact trajectory and terminal reaction therefore satisfy the same complete-window angular balance. The nonzero postimpact director rate is dot a0.x=-rho*r_coeff*t_impulse. A missing opposite-ring term loses both the65/32 response and this reaction moment even if a separate resultant repair is attempted.

This gives a complete free trajectory followed by ONE terminal impact. Positions do not change in the terminal velocity operation, so preimpact reference embedding remains valid at its end. It does NOT prove the next rotating, loaded continuation; that requires section7's new held-map/normal/work/path certificate. A one-event control is not relabeled sustained cloth manipulation. It is also not an active strained-B2 control. The exact impulses/energies above are proposed independent acceptance oracles, with the actual-law, mapping, R1/R2, native and whole-window gates still independently required.

For BOTH F1 and F2, add independent reconstructions under the exact proper rotation(x,y,z)->(-y,x,z), translation(4,-2,1)m, stable-ID storage permutations, and reconstruct-from-published-state. Transform positions, velocities, impulses and angular momentum about a consistently transformed origin; do not compare unadjusted origin-dependent angular momentum after translation. Full stable state/history/path/phase evidence must agree between each authored rebuild pair, normalizing only the exact typed timing field. No values or windows may be fitted after execution.

## 10. Independent negatives and distinguishing their rejection

N1, remote incident overlap in a connected valid mesh: use six faces(0,j,j+1) around center0, cyclic ring1..6. A valid planar reference ring is(2,0),(1,1),(-1,1),(-2,0),(-1,-1),(1,-1), all z0. Center is0 and rho>0 constant. Deform the ring to(1,0),(-1,1),(-1,-1),(2,0),(-2,2),(-2,-2), all z0. Every successive oriented cross product is strictly positive, every director is+e_z and every product tetrahedron is positive. Nonetheless face(0,1,2) and face(0,4,5) overlap away from their common material center. The ring coordinates are distinct. Complete incidence rejects this remote overlap; a shared-ANY-node exclusion or positive-tetra rule would miss it.

N2, positive tetrahedra but an overlapping star with coincident distinct identities: keep N1's valid reference/topology and use current ring coordinates(1,0),(-1,1),(-1,-1) repeated twice under six DIFFERENT stable IDs. All face areas/directors/tetrahedral determinants remain positive, but distinct material volumes coincide. Coordinate equality cannot merge their identities or validate a forged all-pairs certificate. This is separate from N1's distinct-coordinate control.

N3, interior-time crossing with legal endpoints: use two allowed disconnected reference components, each the triangle(0,0),(1,0),(0,1), with z_A(0)=1,z_B(0)=-1 and rho=1/8. Let z_A(t)=1-2*t,z_B(t)=-1+2*t for t in[0,1]. At both endpoints the bodies are disjoint, all areas/director sums and tetrahedra stay strictly positive, but their interiors overlap whenever|2-4*t|<1/4. Topology and local orientation must pass, while the complete temporal incidence proof rejects. This is a crossing control, not a degeneracy or endpoint-overlap substitute.

N4, degeneracy/inversion while director sums remain valid: use the F1 topology, hinge(-1,0,0),(1,0,0), outer vertices(0,+3/8,9/32),(0,-3/8,9/32). Outer normals are(0,-3/5,4/5),(0,+3/5,4/5), hinge normals+e_z. A positive-oriented T2 determinant is4*rho*(3/8-3*rho/5). It is exactly0 at rho=5/8 and negative at rho=3/4, although projected thickness ratios are at least4/5 and every S_i is nonzero. The flat reference with the same positive rho is valid. Reject both current geometries independently of the director-sum guard.

N5, vanishing director sum without a vanished face: with F1's oriented faces put both outer vertices at(0,1,0), hinge unchanged. The two oriented raw areas are+2*e_z and-2*e_z. Shared-vertex S_i=0 although each face is nondegenerate. Reject before division; do not retain an old normal or treat coordinate-coincident outer IDs as the same node.

N6, topology and identity forgeries: reverse the global side diagonal in only one cell; omit orientation parity; forge shared-simplex IDs; duplicate a tetrahedron under a new cell; drop an external facet; label an internal staircase facet external; change rho/reference/law/path/address after signing. Independently recomputed topology, boundary incidence, fingerprints and full coverage must reject each. These are not geometric epsilon tests.

N7, mechanics and capacity forgeries: omit F2's node3 block, use barycentric-only J, or move its lost force to the center. The last may preserve resultant but loses the off-center torque; all must fail the independent J/virtual-work/moment/65/32 checks. Also forge a zero separated-point couple, omit a shared-base cross term, omit native conversion uncertainty, drop an ideal-extremum residual, spend nullspace self-stress as capacity, or count the four top facets as four owners. Preserve the existing R2 adversarial control where a small nodal residual fits a map ceiling yet its amplified pressure/phase uncertainty exceeds the original phase budget.

Additional required negatives cover a director/tetra degeneracy only in an interior time cell; a gap/overlap in the certificate time partition; unproved boundary occupancy; a noncollinear active normal cone submitted to the unique-ray law; remote boundary self-touch submitted to the STRICT embedding checker; exhausted roots/bits/operations/depth; stale native/reference/history receipts; and late rejection after earlier substeps. Unknown is a distinct rejected result, never a claim of penetration or safety. These negatives do not count as either successful physical control and cannot be used to replace them with all-fold rejection.
## 11. Proposed public interfaces, ownership and delivery sequence

These names and files are a concrete proposal for parent adjudication, NOT a source grant. Arrays are copied into immutable owned storage; all caller arrays may be changed after construction without altering the owned object. IDs are nonnegative distinct int64 where uniqueness is required; shape/dtype/bool-as-index/nonfinite/range/duplicate/capacity checks precede any workspace or native mutation. Coordinates and references are metres, velocities m/s, h seconds, force N, impulse N*s, mass kg, J dimensionless, W kg^-1, energy J and angular impulse N*m*s. Exact scalar values are rational interpretations of the stored binary inputs, not decimal re-interpretations.

| Proposed public interface | Owned input/result contract |
|---|---|
| ConformingShellReference(domain, node_ids, face_ids, oriented_nodes, reference_positions, half_thickness_m, *, law, limits) | node_ids int64[N], face_ids int64[F], oriented_nodes int64[F,3] containing stable IDs, reference_positions FP64[N,3], half_thickness_m FP64[N]. Own the derived lifted IDs, oriented tetrahedra int64[T,4], full incidence, boundary facets and reference proof. A caller cannot supply an authoritative boundary mask. |
| ShellDirectorLaw(kind='raw-area-star-director-v1', *, projected_ratio_min=1/4, shape_policy='invariant-1e-10-v1') | Exact schema/version, fixed all-one star weights, normal/winding convention and the declared domain gates. Proposed defaults require parent freezing before execution. No callbacks, history-derived normals or mutable geometry-dependent weights. |
| ShellProofLimits(...) | Explicit section6 ceilings and counters, smaller caller bounds only; fresh counters for producer and verifier. Reject malformed or unsupported policies instead of silently applying a different one. |
| ShellMaterialAddress(reference_id, tetra_id, lifted_vertex_ids, counts, generation) | Four lifted IDs with caller/canonical correspondence; int64[4] nonnegative counts summing2^52; owned exact ideal barycentrics and their enclosure/representation error. Canonicalize points on shared features by their minimal shared reference simplex. Geometry snapshots are separate from stable material identity. |
| shell_director_geometry(reference, positions, *, limits) | FP64[N,3] current positions; raw-area sums, director/lifted-point exact or outward evidence and sparse proposal J. Does not imply embedding, response acceptance or native delivery. |
| certify_shell_embedding(reference, positions, *, limits) / verify_shell_embedding(certificate, reference, positions, *, limits) | One typed strict embedding proof or ShellUnresolved. The verifier independently recomputes topology, all pair obligations, input binding and inequalities. |
| certify_shell_sweep(reference, path, *, h, limits) / verify_shell_sweep(certificate, reference, path, *, h, limits) | path FP64[2,N,3] is the actual declared middle-node endpoint path; h finite positive. Own all curved-director expressions, nondegeneracy conditions, complete time cover and all material-pair obligations. No endpoint-only acceptance. |
| shell_robot_manifold(reference, positions, robot_geometry, *, identity, limits) | Complete true-boundary feature/tie/occupancy evidence for the explicitly supported robot primitive, exact ideal coordinates and actual owned coordinates; result distinguishes complete geometry from unresolved. No implicit choice of a physical cone law. |
| shell_robot_sweep(reference, path, robot_path, *, identity, h, penetration_budget_m, limits) | Both endpoint/body identities and actual continuous paths; finite nonnegative penetration budget, with0 for these controls; complete external boundary and material-occupancy coverage. The robot shape/path capability is explicit, not inferred from a point-port name. |
| ShellMaterialPointPort(base, reference, addresses, *, base_node_ids, limits) | One public all-node physical base; owned base_node_ids int64[N] is the bijection from its public point-row order to reference stable IDs. Require complete unique base coverage and bind this map separately from reference storage order. Sparse block receipts row_ptr int64[K+1], physical node IDs int64[nnz], J FP64[nnz,3,3]; positions/velocities FP64[K,3]; aggregate requested nodal impulses FP64[N,3]. Own original snapshot and map receipts. All stars and repeated-node contributions remain complete. |
| ShellMappingLedger.require(plan, original_positions, original_velocities, actual_phase_receipt, *, budgets) | Independently rebuild the full-star law/map, ideal extrema, requested/represented/actual loads, capacity witnesses, resultant/moment/virtual work and phase defects. Typed rejection retains original evidence; no projected state repair. |

The proposed port implements the existing PUBLIC PointPort contract explicitly: domain and dtype properties; positions()->[K,3]m; velocities()->[K,3]m/s; response(impulses[K,3]N*s)->[K,3]m/s; snapshot()->owned base state; validate_candidate(state)->validated state; apply_impulses(impulses)->owned candidate; and candidate_audit(state)->actual full-domain kinetic/point audit. response performs one full scatter to the original all-node base, its public response, then full gather. If apply_impulses is used for a HOST proposal, it delegates exactly one aggregate request to the immutable base and never mutates a runtime participant; it is not native delivery and its result cannot substitute for the actual replay/terminal receipt. The physical coordinator uses response/scatter evidence and exactly one public replay/terminal delivery. candidate_audit recomputes point motion from the public all-node candidate audit and full-star map; it never reads private runtime arrays. Add a public owned node_impulses(impulses[K,3])->[N,3]N*s audit/scatter method; this returns a request, not proof of delivery. FP64 proposal blocks never change the declared base/native precision or authorize a different state family.
A boundary point represented in several tetrahedra has one material address only when their exact reference coordinates identify the SAME point. Remote touching points retain distinct addresses. A contact owner binds ordered body/domain IDs, reference/topology/law identities, oriented boundary feature IDs, both canonical material addresses, caller correspondence and birth generation. History can persist that stable owner across configuration changes, but every J/path/phase receipt binds its NEW original geometry snapshot. Loss of support or a changed material reference invalidates the relevant certificate/history under a declared rebuild rule; coordinate equality cannot transfer ownership. History schema/version and base/contact field disjointness are explicit and bounded.

Fingerprints bind schema version, binary coordinate bytes or canonical exact numerators/denominators, array shape/dtype/order, stable node/face/cell/lifted IDs, all orientation permutations, reference positions and ordering keys, rho, law/weights/gates, mass/precision/representation identity, source and target path states, h, material addresses/generations, proof/resource policy and phase. A hash is only a binding identifier: independently checked content supplies truth. No scene-name, object-name, time or fixture branch may enter geometry, physical law or acceptance.

Typed unresolved reasons should distinguish invalid_reference_topology, nonconforming_product, degenerate_face, vanishing_director, projected_support_unknown, inverted_or_degenerate_tetra, nonincidence_overlap, unsupported_boundary_touch, unresolved_normal_cone, unsupported_robot_path, incomplete_time_cover, unknown_algebraic_identity, precision_exhausted, operation_budget_exhausted, stale_receipt and actual_phase_uncertainty. A safely proven strict separation and an unresolved complete manifold may coexist; unknown manifold construction is not evidence of geometric penetration. Each rejection reports the unresolved obligation without publishing a partial physical state.

The numerical response admission remains explicit: on the full admitted symmetric response require strictly positive exact LDL pivots and dimensionless kappa_inf=||W||_inf*||W^-1||_inf<10^12. Equality, singularity, an unknown inverse witness or exhaustion rejects; no regularization/eigenvalue clipping. Retain raw/asymmetric W and the original relative asymmetry ceiling1e-12. Any explicit symmetry averaging is a proposal operation whose defect is charged against actual output. Existing local friction effective-response gates1e-12 remain. The quotient's different rank-deficient algebraic domain is not silently substituted for the full-rank physical R2 witness.

For R2, on each phase's complete stored map B and original mass define L=(B M^-1 B.T)^-1 B M^-1, p_eff=L*ell and r=ell-B.T*p_eff. p_eff is a certification decomposition, not unique delivered physical pressure. Preserve proposal p, p_eff, their difference and r. Held ell is exact h*(q(F_effective)-q(F_original)); bound every original predictor residual q(v_pred)-q(v0)-h*(q(g)+q(w_original)*q(F_effective)) divided by q(w_original). Terminal ell is actual(q(v_post)-q(v_pre))/q(w_original), with the requested/native batch separate. Propagate all uncertainty through the full map/inverse; use normal LOWER and tangent UPPER bounds and check final Coulomb/KKT over the full capacity interval. Never turn zero-load rounding into friction capacity or hide r by redistribution. Actual/ideal extrema, original material/path gates and all physical budgets remain independent.

For clarity, keep the precise physical budget units: impulse-map ceiling=tau*1N*s, phase-speed ceiling=tau*1m/s, and independently response-scaled residual. Host positive-energy ceiling1e-8J and absolute angular ceiling1e-8N*m*s; actual ceilings1e-5J and1e-5N*m*s. Check per step AND the noncancelling whole-window sums, never multiplied by step count. Independent enclosure contributions are <=1/64 of the corresponding budget. Dissipative F2 DeltaK is checked against its exact negative oracle as well as no energy creation; do not reinterpret a dissipation allowance as a larger positive-energy budget.

Suggested exact NEW file ownership by dependency:

| Stage / future owner | NEW production paths | NEW test paths |
|---|---|---|
| F0 topology/law/exact static geometry, shell owner | src/phymeo/constraints/conforming_shell.py; src/phymeo/geometry/shell_directors.py; src/phymeo/geometry/_shell_exact.py; src/phymeo/geometry/shell_embedding.py | tests/test_conforming_shell.py; tests/test_shell_directors.py; tests/test_shell_exact.py; tests/test_shell_embedding.py |
| F1 continuous incidence proof, shell owner | src/phymeo/geometry/shell_sweep.py; src/phymeo/validation/shell_embedding.py | tests/test_shell_sweep.py; tests/test_shell_embedding_verifier.py |
| F2 full-star point/robot geometry and independent mapping, shell owner | src/phymeo/coupling/shell_director_port.py; src/phymeo/validation/shell_director_mapping.py; src/phymeo/geometry/shell_robot_manifold.py; src/phymeo/geometry/shell_robot_sweep.py | tests/test_shell_director_port.py; tests/test_shell_mapping.py; tests/test_shell_robot_manifold.py; tests/test_shell_robot_sweep.py |
| Physical coordinator/transactions/native acceptance, parent-owned separate grant | src/phymeo/solvers/conforming_shell_contact.py; src/phymeo/validation/conforming_shell_contact.py | tests/test_shell_physics.py; tests/test_shell_transaction.py; tests/test_shell_tyx.py |

The validation modules may reuse checked rational primitives and immutable public schemas, but not call a producer and compare its self-issued certificate. Independently derive incidence, binding, path polynomials, complete cover, J and actual-law inequalities from the original inputs. Future test oracles include explicit tet lists, F1's bound and free equations, F2's five blocks/65/32/reaction squares, and N1-N7 rather than mirroring implementation output. Suggested unimplemented test names are test_f1_free_changing_fold_full_window, test_f2_full_star_terminal_hold, test_f2_full_star_terminal_slip, test_connected_remote_incident_overlap, test_positive_tetra_overlapping_star, test_legal_endpoints_interior_crossing, test_tetra_degenerate_and_inverted, test_vanishing_director, test_nonconforming_product, test_forged_incidence_boundary, test_full_star_torque_and_reaction, and test_late_failure_restores_original_frame. They are proposals, not authored/collected/passed tests.

Parent owns every OLD seam: scene/state registry and history; runtime/program, stepper, load_replay, terminal_impulse and point_participant_program; public exports and legacy prism/offset/S1/LG/S2-L files; old material/contact/energy/mapping ledgers; robot composition; B2/R7 typed dispatch and native kernels. The None F1 body and F2 two particle participants may require no point-participant/replay source change: use their existing public capabilities. Do not modify them merely to make a new type appear accepted. Actual shell coordinator integration, arbitrary robot participants, typed materials and changed consumers require an explicit parent-reviewed contract and exact ownership. This document's proposed new list grants no edit to any such file now.

Host/actual distinction: host FP64/rational proposal geometry is not delivered native dynamics. If R7 is used, retain immutable FP64 reference R plus actual FP32 displacement d, prescribed RN64 decoded world position and actual FP32 velocity. Bind R,d,d_pred, native input/output, reference_id and operation/codec receipts; never call this native FP64 or a single-FP32 world state. The declared continuous material curve is derived from the actual decoded published endpoint path and recomputes directors there; native velocity and conversion discrepancies remain separate. No host position correction, recentering, new reference handoff or legacy-wire relabeling is allowed. Typed R7 and active B2 contact integration remains a distinct parent-owned extension; standalone quotient algebra does not grant it.

The later execution selection should validate new host geometry/maps/verification, then F1/F2's complete physical windows and reconstruction controls on actual interpret and CUDA, plus the specifically bounded C++ selection fixed by parent before execution. Every mode needs actual child-source/pin evidence, raw exits, source manifests, immutable evidence and native receipts, with one sole serial execution slot. No skips, fake native fallbacks, added samples for reporting or unchanged old acceptance reruns. This documentation task executes NONE of them. Consumers actually affected by future dispatch changes determine old-seam regressions; accepted unchanged local scopes remain accepted on their original versions.

Transactions must open from original owned state/history/inputs and close every public replay/terminal workspace. Aggregate all same-base contact requests before one physical delivery. Validate all terminal batches before any participant mutation. A failure in geometry, actual material path, mapping, capacity, physical balance, history assembly, diagnostics or close restores the entire original frame including earlier accepted substeps, with no hidden warm-start/reference change. A retry starts from exactly those original bytes and is independently revalidated. F1 no-contact execution and F2 exactly-once terminal execution are both required; a rejected negative is not a physical-success substitute.

## 12. What is proved here, what is proposed, and what still needs a contract

Hand-derived commitments in this document are: the conforming oriented staircase product and its boundary cancellation; the raw-area law and complete star derivative; translation/rotation, virtual-work, resultant and torque identities; the declared point-lumped kinetic model and full shared-base terminal response; the sufficiency of complete tetra-intersection equality for strict embedding; the global convex-domain Lipschitz proof; F1's every-simplex/all-time33/64 bound and actual free-particle trajectory; and F2's original free approach, five-block map,65/32 response, hold/slip impulses, momentum, energy and angular reaction. These are analytical derivations, not empirical validation. The reference/topology/domain predicates remain mandatory hypotheses.

Proposed algorithms requiring parent adjudication and implementation evidence are the exact six-root static primitive, complete fixed-size intersection enumeration, factored continuous shared-feature witnesses, global-gradient proof receipts, bounded interval coverage, exact topology/boundary verifier, resource ceilings and new conditioning gates, public full-star port and new geometry/mapping/coordinator interfaces. The controls show a nonempty mathematical success domain; they do not establish runtime cost or a passing native implementation. The static kernel's six-root cap applies per local geometry primitive; current/reference and unrelated-row contributions must be handled by separately bound rational enclosures when their union exceeds that field, rather than hiding extra roots.

The smallest unresolved obligations are explicit:

1. A weaker, complete material relation admitting lawful REMOTE boundary self-touch while proving interior disjointness and one-sided fan occupancy through time. It must distinguish genuine remote contact from shared material incidence and keep distinct capacity owners. Strict embedding remains a correct no-remote-touch domain and cannot be renamed that solution.
2. Nonzero rotating/changing-director HELD continuation: an actual replay-compatible force/work/moment/capacity construction or certified defect bound inside the unchanged budgets, with curved path, actual native inputs and complete contact/ideal-extremum evidence. F2 terminates at birth and supplies no postimpact claim.
3. Physical noncollinear normal cones, distributed pressure ownership and rank-deficient response with contact-admissible pressure. Accepted quotient algebra alone neither supplies a cone law nor authorizes nullspace pressure as friction capacity. All maximal ties remain evidence meanwhile.
4. Typed active B2/R7 material/contact integration, strained ordinary-mass fold trajectories, original material energy/work/path checks and joint shell/robot acceptance. U=0 success is not a strained membrane/bending result. General finite/articulated gripper geometry, path and tree-energy proof are additional to the dynamic-point control.
5. A continuum or independent-director physical model, if required: explicit through-thickness inertia, constitutive parameters and all inertial/energy derivatives, plus a separate discretization/mesh-refinement argument. No continuum consistency, arbitrary convergence or general-tree energy theorem is claimed here.
6. Complete implementation/independent verification/native transaction evidence for the proposed bounded algorithms. Exhaustion or unsupported algebraic identity remains unknown/reject, but cannot substitute for implementing and accepting the explicitly nonempty F1/F2 domain.

Parent should freeze or revise the point-lumped derived-director model, the reference/order and strict-embedding first domain, the new projected/conditioning gates and proof ceilings, and the staged interface/ownership boundary before source authoring. Later remote touching, cones, rotating loaded continuation and active B2 integration remain required P6.5 work, not optional omissions. Legacy local contracts and their acceptance evidence are unchanged.

**PROPOSAL_NOT_FROZEN.** Handoff is one new document and one clean documentation commit, with its exact file SHA256 and Git commit reported outside this file. AUTHORING_SLOT_RELEASED after that checked handoff. No execution slot was held or consumed; no source implementation, old-report edit, parent merge, test or native acceptance is claimed.