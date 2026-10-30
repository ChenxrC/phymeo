# P6.5 conforming shell geometry and complete director map

Status: FROZEN_CODE_AND_TEST_AUTHORING_ONLY. This is S2-FG, the geometry and material-map prerequisite of S2-F. Parent selects the finite proof route below from the [proposal](../../development/p65-s2-conforming-shell-proposal.md). It does not freeze a contact solver or claim completed lawful self-contact, native execution, physical trajectories or P6.5. See the [parent ruling](../../development/p65-s2-conforming-shell-parent-adjudication.md).

## Architecture, model and exact scope

Use the original S2 author in its released worktree, starting at c35a13a645bc68e912d390779ec29732ef07b644. Author tests before implementation, but run neither. Parent retains old shared sources. No parent merge, helper/reviewer, install, import, collection, probe, pytest, lint, formatting, compile/build/profile, native command or push is granted. All executable checks require a later sole serial grant.

Build a new conforming piecewise-affine material body with directors derived from middle-node positions. The intended later dynamics has fixed positive point masses at the middle nodes; offset geometry has no independent inertia. U=0 is a declared model, not an active strained-B2 solution. No kinematic director copy, frozen normal, convex-hull wedge or fictitious internal boundary. Preserve TrianglePrismSurface and every existing caller.

General oriented manifold topology and variable positive vertex thickness can be constructed and differentiated. The first POSITIVE embedding certificate deliberately supports only reference components that are planar convex triangulated disks with a constant vertex half-thickness within each component. Different components may have different positive thicknesses and planes. Admission additionally needs the complete reference checks, current/path domain checks and sufficient global bound below. Nonplanar/nonconvex/nonuniform-thickness references remain explicit unsupported reference charts, not silently accepted and not declared physically invalid.

This finite route admits the nonflat F1 fold and flat F2 vertex-star maps. It has no arbitrary pair radical-field solver or CAD fallback. It uses exact rational outward intervals and a proved global embedding theorem, avoiding unfactored permanent-shared-feature zero predicates. All material tetrahedra, including incident tetrahedra, are covered by that theorem. Strict embedding excludes distinct remote boundary addresses touching. Later weak boundary contact, changing-director held transfer, normal cones, external robot occupancy/CCD, runtime response, B2/R7 and full physical controls remain mandatory separate work.

## Exact NEW ownership: fourteen paths

Production:
- src/phymeo/constraints/conforming_shell.py
- src/phymeo/geometry/_shell_intervals.py
- src/phymeo/geometry/shell_directors.py
- src/phymeo/geometry/shell_embedding.py
- src/phymeo/geometry/shell_sweep.py
- src/phymeo/validation/conforming_shell.py

Tests:
- tests/test_conforming_shell.py
- tests/test_shell_intervals.py
- tests/test_shell_directors.py
- tests/test_shell_embedding.py
- tests/test_shell_sweep.py
- tests/test_shell_certificate_verifier.py

Reports:
- docs/development/p65-conforming-shell-geometry-report.md
- .superpowers/sdd/2026-10-30-p65-material-self-gripper-integration/task-2-conforming-shell-geometry-report.md

No existing file, export, dependency, proposal, report, solver or runtime changes. No copying accepted old private proof helpers and changing their semantics. New low-level arithmetic may be shared by these six modules; it cannot import a solver or run work at import time.

## Owned inputs, topology and material identity

ConformingShellReference(domain, node_ids, face_ids, oriented_nodes, reference_positions, half_thickness_m, *, law=ShellDirectorLaw(), limits=ShellProofLimits()) is an immutable owned data object. Domain is a nonempty string of at most256 UTF-8 bytes. Arrays must be actual NumPy ndarrays with exact int64 or float64 dtypes, not bool/object/implicit conversions:
node_ids[N], face_ids[F], oriented_nodes[F,3] contain stable IDs, not row indices;
reference_positions[N,3], half_thickness_m[N] are finite FP64, with positive thickness.
Require N>=3 and F>=1. All IDs are nonnegative unique within their declared namespace. Face entries refer to distinct declared node IDs; every node is used. Check shapes and hard limits before copying. Copies use immutable backing storage; views of mutable caller memory are forbidden. Certificates and exact scalars use deep-owned frozen tuples.

Every edge has one or two incident faces. Paired edge orientations oppose. Reject duplicate unoriented faces, nonmanifold edges and bow-tie/disconnected links. Each interior vertex link is one cycle; each boundary vertex link one interval. Components and oriented boundary cycles are recomputed. Inputs cannot supply an authoritative boundary mask.

Use lifted IDs (node_id, sign), sign=-1,+1. For increasing stable node IDs i<j<k in a face, the three preliminary tetrahedra are
    (i-,j-,k-,k+), (i-,j-,j+,k+), (i-,i+,j+,k+).
Multiply signs (+,-,+) by the original face orientation parity; swap the final two entries of a negative tetrahedron. Tetra IDs are (face_id, staircase_index). Adjacent cells share the exact side diagonal i- to j+. Recompute the oriented simplicial boundary; internal facets cancel, external facets have exactly one incident tetrahedron. Full pair incidence is the intersection of lifted-ID sets, never coordinate equality.

Retain caller node/face order for input binding. Canonical material identity sorts stable IDs and normalizes only cyclic face order, preserving winding. Storage permutations with the same IDs and associated data preserve canonical reference_id, tetra IDs and boundary identities; full input receipts also bind actual caller order. Renumbering IDs changes the triangulation unless ordering keys are explicitly transported, which this version does not offer.

reference_id binds domain, canonical topology, exact binary reference positions/thickness, director law and schema. A separate receipt fingerprint binds actual input order, state/path bytes, h, address, precision declaration and proof policy. Hash equality is not geometric proof. Signed zero may be retained in byte binding but has its exact rational value in mathematics.

## Director law and complete material differential

Only raw-area-star-director-v1 is supported. All incident oriented raw areas have fixed weight1, including boundary stars:
    c_f=(x_b-x_a) cross (x_c-x_a);
    S_i=sum_{f incident i} c_f; n_i=S_i/sqrt(S_i dot S_i); a_i=rho_i n_i.
No unit-normal averaging, ghost face, old-normal reuse or current-dependent weight. ShellDirectorLaw fixes projected_ratio_min=1/4 and shape_policy=invariant-1e-10-v1.

All current configurations and the entire certified path must prove:
- every face and S_i nonzero;
- ||c_f|| / max_edge ||edge||^2 > q(1e-10);
- ||S_i|| / sum_incident ||c_f|| > q(1e-10);
- (S_i dot c_f)/(||S_i|| ||c_f||) >=1/4 at every incident corner;
- every oriented lifted tetrahedron determinant strictly positive for embedding/sweep admission.
q denotes the exact rational value of the specified stored binary constant. No epsilon zero, clipping, repair or normalization of uncertain inputs. Squared inequalities require their original sign conditions. Unknown rejects; equality at a strict gate rejects.

With [u]cross v=u cross v, face derivative blocks are
    H_f,a=[x_c-x_b]cross;
    H_f,b=-[x_c-x_a]cross;
    H_f,c=[x_b-x_a]cross.
H_i,j sums every relevant incident-face block, and
    D a_i/D x_j = rho_i/||S_i|| (I-n_i n_i.T) H_i,j.
For fixed tetra barycentric material coordinates lambda:
    beta_i=sum_{lifted entries of i} lambda;
    gamma_i=sum sign*lambda;
    y=sum beta_i*x_i + sum gamma_i*a_i;
    J_j=beta_j*I + sum_i gamma_i*rho_i/||S_i|| (I-n_i n_i.T) H_i,j.
Keep every star neighbor, aggregate equal physical-node IDs, including overlap of different stars. No only-three-parent-node restriction.

ShellMaterialAddress(reference_id, tetra_id, counts, generation) owns four nonnegative exact int counts summing to2^52 and a nonnegative generation. Lifted IDs and orientation come from the reference; no user-selected coordinate correspondence. On shared features canonical identity is the smallest lifted-ID simplex with positive counts, with corresponding exact weights; equivalent parent-tetra addresses must give the same material point and map. A body vertex can have a full multi-face star map.

Translation and proper-rotation covariance imply sum_j J_j=I and sum_j J_j*(omega cross x_j)=omega cross y. Therefore J.T*f has resultant f and moment y cross f, and virtual work is exact in the declared mathematical model. These identities are separately tested and independently verified with outward arithmetic; a stored FP64 proposal is not declared exact.

shell_director_geometry(reference, positions, *, limits) returns owned lifts, S/director intervals, law-domain evidence and FP64 proposals, with status director_geometry_only and NO embedding/physics accepted flag.
shell_material_point(reference, positions, address, *, limits) returns the canonical address, point interval/proposal and sparse complete J blocks keyed by stable node ID. It neither accepts contact nor applies impulses.

## Arithmetic, publication and finite resources

Implement a new Fraction-based outward interval engine. Exact finite input FP64 values are converted by their binary integer ratio. Integer/Fraction literals in certificates are normalized and checked. No float inverse, floating determinant sign, finite difference, eigenvalue repair, approximate equality or old two-root solver.

Each interval endpoint is rational. Outward sqrt bounds are proved by integer-square inequalities. Detect exact rational squares (both numerator and denominator perfect squares) exactly; otherwise use dyadic lower/upper bounds with checked integer isqrt. The bound must enclose the original rational, including perfect-square endpoint cases. Division requires a proved nonzero-sign denominator. Dot/cross products and polynomial operations are explicit metered operations.

FP64 point/director/J values are proposals: convert the rational midpoint of each final ideal interval to FP64, reject nonfinite conversion, and retain exact q(proposal) plus max(|q(proposal)-lo|,|q(proposal)-hi|). This is a storage-error enclosure, NOT a claim that the exact ideal was correctly rounded. Intervals represent the mathematical body derived from input middle positions; future physical callers must account for published proposals and native conversion separately. No R7/native precision claim follows.

ShellProofLimits defaults AND hard ceilings:
max_nodes64, max_faces40, max_tetrahedra128, max_star_faces16,
max_pairs8192, max_time_cells4096, max_time_depth32,
max_fraction_bits4096, max_scalar_operations2000000,
initial_sqrt_bits64, max_sqrt_bits4096.
Callers can only tighten limits; all values are positive built-in integers (not bool), initial precision<=maximum. Refinement doubles precision, bounded by maximum. The fraction/integer bit cap can terminate refinement before the nominal precision ceiling; it never permits an oversized shift/intermediate merely to reach4096 bits of precision.

Count every exact rational +,-,*,/ in interval endpoints, matrices, polynomial transforms, norm bounds, topology geometry and checks. Count each integer isqrt as an additional operation, enforce its input/output bit cap and bounded precision schedule. Fraction internal reduction is not a separately counted scalar operation, but every exposed arithmetic intermediate/result is capped. Negation, comparisons, indexing and hashing are bounded by shapes/partitions. No unmetered NumPy exact-arithmetic substitute. Producer and internal checking share one counter; standalone independent verifier uses its own fresh counter, never trusting producer counters. No partial accepted certificate on exhaustion. Reference construction has its own bounded call; a certificate call must re-establish its required reference facts within that certificate's single counter, not reset a hidden budget through a cached constructor or internal checker.

Reported ideal interval widths/error bounds are not a replacement physical tolerance. Positive controls require point/J coordinate error bounds below2^-40 (m for points, dimensionless for J). Physical mapping/phase budgets1e-8 or2e-5 and energy/angular1e-8 or1e-5 remain unchanged future gates; current certificates cannot spend them.

## Independently checked reference chart

For EACH component the reference checker proves planarity exactly from rational middle coordinates and a nonzero oriented raw face normal c_ref. All oriented faces have positive parallel normals. All vertex thickness values in that component are exactly equal. Select a nonzero normal coordinate to project to a rational 2D chart; record the axis/orientation but do not assume a world-coordinate plane.

Recompute a single simple convex oriented boundary polygon, permitting ordered collinear boundary subdivisions but no zero edges/backtracking/repeated geometric vertices. Verify all projected face orientations positive, every triangle inside the closed boundary, and exact total triangle area equals polygon area. Independently check EVERY projected triangle pair: its closed intersection equals exactly its shared abstract edge/vertex, or is empty. Use complete rational convex clipping/segment intersections, including collinear/equality cases; bounding boxes alone are not this check. No positive-area-only test may miss an illicit point/edge overlap. These checks establish coverage of the convex polygon without overlap.

Then D_component is precisely this polygon times [-rho,+rho] along its common unit normal. The fixed staircase triangulation covers D without overlap. Reference tetra determinants are positive; reference geometry intervals must also verify invertibility before gradients are used. Boundary and full incidence derive from the complete product, not from observed current coordinates. Reference components must themselves be disjoint by a checked separating witness; no claimed reference body may already overlap or touch another material component.

The independent verifier reconstructs the above topology, projected intersections, boundary and coverage from original inputs. A supplied convexity flag, dropped face, reordered side diagonal or a digest cannot substitute these checks.

## Global strict embedding proof

For each reference tetrahedron define the fixed barycentric gradients with respect to physical reference coordinates X. Compute them by checked interval cofactors/inversion of the complete reference edge matrix; certify nonzero determinant. Keep all beta/gamma gradients with stable node correspondence.

At a current configuration F(X)=sum beta_i*x_i+sum gamma_i*a_i. On that tetrahedron:
    D(F-Id)=sum_i (x_i-X_i) outer grad(beta_i)
             +sum_i (a_i-a_i_ref) outer grad(gamma_i).
Choose the lowest stable-ID middle node of the component as translation gauge and subtract its displacement from EVERY (x_i-X_i). This changes F by one component translation and does not change its derivative, since sum grad(beta_i)=0. Factor this exact identity before interval evaluation.

A sufficient bound is
    k_T = sum_i upper||delta x_i - delta x_anchor||_2 * upper||grad beta_i||_2
          +sum_i upper||a_i-a_i_ref||_2 * upper||grad gamma_i||_2.
Use outward sqrt/norm products and retain the individual terms; no cancellation of positive charges. Require k_component=max_T k_T <1 strictly, not <=1 or a rounded float comparison. This rank-one bound is chosen instead of a potentially looser undifferentiated matrix norm.

For any two reference points, integrate the piecewise constant derivative of F-Id along their segment in convex D. The finite conforming triangulation permits a common subdivision; lower Lipschitz constant1-k_component proves injectivity. At each tetrahedron I+s*D(F-Id),0<=s<=1 stays invertible, so its determinant retains the positive reference orientation. This covers every same-cell, face/edge/vertex incident and distant tetra pair IN the component, with exactly the image of their shared reference simplex. Positive tetra volumes alone do not prove it.

For DIFFERENT components require one rational nonzero separating vector d and strict disjoint support intervals over all their lifted vertices. No normalization is needed. Producer tries a bounded deterministic set: coordinate axes, both reference raw normals, and the difference of component mean middle positions at the cell midpoint; it may return unknown if none works. Verifier accepts any resource-bounded rational d whose full support inequalities it independently proves. No near-zero vector normalization, ignored component pair or boundary touching accepted. This finite candidate search is sufficient, not a complete collision decision.

certify_shell_embedding(reference, positions, *, limits) returns ShellEmbeddingCertificate only after reference, domain, positive tetra, complete component/tetra coverage, k<1 and cross-component separation checks. Failure to find this sufficient proof is embedding_not_certified, not a fabricated proof of physical penetration. A witnessed invalid topology/zero director/inverted determinant has its distinct rejection category.

## Actual nonlinear path and full temporal coverage

certify_shell_sweep(reference, path, *, h, limits) consumes owned FP64 path[2,N,3], positive finite FP64 h. Normalized t in[0,1] gives x(t)=x0+t*(x1-x0). This is a geometry assertion about those endpoints and h, not evidence that a simulator generated them.

Build exact polynomial coefficients of edge differences BEFORE evaluation, then c_f(t) and S_i(t) of degree<=2, Q_i(t)=S_i dot S_i of degree<=4. This factors shared input identities and common translations. Use exact Bernstein coefficient enclosures after rational subinterval change of variable (or an equivalently complete exact polynomial enclosure, documented in reports). Do not independently interval-subtract x_j(t)-x_i(t) and lose a known common translation.

On every accepted time cell, prove the entire law-domain inequalities, Q lower>0, lifted positive tetra orientation, each component's global k<1 and every cross-component separation. Bound normalized directors nonlinearly using S/sqrt(Q); no frozen endpoint normal, chord-only rule or frozen velocity multiplied by h. Reference gradients remain fixed. Component translation gauge is the exact affine displacement of its anchor. Factor lifted common middle-node terms before interval tetra-edge determinants.

Cells form a gap-free nonoverlapping exact partition of[0,1]; adjacent endpoints are identical rationals. Producer may bisect dyadically, left-first, and refine sqrt precision within caps. max_time_cells counts ALL created/visited cells, including failed parents, not only accepted leaves. Depth and work apply throughout. A failure at an unresolved leaf returns an unresolved result; it never drops pending siblings. Verifier independently proves each accepted cell, complete cover, all component/tetra/pair memberships and limits. It may validate a different valid partition or different legal separating vectors; it must not require producer-specific witness bytes.

Shared-feature identities are covered structurally by the single conforming map and the global theorem, not an epsilon separation gap. Static endpoints do not imply sweep success. Multiple components crossing with valid endpoints must reject the complete sweep.

## Public certificate/verifier contract

New immutable ShellDirectorLaw, ShellProofLimits, ShellMaterialAddress and ConformingShellReference live in constraints/conforming_shell.py. ShellUnresolved(ValueError) carries one short category and bounded detail, never partial accepted state.

ShellDirectorGeometry, ShellMaterialPoint, ShellEmbeddingCertificate and ShellSweepCertificate have versioned schemas conforming-shell-*-v1, declared ideal-from-stored-fp64 precision, original bindings, topology/identity coverage, exact interval witnesses, actual proposal values/error enclosures and policy. Embedding/sweep status is strict_geometry_certified only. No physical accepted, friction pressure/capacity, energy credit, callback or native flag.

Independent functions in validation/conforming_shell.py:
- verify_shell_director_geometry(receipt, reference, positions, *, limits) -> True or ShellUnresolved
- verify_shell_material_point(receipt, reference, positions, address, *, limits) -> True or ShellUnresolved
- verify_shell_embedding(certificate, reference, positions, *, limits) -> True or ShellUnresolved
- verify_shell_sweep(certificate, reference, path, *, h, limits) -> True or ShellUnresolved

Verifiers may share checked scalar interval/vector arithmetic and immutable input types. They MUST NOT call certify_*, shell_director_geometry, shell_material_point, a producer's topology/Jacobian/coverage routines, or compare to a freshly produced certificate. Reconstruct incidence/stars and J independently from original oriented faces, reconstruct proof requirements/cover independently, and compare proposed fields against independently established enclosures/identities. Certificate hash is binding only. For each claimed ideal interval require containment of an independently computed outward enclosure at a legal precision; overlap alone or the reverse containment is not proof. Recheck publication/error bounds from the stored FP64 values. The exact translation/rotation/work identities follow from the independently reconstructed analytic formula; merely finding zero inside a residual interval never proves an identity. Shape and boundedness checks precede consuming untrusted nested evidence. Exact implementation field spelling must be listed in both reports; signatures and semantics above are fixed.

## Frozen independent success controls

The proposal sections8-9 supply independent handwritten equations, not fitted producer outputs. Preserve their values, signs, units, masses and later physical budgets.

F1: unit masses, reference diamond nodes(-1,0,0),(1,0,0),(0,1,0),(0,-1,0), faces(0,1,2),(1,0,3),rho1/16. Initial outer heights1/8, outer velocities+e_z, hinge velocities0,U=F=g=0,T=1/8. Exact free middle path q=1/8+t is an ANALYTIC input here; actual public/runtime production of it is a future physical gate.
For all six tetrahedra, b=|eta| and w=1+zeta/rho-b in T0, w=b in T1/T2. F-Id=q*b*e_z+w*delta a_outer, ||grad b||=1, ||grad w||<=sqrt(1+rho^-2), ||delta a_outer||<=rho*q. Hence k<=q*(1+sqrt(1+rho^2))<33/64 and lower Lipschitz>31/64 over the whole window. No unproved smooth/bilinear interpolation is substituted for those actual six affine pieces. The F1 whole-window certificate must itself have a checked upper k<33/64, not merely quote that handwritten bound beside a weaker computed result. Independent tests demand successful whole-window and every substep certificates for h1/64,1/128,1/256, with two separately reconstructed references. Preserve all required memberships and original h/window; no final-only sampling.

F2: five unit middle masses at0,+e_x,+e_y,-e_x,-e_y, four oriented fan faces, rho1/4. Material point top-center. Its complete map:
J0=I; J1=-(1/8)e_x e_z.T; J3=+(1/8)e_x e_z.T;
J2=-(1/8)e_y e_z.T; J4=+(1/8)e_y e_z.T.
All other entries exactly0, including the central director derivative. The actual point coordinate is(0,0,1/4). Reactions must include the opposite ring node outside an arbitrary parent face. With an independent unit external point the algebraic relative W is diag(65/32,65/32,2). Hand expectations lambda1/2,mu1/2; hold u65/256,t=-1/8,DeltaK=-1089/4096; slip u65/64,t=-1/4,slip65/128,DeltaK=-451/1024. Test the map/response/virtual-work/resultant/moment identities as pure algebra. No runtime contact is implemented or claimed by these tests.
Keep the proposal's free approach T1/16,h1/16,1/32,1/64 and two rebuilds as the later physical control contract; do not present a static-map test as that trajectory passing.

For F1 and F2, test identity, proper rotation(x,y,z)->(-y,x,z), translation(4,-2,1), and node/face storage permutations retaining stable IDs, with separate rebuilds. Compare identity-keyed fields and independently transformed mathematics, not a production run used as the sole oracle. For nonflat F1 also check the complete map along its free tangent against independently differentiated a_outer(q). No finite-difference producer oracle.

## Frozen negatives and evidence boundaries

Retain proposal N1-N7. In this geometry-only checkpoint N1/N2 must reject strict embedding with otherwise valid topology/director/positive-tetra observations; rejection by failed sufficient global proof is UNKNOWN, not a false asserted collision classification. N3 separated endpoints and crossing interiors must fail sweep; endpoints are separately certified. N4 exact zero/inverted tetra despite valid director and N5 exact zero star despite nonzero faces reject. N6 all forged connectivity/boundary/address/path/policy/certificate coverage rejects independently. N7 missing full-star/opposite node, barycentric-only J and resultant-only repair fail map verification; no physical R2 capacity result is claimed until response integration.

Also test nonuniform/nonplanar reference chart unsupported without destroying its immutable topology object; convex boundary with collinear refinements; malformed nonmanifold/bow-tie/cyclic winding; shared-address parent independence; bit/operation/shape/time/depth exhaustion; divisions containing0; sqrt perfect squares and tiny positive nonsquares; pure large common translations factored before interval work; valid endpoints/interior director or tetra degeneracy; missing time cell/component/tetra; remote boundary touch excluded by STRICT geometry; alternate valid separator/partition accepted by verifier. New limits cannot be loosened during debugging.

All source/test execution remains UNPERFORMED. After code-only handoff parent reads hashes/scope, then separately grants focused host environment/static/six full files and same-run certificate evidence. No actual interpret/CUDA/CPP, old suite or broad regression is selected for this host geometry component. Native and full physical F1/F2 selections belong to later complete solver contracts, not omitted P6.5 obligations.
