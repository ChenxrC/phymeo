# P6.5 weak material-boundary geometry with convex cluster witnesses

Status: FROZEN_CODE_AND_TEST_AUTHORING_ONLY. This is S2-FB-G, a bounded HOST geometry prerequisite. It is NOT a physical self-contact solver, a native result, or completed S2-F/P6.5.

Authority: parent review of the complete [boundary proposal and Appendix A.1-A.12](../../development/p65-s2-boundary-contact-proposal.md), child commit 04892aaa2b2200df3026d2ba9d267778a5307980, working SHA256 04afba8ad9f1a60d7ed9304c2b696bda6d32e58c33ecec95f0315b25df3d97d0, blob e14f3088475e0d359af36b127e3804fc364c98e5, imported normally as 7752c5bf427f50a97107b38490f4bf24f59e3c97. The original 62373 bytes/305-line proposal remain historical PROPOSAL_NOT_FROZEN. THIS contract selects its finite implementation; it overrides broader candidate language. Keep the [strict geometry component](2026-10-31-p65-conforming-shell-geometry-design.md) and every accepted old source/test unchanged.

## Ownership and execution boundary

Use ORIGINAL /root/p65_self_contact_module in C:/Users/38347/.codex/worktrees/p65-self-contact-module/phymeo, starting clean 04892aaa. Read this parent contract/plan/brief by absolute path; do not merge parent. Exactly twelve NEW deliverable paths:
- src/phymeo/constraints/shell_boundary.py
- src/phymeo/geometry/_shell_contact_field.py
- src/phymeo/geometry/shell_boundary.py
- src/phymeo/geometry/shell_boundary_sweep.py
- src/phymeo/validation/shell_boundary.py
- tests/test_shell_contact_field.py
- tests/test_shell_boundary.py
- tests/test_shell_boundary_sweep.py
- tests/test_shell_boundary_verifier.py
- tests/test_shell_boundary_points.py
- docs/development/p65-shell-boundary-geometry-report.md
- .superpowers/sdd/2026-10-30-p65-material-self-gripper-integration/task-2-shell-boundary-geometry-report.md

Only pure source/test authoring, text/manual mathematics, file IO and small Git commits are granted. Write tests first but label every collection/static/test UNEXECUTED. NO project import, collection, arithmetic/counting/numerical/offline probe, pytest, lint, formatter, compile/build/install/profile, native/backend command, helper/reviewer/new agent, remote CI, push/reset or other implementation dispatch. No execution slot exists. Do not change old source/tests/exports, dependencies, strict API, old reports/proposals, runtime, R7/B2, ports or robot seams. Parent is their sole owner. A later separate serial validation grant is necessary.

## Mathematical body and admission

Reuse the declared conforming PL material model, raw-area-star director law, stable lifted material IDs and staircase topology. Mass remains at middle nodes in later dynamics; offsets/tetrahedra do not add inertia. A reference may be passed as the existing immutable ConformingShellReference public data object, but new producer and verifier independently reconstruct every required original-input topology/chart fact inside their own aggregate counter. No old constructor/helper result or reference hash is an oracle. Only its public owned input data and low-level checked arithmetic/schema may be consumed.

Reference components remain oriented planar convex triangulated disks with exact constant positive half-thickness in each component; different components must be initially strictly separated. All original shape/star/projected-normal gates remain: nonzero face/star; invariant face and star quality greater than q(1e-10); projected ratio at least 1/4; all current/path oriented tetra determinants strictly positive. General manifold construction does not make nonconvex/nonplanar reference charts supported.

The weak relation is:
    F(a)=F(b) implies a=b as a MATERIAL point, or BOTH a,b belong to the TRUE global reference boundary.
A local tet facet/cluster cut is not automatically global boundary. Coordinate coincidence never merges material IDs. Positive volumes alone are insufficient. Require complete individual material-star occupancy, and no overlapping occupied open cones of distinct remote owners. A complete contact patch may jointly occupy both sides and have NO free robot side; retain that fact without inventing external exposure or choosing a friction normal.

## Selected generic witness algorithm

Implement proof_kind='convex-cluster-matching-v1'. A bounded immutable UNTRUSTED witness_hint supplies cluster membership, face partitions/cycles/projection choices, cross-cluster support descents and full matching maps, exterior-direction hints, and an exact field basis/dependencies. It may be supplied by callers. No automatic successful cluster search is promised. Invalid/missing hints return an explicit unresolved/invalid-witness category. No scene/C1/C2/coordinate/time special case is allowed.

Authenticate all Appendix A.2 hypotheses, not just a theorem tag:
- Clusters exactly partition every original tet. Full internal facets cancel. Interior links and boundary links are manifold; boundary is a connected oriented sphere with correct incidence/Euler characteristic.
- All oriented tets are positive.
- Each boundary plane comes from a noncollinear seed triangle and supports ALL cluster vertices. Its group covers exactly the FULL zero-support vertex set and every assigned coplanar boundary triangle. The complete zero-set ID list canonically reconciles coplanar groups; no partial duplicated plane group.
- Each projected plane group has a complete consistently oriented disk triangulation and a simple convex boundary, all face vertices inside, nonzero triangles, distinct boundary coordinates and monotone collinear subdivisions. For each nonzero boundary edge, the already enumerated support signs contain a strict interior vertex. Full face membership/links are verified.
These checks establish the whole convex hull boundary embedding/coverage. Positive degree then establishes the entire cluster embedding and its full local stars. Output the authenticated facts needed for this theorem, including full facet/stratum identities; do not replace them by Boolean convexity/degree flags.

For EVERY cluster pair, test all vertex support inequalities for its supplied plane. Strict separation proves empty intersection. For equality retain complete support faces and all tied features; at most three checked support descents can reduce the common affine hull to a patch/segment/point. The accepted route requires a COMPLETE geometric face-complex/segment/point matching bijection, including every vertex, simplex, subface and oriented incidence. Check original material preimages throughout the entire matched pieces: shared material identity or both true global boundary. Incomplete matching, general clipping or an unproved descent is unresolved. Do not implement an unbudgeted 56-solve/CAD/fan-arrangement fallback under this grant.

Retain every unordered original tet pair exactly once through a pair-to-proof map. Internal pairs inherit the authenticated cluster embedding and actual shared reference simplex. Cross pairs restrict complete matched support complexes to their original tet faces with exact incidence maps. Cover ALL ordinary material strata, not only contact strata. Derive every active cluster facet and full occupied cone union. For each global material-boundary stratum authenticate a nonzero exterior direction outside every incident cluster cone via a positive dot against an active facet. Remote contacts retain all owners, true-boundary stars, support ties and complete subfaces. Internal cluster interfaces cancel by material identity and cannot own pressure.

Reference authentication may use the SAME independently checked convex disk/complete boundary degree theorem; it must still prove complete original material coverage and positive reference tets. This is a new proof route, not a modification of accepted strict-reference code.

## Exact field, accounting and limits

Implement only bounded STATIC real multiquadratic fields K over rational inputs, plus finite polynomials/rational functions OVER K. No changing-radical Q(t) field, arbitrary algebraic root solver, floating inverse/determinant sign/finite-difference oracle, epsilon zero, regularization or repair.

Use canonical rational coefficient masks and positive independent generators. Authenticate square classes by bounded ratio perfect-square tests with checked integer roots and square-back; validate every dependency and positive branch. Zero means exact canonical zero coefficients. Nonzero sign uses outward rational intervals and the legal precision schedule, never interval overlap. Alternate legal bases/witnesses may be accepted when independently authenticated; producer byte identity is not the oracle.

Use the existing _shell_intervals.Arithmetic only as an unchanged LOW-LEVEL checked primitive backend, or an equivalent new fully metered backend in the new field file. In either case every rational +,-,*,/, unary negation/negative abs, integer-root/floor, exposed intermediate, field inverse, polynomial transform and failed refinement is charged/guarded. Internal Fraction normalization is not separately charged, consistent with the original model. No NumPy exact arithmetic or hidden helper bypass.

Inherit original defaults AND hard ceilings nodes64,faces40,tets128,star_faces16,pairs8192,created_time_cells4096,time_depth32,fraction_bits4096,scalar_operations2000000,sqrt_initial64/sqrt_max4096. Freeze additional hard ceilings field_rank4, polynomial_degree8 INCLUDING intermediates, expression_nodes4096, material/contact_strata4096, clusters8, unique_lifted_vertices_per_current_cluster16, boundary_triangles_per_current_cluster32, face_polygon_vertices16, active_fan_planes_per_stratum64, support_descents3, factor_exponent8. All can only tighten. Original reference components retain their full original shape bounds; current-cluster cuts are not a reference-shape reduction. Bound all nested hint/representative lists before allocation/traversal; finite ID walks cannot conceal numerical work.

Use original doubling from initial precision to min(2p,max), visiting the maximum once. Guard BEFORE oversized allocation/shift/arithmetic; 4096 nominal sqrt precision never exempts the 4096-bit guard. Unknown/exhaustion returns no partial successful geometry. A failed axis does not prove penetration. Distinguish proved invalid geometry from insufficient proof.

Appendix A.4-A.9 specifies the positive-route arithmetic schedule. Parent reviewed the corrected 15*A_2+9*M_2=411 tet primitive and all ledger sums. Manual bounds are C1 one pass510815 / producer plus inner checker1021630, C2 one pass24534 / aggregate49068; optional point query91933 / aggregate183866. These are conditional design bounds, not measured passes. They include reference reauthentication, both sign attempts64/128, all basis publications and endpoints. Actual code must meter everything; if an added operation/search requires a revised hand bound, report the discrepancy honestly before execution, without changing2m or controls.

For unchanged C1, the field is Q(sqrt2,sqrt5); C2 is rational. Appendix A.8 supplies positive geometry/support margins, including nonzero predicate floor2^-16 and det>3/256 for C1. A.9's original-input gauge, degree/denominator and monomial-inverse arguments bound sign refinement through128 and positive-route exposed intermediates below1024 bits. These are mandatory audit expectations, not permission to skip actual field/guard tests on recognized inputs. General rank3/4 or higher-height inputs have no such success promise.

## Complete temporal proof and transport

Public sweep inputs are owned FP64[2,N,3] affine-middle segments plus finite positive FP64 durations converted to exact binary rationals. Require exact joins and complete original binding; each segment and every created/split/failed time node consumes the original time budget. Use exact rational gap-free partitions, complete endpoints and all contact-event/tie transitions.

path_kind='constant-derived-director-v1' requires actual quadratic S_i(t) from original middle paths, coefficientwise S_i(t) cross S_i(0)=0, nonzero S_i(0), and S_i(t) dot S_i(0)>0 throughout EACH CLOSED segment. All face/star/quality/tet/cluster/occupancy gates still cover the entire path. Constant directors cannot be asserted from endpoints. More general changing normalized directors are explicitly unsupported here.

Then lifted vertices are affine over K. Whole-cell planes/support/face/J predicates use checked finite polynomial/rational-function coefficients and Bernstein bounds or authenticated endpoint factors with strict signed remaining factor on closed cells. Denominators and projection coordinates must stay valid throughout; exact persistent zeros are coefficient identities. Rational event partitions are allowed; unresolved irrational events remain unknown. No sampling, chord replacement or frozen velocity*h argument.

A component_translation transport optimization is allowed ONLY after verifying exact equal displacement for every node of the ORIGINAL component and recomputing/factoring complete edge/star/H polynomials. This proves S,H,J constant and transports all per-component geometric facts. Cross-component relative supports/intersections/events must still be proved over the full path. Every endpoint point is reconstructed from ORIGINAL endpoint middle coordinates plus authenticated offset, with its OWN proposal/error. J endpoint copies require the proved constant-map identity. No hash-only endpoint reuse/global cache. C2's two whole segments and common join are both mandatory.

## Full material maps and independent verification

Every successful geometry call includes BOTH signs of EVERY lifted node, complete exact points and full-star J basis blocks (including zero blocks for participating neighbors), original stratum/preimage maps and stored proposal/error bounds. Complete linear barycentric preimage maps represent every material point of a matched contact stratum; do not reduce a patch to one sample.

Use the exact fixed-material derivative from the strict director law:
    J_j=beta_j*I + sum_i gamma_i*rho_i/||S_i||*(I-n_i*n_i.T)*H_ij.
All stars/shared physical node contributions are kept. For general constant-director paths retain the exact rational-function J and denominator proof. Stored endpoint matrices alone are not a whole-path map proof.

An ExactBoundaryAddress uses reference_id, generation, canonical minimal lifted simplex, positive exact field weights summing to1, and ALL incident tet representatives. It is a separate type from the unchanged dyadic ShellMaterialAddress. Authenticate true global boundary and every representation from original reference data; exact material identity is never world-coordinate equality. shell_exact_boundary_point is an independently checked MAP-ONLY call, not a geometry/physical acceptance shortcut. Its original-input reference/law/J proof uses its own complete counter; batching cannot reset a mandatory query budget.

FP64 outputs are RN64 of rational enclosure midpoints, with exact q(stored) and worst-case distance to the ideal interval. They are proposals, not ideally correctly rounded radicals/native delivery. Positive controls require every point component error<2^-40 m and every J component error<2^-40 dimensionless. Later weighted floating sums/conversion and physical work/moment/capacity need their own accounting.

Parent API/independence decision: one separately implemented HIGH-LEVEL verification engine in validation/shell_boundary.py may serve BOTH the producer's inner check and the public independent verifier. This supersedes Appendix A.10's unnecessary requirement for a third high-level algorithm. It must share NO producer high-level topology, field-admission, cluster, pair/fan, path, J or coverage routines. Only low-level checked arithmetic and immutable schemas may be shared. Inner checking starts a fresh proof-value table on the producer's SAME counter; public verification starts a fresh table and its OWN counter from original inputs. This preserves both independent logic and the two-pass/one-pass bounds. Public verifier must not call the producer. Tests must demonstrate that producer facts cannot act as its oracle.

Claimed intervals must contain the independent outward enclosure. Hashes only bind original state/reference/order/policy/segments/durations/address/hints, never prove them. Legal alternate hints/partitions and outward-enlarged enclosures may validate if every exact identity/coverage/gate holds. Missing, duplicate, forged or stale pair/stratum/time/address members reject even if counts match. Observed producer operation counts are informational; verifier computes its own complete work and enforces policy independently.

## Public schema and statuses

Freeze the historical section12 schemas with the Appendix A.11 cluster/matching/transport delta, replacing unused fan-arrangement fields by full active-facet/cone-union/exterior witnesses for this proof_kind. Add explicit immutable ConvexClusterBoundaryWitness and full-support MatchingIntersectionWitness records; no opaque success flags. BoundaryProofLimits contains inherited ShellProofLimits plus the above extra caps. Bound/own all arrays and deeply freeze exact nested values, reject bool/object/coercive implicit numeric IDs/weights.

Required public functions:
    certify_shell_boundary(reference, positions, *, witness_hint, limits=BoundaryProofLimits())
    verify_shell_boundary(certificate, reference, positions, *, witness_hint, limits)
    certify_shell_boundary_sweep(reference, segments, *, durations, witness_hint, limits=BoundaryProofLimits())
    verify_shell_boundary_sweep(certificate, reference, segments, *, durations, witness_hint, limits)
    shell_exact_boundary_point(reference, positions, address, *, limits=BoundaryProofLimits())
    verify_shell_exact_boundary_point(receipt, reference, positions, address, *, limits)

The exact-address receipt carries/authenticates its field basis/dependencies; it must not require a prior trusted geometry receipt. geometry statuses are weak_material_geometry_certified and weak_material_path_certified, precision ideal-from-stored-fp64; point status exact_boundary_map_only. No accepted flag may mean physical success. Classification retains empty/material_incidence/true_boundary_contact/incidence_and_boundary_contact. Errors retain section12 distinctions and explicit unsupported_witness/matching/constant_director categories. Inability to prove a gate is UNKNOWN, not fabricated collision. No pressure/friction/state-update field.

## Fixed independent controls and tests

Original proposal sections8/9 and Appendix A.8/A.9 are normative fixed data, with no parameter fitting:
C1: 17 nodes,20 faces,60 tets,1770 pairs, unit declared masses,rho1/32, connected nonflat five-panel loop with distinct material ends matching in a full patch. Reference s=(0,1,3,5,7,8); current xz columns(0,0),(1,0),(1,2),(-1,2),(-1,0),(0,0), y=+/-1, original five centers/four-triangle fans. T1/8, U=(1/4,1/8,-1/16), complete persistent translation; reference IDs and triangulations exactly as published.
C2: two unit-mass flat rectangles A[-2,-1]x[-1,1], Bcontact[-1,0]x[1,3], rho1/32, each two triangles. Original B is shifted+1/8 ex; delta(t)=1/8-t then t-1/8 on two segments joined at1/8, T1/4. Retain BOTH ex/ey support rays and all endpoint features.
These are supplied analytic GEOMETRY paths, not simulated folding, force-generated reversal or actual contact response.

For BOTH controls require whole-window/all whole segments AND EVERY h-step at h1/64,1/128,1/256; two independently constructed instances; identity, Q(x,y,z)=(-y,x,z), translation(4,-2,1), and stable-ID storage permutation. No reduction of fixed grid. Authenticate untrusted hints for each variant; no fixture recognition. Compare complete deterministic payloads of the two rebuilds with no timing normalization. Evidence identities/directories are predeclared from authored cases, not selected after seeing equal hashes. Future execution must save full reference/path/duration/field/cluster/pair/stratum/fan/contact/preimage/J/publication/resource/time receipts and independent verification outcomes.

Write tests FIRST, all currently UNEXECUTED:
- exact rational/field operations, square classes, inverse/zero/sign/precision/bit/ops and complete aggregate accounting;
- C1/C2 independent hand formulas, complete membership/matching/fans/all-basis maps and original point/J error bounds;
- translation/rotation/virtual-work/resultant/moment identities with independent analytical or exact elementary expectations, never finite differences or producer reruns as oracle;
- rational1/3 exact boundary address, alternate incident representations, sparse full-star/opposite neighbor participation, immutable input ownership;
- all applicable historical section11 negatives: positive-local-volume global overlap, zero/inverted tet/star, remote interior disguised as boundary, wrong fan, internal staircase owner, lost tie/owner, shared-vertex plus extra penetration, coincident-ID merging, C2(-eta,-eta) positive volume2*rho*eta^2, partial pair/stratum/time cover, stale binding and exact-zero/branch forgery;
- forged cluster sphere/interior links/face disk/zero-support keys/support/descent/matching/exterior/transport hints; same counts with wrong membership;
- valid endpoints with interior crossing, false constant director, zero star inside a path, unsupported changing-director F1 without accepting a chord;
- legal alternate witnesses/partitions/enlarged outward intervals, mandatory independent containment, undercharged/reset counters, all resource failures with no partial success.

Never rerun accepted FG99/oldS2/R7/G2/adapter/quotient under this authoring grant. A later focused HOST-only selection will be separately authorized after twelve-file clean handoff. Native/physics/integrated review/build/wheel remain later distinct stages.

## Handoff and still-open physical work

Finish both new reports with exact public schema/API, mathematical/implementation limitations, all tests UNEXECUTED, twelve path SHA256/Git blobs, small commits and clean CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED. A contradiction with fixed geometry/proof/resource assumptions must be sent as minimal pure text to parent, not patched by changing controls or budgets.

General clipped contact, changing-director/irrational-event paths, actual loaded folding, complete normal-cone/friction/capacity, exactly-once force/terminal delivery, original-input energy/angular, ordinary-mass active B2/R7, external finite/articulated robots, transactions/native evidence, dual-arm cloth transfer and whole P6.5 remain mandatory. Preserve original physical tolerances and zero penetration. This finite new geometry component does not claim their completion.

## 2026-10-31 authoring clarification: public verification resource receipt

This fills the previously unspecified return type; it does not alter geometry, proof domains, arithmetic/cell limits, the twelve-file allowance or NO EXECUTION SLOT. The original author may apply it within the ongoing CODE_AND_TEST_AUTHORING_ONLY task.

All three new public verify functions return an immutable BoundaryVerificationReceipt on success and the literal False on rejected/unknown input under their declared error handling. A successful receipt has __bool__ returning True. Define it only in the already allowed NEW constraints/shell_boundary.py. The old verification APIs stay unchanged.

Required fields:
- schema='shell-boundary-verification-v1';
- verification_kind in {'static','sweep','point'};
- reference_id and original_input_binding from the independently authenticated original inputs;
- verified_certificate_binding, a deterministic binding of the complete bounded canonical claim actually checked, including its resource fields, not merely the original-input hash;
- policy, the immutable verified BoundaryProofLimits;
- status, exactly the independently expected weak_material_geometry_certified, weak_material_path_certified or exact_boundary_map_only for its kind;
- precision='ideal-from-stored-fp64';
- operations, created_cells and peak_fraction_bits from THIS successful public invocation's fresh meter.

Resource values are builtin integers and satisfy the original limits. Static/point created_cells are their actual zero; sweep created_cells is the independent verifier's actual total, including created parent/failed/sibling cells required by the proof. Do not substitute the producer-plus-inner combined count or copy any claimed producer/checker count. A receipt is constructed only after the entire independent high-level check and its resource enforcement succeed. Full certificate binding/serialization must be bounded, deterministic and honestly accounted for; if it performs checked arithmetic, those operations remain on the same meter. No hidden second verifier invocation, test-only observation hook, global cache or post-hoc inferred count.

The internal independent _check_* engine can retain its boolean/internal return and receives the continuing producer counter for inner checking. It need not construct a public fresh-counter receipt, reset a counter, or repeat any geometry for this API clarification. Public wrappers allocate one fresh meter, run the checker once, and publish that same call's observations.

Update the existing five authored tests and both final reports within their current paths. Save each successful public receipt alongside the full claim and original inputs in the predeclared payload, so its certificate binding, kind/status and observed work can be audited without a rerun. Add independent checks of immutable receipt data, exact kind/status/bindings, differing valid claims sharing original inputs, genuine meter rather than claimed counts, and literal False for rejection/resource failure. No negative case acquires a partial successful receipt. False means verification did not certify the claim; it is not a penetration proof. Truthiness and the scoped status never imply physical, capacity, energy or native acceptance. All authoring remains UNEXECUTED.
