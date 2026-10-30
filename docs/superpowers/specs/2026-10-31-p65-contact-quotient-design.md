# P6.5 exact contact quotient algebra

Status: FROZEN_CODE_AUTHORING_ONLY. Parent mathematical ruling for a bounded prerequisite, based on the exact derivation in [the adjudication](../../development/p65-b2-contact-adjudication.md) and sections 11-15 of [the proposal](../../development/p65-b2-contact-integration-proposal.md). This freezes standalone algebra only. Complete bending/contact physical support, ordinary-mass success controls, phase/cone admission and joint energy accounting remain NOT_FROZEN.

## Scope and ownership

The original B2 owner implements four NEW paths only:

- src/phymeo/validation/contact_quotient.py
- tests/test_contact_quotient.py
- docs/development/p65-contact-quotient-report.md
- .superpowers/sdd/2026-10-30-p65-material-self-gripper-integration/task-1-quotient-report.md

No existing source, tests, exports, reports, proposal, constraints, solver, runtime, native kernels or dependencies change. Parent retains all shared seams. This component is not connected to a solver or public package export. It cannot accept contact, modify state/requests, supply friction capacity, credit external work, or replace S2 R1/R2. It certifies an exact algebraic decomposition whose physical use needs later independent gates.

Only the existing B2 agent performs code/test authoring, file IO and Git. S2 owns the sole execution slot. No import, collection, numerical/offline probe, pytest, lint, formatting, compile/build/install/profile or helper/reviewer is authorized here. No merge of the parent tree is required to read this contract. Existing user authorization permits this prerequisite without another phase-permission cycle.

## Exact input domain and resources

B is an m-by-n dimensionless actual stored attachment-difference map retaining EVERY owner row, including duplicate and opposed rows. W is the original frozen full n-by-n inverse mass, in 1/kg, symmetric positive definite. All coordinates are translational in m, all delivered impulses ell and uncertainty radii e are in N*s. Mixed angular/translational coordinates, variable mass and a changing map are outside this contract.

Inputs are finite list/tuple containers of exact built-in int or fractions.Fraction scalars; bool, float, implicit array conversion, generators, NaN and infinities reject. Owned immutable normalized Fraction tuples preserve inputs. e is coordinatewise nonnegative. Ordered owner_ids and coordinate_ids are unique, nonempty strings, one per row/column; duplicate numerical rows are allowed. Identity order is bound, not inferred from coordinates. Each ID has at most 256 UTF-8 bytes. Matrices must have the declared nonempty rectangular shapes.

Public frozen QuotientProofLimits defaults and hard ceilings are max_owner_rows=192, max_coordinates=256, max_fraction_bits=4096 and max_scalar_operations=2000000. Callers may select smaller positive limits, never larger ceilings or another conditioning threshold. Shape/length/ID checks happen before copying unbounded containers. Every normalized input and every computed rational has numerator and denominator bit lengths at most max_fraction_bits. Each exact rational add, subtract, multiply or divide consumes one scalar operation, including dot products, elimination, residual and identity checks; negation, absolute value, comparisons, indexing and hashing do not. No unmetered matrix arithmetic bypass. Bounded shape loops govern comparisons. Exhausted count or intermediate bit length rejects before returning any certificate. Producer and verifier each enforce their own fresh full counter; neither trusts a producer's claimed count.

The conditioning ceiling is the fixed dimensionless strict value 10^12. There is no epsilon rank, regularization, eigenvalue clipping, floating inverse or fitted self-stress. Zero rank, including an all-zero unloaded B, is explicitly unsupported and rejects. A future empty/unloaded type would require a separate contract; do not manufacture H, an inverse or capacity here. Resource ceilings delimit this prerequisite, not claimed coverage of arbitrary P6.5 meshes.

## Algebra and proof

Use exact elimination with deterministic first-nonzero pivots to establish r=rank(B)>0 and a complete full-column-rank basis N of ker(B.T). For full row rank N has zero columns and Pi_N=0. Otherwise:

    A = B W B.T
    Pi_N = N (N.T N)^-1 N.T
    Pi_R = I - Pi_N
    s = ||A||_inf > 0
    H = A/s + Pi_N
    L = (1/s) H^-1 B W

Exact positive LDL proves W and H positive definite. Require strict kappa_inf(H)=||H||_inf ||H^-1||_inf < 10^12. W need not have an additional condition ceiling; W positivity, bounded exact arithmetic and H conditioning all remain mandatory. Preserve full off-diagonal W and all shared-coordinate cross terms.

Verify exact dimensions/rank/nullspace completeness; symmetric idempotent Pi_N, Pi_N B=0 and A Pi_N=0; H H^-1=I; and L B.T=Pi_R. H adds only the explicit nullspace certification gauge. It is not a regularized physical response matrix and never replaces A in a solver.

For observed delivery ell:

    p_eff = L ell
    r_impulse = ell - B.T p_eff
    Q = B.T L
    e_pressure = |L| e
    e_remainder = |I-Q| e

Check Pi_N p_eff=0, B W r_impulse=0, Q^2=Q, Q.T W=W Q and the exact energy identity

    ell.T W ell
      = (B.T p_eff).T W (B.T p_eff) + r_impulse.T W r_impulse.

Thus p_eff is a zero-nullspace minimum Euclidean owner-norm certification representative of the represented part; it is not unique measured physical pressure. Negative components are legitimate algebraic outputs, not accepted unilateral contact. r_impulse remains explicit even if its stored-map response vanishes. Half its quadratic form is nonnegative kinetic energy, not an external work credit. For any |delta ell|<=e, the two error vectors conservatively enclose the linear changes of p_eff and r_impulse at this SAME fixed B,W. No nonlinear held-trajectory or ideal-extrema guarantee follows.

Complete owner rows remain present. Do not add proposal null self-stress, pick a favorable cone representative or drop redundant rows. No proposal pressure is an input to this algebra API. Future physical callers must independently retain and charge the complete requested pressure, ideal-map differences, phase uncertainty, actual/ideal extrema and energy/angular defects. An unchanged actual delivery plus different null self-stress cannot change this certificate.

Full row rank reduces exactly to L=(B W B.T)^-1 B W. Uniform positive scaling of W preserves H,L and p_eff for fixed ell. Row/coordinate permutations with their IDs and W transform the result covariantly. Any complete exact basis of the same nullspace yields the same Pi_N,H,L; the verifier validates spanning identities and does not require producer-specific basis vectors.

## Frozen public API and certificate

- QuotientUnresolved(ValueError): bounded failure with a short category; no partial certificate.
- QuotientProofLimits: immutable validated resource policy above.
- ContactQuotientCertificate: frozen deep-owned data, schema exact-contact-quotient-v1 and status exact_algebra_certified.
- certify_contact_quotient(difference_map, inverse_mass, delivered_impulse, impulse_uncertainty, *, owner_ids, coordinate_ids, limits=QuotientProofLimits()) -> ContactQuotientCertificate.
- verify_contact_quotient(certificate, difference_map, inverse_mass, delivered_impulse, impulse_uncertainty, *, owner_ids, coordinate_ids, limits=QuotientProofLimits()) -> bool. Return True only after complete independent verification; reject invalid/unsupported evidence with QuotientUnresolved.

Certificate fields bind ordered IDs, owned exact B/W/ell/e, resource policy, canonical input fingerprint, rank, nullspace basis N, A, Pi_N, s, H, H_inverse, L, p_eff, r_impulse, both propagated error vectors and kappa_inf. Field spelling may be clear Python names; the two reports list the exact implementation spelling. No physical accepted flag or executable callback. Fingerprints canonically encode normalized numerators/denominators, shapes, ordered IDs, schema, units/domain and policy; equivalent rational spellings yield identical bindings. No floating JSON conversion.

The verifier does not call the producer and compare its output. It independently re-establishes input binding, rank/completeness, positive LDL, conditioning, matrix identities, delivery decomposition, propagated uncertainty and quadratic identity using checked scalar helpers. Reuse of bounded scalar/matrix primitives is allowed; certificate-provided derived fields, basis, fingerprint and condition are never trusted. A valid alternate scaled nullspace basis is accepted if all identities and bounds hold. External input or ID changes reject even if p_eff happens to stay equal. Do not copy parent-owned S2 helpers or import project solvers; production dependencies are Python standard library only.

## Independent authored controls

Use explicit small rational expected values, not expected results produced by the implementation under test.

1. Full rank: B=I3, W=diag(1,2,4), ell=(1,-2,3), e=(1/16,1/8,1/4). L=I, p_eff=ell, remainder=0, pressure error=e, remainder error=0, kappa(H)=4.
2. Duplicate rows: B=((1,),(1,)), W=(1), ell=(2), e=(1/4). Pi_N=((1/2,-1/2),(-1/2,1/2)), s=2, H=I2, L=(1/2,1/2). p_eff=(1,1), remainder=0, errors=(1/8,1/8) and 0. Both owners remain.
3. Opposed rows B=((1,),(-1,)), W=(1), ell=(2) gives p_eff=(1,-1), with no fitted positive pressure. Verify that changing an external illustrative null self-stress is irrelevant to the same B/W/ell inputs; no proposal parameter may enter.
4. Off-range: B=((1,0),), W=I2, ell=(2,3), e=(1/4,1/8) gives p_eff=(2), r=(0,3), errors=(1/4),(0,1/8); residual half-energy=9/2.
5. Shared cross terms: B=((1,0),), W=((2,1),(1,2)), ell=(0,2) gives A=(2), L=(1,1/2), p_eff=(1), r=(-1,2). The quadratic identity is 8=2+6; deleting W cross terms is detected.
6. B=I2, W=diag(1,1/k): k=10^12-1 succeeds, k=10^12 and k=10^12+1 reject. Equality is not accepted.
7. Uniform W scaling, nontrivial owner/coordinate permutations with IDs, and a complete scaled nullspace basis verify covariance/basis independence.
8. Individually forge/incompletely span N; alter A/Pi_N/H/H_inverse/L/p/r/error vectors/kappa/fingerprint/IDs or bound inputs. Every false certificate rejects even when an unchanged physical delivery hides an owner change.
9. Ragged/dimension mismatch, nonsymmetric/nonpositive W, float/bool/scalar coercions, negative e, duplicate/empty IDs, zero rank, invalid/too-large policy, input/intermediate bit exhaustion and scalar-operation exhaustion reject without partial evidence.
10. Caller container mutation cannot alter certificate-owned inputs; equivalent Fraction representations bind identically. Verify genuine public producer and independent verifier, plus meaningful negative paths.

These are algebraic controls only; they do not replace two ordinary-mass active-bending/contact success trajectories or a native/CPP test.

## Handoff and later verification

Author tests before implementation; code-only restriction means no observed RED exists. Preserve that distinction. Deliver CODE_READY_TESTS_PENDING with small logical commits, exact four-path diff, hashes, authored test function/parameter inventory, precision/resource limitations and clean status. Do not fabricate collection/test counts or reuse B2/S2 prior passes for the new source.

Parent then reads the code/hash/allowlist and, only after the current S2 formal release, issues a separate sole serial grant for environment/import provenance, scoped Ruff/format and the complete new host test file with fresh absolute basetemp. No old host75, R7, G2, LG69, adapter or physical suite is rerun for an unconnected stdlib module. No native claim or extra stage reviewer follows from this prerequisite.
