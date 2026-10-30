# S2-LG pair geometry authoring record

Status: CODE_AUTHORING; no executable validation has been performed or authorized.
Contract merged normally at d79316c53f74c7d6f142b8c9a54f565a32ed27f9.
The frozen parent specification including R4 governs this checkpoint.

## Independent fixtures frozen before implementation or execution

Coordinates below are exact binary rationals. Time t is dimensionless on [0,1];
h is positive seconds. Thickness, gaps, and penetration budget are metres.
These fixtures and inequalities are handwritten, not fitted to producer output.

1. Independent radicals: A=((0,0,0),(1,0,0),(0,1,1)),
   B=((0,0,4),(1,-1,4),(0,1,3)), rho=(1/4,1/8).
   Raw areas are (0,-1,1) and (1,1,1), squared norms 2 and 3.
   In direction -ez, g=2-sqrt(2)/8-sqrt(3)/24>2-3/16-1/12>0.
   This independently derived axis proves separation; it is not assumed maximal.
2. Moving safe A: A(t)=((0,0,0),(1,0,t),(0,1,0));
   B(t)=((0,0,5),(1,0,5),(0,1,5-t)); rho=(1/4,1/4).
   Both normals vary: cA=(-t,0,1), cB=(0,t,1), so both area norms>=1.
   Every A vertex has z<=1+1/4 and every B vertex has z>=4-1/4;
   their vertical gap is >=5/2 throughout. A complete CCD partition must certify.
3. Moving safe B: A(t)=((0,0,0),(1,0,2*t-1),(0,1,0));
   B(t)=((t,0,8),(1+t,0,8),(t,1,9-2*t)); rho=(1/8,1/4).
   cA=(1-2*t,0,1), cB=(0,2*t-1,1), both norms>=1.
   A top z<=9/8; B bottom z>=27/4; gap>=45/8.
   Both normals rotate and B translates. A complete partition must certify.
4. Endpoints separated, interior crossing: A=((0,0,0),(2,0,0),(0,2,0));
   B(t)=A+(0,0,4*t-2); rho=(1/4,1/4). Endpoint vertical gap=3/2.
   At t=1/2 both volumes coincide. Complete SAT maximum=-1/2 (vertical width
   1/2 is smaller than every lateral support width). This must violate budget0,
   or reject honestly if a configured proof budget cannot establish the witness.
5. Two rotating normals with endpoint-chord failure:
   A(t)=((0,0,0),(1,0,2*t-1),(0,1,0)), B(t)=A(t)+(0,0,18),
   rho=(10,10). cA=cB=(1-2*t,0,1), both norms>=1.
   Endpoint vertical gap=17-10*sqrt(2)>2 (sqrt(2)<3/2).
   At t=1/2 the XY footprints coincide and vertical overlap is2. Lateral widths
   are1,1,1/sqrt(2), hence complete SAT maximum=-1/sqrt(2)<0.
   Each normal has |N''(1/2)|=4, so both curvature contributions are nonzero.
   A receipt omitting either rho*M term is invalid. Tests will independently
   reconstruct the two-term inequality and reject a corrupted receipt. Because
   the general factor3 bound is conservative, no claim is made that this exact
   fixture necessarily becomes falsely certified by every one-term omission.
6. Interior area collapse: A(t)=((0,0,0),(1-2*t,0,0),(0,1,0));
   B is fixed at ((0,0,5),(1,0,5),(0,1,5)), rho=(1/4,1/4).
   cA=(0,0,1-2*t) vanishes at1/2; spatial separation cannot hide invalid area.
7. Exact contact: stationary copies of ((0,0,0),(2,0,0),(0,2,0)),
   separated vertically by1/2, rho=(1/4,1/4). Exact gap0, constant normals,
   and curvature0 yield a certified full [0,1] interval without an epsilon.
8. Explicit budget exhaustion: use fixture2 with max_time_cells=1 and
   max_time_depth=1 plus initial_sqrt_bits=max_sqrt_bits=1. The certificate may
   succeed if its exact inequalities suffice; an independent forced refinement
   case uses sqrt(2)-a dyadic convergent with sign separation requiring more
   than1 bit. Tests require honest unresolved and a full pending-domain record,
   never a manufactured unsafe conclusion or assumed budget failure.

## Scope and proof decisions

New independent pair-local arithmetic preserves the old Quadratic unchanged.
All maximal exact SAT ties are retained; deterministic order is evidence order,
not a friction-normal law. Identity owns explicit IDs, oriented node triples,
caller/canonical permutation, and exact input correspondence. Incident inputs
are diagnostic and never receive an incidence exemption. Full folded material,
response, normal-cone friction, persistent anchors, and transaction integration
remain required later work. There is no general convergence or native claim.

Planned files: geometry/_prism_pair_exact.py, geometry/prism_prism_manifold.py,
geometry/prism_prism_sweep.py; matching three tests; this report; Task2 report.

Parent ruling during authoring: expose verify_prism_prism_sweep in the same
sweep module. It independently recomputes the fingerprint, complete partition,
both area proofs, both curvature contributions and signed lower inequalities
from the supplied path. Constructor-created receipts are not self-authenticating.
Either single-contribution deletion must be rejected by this verifier. This
negative control is distinct from proving that the crossing path would become
falsely accepted by every possible omitted-term implementation.

Additional explicit proof-budget control, frozen during authoring and still
before execution: A(t)=((0,0,0),(1,4*t-1,0),(0,0,1)),
B(t)=A(t)+(0,0,5), rho=(1/4,1/4). Raw area c=(4*t-1,-1,0)
has norm>=1 and the exact vertical gap is4 at every time. The whole-interval
midpoint projection uses d0=(1,-1,0), giving c(t).d0=4*t and minimum0,
so that area proof cannot accept the root cell. On each half the midpoint
projection is strictly positive. max_time_cells=1 must leave both halves
pending; default limits must certify the two-half partition. This deterministic
budget fixture replaces any assumption that low sqrt precision alone must fail.

Parent static completeness ruling: a positive retained SAT axis is valid
separation evidence, but empty selected support clipping leaves the complete
manifold unresolved. That status is not a penetration claim. Proposal A checks
its independent support equation, never assumes that -ez is maximal. Sweep
certificates use full fixed-axis support inequalities without requiring a
complete static clip. The independent sweep verifier remains required.

Additional independent static two-root contact control: A as fixture1,
B=((0,0,5),(1,-1,5),(0,1,4)), rho=(5,2). Direction d=(-2,1,1)
is perpendicular to cA and cB. A's max-d edge and B's min-d edge meet at
(0,-2,4)=A2+3*cA=B1-cB. Material endpoints are exactly
(0,0,1,3*sqrt(2)) and (0,1,0,-sqrt(3)); both signed offsets are strictly
inside the supplied thickness. This proves actual edge-edge touching and
both material reconstructions independently of the implementation.

## Code checkpoint and public interfaces

Status: CODE_READY_TESTS_PENDING. This is authored code, not a passing or
numerically verified result. No project import, collection, pytest, lint,
formatter, compile, build, installation, profile, native check or remote CI
was performed. No historical TDD RED is asserted. No executable-validation
slot was requested or used during this authoring grant.

- `_prism_pair_exact.py`: frozen `PrismPairProofLimits`, reduced `PairField`,
  immutable `PairNumber`, typed `SignUnresolved`, rational square/root bounds,
  normalized-gap comparisons. Pair basis order is canonical for the supplied
  unordered radicands after rational/common-root reduction. It does not attempt
  global square-free factorization or implicitly convert different contexts.
- `PrismPairIdentity(domain,face_ids,node_ids)` owns immutable bytes behind
  read-only int64(2,) and int64(2,3) views. Faces are distinct; node triples
  are oriented, distinct and nonnegative. Cyclic-minimum node order and sorted
  explicit face IDs determine the canonical pair, not coordinates.
- `prism_prism_manifold(x_a,x_b,*,identity,half_thickness_m,proof_limits)`
  accepts FP64(3,3) inputs and positive FP64(2,) half thickness. It returns
  immutable `PrismPairManifold`: separated/touching/penetrating/unresolved,
  both fingerprints, explicit caller-to-canonical permutation, every exact
  SAT axis, every maximal tie, complete clipped extrema when resolved, both
  exact material endpoints, caller-ordered vertices, exact gap enclosures,
  and a typed remaining proof obligation when unresolved.
- Every `PrismPairAxis` retains rational canonical direction, caller-oriented
  direction, rational squared norm, exact field gap numerator, caller support
  indices and a canonical key. There is no primary axis or friction normal.
  Its points carry caller-order endpoints/weights and canonical material keys.
- `prism_prism_sweep(path_a,path_b,*,identity,half_thickness_m,h,
  penetration_budget_m,proof_limits)` consumes FP64(2,3,3) paths, positive
  duration seconds, nonnegative penetration budget metres. Its immutable result
  is certified/violation/unresolved. Certified segments form an exact rational
  [0,1] cover; pending cells preserve all remaining domains. A violation owns
  a rational time and a complete-SAT upper-gap witness. Area coefficients,
  area_a/area_b and normal_contributions use canonical side order; the explicit
  caller permutation and caller-oriented segment direction record exchange.
- `verify_prism_prism_sweep(certificate,path_a,path_b,*,identity,
  half_thickness_m,h,penetration_budget_m,proof_limits)` verifies safety receipts
  only. It returns False for violation/unresolved, altered bindings, incomplete
  partitions, inconsistent budgets or forged segment quantities. It does not
  invoke the sweep producer; it reconstructs path geometry and every fixed-axis
  inequality. Producer and verifier share the local exact bound calculator;
  independent written tests reconstruct its two-term equations separately.

Coordinates and signed offsets are metres, barycentric weights dimensionless.
Times in proofs are dimensionless t in [0,1]; h is bound into the fingerprint.
Normal differentiation uses that normalized time, so width^2 and curvature
refer to the same variable. No velocity, force, impulse or energy is produced.

All mutable inputs are copied into exact tuples/owned immutable bytes. Exact
shared-ID coordinates must coincide at every supplied endpoint. Canonical
fingerprints bind IDs, cyclic orientation, full coordinates/path, thickness,
proof policy, h and penetration budget where applicable. Caller fingerprints
also bind vertex/caller permutations. Reverse winding has a distinct identity
fingerprint. No float topology or expanded geometry tolerance is introduced.

The temporal lower bound is valid across support changes: each of the 36 vertex
pair projected differences satisfies the same endpoint/curvature inequality,
so their minimum does too. Both area bounds precede acceptance. The bound is
L=min(g_a_lower,g_b_lower)-norm_upper(d)*(rhoA*MA+rhoB*MB)*width^2/8.
Acceptance uses L+budget*norm_lower(d)>=0. Negative L divides by norm_lower;
nonnegative L divides by norm_upper for an outward normalized lower bound.
A single negative axis is never a violation witness.

## Logical commits and preserved scope

- 1c0d080: independent fixture equations frozen before implementation/execution.
- 3cc2ade: exact pair field and independent algebra tests.
- e69e5ab: identity-bound static geometry and material/coverage tests.
- d561b23: independent common-root static cancellation control.
- c94b91f: continuous proofs, independently recomputed receipt verification,
  moving/corrupted-receipt tests and additional pre-execution derivations.

The three new modules import only each other, NumPy and Python standard library.
No old exact/clip/manifold/sweep module, original test, S1 foundation, export,
response, anchor, history, runtime, dependency pin, workflow or parent document
was changed. Read-only Git comparison against d79316c confirmed exact legacy
and S1 source/test parity. Therefore old full suites and legacy direct consumers
are not rerun merely for this additive checkpoint. New tests remain unexecuted.

Pending validation must first preserve the original frozen six-file snapshot,
raw commands/logs/RAW_EXIT, and source hashes under a sole execution grant.
Formatting/lint and all three complete authored test files remain required.
No parseability, collection count, numerical pass, native parity, performance,
arbitrary convergence or full collision acceptance is asserted here.

Full folded cloth still needs the incident-material/shell decision, complete
common-load response, all cross terms, normal-cone friction, actual endpoint
mapping/ledger audit, persistent anchors, history and transaction integration.
Those are deliberately outside this geometry-only checkpoint and remain open.
