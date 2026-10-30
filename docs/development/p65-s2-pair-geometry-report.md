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
