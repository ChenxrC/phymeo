# S2-L same-base response proposal

Status: **PROPOSAL_NOT_FROZEN**. Documentation only, for parent adjudication.
Starting lane HEAD: cb41e0379562c6574847fb6c356ef15d11af5006.
Authority: parent e2dec9013797f5103b0fda2283f3553d46b66de0 and its
`task-2-response-proposal-brief.md`. S2-LG is accepted geometry infrastructure;
this proposal neither reimplements nor re-reviews its 69 host cases.

Only this new document is written. No source, test, old report, shared seam,
merge, import, test/collection, offline numerical calculation, probe, formatter,
lint, compile, build, profile, installation or remote action was performed.
The formulas and candidate fixtures below are handwritten. They are proposed
expectations, not observed results. No execution slot is held.

## 1. Recommendation and finite successful domain

Recommend **A: one public replay coordinator with immutable phase attachments**.
All self rows sharing a cloth domain enter one response problem and one physical
base. The coordinator proposes held world impulses, replays the actual base
program from its original snapshot, certifies the selected trajectory, then
requests one aggregate native terminal velocity application. Its output is the
actual base result, never a host-created replacement state.

The first version should have an explicit `unique_exposed_ray_v1` response law:
nonincident face pairs, nondegenerate representable triangles, all maximal SAT
evidence retained, a unique oriented exposed normal ray whenever a nonzero
contact load is accepted, complete point/segment/polygon extrema, and a finite
independently audited response solve. Different pairs may share physical nodes;
all such response cross terms remain mandatory. For a charged held interval,
initially close material anchors, their supporting features, and their normal
rays must have a certified continuation. The first implementation may restrict
charged held continuations to constant face normals and a fixed support feature;
positive raw-area projection plus exact coefficient parallelism proves that
restriction. Unloaded motion still uses the full two-normal S2-LG certificate.
Terminal birth at exact contact is supported and can create nonuniform outgoing
nodal velocities. Continuing that new state through a later rotating loaded
step requires the corresponding future continuation proof; it is not silently
promised by the impact window below.

This finite domain contains sustained full-face pressure, two-sided collision,
and genuine friction sticking/sliding with normal-offset rotational response.
It does not reject every useful self-contact case. It deliberately does not
claim an arbitrary nonsmooth normal-cone law or general convergence. Positive
separation with several tied SAT axes is harmless while all loads are zero;
noncollinear active load-bearing ties return an explicit unsupported obligation.
That is a scoped response boundary, not a completion claim for full S2.

Initial negative penetration beyond the declared budget, incomplete static
clipping, exhausted sign/time proofs, degenerate material geometry, unresolved
active ties, or failed actual-state audits reject atomically. A static result
with positive retained SAT evidence but unresolved clipping can prove that an
unloaded pair is separated; it cannot supply a complete loaded manifold.

## 2. Observed public seams and two viable integration approaches

The following behavior was read from the current parent integration as text.
Paths are relative to src/phymeo.

| Public seam | Observed contract and consequence |
| --- | --- |
| coupling/cloth_self_contact.py, SelfContactBinding | All attachments go through one TriangleOffsetPointPort and one same-domain MultiDomainPointOperator. Public arrays record actual normalized weights, points and Jacobians. node_impulses is an audit/request aggregate, not an apply operation. |
| coupling/domain_graph.py, MultiDomainPointOperator | Sums signed loads before response/apply; identical endpoints cancel. All same-domain and shared-point cross terms are retained. apply returns one owned candidate per physical domain. |
| coupling/offset_port.py, TriangleOffsetPointPort | Full derivative of b.x+s.N; host FP64 geometry with the base's actual backend. Scaled area admission remains 1e-10. Rebuild on configuration/attachment changes. Existing actual-point representability/moment checks remain unchanged. |
| constraints/contact_impulse.py, solve_contact_impulses | Fixed-configuration, closed same-world-point, zero-restitution solve. It queries a full response, performs final actual-law/energy audits, and calls graph.apply_impulses itself. It is not a proposal-only helper and must not be followed by another application of the same impulse. |
| constraints/friction.py, solve_friction_disk | Public fixed-normal 2x2 tangent subproblem; impulses N*s, response 1/kg, energy J. Sticking is tried before dynamic-disk sliding. It does not advance a scene or solve coupled normal capacities. |
| runtime/load_replay.py, SceneLoadReplay | Owned original state/history/input snapshot; declared integrated load divided by h into an input rate. evaluate returns actual represented input difference times h, not just the requested load. Replays never accumulate candidate states. |
| runtime/terminal_impulse.py, PointVelocityImpulses/TerminalImpulseProgram | One live native terminal operation after base convergence; batches domain requests, records applications=1, selected pre-state/diagnostics and an independent terminal ledger. Borrowed base factory ownership remains with its owner. |
| coupling/participant.py, ParticipantWorkspace | Public predict/material sweeps/point_port/apply_position_load/prepare_velocity/apply_velocity_impulses/result/close. Configuration loads are kg*m; velocity impulses are N*s. Iterative sweeps may not advance time. |
| runtime/scene_stepper.py, SceneStepper | Clones registered state/history; closes before publication; failed later substeps return the entire original frame. Diagnostics/close failures cannot publish a partial candidate. |

**A: replay coordinator (recommended).** New response code uses public S1 arrays
and response queries to form a proposal only. A separately verified load plan
aggregates all attachment J-transpose loads into a nodal FP64 array. Held loads
enter one SceneLoadReplay force channel; terminal loads enter one
PointVelocityImpulses batch for the cloth domain. The graph's apply_impulses is
NOT also called in this route, and audit node_impulses is never applied as an
extra request. This is an explicit extension from S1's instantaneous application
route to the common replay route; parent must freeze that routing contract.
Neither two independently advanced cloth states nor an already-applied graph
candidate is spliced into the native result.

The existing closed-contact solve can remain an independent fixed-configuration
reference under later authorized tests. Production needs a new proposal-only
solver, because calling that solver and then TerminalImpulseProgram would make
two actual applications. Host proposals do not become backend acceptance until
the chosen native program has executed and its actual output passes all gates.

**B: a common participant contact block.** A new SceneProgram may own one public
ParticipantWorkspace per domain, interleave complete material sweeps with one
aggregated contact configuration correction, reconstruct once, then apply one
aggregate terminal impulse. The contact multiplier Lambda is kg*m, with
p=Lambda/h only when the prescribed position/velocity reconstruction establishes
that identity. For constant J and a material-free mass model,
Delta x=M^-1 D.T Lambda and Delta v=Delta x/h, so Lambda=h*p exactly reproduces
A's contact increment. This gives a concrete finite successful domain, including
the stationary pressure control below. The whole aggregate is applied once per
private correction increment, not once per endpoint or pair; iteration counts
and all intermediate updates remain visible.

B is not automatically equivalent to held-force replay for deforming material.
External contact position corrections change the ordered material path and its
work. It needs a separately named discrete law, recorded external configuration
updates, current-map/geometry proofs, and parent-owned integration into the
material path ledger. It may not inject corrections into an old trace and call
that trace unchanged. A bounded global normal/tangent block is feasible, but no
arbitrary-convergence claim follows from a local disk solve. B is viable as an
explicit participant-based alternative, not a shortcut to A's acceptance.

A preserves existing original-input replay and selected-pre semantics with the
smallest new mathematical boundary. B has a larger material/representation
contract and is not recommended for the first response checkpoint.

**Concrete missing robot-free seam.** ClothProgram currently lacks the
terminal_point_impulses feature. ArticulatedCouplingProgram supplies it for an
empty contact set, but its constructor requires a robot domain. Do not introduce
a dummy robot or use private workspace/native fields. Parent should add a small
public particle-participant SceneProgram adapter that delegates public methods,
owns exactly one participant per domain, supports one aggregate terminal batch,
and binds backend/precision/history. It can initially support a node-only
material-free/Distance scene. Composite/B2 integration is separately typed below.

## 3. Shared-base equations and phase law

For k material pairs on n physical nodes, both endpoints are
x_A,k=sum_i b_A,ki*x_i+s_A,k*N_A,k and similarly for B. Signed s is metres;
barycentric b is dimensionless. Let J_A,J_B have shape(k,3,n,3) conceptually,
with sparse three-node blocks per endpoint, and D=J_A-J_B. World impulses
p[k,3] are N*s, +p at A and -p at B. With original positive nodal masses,

    ell = D.T p                         [N*s, shape(n,3)]
    W = D M^-1 D.T                      [1/kg, shape(3k,3k)]
    u_after = u_before + W p             [m/s]
    Delta K_terminal = p.u_before + .5 p.W.p  [J].

Every block is
W_kl=J_A,k M^-1 J_A,l.T+J_B,k M^-1 J_B,l.T
     -J_A,k M^-1 J_B,l.T-J_B,k M^-1 J_A,l.T.
No cross term disappears merely because both endpoints have the same domain,
because a pair is nonincident, or because two different pairs share a face.
The physical nodal sum is formed BEFORE inverse mass or its energy quadratic.
All external contact rows touching that same base must join this common solve
under the future parent coordinator. Sequential independent contact programs
that each advance the cloth are not an equivalent integration.

At phase j=0 (held) or 1 (terminal), the owned actual world load is
p_j=lambda_j*n_j+t_j, lambda_j>=0, n_j.t_j=0. n is a certified unit-direction
representation of the exact rational SAT ray, not an epsilon-based axis label.
Use actual represented normal projections in audits, not a requested scalar
lambda when rounding changes its world projection.

Held Signorini is endpoint-gap complementarity for the selected replay: no
accepted depth below -b, lambda_H>=0, and its dimensionally normalized residual
vanishes within the declared bound. Its full displacement response is supplied
by actual replay. h*W is a useful proposal matrix for a material-free frozen
map, never a proof of a nonlinear material trajectory derivative. Terminal
Signorini uses the selected pre-state and final relative normal velocity,
lambda_T>=0, u_n^+>=0, lambda_T*u_n^+=0 within explicit residual budgets.
Restitution is zero in the first contract.

Within a given owned tangent disk, first test a sticking solution with
u_t=0 and ||t||<=mu_static*normal_capacity. Otherwise require
||t||=mu_dynamic*normal_capacity and t anti-parallel to the actual final slip.
For first controls mu_static=mu_dynamic=1/2. Unequal coefficients may later be
admitted under the same stick-first convention, not presumed to have a unique
coupled Coulomb solution. The per-row disk is a bounded conditional subproblem;
normal/tangent coupling and every other row remain in W and final audits.
A joint unconstrained SOC energy minimization is not silently substituted for
Signorini plus Coulomb: its normal stationarity is generally a different law.

Match existing two-phase ownership semantics: an initially closed owner spends
held capacity mu*lambda_H in the held phase and terminal capacity mu*lambda_T
in the terminal phase. It does not spend lambda_H twice. An initially open owner
has zero held tangent load; if it is proved to reach contact, its terminal disk
may use mu*(lambda_H+lambda_T) once. Each phase's normal projection is evaluated
in its own n_j BEFORE summation. A previous timestep's impulse is never new
capacity, and a new/retired material owner cannot inherit capacity from a row
number or a neighboring clip vertex. A constant-normal first domain avoids an
unproved normal-rotation transport of a loaded disk.

## 4. Ties, pressure ownership and material identities

S2-LG returns all maximal SAT ties. Evidence ordering is not a law. Duplicate
canonical evidence for the same physical material owner/ray must be deduplicated
or rejected before constructing a constraint; it never creates another disk.
Each distinct retained material pair owns ONE vector impulse and ONE phase
capacity. The three corners of a face contact can own three distinct pressures;
the total friction bound is mu*sum(lambda_k), not mu times that total on every
corner/axis. An exact duplicate owner cannot inflate this sum.

Noncollinear ties need a new nonsmooth-contact model before acceptance. In
particular, near a corner the nonoverlap set is a union of separating branches;
blindly imposing every tied separating-plane velocity inequality as if all
were independent physical contacts can reject a geometrically separating
motion. Nor does giving each generator an independent copy of a pressure
budget define a physical normal cone. One possible later law would allocate
nonnegative generator weights whose SUM is a single owned normal measure and
solve an explicit cone/tangent variational problem. Its traction meaning,
material owner and friction dissipation must be specified first. That problem
is left to parent adjudication; unique_exposed_ray_v1 rejects these loaded ties
explicitly while retaining their complete geometry evidence.

General redundant face pressures need not be unique even if nodal response is.
The first solve should either certify the admitted active response is sufficiently
positive definite or reject a rank/conditioning obligation. Never regularize a
singular system and call the pressure physically unique. If parent later admits
nonunique systems, a minimum-norm/lexicographic secondary choice is an algorithmic
selection, must be named, and still needs capacity/dissipation invariance audits.
The analytic controls below have a unique normal solution under their stated
matrix; that does not extend to arbitrary contact networks.

A persistent owner key includes schema version, domain, explicit face IDs,
cyclic-oriented node triples, both material addresses and birth generation.
Its identity is not its rounded world point, SAT axis array position, or nearest
new extreme. A/B exchange swaps endpoint slots and oriented normal, preserves
canonical ownership, and cannot swap thickness/phase budget accidentally.

For numerical history, propose each barycentric address as int64 dyadic counts
summing to 2^52, plus a signed FP64 offset and its exact bit token. Deterministic
inside-simplex quantization and bounded offset rounding carry independent exact
world-error enclosures. Record actual weights returned by the public offset
port after its normalization. The ideal LG endpoints and every complete ideal
manifold extreme remain separate evidence; quantization cannot erase an extreme,
change the primal prism, establish topological equality or silently select a
neighboring contact. Stored history uses only the registry's supported integer,
bool and FP64 fields; exact field tokens are serialized diagnostics, not object
arrays smuggled into history.

Freeze both material addresses during a phase. Retain old loaded owners only
while their finite support/contact continuation is proved. Rebuild from the
same original snapshot when a bounded support change requires new owners; a
new owner starts with zero inherited capacity. A loaded owner leaving support
without a proved redistribution law rejects/retries; it is not silently moved
to a centroid or nearest extreme. Every final manifold extreme and required
velocity direction is audited, including extrema carrying zero solved pressure.

## 5. Independent actual-map, couple and work ledger

A new ledger must rebuild each ideal offset Jacobian from the original node
coordinates and the actual owned material address, rather than trusting S1's
reported J as its oracle. For one face, c=e cross f, C=c.c and N=c/sqrt(C):

    delta N = (I-N*N.T) delta c / sqrt(C),
    J_i = b_i I + s (I-N*N.T) (partial c/partial x_i) / sqrt(C).

Use reduced pair fields for each pair/time and rational outward intervals for
sums across unrelated pairs. The old single-Quadratic thick mapping ledger
cannot be treated as a universal many-root accumulator. Exact rational stored
J entries/world impulses give a second, distinct actual-map aggregate. Compare
ideal-J, actual-J, requested nodal load, represented replay input load, and actual
terminal response at their own boundaries; do not collapse them into one number.
Both endpoint errors and all shared-node contributions are included.

S1's node_impulses can cross-check requested nodal aggregation; it is not evidence
that the native program applied that request. A proposed production
SelfContactLoadPlan reads only public binding arrays/domain_impulses, sums all
signed attachment loads through actual J blocks, and produces one owned nodal
request. The independent ideal ledger checks it. Replay/terminal receipts then
check actual application. This explicit request route, not a private _loads
call or second graph.apply_impulses, is the recommended parent ruling.

For aggregate nodal load ell and pair loads p, independently verify

    sum_i ell_i = 0                                  [N*s],
    sum_i x_i cross ell_i = sum_k (x_A,k-x_B,k) cross p_k [N*m*s],
    sum_i ell_i.v_i = sum_k p_k.(J_A,k-J_B,k)v         [J].

The last line is impulse/velocity pairing. For force power in watts divide the
held impulse by h. The couple is NOT automatically zero: separated endpoints
produce (x_A-x_B) cross p. Normal load with separation parallel to n can have
zero couple; finite-gap friction generally does not. Record the actual couple
and bound its absolute contribution. Do not cancel opposing unproved errors or
claim conservation merely from +p/-p. Exact common-point controls have zero
pair couple; represented anchor and map errors still get independent bounds.

Held internal work is W_H=sum_i(ell_H,i/h).(x_pre,i-x0,i), using the ACTUAL
represented input difference returned by ReplayTrial. Its comparison with
material-point displacement work includes the fixed-map/nonlinear-path defect;
J0 times nodal displacement is not generally an exact normal-offset chord.
The first charged constant-normal controls have an exact affine map along their
path. Rotating loaded continuation needs a new bounded work/anchor-path proof.

At terminal time, sum all nodal impulses first and use the original mass model:

    W_T = sum_i ell_T,i.v_pre,i + .5 sum_i w_i |ell_T,i|^2,
    E_T_defect = (K_post-K_pre)-W_T.

Cross terms from multiple rows/endpoints sharing a node live inside that square.
Audit actual decoded post-state, original inverse masses, conversions and
selected pre-state. No per-row diagonal energy sum can replace this identity.

The physical whole-step/window balance is

    Delta(K+U_material) - W_original_external,

where W_original_external uses original force/gravity/robot inputs, never the
effective replay inputs. Self-contact W_H and W_T are internal transfers and
dissipation diagnostics; they are NOT subtracted as external power to forgive
energy creation. Maintain both the raw physical defect and detailed internal
identities. Keep per-step positive energy and their whole-window positive sum,
and component/domain absolute angular/mapping defects, without cancellation or
averaging. Material-free U=0. Distance U uses the original compliance/length law;
B2 U and path semantics require parent typed dispatch, not a fake distance law.

## 6. Candidate independent physical controls, handwritten before execution

These are explicit proposal fixtures. Values must not be fitted after producer
runs. All nodes belong to ONE `cloth` ParticleModel/SceneState, not two domains.
Faces have IDs 11/29, oriented node triples (0,1,2)/(3,4,5), both +z winding, and
nonincident material labels. All arrays/configuration use explicit FP64 storage.
Gravity is zero. No robot, handoff adapter, prescribed contact impulse, external
collision body or host state correction supplies the expected contact response.

For every accepted window, propose two independent fresh model/history rebuilds,
canonical A/B exchange and cyclic node reorder, and a proper signed coordinate
rotation (x,y,z)->(-y,x,z) plus translation (4,-2,1) m. Transform every original force/velocity and expected vector as well. Geometry/identity predictions
transform with these operations; the solver must not branch on case names.
Backend-dependent thresholds do not change fixture forces, windows or laws.

### A. Sustained held pressure with unequal masses/thicknesses

Use the common XY triangle v0=(0,0), v1=(3,0), v2=(0,4) metres. A's middle
vertices have z=+1/8 m and half thickness rho_A=1/8 m; B's have z=-1/4 m and
rho_B=1/4 m. Their complete contact polygon is the same 3-4-5 triangle at z=0.
The unique contact ray toward A is +ez. Actual material addresses are the three
corner weights, with signed offsets s_A=-1/8 and s_B=+1/4.

Each A node has mass 2 kg (w=1/2); each B node mass 4 kg (w=1/4). Initially all
velocities are zero. Original nodal forces are -2ez N on A and +2ez N on B.
Set both friction coefficients to 1/2; the exact tangent impulse is zero.
Use T=1/4 s and h in {1/64,1/128,1/256} s: 16,32,64 steps, respectively.
Initialize fresh contact history, not a copied multiplier warm start.

Normal-offset derivatives do not contribute to a purely normal nodal load.
At each matching vertex the full normal response is w_A+w_B=3/4 kg^-1. The
free endpoint gap is -(2 h^2)(3/4)=-(3/2)h^2 m. A held corner impulse p_H adds
h*(3/4)*p_H, so endpoint complementarity gives independently

    p_H,k = 2h N*s,  k=0,1,2;  t_H,k=0;  p_T,k=t_T,k=0.

This also follows directly from the two original force balances: each actual
replay contact force exactly cancels its own original +/-2 N. It is not a
producer-derived target. Positions and velocities remain exactly initial in
the ideal arithmetic, all corner gaps remain 0, total held impulse magnitude
per pair per step is 6h N*s, and over the window it is 6T=3/2 N*s. Both contacting
faces have positive pressure while the single physical base is advanced once.

K and original external work are 0; angular change, internal held displacement
work and pair couple are 0. Normal pressure is unique in this fixture because
the three normal response blocks are independent positive scalars. Do not infer
uniqueness of arbitrary patch pressures from it.

Run first with materials=None. A separate Distance version may add exactly
ordered within-face edges(0,1),(1,2),(2,0) and the corresponding B edges, rest
lengths 3,5,4 m, compliance 1/16 s^2/kg, with initially zero multipliers. They stay
at exact rest, so U and every ideal material correction remain 0. This supplies
an accepted material-bearing response control without depending on B2 bending
or its pending representation implementation.

### B. Force-free impact with genuine full-offset sticking/sliding

Use v0=(-1,-1), v1=(1,-1), v2=(0,1) m, mass 1 kg per node, and rho_A=rho_B=1/4 m.
Let T=1/16 s; take h in {1/16,1/32,1/64} s, giving 1,2,4 steps with the same
physical initial state and the same impact time. At t=0 set

    A_i=(v_i.x-u*T/2, v_i.y,  rho+T), v_A=( u/2,0,-1) m/s,
    B_i=(v_i.x+u*T/2, v_i.y, -rho-T), v_B=(-u/2,0, 1) m/s.

Original force and gravity are zero; materials=None. Before T the exact vertical
prism gap is 2*(T-t)>0, irrespective of the lateral footprint. Thus no held load
is needed. At T both footprints coincide and touch at z=0; complete LG supports
supply all three corner owners. Their zero-restitution normal terminal impulses
are eta_k=1 N*s because u_n^-=-2 m/s and W_nn=2 I kg^-1.

The full normal-offset tangent response is not centroid translation. For these
barycentric shape gradients,

    g0=(-1/2,-1/4), g1=(1/2,-1/4), g2=(0,1/2) [1/m],
    sum_i g_i*g_i.T = diag(1/2,3/8) [1/m^2].

Equal/opposite signed offsets make the aggregate normal/tangent cross block
cancel in THIS symmetric fixture. The tangent block between corners j,k is
2*(delta_jk I + rho^2*sum_i g_i*g_i.T) kg^-1. For equal x impulses t at all
three corners, the relative x response is therefore

    u_after = u + k_x*t,  k_x=2*(1+3*rho^2/2)=35/16 kg^-1.

The value 2 kg^-1 from a centroid-only model is wrong. With mu_s=mu_d=1/2 and
eta=1 per owner, each disk radius is 1/2 N*s and the exact stick/slip boundary
is u_c=(35/16)*(1/2)=35/32 m/s.

- Hold/stick member: u=35/64 m/s. Each terminal tangent impulse is
  t=-1/4 N*s and the actual material-point relative tangent speed is 0.
- Slip member: u=35/16 m/s. Each t=-1/2 N*s saturates its own single capacity,
  and the actual final material-point relative tangent speed is 35/32 m/s.

This is terminal friction sticking/sliding, not a claim of sustained static
holding after the impact. The window ends at T. A later rotating loaded step
needs the additional continuation contract described above.

The full-map nodal oracle exposes the rotational response. With terminal t as
above, both faces have v_z,i=3*rho*t*g_i.x; A's v_x=u/2+t and B's v_x=-u/2-t.
Thus the stick member's centroids still have opposite x speeds 3/128 m/s, while
its offset contact points stick. A centroid-velocity stopping check would give
a false result. The outgoing common face rotation is part of the physical
response, not numerical error to be removed.

Normal energy change is -3 J. The tangent change is
3*(t*u + .5*(35/16)*t^2), giving -105/512 J for stick and -315/128 J for slip.
The full expected kinetic changes are respectively -3-105/512 J and
-3-315/128 J; W_original_external=0. Initial and final total angular momentum
are equal. In particular L_y=3*rho*u, and the endpoint impulse couples vanish
because each A/B contact pair acts at the same actual ideal world point.
Both spin and orbital contributions must be included in this comparison.

These values were derived from the full nodal Jacobian and mass equations, not
from running a handoff, fitting a friction coefficient or reading a producer
matrix. All chosen inputs and requested ideal impulses are binary rational.
Native arithmetic still has to meet the declared actual-state audits; these
hand calculations do not certify a future native implementation.

### C. Shared-base cross terms, tie/identity and representation negatives

For a concrete shared-base control, use three copies of the XY triangle from B,
all in one nine-node mass-1-kg model, rho=1/4. Middle planes are A:z=0,
B:z=-1/2, C:z=+1/2, with initial velocities 0,+ez,-ez respectively. Pair A/B
uses +ez and A/C uses -ez. They are nonincident pairs but share A's physical
nodes between rows. Both interfaces are initially touching; take h=1/64 s,
materials=None, no original forces, zero tangent loads, one capture step.

For each matching corner the complete normal matrix and free velocity are

    W_n=[[2,-1],[-1,2]] [1/kg],  u_n=(-1,-1) [m/s].

Both held impulses must be (1,1) N*s: A's aggregate is exactly 0, B/C stop, and
the two interfaces remain at gap 0. Terminal loads are 0. Three corner blocks
give Delta K=-3 J. Replacing W_n by its diagonal gives (1/2,1/2), leaving both
actual closing speeds -1/2 m/s and endpoint gaps -h/2; the final actual/CCD gate
must reject. Independently applying two candidate states or summing per-row
energy without the -1 cross terms must also fail. This is not an adjacent-face
exemption: each paired face has disjoint node IDs; cross terms arise from a
shared physical endpoint face across different rows.

Tie negative: use equal-thickness horizontal right triangles
((0,0),(2,0),(0,2)); translate the second by (2,0,2*rho). The prisms touch at a
single point with multiple noncollinear maximal separating rays. A loaded
unique_exposed_ray_v1 request must reject with the full tie evidence intact,
not select the first axis or allocate mu*P to every axis. A duplicated canonical
owner/ray, or repeated pair ID, must be deduplicated exactly or rejected before
capacity accounting; never silently gain a second friction budget.

Identity negatives: swap coordinates without swapping the oriented node IDs;
reuse history after reversing a node winding; alter either thickness or one
phase's owner generation; change a path endpoint after CCD; retain a loaded
anchor after it leaves support. Every stale binding must reject and leave
complete state/history/inputs unchanged. Canonical A/B exchange WITH all data
swapped is a positive covariance control.

Mapping negative: replace J by barycentric weights alone, omit either endpoint's
normal derivative, independently advance duplicated same-domain states, or apply
node_impulses a second time. The B response coefficient/energy/angular oracle
or C closing-speed audit must detect it. Couple negative: for a diagnostic pair
of represented endpoints separated by delta*ez and impulse t*ex, the expected
couple is delta*t*ey, never 0 by construction. Unknown interval/representation
proof and an incomplete LG manifold remain rejection, not a passing state.

## 7. Proposed interfaces, shapes and ownership

These are proposed new contracts, not existing callables or an implementation
grant. All configurations are owned, immutable and fingerprinted; invalid dtype,
shape, nonfinite values, identities, duplicate ownership and unknown schemas
reject before numerical work.

- `SelfContactResponseLaw(mu_static,mu_dynamic,*,normal_domain,
  phase_capacity="owned_split_v1",restitution=0.0)`: explicit dimensionless
  coefficients, the finite ray-domain above, and named phase ownership.
- `SelfContactResponseLimits(max_contacts=128,max_anchors_per_pair=64,
  max_rebuilds=8,max_outer_replays=200,max_line_search_halvings=16,
  max_normal_iterations=200,max_tangent_iterations=96,geometry_limits=...)`:
  finite positive integer resources; exhaustion carries remaining obligations.
  Geometry limits preserve the accepted PrismPairProofLimits defaults
  64/4096 sqrt bits and 4096/32 time cells/depth, unless parent freezes another
  explicit policy before any tests. No automatic budget escalation.
- `SelfContactBudgets(penetration_m,anchor_error_m,velocity_mps,
  impulse_ns,energy_j,angular_impulse_nms)`: physical units are explicit, with
  separate independent map/enclosure budgets; no hidden single dimensionless
  epsilon substitutes for geometry, force, work and angular checks.
- `make_self_contact_anchors(manifolds,*,identities,history,limits,budgets)`:
  complete ideal extrema plus owned finite material anchors and a continuation
  obligation. Material triangle row indices are int64(2k,3), actual weights
  FP64(2k,3), offsets FP64(2k,), pairs int64(k,2), canonical owner tokens and
  active masks bool(k,). Ideal endpoint coordinates/material values remain exact
  immutable diagnostics. The function does not claim temporal continuation.
- `propose_self_contact_phase(binding,normal_rays,free_data,*,phase,
  capacities,law,limits)`: proposal-only, with FP64(k,3) world impulses and
  per-owner normal/tangent/capacity receipts. Full W has shape(3k,3k); no
  diagonal-only approximation is an acceptance witness. No state is applied or
  returned. Friction disk calls are conditional proposals, not whole-graph
  physical certification.
- `make_self_contact_load_plan(binding,world_impulses,*,phase)`:
  owned nodal request FP64(n,3), actual attachment arrays, original endpoint
  identities and phase fingerprint. It aggregates all rows once and is checked
  against S1 audit node_impulses plus the independent ideal map ledger.
- `SelfContactMappingLedger.require(plan,*,original_positions,
  original_velocity,actual_replay_or_terminal_receipt,budgets)`:
  reconstruct both ideal J maps, actual-J request sum, map error, net impulse,
  actual couple, virtual-work pairing and represented-load discrepancies.
- `SelfContactResponseLedger.require(original,selected_pre,final,*,
  original_inputs,h,geometry,anchors,held,terminal,material_evidence,budgets)`:
  complete phase laws, all extrema, full LG trajectory certificate verified via
  verify_prism_prism_sweep, actual work/energy/angular balances and publication
  binding. Unknown/unsupported is never safe.
- `ClothSelfContactProgram(scene,base_program,topology,*,law,materials,
  budgets,limits,registry,inner_settings,history_prefix="self_contact")`:
  a SceneProgram over public base/replay/terminal methods. First materials value
  is None or explicitly bound DistanceConstraints; B2 needs a later typed reader.
  Base factory is borrowed; program owns all replay/workspace lifetimes it opens.

An outer replay normally has no still-live native point port after close. Thus
its public ParticlePointPort/S1 graph may supply a host mathematical response
PRECONDITIONER from the actual published snapshot and original mass. It must
never apply that host candidate or label its response a native measurement.
Actual native conversion, velocity response and work are certified only from
the real terminal/replay output. If parent later provides a live public backend
response view it can replace this preconditioner, but private _NodePort or native
candidate access is not an acceptable substitute. Host geometry/load proposals
already belong to the explicit contact architecture; material dynamics and
native publication remain actual backend operations.

History is bounded by the configured contact/anchor counts. Propose active
flags, explicit face/node IDs, dyadic barycentric counts, offsets/offset bits,
birth generations, phase normals and world impulses. Store impulses for identity
and warm-start proposals only, never as reusable physical capacity. Derive schema
shape/dtype before workspace creation; mutation invalidates the bound program.
Full exact diagnostics use explicit rational/root tokens accepted by the runtime
fingerprint serializer, not raw PairNumber/Fraction object arrays.

## 8. Lifecycle, failure and retry

1. Snapshot all original domain state, inputs, history and material/representation
   identity. Enumerate nonincident candidates with S1, retain its explicit
   shared-vertex exclusion policy, and record every pair disposition. Self-only
   acceptance is scoped to the declared nonincident pair set.
2. Validate independent face areas even if broad phase finds no pair. Bind R4
   identities. Build complete LG manifolds/anchors and reject unsupported loaded
   ties before any pressure/friction capacity is allocated.
3. Build one phase-0 attachment batch over the original full node base. Propose
   a complete held load vector, aggregate once, replay from the same original
   state/history, and inspect actual represented inputs and native outputs.
   Zero-load/p-load/zero-load trials must not accumulate previous candidate state.
4. Rebuild/line-search only within frozen resource caps. Every trial has a new
   state/path fingerprint. Recompute/verify CCD and anchor continuation for that
   exact path. A previous certificate never validates a modified candidate.
5. Select one complete held pre-state. Freeze both terminal material endpoints
   and phase-1 normals. Construct terminal proposals using the full shared-base
   matrix, then run the native terminal pass once for its aggregate domain batch.
   The terminal replay's complete pre-state, pre-diagnostics, history, inputs,
   stage iterations/metrics and material/representation path must match the
   selected held pre-state; changing only a reported summary is insufficient.
6. Verify every normal/tangent owner, every ideal extreme and supported ray,
   actual mapping, actual couple, all-node kinetic identity, original-input
   physical balance, and complete path coverage. A local solver residual alone
   is not acceptance. No centroid-only or row-diagonal final audit is allowed.
7. Own/freeze all diagnostics and history, close child workspaces/replays, and
   allow SceneStepper to publish one candidate. Close/diagnostic/late ledger
   failure discards every candidate and restores the original public state.
   Fresh retries rebuild private resources and must agree with a fresh run.

Fault controls must cover a late geometry obligation, changed owner, native
request failure, selected-pre mismatch, material-path failure, diagnostics and
close failure, and failure after an earlier accepted frame substep. Resource
lifetimes and persistent history are part of acceptance, not optional cleanup.
No new test or implementation is run under this proposal task.

## 9. Candidate physical/proof budgets and backend dependency

For controls A/B/C propose penetration budget 0 m, since their exact supported
solutions require no overlap and have exact dyadic geometry at publication.
This is an intentionally strict physical requirement, not a claim that general
native contact can meet it. A later positive budget must be explicitly adjudicated
as a law change, not fitted to a failing fixture. Candidate anchor world error
ceiling is 2^-40 m per endpoint; these controls have exact corner/offset addresses
and zero ideal conversion error. Existing offset scaled-area and stored-point
representability thresholds remain unchanged.

Propose reference_length=1 m, reference_speed=1 m/s, h/windows exactly as section 6,
outer StepSettings.substeps=1 per prescribed step, normalized host tolerance 1e-8
and native tolerance 2e-5, each declared before execution. Phase velocity residual
has units m/s and is bounded by tau*reference_speed. A row impulse residual is
converted using its independently positive response scale, not compared directly
to a velocity tolerance. For the proposed controls, set the explicit load-map impulse error ceiling to tau*Jref with Jref=1 N*s; retain the separate response-scaled phase residual too. No impulse-unit error is divided by a speed without a mass/response scale. Retain existing response symmetry/conditioning gates
where reused, without weakening them to accommodate a backend. Actual normal
and tangent support/cone signs use exact/outward proof bounds.

Adopt the parent's stricter S2 response ceilings: host 1e-8 J positive energy
and 1e-8 N*m*s absolute angular defect; actual native 1e-5 J and 1e-5 N*m*s.
Each applies per step AND to the noncancelling whole-window sum. The A exact-rest
Distance control does not borrow B2's 1e-3 ceilings. These hand-derived controls
provide no mathematical reason to request a larger budget. The whole-window
limit is not multiplied by the number of steps. Mapping/codec/
normal-direction enclosure widths must be separately allocated within at most
1/64 of the applicable physical residual budget. If the independent bound is
too wide, refine within the frozen proof cap or reject; do not omit the term.
The analytically zero work/torque controls and exact expected nonzero energy
losses remain the independent physical oracles.

These are candidate NEW response budgets for parent review, not modifications
to any accepted legacy geometry/mapping/material gate. All original B2/G2
physical budgets remain binding when those systems are combined.

**First actual backend path:** host geometry/independent ledgers plus one
selected native particle participant, explicit numpy/interpret/cuda/cpu_cpp
mode, no fallback. None/Distance controls can be tested without B2. Prediction,
material corrections where present, velocity reconstruction and terminal
application must have operation-specific actual execution receipts. Native
velocity is the actual FP32 result promoted to FP64 in the existing v1 route;
no host recomputation of v, x or lambda is allowed to make an audit pass.

**B2 dependency is now concrete.** Parent has frozen the proposed
reference-displacement-f32-v2 option in
`2026-10-30-p65-b2-native-representation-design.md`, but its full native acceptance
is still pending at this documentation checkpoint. Immutable trial-start R plus
actual native FP32 d decodes X=RN64(q(R)+q(d)); world X is not a single-FP32
promotion, while velocity remains native FP32 promotion. Contact geometry and
physical ledgers must all use that same canonical X. They may not use R+d before
decode, a host-repaired world state, or old require_fp32(X) assertions to relabel
v2. Keep immutable R/d/path/decoder evidence bound across held and terminal
replays, with fresh zero displacement from the same original snapshot per trial.
Terminal impulse changes velocity only and preserves encoded position/reference.

The current old thick mapping/material dispatch is distance/single-Q/v1-specific.
A new pair/many-root ledger and parent-owned typed material/terminal dispatch
are required. Do not modify old Q or repurpose a v1 raw trace. The new contact
program must reject composite/v2 inputs until that reader is present and the
actual backend producer is accepted. B2's failed original native receipts stay
failed; S2 geometry or a material-free response success cannot certify B2.

## 10. Proposed exact file ownership and test selection

If parent freezes A, propose this bounded response lane allowlist, all NEW files:

1. src/phymeo/constraints/cloth_self_response.py — law/limits/budgets, owned
   phase/load receipts and bounded proposal-only coupled normal/tangent solve.
2. src/phymeo/geometry/cloth_self_anchors.py — exact extrema to owned addresses,
   identity/lifecycle and constant-normal held-support continuation proofs.
3. src/phymeo/validation/cloth_self_mapping.py — independent two-endpoint ideal
   Jacobian, actual aggregate, couple/work and rational multi-field enclosures.
4. src/phymeo/validation/cloth_self_response.py — phase ownership, complete
   final laws/CCD binding and original-input physical ledger.
5. src/phymeo/solvers/cloth_self_contact.py — public configuration/history and
   SceneProgram facade.
6. src/phymeo/solvers/_cloth_self_contact_workspace.py — common response,
   bounded held/terminal iteration, complete acceptance and lifecycle.
7. src/phymeo/solvers/_cloth_self_contact_replay.py — public replay/load-plan/
   terminal receipts and selected-pre binding, no private native mutation.
8. tests/test_cloth_self_response.py — full W/phase law/pressure/capacity tests.
9. tests/test_cloth_self_anchors.py — exact material ownership/extrema/ties.
10. tests/test_cloth_self_mapping.py — independent map/couple/work and corruptions.
11. tests/test_cloth_self_physics.py — complete A/B/C controls and rebuilds.
12. tests/test_cloth_self_transaction.py — late failure, full rollback and retry.
13. tests/test_cloth_self_response_tyx.py — actual selected backend/state receipts.
14. docs/development/p65-s2-response-report.md — original/final evidence and limits.

This is an implementation proposal only; none of those files is authorized or
created by this task. Existing S1/S2-LG sources/tests/exports remain untouched.
No new dependency, plugin, backend pin or workflow change is proposed.

Parent-exclusive new generic seam:
`src/phymeo/runtime/point_participant_program.py` and
`tests/test_point_participant_program.py`, for the robot-free participant scene
adapter. Alternatively parent can explicitly extend ClothProgram's public
terminal protocol; that would be an old shared edit requiring its own narrow
consumer checks. The self-contact lane must not make that choice silently.

Parent-exclusive existing seams if the combined stage requires changes:
`runtime/program.py`, `runtime/terminal_impulse.py`, `runtime/load_replay.py`,
`runtime/scene_stepper.py`, `coupling/participant.py`,
`backends/tyx_participants.py`, `solvers/articulated_coupling.py`,
`validation/terminal_impulse.py`, `solvers/_material_contact_balance.py`, and
`solvers/initial_robot_material_contact.py`. Parent also owns typed composite/v2
material dispatch and the future coordinator combining self and robot rows over
one cloth state. The B2 owner retains its separately granted represented-kernel/
path files. No blanket shared allowlist is granted by naming these seams.

A later serial validation plan should select the complete six new lane test
files plus the new adapter tests; exact shared consumers only for files actually
changed. Do not repeat LG 69 or S1's unchanged suites merely for a new report.
Freeze original source/evidence first; preserve first RED/GREEN and each raw exit.
Host comes before actual interpret/CUDA and one bounded cpu_cpp control. The
physical A/B matrix, all h values, both independent rebuilds and corruption/
rollback controls may not be replaced by a smaller success case. Unsupported
backend/domain outcomes remain explicit, not skipped tests relabeled acceptance.
No numerical pass, native performance, TYX003 closure or final P6.5 stage review
is implied by this design task.

## 11. Parent decisions and remaining scope

Parent must freeze: A versus the separately named B law; the public robot-free
base adapter and exclusive request/application route; the finite unique-ray/
constant-normal charged domain versus a larger proved cone/continuation model;
phase capacity/pressure selection; exact history/address encoding; candidate
budgets/fixtures; and typed material/representation integration ownership.
The terminal stick/slip window is a real contact response control, but it does
not replace any later requirement for sustained friction hold/slip with rotating
loaded cloth. Such a control requires the additional supported law, not a
mislabeling of this impact result.

S2-F remains a separate required model decision. Ordinary face-normal prisms
adjacent across a material hinge can overlap locally even under a lawful fold.
S1's any-shared-vertex exclusion is not full folded-cloth coverage; blanket
incidence exclusion and rejecting every ordinary fold are not solutions.
Conforming shell/director geometry, its full Jacobian, incident-material/far-end
crossing proof, and external robot contact for that model remain outstanding.
Neither this nonincident response proposal nor accepted S2-LG proves global
self-collision safety, general-tree energy, arbitrary contact convergence or
complete P6.5.

Proposal remains **PROPOSAL_NOT_FROZEN**. All implementation and executable
validation require parent adjudication and a later explicit grant.
