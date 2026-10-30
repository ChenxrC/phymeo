# P6.5 S2 self-contact proposal: pair geometry and the material-incidence decision

Status: **PROPOSAL, NOT FROZEN**. This continuation performed only source reads,
Git inspection and handwritten derivation. No project imports, tests, numerical
probes, lint/formatting, build, installation, profiling, remote CI or helpers ran.
This document is the sole authorized change.

Investigated integration: `712fdadcaff83c661f769372aee85f074b2f8830`,
`C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo`.
Proposal branch: `codex/p65-self-contact-module`, following report commit
`5056ce8c0dce04e17951fe80fc9e8bf41f99d68c`. S1's implementation and its preserved
80/80 focused evidence are not repeated or reinterpreted here.

## 1. Two separately scoped deliverables

**S2-L, buildable primitive:** exact static nonincident prism/prism geometry,
symmetric supports/clipping and two endpoint material maps, conservative
continuous certification, and common-load contact/friction assembly over one
base. The arithmetic and public data contracts below are concrete enough for a
parent-approved implementation plan. It can proceed without pretending the
incident-material problem is solved. Parent must still freeze tie/friction
semantics and numeric/resource acceptance limits before coding/execution.

**S2-F, full folded cloth:** must additionally settle the physical material body
at shared vertices/edges and certify lawful local folds versus remote overlap.
Legacy face-normal prisms overlap near an ordinary bent hinge, while their
through-thickness maps do not agree on the shared edge. Blanket incident-pair
exclusion misses remote crossings; blanket rejection of ordinary incident
prism overlap prevents folding. Neither is a full solution. A new conforming
shell-cell model and a rigorous legacy material-incidence allowance are compared
below. This decision is unresolved and must remain a visible prerequisite of
full folded-cloth and combined B2/S2/G2 acceptance.

| Parent decision | Proposed direction / unresolved choice |
| --- | --- |
| D1: material body | Explicit NEW conforming shell-cell type if no rigorous legacy incidence allowance is supplied; retain `TrianglePrismSurface` unchanged. Director law, thickness field and fold domain remain decisions |
| D2: arithmetic | Rational SAT axes plus reduced two-root exact static predicates/clipping; rational interval time proofs. Approve precision/work budgets |
| D3: corners and ties | Canonical unordered pair/feature selection with all exact tie evidence; no patch priority. Freeze single-axis support restrictions versus a normal-cone friction law |
| D4: physical path | Explicit affine nodal substep path, with actual rotating normals/directors. If B2 declares another physical path, certify its every segment instead |
| D5: shared program | One physical particle state, aggregate held replay and terminal batch, composite B2 path/energy accounting; parent owns old shared seams |
| D6: acceptance | Freeze independent scenarios, h-halving/duration/mesh/failure gates and physical thresholds before results; no arbitrary convergence or general-tree energy claim |

No choice in this proposal is represented as already approved.

## 2. Current interfaces actually inspected

Paths are relative to `src/phymeo/` in the inspected integration snapshot.

| Current API/path | Relevant actual boundary |
| --- | --- |
| `constraints/triangle_prism_surface.py:TrianglePrismSurface` | Owned key/domain/three nodes and positive scalar rho; body is a face-normal triangular prism |
| `geometry/_prism_exact.py:Quadratic` | `a+b*sqrt(c)` with exact signs; `pair` rejects different nonzero quadratic fields |
| `geometry/prism_patch_manifold.py:triangle_prism_patch_manifold` | Complete prism/finite-patch SAT and clipping with rational coordinate-prismatic robot FK. Exact maximal ties favor `patch_face`; `prefer_initial_axis` preserves that priority |
| `geometry/_prism_clip.py:clip_support, material` | Projected convex feature intersection and prism `(b0,b1,b2,s)` recovery, hard-wired to single-Q arithmetic |
| `geometry/prism_patch_sweep.py:triangle_prism_patch_sweep` | Affine nodal/prismatic paths, exact samples, area polynomial and normal-curvature bounds, fixed rational-direction certificates; `certified/violation/unresolved` |
| `geometry/_patch_interval.py` | Rational outward interval operations; exact internal Fractions retained; division through zero rejects |
| `solvers/_contact_replay_audit.py:sqrt_bounds` | Fixed 256-bit rational square-root enclosure; not an exact equality/tie decision by itself |
| `coupling/offset_port.py:TriangleOffsetPointPort` | Full normal-derivative fixed-configuration map over one backend-owned base; host FP64 geometry |
| `coupling/cloth_self_contact.py:make_self_contact_operator` | S1 combines all offset attachments once with a same-domain `MultiDomainPointOperator`; `node_impulses` is a requested-load receipt, not application proof |
| `solvers/_thick_contact_geometry.py:MaterialAnchor, convert_material` | One prism material endpoint, ideal address/conversion error and 52-bit dyadic barycentric conversion; current float-material token is not a new geometric topology predicate |
| `solvers/dynamic_robot_thick_contact.py:DynamicRobotThickContactProgram` | Robot-specific two-phase program; all-dynamic FP64 particles, distance-only material contract, bounded history/anchor/rebuild/CCD resources |
| `validation/thick_contact_mapping.py:ThickContactMappingLedger` | Independently reconstructed maps and actual force/moment/power checks; several accumulators are single-Q, unsuitable unchanged for unrelated normal fields |
| Same file: `phase_capacity, phase_audit` | Each actual phase projects onto its own normal; cone/dissipation/positive-work checks; initially open held tangent capacity is zero |
| `constraints/friction.py:solve_friction_disk` | One fixed-normal SPD 2D tangent problem, not a full coupled graph solve |
| `runtime/load_replay.py:SceneLoadReplay.evaluate` | Replays from one complete initial state/history; returns effective inputs, represented integrated loads, duration and owned result |
| `runtime/terminal_impulse.py:PointVelocityImpulses, TerminalImpulseProgram` | Owned world-impulse requests; grouped live terminal operation and independent terminal ledger after the base velocity stage |
| `runtime/program.py:SubstepWorkspace` | Predict, position/velocity sweeps and metrics, result/diagnostics/close; no external commit |
| `runtime/scene_stepper.py:SceneStepper` | Validation/verifier/diagnostics/close precede publication; later-substep failure returns original whole-step state/history |
| `validation/material_path.py:MaterialPathLedger` | Explicitly distance-only equation/path audit; not a general material energy certificate |
| `solvers/_material_contact_balance.py:MaterialContactBalance` | Existing frozen-M0 robot energy scope and particle angular-discretization budget; not general articulated-tree energy |
| `core/scene.py:HistoryField`, `runtime/program.py:own_diagnostics` | History only FP32/FP64/int32/int64/bool; exact diagnostics need supported canonical tokens, not arbitrary algebraic objects |

Existing workspaces are implementation details, not public extension hooks.
Parent owns any extraction of reusable seams. No new module may mutate private
runtime state or nest two independently advanced cloth states.

## 3. S2-L exact static prism geometry

For each rational snapshot define

```text
e=x1-x0, h=x2-x0, c=e cross h, C=c dot c > 0,
N=c/sqrt(C), P=conv{x_i +/- rho*N : i=0,1,2}.
```

Finite FP64 inputs denote their exact binary rational values; internal rational
times never round through float. Thickness is positive metres. This is the
legacy primal volume, with no sphere skin, epsilon-expanded body or hidden
smoothing. Nonfinite/unrepresentable inputs reject. Exact nondegeneracy and the
existing offset-port scaled-area admission (`1e-10`) are separate requirements;
do not tune or weaken the latter to admit a test.

### Rational directions, despite two irrational normals

Every physical edge is parallel to one of `e,h,h-e,c`; c may replace N for edge
direction. Face normals are c and the three `edge cross c` directions. Complete
prism/prism SAT therefore uses both bodies' face normals and all cross products
of their four edge-direction classes. **Every axis is rational at an exact
rational snapshot.** Remove only exact zero vectors; divide by the first nonzero
component to deduplicate an unoriented rational axis, then consider both signs.

```text
g_d = min_A(d dot a)-max_B(d dot b), q_d=d dot d>0,
g_star=max_d g_d/sqrt(q_d).
```

The support gap still involves both radicals. `g_star` is the signed SAT gap/
penetration measure, not a claimed Euclidean closest distance for separated
bodies. Static classification uses all axes; samples alone are not CCD.

### Recommended pair-local field and rigorous alternatives

Use a new field `K=Q(sqrt(C_A),sqrt(C_B))` with up to four rational coefficients:

```text
z=a+b*sqrt(C_A)+c*sqrt(C_B)+d*sqrt(C_A*C_B).
```

Reduce the degree first. Rational-square tests of the numerator/denominator of
C_A, C_B and C_A/C_B detect rational or common-root cases by integer square roots;
full integer factorization is unnecessary. If both are nonsquares and their
ratio is nonsquare, the two square classes are independent. Only in the reduced
basis can coefficient zero mean exact algebraic zero. Arithmetic/equality/hash
are context-aware, never float comparisons of roots or coordinates.

Multiplication uses u*u=C_A and v*v=C_B; inverse uses exact conjugation or the
rational multiplication matrix after proving a nonzero denominator. Sign can
use algebraic comparisons, or exact-zero detection followed by successively
refined rational root intervals until zero is excluded. A bounded-work failure
returns `unresolved`. A finite enclosure overlapping zero is not equality.

Compare normalized gaps by signs, then the sign of
`g_a*g_a*q_b-g_b*g_b*q_a`, reversing for negative gaps. This avoids adding a third
radical to K. Keep rational d and q_d as proof data; a normalized stored FP64
normal receives a separate enclosure/representation audit. Clipping uses d and
q_d, not a normalized float normal.

- Recommended: exact reduced K for static supports/ties/clipping, certified
  rational intervals for temporal proofs and global audits.
- Interval-only predicates are sound when separated bounds decide every branch,
  but exact contact/coplanarity/ties require symbolic fallback or explicit
  unresolved support. No epsilon/equality shortcut can close that gap.
- General algebraic-number machinery could cover more fields, but adds a larger
  dependency/proof scope. No dependency change is proposed without approval.

Each rational time has its own field. A whole mesh has many independent normal
roots; do not combine unrelated pairs or time samples in one K or the old Q.
Independent per-row ideal map enclosures can be summed with rational intervals
for global nodal/moment/power audits.

## 4. Symmetric support selection and owned material anchors

Canonicalize the unordered pair by stable material face IDs, not caller side.
Unique sorted node-ID triples plus domain can identify faces, with oriented
cyclic order in the configuration fingerprint; explicit int64 IDs are another
parent option. Row reorder must preserve keyed receipts or reject incompatible
history remapping. No object-name, coordinate or time rule chooses a law.

Enumerate all exact support vertices. Intersect both projected convex features
symmetrically in an exact coordinate-drop chart, covering point/segment/polygon
and collinear overlap. Recover two exact 3D support points and two `(b0,b1,b2,s)`
addresses; prove b>=0, sum(b)=1, -rho<=s<=rho and reconstruction in the original
prisms. Contact with failed clipping is unresolved, never separation.

Retain every exact maximal-gap tie. A proposed primary-axis rule uses canonical
unordered feature provenance, then rational axis ordering. A prior axis can win
only within the proved exact tie set and with the same rule under A/B exchange.
Swapping operands swaps endpoints and negates the oriented normal. There is no
patch-face preference. In a genuinely ambiguous cone, a world-axis tie breaker
may break rotational covariance; report the ambiguity instead of claiming one
unique physical normal.

D3 must settle corner friction. Single-primary-axis support is limited to cases
where omitted active directions/extrema also pass the complete final audit.
Discover/solve required extra constraints or reject. A normal-cone formulation
needs an explicit definition of each anchor's normal budget and tangent law;
copying capacity to every tied axis multiplies friction and is forbidden.
Sorting axes or solving one local disk proves neither contact coverage nor a
unique friction law.

Freeze BOTH actual material endpoints within each phase. Proposed legacy
representation uses three int64 dyadic counts summing to 2**52 and a finite
signed offset, with exact FP64 bit tokens (signed zero canonicalized). Record
ideal addresses and independent world conversion-error bounds for both ends.
Allocate their total error within the explicit geometry budget; do not change
the ideal primal geometry. Tokens identify owned material addresses; equality
of rounded world points never establishes geometric topology.

Anchor keys combine pair ID, integer feature provenance, material tokens and
schema version. Keep owned anchors rather than attaching them to the nearest
moving clip vertex. New anchors are births; retired anchors lose capacity.
Never copy held load/capacity to a new extreme. A loaded anchor leaving finite
support requires the declared rebuild/failure path, not silent reprojection.
Independently retain all ideal manifold extrema so conversion/deduplication
cannot erase required constraints.

## 5. S2-F incident material: explicit alternatives and blockers

### What existing combinatorics proves, and what it does not

Common node IDs/oriented edges identify shared middle-surface material simplices.
Sparse barycentric weights on those IDs can prove the same s=0 material address.
A shared vertex/edge does not exempt remote feature collisions. S1's documented
`exclude_shared_vertex` stays unchanged, but new S2 enumeration must supplement
all incident pairs or supply an independently certified omission disposition.

At s!=0 the two maps on an edge are `x_edge+s*N_A` and `x_edge+s*N_B`; different
normals give different points. Ordinary bending produces finite near-hinge
volume overlap. Excluding only the exactly shared middle edge does not remove
it. Rejecting it prevents ordinary folding. Exempting the whole incident pair
misses remote crossings. Since an intersection of convex prisms is convex,
"the intersection component connected to the hinge" cannot separate an ordinary
wedge from a far-end overlap in the same component.

### Alternative L: rigorous legacy material-incidence allowance

Preserve the prisms, but explicitly define a reference-material overlap/gluing
relation over pairs of material addresses. Prove throughout the candidate path
that every exempt overlap belongs to that relation and that all other material
address pairs are covered by contact/CCD. This requires a reference chart,
transported labels and local injectivity/support conditions. A distance collar,
k-ring, angle threshold or tolerance chosen for an example is not a material
relation. Initial/rest overlap alone does not prove a later overlap is local.

The current face-normal prism configuration supplies no such relation, and this
proposal does not invent one. A future `IncidentOverlapCertificate` would carry
reference-model fingerprint, pair IDs, exact shared-address relation, covered
parameter/time cells and proof kind. Constructing the record cannot grant
acceptance; the verifier must check its claims. Until a valid relation exists,
nontrivial incident legacy overlap remains unresolved. That is a truthful partial
support boundary, not the full folded-cloth implementation.

### Alternative C: NEW conforming shell-cell representation

Keep all legacy `TrianglePrismSurface` behavior unchanged. Define shared global
top/bottom material vertices `z_i^+/-=x_i +/- a_i`, with middle positions x_i
and shared offset vectors a_i in metres. Every incident cell uses the same
vertices. Per-vertex thickness is explicit new data, not a silent average of
unequal legacy per-face thicknesses.

Director choices are different physical laws:

1. **Derived director:** for example `a_i=rho_i*S_i/|S_i|`, where S_i is an
   explicitly specified oriented sum of incident raw areas. This is a proposed
   new geometric law, not evidence that averaged normals are physically valid.
   Prove/reject vanishing or ill-conditioned S_i, orientation, projected thickness
   and cell/star validity. The Jacobian includes every node in the incident
   face star; the three-node normal-offset port cannot represent it. Scatter
   the complete loads once into the same particle base.
2. **Independent director:** needs actual orientation variables, angular
   velocity/inertia, material energy and native equations. Copying a predicted
   director kinematically creates unaccounted work. This expands B2/state-family
   scope and cannot be hidden in the first implementation.

The bilinear wedge `F_f(b,s)=sum_i b_i*(x_i+s*a_i)`, b in the reference simplex,
s in [-1,1], shares its side map with neighbors. Its side quadrilateral is
usually warped, however. The convex hull of its six endpoints is not generally
that material volume; replacing one by the other changes the law and map.

**Conditional convex-cell proof:** if two convex cells have the exact same
planar common facet S, and every other A vertex lies strictly on one side of
its plane while every other B vertex lies strictly on the other, their
intersection is exactly S. Check nonzero facet area, orientation, volume,
coplanarity and all support inequalities. Shared endpoints or averaged normals
do not establish these hypotheses. Two convex cells with disjoint interiors
cannot share only a genuinely warped nonplanar side sheet: their intersection
must itself be convex. Convex six-vertex cells therefore require restrictive
planar-side/support conditions.

**Conforming simplicial option:** fix a material product triangulation of each
reference triangle x [-1,1] into tetrahedra, choosing side diagonals from one
global vertex-ID order. Adjacent cells share exactly the same triangular side
facets. This is a new piecewise-affine material body, not the bilinear wedge or
legacy normal prism. Certify signed tetrahedral determinants, boundary embedding,
shared-facet orientation and cell/star injectivity. Positive determinants alone
do not prove a globally embedded mesh. Do not fill a nonconvex cell by its hull.

For convex primitives, compute exact intersection and verify it is contained
in the declared shared material simplex. A remote intersection vertex outside
that simplex or positive-volume overlap is not exempt, even if connected to
the shared side. Internal tetrahedral subdivision facets are not friction
interfaces. For face neighbors the opposite-halfspace condition can provide
whole-interval incidence proof. Edge/vertex neighbors require tangent-cone/star
or verified material-parameter subdivision proofs. A positive ordinary gap
cannot hold at a permanent shared vertex; factor exact shared-simplex identities
or certify the parameter domains instead of adding a spatial exclusion radius.
Any uncovered material/time region remains unresolved. Disconnected material
links at a vertex need explicit junction semantics or rejection; S1's
edge-manifold check alone does not settle that issue.

**External robot contact also changes:** implement shell-cell/patch SAT,
external-boundary clipping, CCD, material coordinates and full director-map
Jacobian. A shared internal facet is not an external contact surface. Derived
normalized directors introduce multiple roots per cell, beyond the two-root
legacy field; use certified intervals with exact degeneracy/tie fallback or a
separately approved algebraic scope. Storing rational offsets makes static
geometry rational only if those offsets really are physical state/configuration
under the approved director law; it does not provide their missing dynamics.
The legacy robot-prism routine and normal-offset port cannot be relabeled.

Recommendation for D1: consider a conforming piecewise-affine cell with a precise
derived-director law as the concrete NEW-model candidate. Freeze it only after
admissible folding, kinematic derivatives and external contact are specified.
Keep L only if a real material-gluing/allowance proof is provided. Do not wait
for this decision to obscure the independent S2-L arithmetic primitive, and do
not claim either alternative is already proven by this document.

## 6. Continuous proof for both moving normals

For the declared affine nodal t in [0,1], each raw area is quadratic:
`c_f(t)=c0+c1*t+c2*t*t`. On rational interval I choose d0=c_f(mid). If the
exact polynomial minimum of `c_f(t) dot d0` on I is positive, then

```text
r_min = min_I(c_f dot d0)/upper_sqrt(d0 dot d0) > 0
```

is a lower bound for |c_f| throughout I. This proves interval nondegeneracy;
three samples do not. An inconclusive projection bound requests subdivision,
not a declaration of degeneracy. Exact zero-area witnesses reject/unresolve.
Representable mapping and its scaled-area admission are additional obligations.

Differentiating N=c/|c| gives

```text
N'=(I-N*N.T)c'/|c|,
N''=(I-N*N.T)c''/|c| - N*|(I-N*N.T)c'|^2/|c|^2
    -2*(N dot c')*(I-N*N.T)c'/|c|^2,
|N''| <= |c''|/r_min + 3*max_I|c'|^2/r_min^2 = M_f.
```

c' is affine, so the norm maximum is bounded by endpoint maxima; c'' is
constant. Use rational outward square-root bounds. For a fixed rational nonzero
d, let lower_g_d(a), lower_g_d(b) be exact/outward endpoint support-gap bounds.
Every vertex-pair projected difference has second derivative bounded by
`upper_norm(d)*(rho_A*M_A+rho_B*M_B)`. Thus

```text
L_I=min(lower_g_d(a),lower_g_d(b))
    -upper_norm(d)*(rho_A*M_A+rho_B*M_B)*(b-a)^2/8
```

bounds the entire support gap below, even across min/max support switches:
the interpolation inequality holds for every vertex pair. Accept an interval
only if `L_I+penetration_budget_m*lower_norm(d)>=0`, with both area proofs.
Normalize negative L_I by lower_norm(d), nonnegative L_I by upper_norm(d) for
conservative output. Approximate sampled axes may propose d, but the recorded
rational fixed-direction inequality is the proof.

A rational sample whose complete SAT upper gap is below the negative budget
is a penetration witness. Otherwise subdivide or return unresolved on budget,
precision, depth, degeneracy or unrepresentability. Certified output must own
an ordered gap-free [0,1] cover. Unresolved output retains the complete pending
partition, including siblings. Fingerprints bind topology, actual endpoints,
thickness, path/law version, history, budgets and limits. Changed candidates
invalidate evidence; unknown is never safe.

Every broad-phase omission needs exact swept-box separation or the selected
incident-material proof. Check face validity independently of candidate culling
so an isolated degenerate face cannot disappear behind an empty pair list.
S1 may supply nonincident candidates only; its excluded pairs require explicit
supplementation, with no changed S1 claim.

For derived-director shells, S_i(t) is also a known quadratic area sum and can
receive its own normalization bounds. Enclose each actual top/bottom vertex
curve, then separately certify cell noninversion/support and pair separation/
shared-simplex containment. The two-prism formula alone is not that shell proof.
A non-affine physical path needs its own enclosures. Finite proof budgets are
allowed to reject; arbitrary convergence is not promised.

## 7. One common-load response, phase friction, and transaction

At each phase stack all self and external rows over one base per physical
domain. For a self row `D_k=J_A,k-J_B,k`, including shared node stencils. For
constant particle M retain the full equations

```text
ell=D.T*p, W=D*M^-1*D.T,
u_after=u_before+W*p,
Delta_K_terminal=p.T*u_before+0.5*p.T*W*p.
```

The self block includes AA+BB-AB-BA; preserve every cross-row/shared-node and
shared-robot term. S1 provides the legacy one-port attachment route. A derived
shell needs a new sparse-Jacobian public port over that same base and an
independent mapping ledger, not a second response/apply implementation.

Parent common-workspace sequence:

1. Snapshot all domains/history once; validate initial geometry/material
   legality; construct owned phase anchors.
2. Aggregate all held world impulses through phase-0 maps and every domain's
   nodal loads once. Public `SceneLoadReplay` adds integral/h to force channels
   (N*s divided by s gives N); robot requests use the supported public path.
3. Bounded rebuilds and line searches replay from the same initial state. The
   instantaneous terminal W is not a certified derivative of nonlinear
   distance+bending held replay; it is only an update proposal metric. An
   exhausted/stalled solve rejects; a selected "best" trial still needs every
   independent certificate.
4. Audit the complete selected held pre-state, effective inputs, ordered
   material traces, multipliers, contact requests, energy/moment and common CCD.
5. Reconstruct both terminal maps and issue one grouped `PointVelocityImpulses`
   batch per domain through the public terminal capability. Multiple private
   replay trials are allowed; two physical state advances are not.
6. Recheck actual/ideal active features, material residuals, phase laws and
   fingerprints; publish state and history together only after close succeeds.
   Late-substep/close failure rolls back the complete original step/history.

Preserve the current two-phase convention as the proposed first law, with
explicit pair mu>=0. Initially closed anchors get held tangent capacity
`mu*p_held_normal`; initially open anchors get zero held tangent. Terminal
capacity is `mu*p_terminal_normal` for initially closed anchors and
`mu*(p_held_normal+p_terminal_normal)` for newly closing ones. Project each
actual phase on its own normal BEFORE addition. Do not give a new anchor its
predecessor's capacity or reuse one normal budget on tied directions.

For each actual phase/candidate audit nonnegative normal impulse, normal
complementarity with the declared gap/h or relative-speed law, and

```text
|t|<=capacity,
t dot u_t + capacity*|u_t|=0  (maximum dissipation within frozen audit bounds),
t dot u_t<=0.
```

A local friction disk can propose a row update; it does not certify the common
candidate. Do not add diagonal padding to singular rows. Prove exact material
identity or reject unsupported/infeasible zero-response constraints.

Independent map reconstruction must establish requested-versus-represented
loads and actual state response, not trust a receipt alone. For self rows:

```text
sum_i ell_i=0,
sum_i x_i cross ell_i=sum_k (x_A,k-x_B,k) cross p_k,
sum_i v_i dot ell_i=sum_k p_k dot (J_A,k*v-J_B,k*v).
```

Separated endpoint forces can have nonzero moment; a fictitious zero target is
invalid. Bound representation/mapping errors and the actual paired-point couple
under explicit physical budgets. Sum many normal fields by independent rational
intervals. Record both requested and actual represented integrated loads.

Full energy includes actual particle kinetic change, distance and B2 bending
potential changes, minus original external work, without double-counting contact
work. B2 must supply bounded angle/energy and the ordered composite path:
`Fraction(float(atan2))` is not exact ideal bending energy. Preserve NUM005
angular-discretization accounting. Keep old distance-ledger type gates until an
independently verified composite protocol exists. Existing robot claims retain
frozen-M0/limited support, not general-tree energy. A geometry pass alone never
certifies full-step force, work or material acceptance.

## 8. Proposed public contracts and support domain

Names are provisional. Exposed data are owned/read-only. Reject invalid shape,
dtype, indices and nonfinite input before mutation. IDs int64; positions,
velocities and world impulses FP64. Units remain m, s, kg, N*s, J and N*m*s.
Numerical budgets are explicit acceptance data, never expanded primal geometry.

| Proposed API/data | Shape and responsibility |
| --- | --- |
| `PrismPairSpec(topology, face_ids, friction_by_pair)` | Canonical stable pair IDs and explicit dimensionless mu; no hidden incidence exclusion |
| `prism_prism_manifold(x_a,x_b,*,half_thickness_m,proof_limits)` | Two FP64 `(3,3)` node arrays and FP64 thickness `(2,)`; exact `separated/touching/penetrating/unresolved`, complete supports/axes, gap enclosures and both material endpoints; any response-activation budget is separate from exact classification |
| `PrismPairPoint` | World endpoints `(2,3)`, exact point tokens, ideal material `(2,4)`, integer feature provenance, conversion-error bounds in m |
| `prism_prism_sweep(path_a,path_b,*,half_thickness_m,penetration_budget_m,proof_limits)` | Each endpoint pair FP64 `(2,3,3)`, declared affine nodal law and h>0; rational time partition, both area proofs, directions/witness; `certified/violation/unresolved` |
| `SelfContactAnchorBatch` | Face-pair IDs `(k,2)`, actual material `(k,2,4)`, dyadic counts `(k,2,3)`, phase flags `(k,)`, ideal/conversion tokens; flatten into `2k` existing offset rows and `(k,2)` row pairs |
| `SelfContactMappingLedger.require(...)` | Independently reconstructed maps, signed requested/represented `(n,3)` loads, full response and actual-state/moment/power/energy audit; no state updater |
| `ClothSelfContactStage` | Parent common program extension contributing rows/evidence through public state/material/replay interfaces; no nested separate time stepper |
| `ConformingShellTopology` (D1 conditional) | New reference cells, global top/bottom IDs, exact side triangulation, vertex half thickness `(n,)`, explicit director rule/version and support limits; no implicit legacy conversion |
| `ShellDirectorPointPort` (conditional) | Owned cell/tetra material coordinates; sparse Jacobian blocks `(nnz,3,3)` plus int64 row pointers/node IDs; world points `(r,3)` over one particle base |
| `shell_cell_pair_manifold/sweep`, `shell_cell_patch_manifold/sweep` (conditional) | New cell-specific geometry, external boundary and incident-material proofs; cannot call legacy prism code and relabel results |

Provisional fixed-capacity history: `pair_ids (p,2) int64`,
`pair_active (p,) bool`, `anchor_active (p,r) bool`,
`material (p,r,2,4) FP64`, `material_counts (p,r,2,3) int64`,
integer feature/address tokens, phase/birth flags and
`phase_world (p,r,2,3) FP64`. Inactive slots must be clean. Offset FP64 bit tokens
can use signed-int64 storage; uint64 is not an existing HistoryField dtype.
Overflow/exhausted capacity rejects atomically. A shell's tetrahedral material
coordinates need their own schema, not misleading legacy `(b,s)` fields.

Exact rational/field diagnostics serialize canonical numerator/denominator and
context tokens through supported builtin int/tuple/string data. History may not
hold opaque object state. Fingerprints include topology, law, path endpoints,
material addresses, thickness, budgets, limits and selected candidate. A record's
constructor is not a verifier; independent functions recheck its assertions.

Initial S2-L solver domain: consistently wound nonincident legacy prism pairs
in one all-dynamic FP64 particle domain, with admissible bounded geometry,
nonnegative explicit friction and the approved common material protocol.
Disconnected sheets in one domain are valid. Full-mesh support requires D1's
incident coverage. Fixed-node reactions, remeshing/topology change, torn seams,
nonorientable meshes and independent-director dynamics need separate contracts.
A conforming model also requires a valid material vertex link/star, not just an
edge-manifold mesh. Later interpret/CUDA checks must identify actual native
state execution while labeling geometry as host; no device-geometry claim.

## 9. Independent scenarios and acceptance equations to freeze

These are handwritten, unexecuted oracle proposals; no results or thresholds
were tuned. Freeze both physical cases B/C, the arithmetic/CCD controls A, and
lawful-fold plus remote-incidence controls D before acceptance.

### A. Two independent radicals and swept controls

A vertices `(0,0,0),(1,0,0),(0,1,1)` give `c_A=(0,-1,1), C_A=2`.
B vertices `(0,0,H),(1,-1,H),(0,1,H-1)` give `c_B=(1,1,1), C_B=3`.
For rho_A=1/4, rho_B=1/8 and d=-e_z, the independent support expression is

```text
g_d=H-2-sqrt(2)/8-sqrt(3)/24.
```

At H=4 it is strictly positive, for example using sqrt(2)<3/2 and sqrt(3)<2.
Require this axis to certify separation and match the expression; do not assume
it is the maximum SAT axis. Swap operands, permute vertices, reverse complete
winding with corresponding signed material addresses, and apply exact signed
coordinate rotations/translations. Add common-root/dependent-field cases,
exact cancellation and dyadic near-zero comparisons.

Separate affine parallel-footprint paths have disjoint endpoint prisms but an
interior crossing. Add independently rotating normals and interior area collapse.
Require complete violation evidence or honest unresolved output for unsafe/
unproved paths, and full [0,1] coverage for certified paths. Dropping either
normal's curvature term must be caught by a separately frozen witness fixture.
Exact touching/coplanar clips test the algebraic zero path, not epsilon logic.

### B. Two free sheets collide within one particle domain

Two congruent XY triangles have disjoint node IDs, one base, uniform per-sheet
velocities and masses M_A=2 kg, M_B=5 kg, each split equally among three nodes.
Use rho_A=1/8 m, rho_B=1/16 m, relative closing speed V=1 m/s, no tangential
motion/external force or initial material stress. At cap contact, each of three
corner normal impulses is one third of the resultant. Normal-offset derivative
terms vanish for these normal loads since `N dot dN=0`.

```text
m_red=M_A*M_B/(M_A+M_B)=10/7 kg,
P_total=m_red*V=10/7 N*s,
v_A^+=v_B^+=conserved center-of-mass velocity,
Delta_K=-m_red*V^2/2=-5/7 J.
```

For an initially open gap 0<g0<h*V under the existing two-phase semi-implicit
held replay, require independently

```text
P_held=m_red*(V-g0/h), P_terminal=m_red*g0/h,
g_held_end=0, P_held+P_terminal=m_red*V.
```

Choose dyadic h and g0. This is a normal-only full-step oracle, not a claim of
scalar reduced-mass tangent response for thick surfaces. Check actual nodal
requests/velocities, global impulse/moment/energy, single aggregate terminal
application and unchanged original snapshots. Accepted CCD reaches contact
without passing through it.

### C. Closed self-contact balance and an impossible sticking control

Use two all-dynamic patches of the same domain, unequal positive masses,
coincident opposing caps and the complete three-corner manifold. Independently
write the unit-XY attachment blocks

```text
J_i=b_i*I+s*v_i*e_z.T,
v_0=(1,1,0), v_1=(-1,0,0), v_2=(0,-1,0).
```

Prescribe ORIGINAL nodal force inputs opposite the independently reconstructed
contact loads for compression F=2 N and shear T=1/2 N, distributed through the
full manifold. Include the offset-induced torque; centroid-only loading is a
different fixture. Do not prescribe the candidate state. For mu=1/2 a stationary
solution is admissible: resultant normal impulse Fh, tangent magnitude Th,
`T<=mu*F`, every nodal force/moment balanced, zero kinetic/material energy change
and zero original input work. Row multipliers can be nonunique; compare actual
loads, state and constraints instead of arbitrary multiplier distribution.

For the identical input with mu=1/8, stationary balance is impossible because
`T>mu*F` and `|sum tangent|<=sum mu*p_n=mu*F*h`. Require slip or honest unresolved
output, never false sticking. A later successful dynamic slip acceptance needs
its own frozen admissible small-step oracle; rejection alone does not pass that
gate. The full W determines acceleration, not a point-mass friction formula.

An auxiliary fixed centroid attachment with node masses M/3 has independently
known tangent eigenvalues

```text
k=1/M_A+1/M_B, R=rho_A^2/M_A+rho_B^2/M_B,
k_plus=k+9*R along (1,1,0), k_minus=k+3*R along (1,-1,0).
```

This checks thickness/cross-term friction updates and exact kinetic work. It
must not replace the complete-manifold physical scenario.

### D. Common material boundary, ordinary fold, and far-end overlap

A conforming flat pair with triangles `(0,0,0),(1,0,0),(0,1,0)` and
`(1,0,0),(0,0,0),(0,-1,0)`, all a_i=rho*e_z, shares the rectangle at y=0.
Exact opposite halfspaces prove intersection is only that common material side;
there must be no internal-boundary self-contact load.

Retain shared labels while moving a nonshared feature outside the declared
shared simplex. Intersection containment must detect the remote overlap even
when connected to the shared side. A simple shared-vertex primitive control uses
A `(0,0),(2,0),(0,2)` and B `(0,0),(1,1/2),(1/2,1)` in XY with equal positive
thickness. Both triangle areas are nonzero; B overlaps A away from the common
material vertical edge. Blanket shared-vertex exclusion is wrong. This pair
control is not by itself a valid global manifold mesh: add a reference-valid
connected star/fold and certify surrounding cells after D1 is chosen.

Also perturb directors to make a side nonplanar, invert a tetrahedron, create
star overlap with positive individual determinants, and produce an interior
remote crossing with legal endpoints. Each must fail its specific support/
incidence certificate. A lawful ordinary fold within the declared D1 domain
must SUCCEED as well; a scheme that rejects every fold fails the intended goal.
Its exact material embedding and force-balance oracle depend on the unresolved
physical model and cannot honestly be frozen now.

### Required common acceptance, not results

Parent freezes h/h2/h4, duration/repeats, mesh refinement, independent
perturbations, fixed physical budgets and expected failures before execution.
Require whole-path penetration/nondegeneracy/incidence coverage; normal
complementarity; cone/maximum dissipation; independently reconstructed actual
map force/moment/power; full terminal kinetic work; actual material/kinetic/
potential change minus external work; and complete state/history rollback after
late native/map/material/CCD/anchor/close failures. Transform tests respect exact
input representation. Native interpret/CUDA support requires later serial proof.
No module or pass count closes combined folded cloth and finite-grasp acceptance.

## 10. Exact proposed ownership and affected tests

Only this document is authorized now. Future filenames/owners below are a
proposal for approval, not permission to edit or execute them this turn.

**Self-contact lane, new S2-L source under `src/phymeo/`:**
`geometry/_prism_pair_exact.py`, `geometry/prism_prism_manifold.py`,
`geometry/prism_prism_sweep.py`, `geometry/self_contact_pairs.py`,
`constraints/self_contact_pairs.py`, `coupling/self_contact_anchors.py`,
`validation/self_contact_mapping.py`.
New tests: `tests/test_prism_pair_exact.py`,
`tests/test_prism_prism_manifold.py`, `tests/test_prism_prism_sweep.py`,
`tests/test_self_contact_pairs.py`, `tests/test_self_contact_anchors.py`,
`tests/test_self_contact_mapping.py`. Keep S1's public policy and tests intact.

**Conditional new conforming-shell source after D1:**
`constraints/conforming_shell.py`, `geometry/shell_cell_geometry.py`,
`geometry/shell_cell_sweep.py`, `geometry/shell_cell_patch.py`,
`coupling/shell_director_port.py`, `validation/shell_director_mapping.py`, with
`tests/test_conforming_shell.py`, `tests/test_shell_cell_geometry.py`,
`tests/test_shell_cell_sweep.py`, `tests/test_shell_cell_patch.py`,
`tests/test_shell_director_port.py`, `tests/test_shell_director_mapping.py`.
Own the reference atlas/director law here, not in rendering or scene scripts.

**Parent owns all existing shared seams:**
`runtime/program.py`, `runtime/load_replay.py`, `runtime/terminal_impulse.py`,
`runtime/scene_stepper.py`, any core scene/history schema changes, old material
and contact ledgers/workspaces, B2 composite path/energy/native participant order,
external robot contact composition, backend adapters/kernels and exports.
Prefer a new parent-owned `solvers/cloth_contact_program.py` and common workspace,
with `tests/test_cloth_contact_physics.py`,
`tests/test_cloth_contact_transaction.py`, `tests/test_cloth_contact_tyx.py`.
Do not remove old distance-ledger gates to claim composite support.

**Existing regression impact for later serial scheduling, not runs now:**
`tests/test_prism_exact.py` (different-Q rejection unchanged),
`tests/test_thick_patch_geometry.py` (legacy primal/patch priority unchanged),
`tests/test_thick_contact_mapping.py`, `tests/test_thick_robot_contact.py`,
`tests/test_thick_robot_transaction.py`, `tests/test_thick_robot_tyx.py`,
`tests/test_offset_port.py`, `tests/test_domain_point_graph.py`,
`tests/test_cloth_self_candidates.py`, `tests/test_cloth_self_operator.py`,
`tests/test_material_path.py`, `tests/test_material_ledger.py`,
`tests/test_terminal_impulse.py`, plus new B2/G2 focused/native/transaction cases.
Parent selects affected subsets after final seams are known; no old full-suite
rerun is proposed here.

## 11. Parent resolution before implementation

1. Sequence buildable S2-L independently, with exact support limits, two-root
   arithmetic, symmetric tie/friction semantics and fixed resource ceilings.
2. Choose an actual incident-material model. For a new shell, choose derived or
   independent directors, thickness/material interpolation, admissible folds,
   cell/star injectivity and remote-crossing certificates, plus robot-contact
   consequences. Do not freeze blanket rejection as folded-cloth support.
3. Freeze the physical substep trajectory and the common B2/contact ledger,
   material history, phase-map and terminal interfaces. Preserve one base and
   actual cross-term/force/moment/work accounting.
4. Freeze independent successful impact, high-mu balance, low-mu slip and lawful
   fold cases, physical thresholds and serial backend acceptance.

The missing incidence/shell proofs are explicit blockers of S2-F, not reasons
to keep the S2-L primitive vague. This proposal certifies neither a new shell
model nor general robot energy, promises no arbitrary convergence, and does not
close S2 or P6.5.
