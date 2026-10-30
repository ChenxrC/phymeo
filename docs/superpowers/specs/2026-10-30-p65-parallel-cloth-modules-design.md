# P6.5 parallel cloth modules: isolated code, serial verification

Status: first parallel module contracts frozen after the one new builder check and parent rulings, 2026-10-30. The human explicitly approved parallel module coding and serial tests because of resource limits. This changes scheduling, not physics or acceptance. Original checkout Task3 and its pending exact CI keep their frozen source. New modules live in isolated managed worktrees from 144656141959483ce577fc0a9e476d235f031305.

## Decision and architecture

Use three development lanes: bending material; same-cloth contact; finite gripper configuration and load-bearing integration. They share the existing particle state, physical units and public ports. The parent owns composition into a single substep, material/contact ledgers, common history and atomic commit/rollback.

Alternatives considered: fully sequential work wastes independent development time; three independent time steppers would duplicate state/forces and cannot certify a joint step. Chosen: independent production modules plus one integration owner and one global execution queue. The user's explicit code-first scheduling permits tests to be authored before/alongside code but executed later. Do not fabricate a historical TDD RED or report unexecuted tests as passing.

The first parallel checkpoints below are useful production building blocks, not completion of bending+self-collision+gripping. They are deliberately independently callable so later shared integration does not require rewriting their mathematics. Full native integration, complete self-contact narrow phase/CCD/response, folded cloth and full gripper load-bearing acceptance remain mandatory downstream gates.

## Ownership and execution

- Delivery checkout C:/Users/38347/Documents/ChatGPT/phymeo remains exclusively owned by thick_friction_implementer until its explicit release. No new module files, test collection, dependencies or commits there.
- Integration worktree C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo, branch codex/p65-parallel-modules, belongs to parent.
- Each coding agent creates/uses its own managed worktree at the frozen contract commit, owns only the allowlisted new files, and commits only there. No push, helpers, other-worktree edits or old shared source edits.
- Test execution, numerical probes, imports that execute physics, compilation/build, native backend checks, performance runs and wheel smoke require an explicit parent queue grant. NO background competing runs. Code reading/writing and Git operations may proceed concurrently.
- Currently queue owner is original Task3 CUDA89050; after it releases, parent Task4 packaging/exact CI closure remains current delivery priority. Module agents return code and written tests with status CODE_READY_TESTS_PENDING.
- Parent selects one immutable source per queue entry, fresh basetemp, raw exit, command/backend/pin manifest, and closes its resources before granting the next entry. Any changed source invalidates only affected evidence, with rationale.
- Keep all eight CPU CI cells, timeout30, test/example/required/fail-fast commands and dependency pins unchanged. No current CI rerun against an unchanged failed head.
- No new code merged/pushed into the delivery branch before current thick-friction cost repair passes its accurate CI gate. Parallel worktree coding is explicitly permitted before that gate.

## Shared contracts

Coordinates metres, time seconds, mass kg, world impulses N*s, generalized actuator effort N for prismatic and N*m for revolute. Every array is owned and read-only when exposed as configuration/evidence; trial state is private. Caller state is not silently repaired.

Existing thickness means the triangular prism conv(x_i +/- rho*N), with positive half thickness rho. Broad-phase bounds may conservatively enclose that volume; they do not change it to a sphere skin. New geometry must identify what it proves and never treat an unknown result as separation. The existing one-quadratic-field implementation cannot silently stand for two independently oriented prism normals.

One physical particle domain has one base participant/candidate. Self-contact A and B endpoints are distinct rows over that SAME base. Aggregate all signed J.T loads before one native apply; preserve cross terms. Internal force cancellation alone is insufficient: audit actual applied node requests, resulting moment and power. Separated endpoint forces have moment (xA-xB) cross p; do not falsely demand zero torque for arbitrary separated tangential impulses.

Parent integration must add a material protocol with explicitly ordered distance/bending projection traces. Existing MaterialPathLedger's distance-only checks stay unchanged until an independently verified composite ledger is present; do not merely remove isinstance gates. Each trace owns element identity, pre/post positions, multiplier changes and actual evaluation gradients. One declared order, full path continuity, actual kinetic/potential changes and NUM005 angular-discretization accounting are required. Stage modules may not advance time themselves.

Contact history uses stable pair/material identities; candidate/commit/rollback are owned by the unified program. Renderer and scenario code never writes physics state, pins cloth, welds endpoints, or selects laws by object name, coordinates or time.

## B1: signed-dihedral bending operator and trace

Allowlist:
- src/phymeo/constraints/bending.py
- src/phymeo/constraints/bending_trace.py
- tests/test_bending.py
- tests/test_bending_trace.py
- docs/development/p65-bending-module.md

Public interfaces:
- BendingConstraints(indices, rest_angle, compliance): indices int64 (m,4) rows (a,b,c,d) for oriented faces (a,b,c) and (b,a,d), four distinct nonnegative indices. rest_angle FP64 (m,) radians in (-pi,pi); compliance FP64 (m,) strictly positive, unit 1/(N*m). Owned immutable arrays, reject duplicates representing the same hinge regardless edge reversal.
- bending_angle_gradient(positions): FP64 (4,3) -> (theta: float, gradient: FP64 (4,3)). e=b-a; nL=(b-a) cross(c-a); nR=(a-b) cross(d-b). theta=atan2(e_hat dot(nL_hat cross nR_hat), nL_hat dot nR_hat). Analytic gradient of this exact convention; no finite differences in production.
- bending_energy(position, constraints) -> float Joules: sum ((theta-rest_angle)^2/(2*compliance)).
- project_bending(position, inverse_mass, constraints, alpha, multipliers) -> tuple[BendingProjectionTrace,...]. FP64 mutable caller-owned working arrays, alpha=compliance/h^2 already supplied; deterministic element order; dlambda=(-C-alpha*lambda)/(sum(w_i*|grad_i|^2)+alpha), C=theta-rest_angle; each dx_i=w_i*dlambda*grad_i. Lambda has unit kg*m^2. Return complete immutable traces for every element. Validate all inputs before mutation; compute in private temporaries and publish arrays only if the complete sweep succeeds.
- bending_residual(position, constraints, alpha, multipliers, reference_angle=1.0) -> (dimensionless max constitutive residual, max angular error rad); reference_angle finite positive.
- BendingProjectionTrace: element_index, node_indices, positions_before/after (4,3), gradient (4,3), theta_before/after, constraint_before/after, lambda_before/after, delta_lambda, alpha, denominator, branch_status. A returned trace has branch_status="smooth"; a branch failure publishes no sweep. Trace tuple order is the complete element order for that sweep; integration adds the ordered sweep index. No fabricated energy/torque acceptance flag.

Supported smooth branch: abs(theta-rest_angle)<pi, theta strictly inside (-pi,pi). Evaluate/check branch at every recorded pre/post state; reject crossing, exact antipodal/branch-cut state, degenerate/nonrepresentable inputs. Scaled face cross magnitude must exceed 1e-10 using face edge scale (same explicit dimensionless admissibility as existing offset port); do not tune per example. No atan2 wrapping of C, no rest-angle reset and no silent fallback. Hard zero-compliance bending is outside B1 and rejects.

Independent tests to AUTHOR (execute only with serial grant):
1. Flat hinge a=(0,0,0), b=(1,0,0), c=(0,1,0), d=(0,-1,0), rest0: theta0/energy0/zero correction.
2. d=(0,0,-1) gives theta+pi/2 by stated convention; compliance2 gives energy pi^2/16. Swapped complete face convention transforms sign consistently. Rigid translation/rotation and scale preserve theta; gradients transform with inverse scale. Independent centered differences only in tests; translation sum grad=0 and sum x cross grad=0.
3. Two adjacent hinges sharing nodes: independently accumulated path and finite difference energy gradient. Late second-hinge rejection leaves both positions and multipliers unchanged; trace reconstruction matches every applied update. Positive unequal masses, all-fixed soft element, invalid shape/dtype/indices/compliance, branch crossing, extreme and degenerate inputs.
These are independent non-handoff hinge and multi-hinge sheet scenarios. They do not yet certify native bending or cloth/contact integration.

## S1: topology, conservative swept candidate pairs, same-domain attachments

Allowlist:
- src/phymeo/constraints/cloth_self_contact.py
- src/phymeo/geometry/cloth_self_candidates.py
- src/phymeo/coupling/cloth_self_contact.py
- tests/test_cloth_self_candidates.py
- tests/test_cloth_self_operator.py
- docs/development/p65-self-contact-module.md

Interfaces:
- ClothSelfContactTopology(domain, triangles, half_thickness_m): owned integer triangles (f,3), distinct vertices per face, no duplicate unoriented face, no edge incident to more than2 faces, consistent opposite shared-edge orientations. Positive finite FP64 half_thickness_m (f,), owned. Exclude face pairs sharing ANY vertex in S1; record this exact incidence policy, not an arbitrary k-ring. Isolated/nonincident faces remain eligible.
- candidate_self_pairs(topology, start_positions, end_positions=None): FP64 finite (n,3) -> sorted tuple of canonical face-index pairs (i,j), i<j. Static when end omitted, affine nodal path otherwise. Compute each face's axis-aligned bounds over BOTH endpoint node coordinates and expand by its rho using exact Fraction.from_float arithmetic; intervals overlap inclusively on all3 axes. Deterministic O(f^2) reference is acceptable, no global cache. This is conservative for the entire rotating-normal prism because each component of rho*N is in [-rho,rho]. It only returns candidates; no CCD/safety/collision-response status. A degenerate intermediate triangle is NOT certified safe by this broad phase.
- make_self_contact_operator(base, triangles, weights, offsets, pairs): combine ALL attachments into one TriangleOffsetPointPort(base, triangles, weights, offsets), then existing MultiDomainPointOperator({base.domain: offset_port}, left, right), with each pair mapped to the same domain. Return a SelfContactBinding with .operator (the existing operator), immutable owned .triangles/.weights/.offsets/.pairs/.world_points/.jacobians and node_impulses(impulses) -> owned FP64 (n,3). node_impulses is the requested aggregate computed from the exposed maps for audit, not a separate apply or an independent physics certification. The actual operator still applies exactly once through the existing base. Do not implement another base response/apply. No finite-thickness side selection or manifold generation in this factory.

Independent tests to AUTHOR:
1. Two disjoint triangles separated by >rhoA+rhoB along an axis excluded; equality included; swept crossing with disjoint endpoint boxes detected by endpoint-union bounds. Unequal thickness, negative/large translated coordinates, zero/invalid thickness and reversed face winding, deterministic reorder receipts. No epsilon expansion.
2. Folded strip topology: incident faces excluded, nonincident folds remain candidates; disconnected sheets remain eligible. Explicit duplicate/nonmanifold/inconsistent-winding rejection. AABB overlap false positive must remain a candidate, never become collision certification.
3. One real ParticlePointPort base with all offset rows; coincident +p/-p mapping and shared-node/canceling cases, one aggregate apply, full off-diagonal response vs independently assembled J_A-J_B, unchanged base snapshot. For separated rows audit the actual moment above, and full kinetic work including cross terms. Late candidate failure does not publish state.
Native response retains base backend but S1 must not claim new device geometry. Full finite-thickness prism-prism narrow phase, two-normal exact/interval proof, full CCD, owned anchor friction, and accepted full step are mandatory S2 integration, not secretly delegated to this broad-phase module.

## G1: finite parallel-jaw builder and load-bearing configuration

Allowlist:
- src/phymeo/builders/parallel_jaw_gripper.py
- tests/test_parallel_jaw_gripper.py
- docs/development/p65-gripper-module.md
A subsequent G2 experiment will use public runtime APIs and remain behind serial integrated acceptance.

make_parallel_jaw_gripper(*, half_gap_m, patch_half_extent_m, carriage_mass_kg, finger_mass_kg, inertia_diagonal_kg_m2, effort_limits_N, gravity_m_s2=(0,0,-9.81)) -> ParallelJawGripper(model, initial_state, patches, effort_limits_N).
The model is 3 real massive prismatic links: shared +Z carriage parent -1, two children at Y=-half_gap,+half_gap, axes +Y and -Y (positive q closes), identity fixed orientations, zero q/v, rectangular finite local XZ patches centred at each child origin. All dimensions/masses/limits finite positive; inertia input (3,3) passes RobotModel checks; patch extent is (half_x,half_z). Existing exact coordinate-prismatic support is preserved. No massless kinematic carriage, prescribed actual state or fake hard stops.
gripper_inputs(gripper, *, lift_effort_N, closing_effort_N) -> RobotInputs: lift scalar and two closing scalars may have either sign (opening allowed), each within owned limits or reject; forces/torques zero; no clipping hidden from caller.
Opening diagnostic from q is 2*half_gap-q1-q2, in metres. It is a diagnostic/support-range check, not an implemented joint limit. Do not label current model as general articulated thickness support.
Tests to AUTHOR: symmetric and asymmetric masses/extents; inherited carriage FK and finite patch orientation; positive commands close; commands exceeding limits rejected; ownership/input immutability; all invalid geometry/inertias. Parent G2 freezes high-mu hold/low-mu drop with finite lift/closure effort and independent force balance before executing. Builder success does not mean load-bearing success.

## Integration and acceptance still required

B2: implement actual bending backend and composite distance+bending full path ledger/native transaction without changing old laws. S2: full same-cloth prism geometry/CCD and coupled response; endpoint/midstep degeneracy/unknown must reject. G2: independent no-self-contact finite gripper hold/slip, then folded-cloth integration. General robot and noncoordinate/rotor thickness still need their own support proof; do not equate the 3-prismatic builder with that capability.

At integration, same substep must satisfy material residuals, all self/external contacts, actual force/moment/power, energy budgets, complete CCD and history rollback together. For all new capabilities retain at least2 independent non-handoff scenarios, h halving, necessary duration/repeat/mesh/perturbation/failure gates; exact thresholds frozen before numerical acceptance, no post-result tuning. Interpret/CUDA mandatory for claimed native capability. Final H2/P6.5 requires integrated folded cloth and finite grasp, never a count of modules/tests.

Read-only builder check occurs once before coding these contracts. Parent resolves mathematical/interface findings; each scope's genuine unmet physics remains visible. A fresh integration review after completed new implementation does not repeat the already closed thick-friction review.

## Parent rulings from the single builder check

B1-B5 were resolved without new user decisions because user already approved this architecture and execution schedule.
- Bending trace includes alpha/denominator/branch and actual pre/post state, not just final lambda. A future angle/gradient audit must establish its own bounds; Fraction(float atan2) is not an exact ideal angle. No native/composite physics acceptance from B1.
- The same physical base is wrapped once with every offset anchor and existing same-domain MultiDomainPointOperator routing. The immutable public receipt exposes actual map and requested nodal loads for later audit; no second state owner.
- S1 excludes all incident face pairs by its explicit policy. Adjacent faces' far-end crossings are therefore outside S1 coverage; candidate generation is not arbitrary self-collision certification. Full future support must address this before folded-cloth acceptance.
- Independent prism normals require a separate number-field/interval proof and symmetric normal-selection rule. Old patch-face priority and single-Q arithmetic cannot be silently reused.
- G1 finite prismatic geometry does not publish controller predictions as state. Initial simple finite hold can precede self-contact completion, but real load-bearing, folding, general-robot support and complete P6.5 remain separate acceptance gates.

Current scripts/tests are deferred by the user's scheduling instruction. A code checkpoint is explicitly unverified, not a passing product or closed stage.
