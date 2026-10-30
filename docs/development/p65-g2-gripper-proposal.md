# G2 finite-effort grasp: parent integration proposal

Status: PROPOSAL_NOT_FROZEN, 2026-10-30. This document is a handwritten force/impulse derivation against current APIs, not executed simulation evidence. G1 host model/input tests passed at a16bf4a; B1/S1 are separately verified foundations. The parent is preparing this proposal while the original B1/S1 implementers prepare B2/S2 proposals without executing anything. The next implementation contract requires combined interface review and explicit parent mathematical rulings; user continuation is already authorized.

## Existing code and scope

- builders/parallel_jaw_gripper.py: actual massive carriage and two finite coordinate-prismatic fingers, bounded effort requests; no state prescription.
- solvers/dynamic_robot_thick_contact.py: DynamicRobotThickContactProgram accepts all-coordinate-prismatic trees, finite patches, triangle prisms and DistanceConstraints or None; original MaterialContactBudgets and CCD remain mandatory.
- tests/test_thick_robot_contact.py: existing public SceneModel/SceneState/SceneInputs, NumPy/tyx participants, ArticulatedCouplingProgram/TyxArticulatedCouplingProgram, SceneStepper assembly; historical A's two independent jaws are not a shared vertical carriage hoist.
- tests/test_thick_robot_tyx.py: actual backend, FP32 publication, launch counts and late-failure instrumentation conventions. Reuse interfaces, not prior results as G2 evidence.

Preferred route: configuration and actual trajectory recording over these public components. Do not add a special gripper solver or duplicate a material/contact step. A new reusable builder is justified only if assembly exposes a genuine repeated interface gap. The first implementation should stay within an example/experiment configuration, new independent tests, and its evidence report. Parent owns any discovered shared correction and requires a genuine original-form failing control before changing it.

This route allows no-self-contact load bearing before B2/S2. It does not satisfy bent-sheet material, self-collision, arbitrary articulated thickness or final folded-cloth acceptance. Alternatives are (a) delay every grasp experiment until all shell geometry is complete, which adds a dependency unnecessary for the planar controls; or (b) separate prescribed-gripper demonstration, which violates finite effort and one physical state. Retain the public shared-runtime route.

## Common physical configuration

SI units. All nodes dynamic FP64; three masses1kg at (-1,0,-1), (0,0,2), (1,0,-1), initially at rest. Half thickness rho=1/8m. Single TrianglePrismSurface, no material constraints in the first independent translation oracle. This explicitly tests finite-thickness contact/load bearing of a dynamic face, not constitutive fabric deformation. A distinct material-active case and refined/folded sheet remain later integration requirements and cannot be inferred from the free-face result.

G1 carriage and both fingers each mass2kg; all diagonal COM inertias1kg*m^2. Three axes +Z/+Y/-Y. World gravity (0,0,-1)m/s^2 for robot and nodes. Rectangular XZ patch half extents(4,4)m remain finite and fully contain the relevant projected triangle throughout the proposed windows. Initial q/v=0. The shared vertical generalized mass is6kg; each closure coordinate has mass2kg; the exact matrix is diag(6,2,2) because axes are perpendicular and no revolute coordinates exist. This statement is specific to this model, not an assumed general-tree M0/M1 energy proof.

Both fingers receive constant +2N closing efforts. Positive q closes their gap; after contact their physical closure velocity is zero through contact response. Lift efforts below are finite actual generalized force requests, with G1 limits(16,4,4)N. No coordinate/velocity override, artificial hard stop, pin, weld or time-dependent contact law. Scenario choices configure initial gap, friction and external force only.

Define D_h(t)=t(t+h)/2 for t=n*h. Semi-implicit constant acceleration gives v=a*t and displacement a*D_h(t), derived independently of solver traces. Every state, contact impulse and external work assertion must use the actual returned state and receipts; oracle functions must not call production response/geometry routines to manufacture expectations.

## A: initially closed finite support and low-friction slip

Initial half gap=rho. Lift force9N, closing force2N each. Robot weight6N and cloth weight3N. All contact points start closed. Six symmetric corner anchors are the expected candidate selection; each face's normal aggregate is2h N*s. Uniform per-row held normal2h/3 is the current symmetric candidate path, not a unique Coulomb pressure distribution claim.

With mu=1, total available tangential force4N exceeds cloth weight3N. Cloth and carriage stay still. Each held tangential row is +Z*h/2 on cloth; the robot gets -3h total through its shared vertical coordinate. Held normal balances the finite closing inputs; terminal impulses are zero for the oracle. Total vertical force is9-6-3=0. Actual actuator work, gravitational potential change and kinetic change are zero in this stationary control. Stored inactive history remains zero.

With mu=1/4, total upward friction on cloth is1N, each held tangent is +Z*h/6. Cloth acceleration is -2/3m/s^2; carriage acceleration is (9-6-1)/6=+1/3m/s^2. Therefore:

- cloth v_z=-2t/3, displacement_z=-(t^2+t*h)/3;
- robot carriage v_z=t/3, q_z=(t^2+t*h)/6;
- both closing q/v stay0; terminal impulses remain0;
- relative cloth-to-patch displacement_z=-(t^2+t*h)/2.

The moving massive robot is essential to this control: its response must include the sum of both fingers' tangent reactions, not two independent vertical carriers. At T=1s the entire face remains within the finite patches. Independent kinetic energy is K=3*(2t/3)^2/2+6*(t/3)^2/2=t^2 J. Gravity potential is 3*d_z_cloth+6*q_z=0; external lift work is9*q_z for the constant mass/translation path. Dissipation and discrete integration terms must be checked through the unchanged ledger, not assumed to vanish or replaced by this aggregate identity.

Continuous displacement references omit the t*h term. The exact discrete-versus-continuous error is t*h/3 for cloth and t*h/6 for the carriage; halving h halves these errors at common T. This temporal-discretization statement does not loosen runtime residuals or energy/CCD budgets.

## B: open-gap capture followed by actual upward hoisting

Initial half gap=rho+delta, delta=1/16m; mu=2, constant lift force12N and closing forces2N each. Both jaws start separated from the cloth by delta. There is no scripted contact activation. Before contact, closure acceleration is1m/s^2, carriage vertical acceleration is(12-6)/6=1, cloth vertical acceleration=-1.

Let n_c be the smallest positive integer satisfying h^2*n_c*(n_c+1)/2 >= delta, and tau=n_c*h. The proposed grids h=1/8 and1/16 have strictly positive overshoot and both give tau=3/8s; no exact-on-grid contact tie is used for this acceptance. Up to step n_c-1, q_close=D_h(t), v_close=t, q_z=D_h(t), robot v_z=t; cloth displacement=-D_h(t), v_z=-t. Contact forces are zero.

At the first contact step, q_close reaches delta through the actual held correction. Set u=(delta-D_h(tau-h))/h. Each finger's held normal aggregate is P=2*(tau-u), terminal aggregate E=2*u, hence P+E=2*tau. Each expected symmetric row carries one third. The initially open contact's held tangent is exactly zero under the existing two-phase law. Full terminal friction capacity uses mu*(P+E), with each phase's actual normal projection preserved.

Just before terminal correction the vertical velocities are -tau for cloth and +tau for the6kg shared robot. The total vertical momentum is3*tau, so sticking yields a common velocity tau/3. Cloth receives upward terminal impulse3*(tau/3+tau)=4*tau, i.e.2*tau/3 per symmetric row; the robot receives its opposite. Capacity per row is mu*2*tau/3=4*tau/3, strictly larger than this demand. Vertical positions at that contact step remain -D_h(tau) and+D_h(tau); terminal velocity correction must not alter them.

After capture, the combined9kg system has acceleration(12-9)/9=1/3. Required cloth upward friction is3*(1+1/3)=4N, i.e.2/3N per row, below mu*2/3=4/3N capacity. For s=t-tau>=0:

- common v_z=t/3;
- cloth displacement_z=-D_h(tau)+(tau/3)*s+D_h(s)/3;
- carriage q_z=+D_h(tau)+(tau/3)*s+D_h(s)/3;
- q_close=delta, v_close=0;
- after the impact step held normal per row2h/3 and held tangent per row2h/3; terminal0.

At T=1, h=1/8, the proposed exact oracle is cloth displacement1/16m and carriage q_z=1/4m. At h=1/16, D_h(tau)=21/256, (tau/3)*s=5/64 and D_h(s)/3=55/768, giving cloth displacement13/192m and carriage q_z=89/384m. Both predicted cloth displacements are above the original position through finite forces; this has not yet been observed in an execution. These values come solely from the displayed rational formulas and must not be fitted to solver output.

For the continuous acceleration/impact reference tau_star=sqrt(2*delta), final cloth displacement is T^2/6-4*delta/3. The discrete expression simplifies to (T^2+T*h)/6-(2/3)*tau*(tau+h). Since0<=D_h(tau)-delta<tau*h, its error is bounded a priori by h*(T/6+4*tau/3). Event-grid errors need not halve by an exact factor; compare both grids to the full discrete equations and this independent O(h) bound, not a tuned observed ratio. Runtime physical budgets remain unchanged.

## C: material-active deformation inside finite closed jaws

This independent handwritten control uses the existing DistanceConstraints implementation in the real G1 shared-carriage assembly. It is not a B2 bending or folded-sheet result. Keep the same three dynamic nodes, masses, finite patches and initial closed gap. Override the external configuration to zero gravity, zero lift, +2N closing effort per finger and mu=0; no runtime or contact-law change is involved. Add one cloth edge(0,2), rest length2-delta_m with delta_m=1/64m and compliance c=1/16m/N. Initial velocities are zero.

For a single step h, alpha=c/h^2 and the independent distance XPBD multiplier is lambda=-delta_m/(2+alpha). Node0 moves by -lambda along X and node2 by +lambda along X; node1 is unchanged. Their X velocities are respectively -lambda/h and lambda/h. The post-projection edge extension is delta_m+2*lambda=delta_m*alpha/(2+alpha). The symmetric deformation stays in the initial XZ plane, so its normal remains Y and the normal contact velocity contribution from the X correction is zero. Each jaw's aggregate held normal is2h; the symmetric candidate predicts2h/3 per row. Robot coordinates/velocities, tangent impulses and terminal impulses are zero. These are predictions to check on returned state and receipts, not injected state.

The independent material-energy expressions are K_after=lambda^2/h^2, U_before=delta_m^2/(2*c), U_after=(delta_m+2*lambda)^2/(2*c), giving Delta(K+U)=-delta_m^2/[c*(2+alpha)]. Material angular impulse is zero in this symmetric geometry. In particular:

- h=1/4: alpha=1, lambda=-1/192, endpoint X velocities +/-1/48 and Delta(K+U)=-1/768J;
- h=1/8: alpha=4, lambda=-1/384, endpoint X velocities +/-1/48 and Delta(K+U)=-1/1536J.

These exact equations are derived without executing production code. Run both step sizes and two independent rebuilds on host and actual backends only after contract freeze and a serial execution grant. Check the complete material trace and the unchanged independent ledger, rather than substituting the aggregate energy formula for either. The proposed G2 tolerances and budgets below remain fixed; if native precision or convergence cannot meet them, retain the failing original-form evidence for parent adjudication. Do not silently borrow the looser historical material special-case budget. C establishes active distance material in this finite gripper assembly only; mesh refinement, bending, self-contact and folded-sheet load bearing remain separate acceptance requirements.

## Acceptance to freeze before implementation

Proposed host T=1s; actual interpret/CUDA T=1/2s, both h=1/8 and1/16, two fresh scene/workspace rebuilds per A(mu1), A(mu1/4), B(mu2). Device B must demonstrate positive upward displacement after capture and common positive post-capture velocity, even though its shorter window need not end above initial height. Full state and padded history must be retained and compared on each rebuild; no extra physical rerun solely to collect reports.

Use existing tolerance1e-8 host and2e-5 actual devices; existing max_energy_gain_j/max_material_angular_defect_nms1e-8 host and1e-5 devices; outer/inner512 position and512 velocity iterations, anchor64/rebuild8, original CCD budgets. Do not multiply tolerances by rows, weaken actual-port membership, promote unresolved CCD or swallow unknown native failures. New inability to converge must be reported against this support case, not solved by widening budgets or writing host state.

Native runs use --require-tyx and explicit mode, actual launch evidence and FP32 publication checks. No host fallback. Every accepted step must preserve original per-domain force/moment/power, complete mapping, material/robot/terminal ledgers and accepted trajectory CCD. Retain late terminal/close failure after earlier accepted substeps, whole outer state/history/input rollback, and retry equal to a fresh run; reusing old failure infrastructure must not assume this new assembly has passed.

Variations still to freeze with builder review: symmetric mass/effort scaling with independent force balances, finite patch dimension/translation changes that stay in the declared support interior, scene/domain/patch order and immutable input controls, review of material-active control C, and later mesh/folded-sheet gates. Do not claim this first free-face G2 subset completes all original G2 or H2 requirements. The full B2/S2/G2 integration acceptance keeps mesh, perturbation, duration, repeat, failure and transaction gates.

## Proposed ownership and outstanding review questions

Parent configuration ownership: new examples/parallel_jaw_cloth.py, tests/test_gripper_cloth.py, tests/test_gripper_cloth_tyx.py, and docs/development/p65-gripper-load-bearing.md. Exact files/API surface should be finalized after reading example conventions and the B2/S2 proposals. No existing solver/runtime/participant/kernel/port/ledger/law change is included by default. No implementation or experiment has run for this proposal.

Required pre-implementation decisions: verify B's event-normal and terminal-capacity equations against exact current public phase semantics; verify the fully stated rational endpoint values; verify C's material and normal-decoupling equations without claiming B2 exists; agree how later conforming self-contact geometry connects to external robot contact. A single new combined design check may examine the completed next contract; do not repeat the closed B1/S1 foundation builder or old thickness fresh review.
