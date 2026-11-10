# MuJoCo-TYX T4: joint constraints and unchanged G1 admission

Status: **RESEARCH / INTERFACE PROPOSAL, NOT IMPLEMENTED OR VALIDATED**.

Base: `73a85f8affccde5ebf21a4629845d06af6949479`. Author branch: `codex/mujoco-tyx-t4-source`. This lane performed source, XML/JSON and static topology reads only. No project import, native compilation, probe, test, formatter, build or install was executed by this lane. The parent separately compiled the unchanged G1 asset once; that receipt is identified below. Its inventory is not a physics result.

The next deliverable is one general constraint system for native pyramidal contact, scalar joint limits and DOF dry friction, followed by an unchanged G1 consumer. It must preserve the nine accepted primitive manifolds, the complete current joint Jacobians, physical mass coupling, Euler integration and atomic publication. A G1 pass alone is insufficient; independent mixed-constraint cases must pass first.

## 1. Source and asset boundaries

Relevant implementation entrances are [static normal admission](../../src/phymeo/backends/mujoco_tyx/_normal_model.py), [floating admission](../../src/phymeo/backends/mujoco_tyx/_floating_model.py), [normal geometry](../../src/phymeo/backends/mujoco_tyx/_normal_geometry.py), [normal solve](../../src/phymeo/backends/mujoco_tyx/_normal_solver.py), and [resident glue](../../src/phymeo/backends/mujoco_tyx/_normal_device.py). The accepted base and remaining scope are in [the continuation entry](p65-continuation.md).

Source references are fixed, not moving release documentation:

- [MuJoCo 3.3.7 engine_core_constraint.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_constraint.c): DOF friction, both scalar limit sides, native row parameters and pyramidal expansion.
- [MuJoCo 3.3.7 engine_init.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_init.c): omitted XML option defaults.
- Local MuJoCo-Warp **0.0.2**, `artifacts/p65-unified-scene-20261110-01/optional-deps/mujoco_warp/_src/constraint.py`: useful row construction reference, with the CPU differences below retained.
- Unitree source repository `unitreerobotics/unitree_rl_mjlab`, commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`, recorded by `sources.json` in the existing G1 research archive. Original primitive robot XML is [the fixed upstream g1.xml](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/assets/robots/unitree_g1/xmls/g1.xml).

The absolute G1 research directory is `C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/artifacts/p65-g1-research-20261109-01`. Three files there must not be confused:

| Asset | Actual geometry and dynamics | Meaning for T4 |
|---|---|---|
| `g1.xml` | Original primitive collision robot: 30 non-world bodies, 29 scalar joints plus freejoint, 33 physical primitives and 35 visual mesh geoms; no actuator block or floor | Source of the primitive collision model; it is not the finished policy scene |
| `g1_primitive_actuated.xml` | Existing Phymeo research asset: same 33 physical primitives, visual mesh geoms removed, floor added, 29 original motor controls; scalar armature .01, damping .05, dry friction explicitly 0 | Selected unchanged G1 consumer for T4/G1; no physical mesh substitution is needed |
| `scene_g1.xml` | Separate deployment scene: 24 collision mesh geoms, 4 collision cylinders, 8 foot spheres, floor, plus 35 visual meshes; default joint dry friction .2 and 29 joint-wide actuator force ranges | Existing native CPU dance reference belongs to this different scene. T4 primitive support cannot claim this mesh scene is supported |

The selected primitive file SHA256 is `f7e856e057a095a9f8708d302531d40ff2519440012c24488407ae8d0f0586aa`, as recorded by the parent's new `artifacts/p65-mujoco-tyx-t4-20261110-01/g1-inventory.json`. That inventory reports `physics_execution="none: native model compilation only"`. This lane read its fields and independently counted the XML/topology; it did not rerun the compiler or reconstruct old evidence.

The deployment scene uses an explicit free `joint` with inherited defaults, rather than the primitive asset's `freejoint` shortcut. Do not infer its free-root damping/armature/frictionloss from the primitive model. A later admission pass must use its compiled native arrays. Collision meshes and cylinders remain separate prerequisites for that scene.

## 2. Exact selected G1 requirements

The primitive XML has no `option` element. The parent's MuJoCo 3.3.7 inventory confirms the native defaults: Euler, pyramidal cone, timestep .002, iterations 100, tolerance 1e-8, impratio 1, no noslip and zero enable/disable flags. The TYX numerical solver remains explicitly different from native Newton; retain the declared FP32 solver tolerance and expose its actual stored value.

| Field | Selected primitive asset |
|---|---|
| Native dimensions | nq=36, nv=35, nu=29, njnt=30, nbody=31 including world |
| Geometry | ngeom=34: 31 capsules, 2 spheres and one plane; no mesh collision or visual geoms remain |
| Joint limits | 29 limited hinges with original ranges, margin 0; at most 58 authored lower/upper slots, normally at most one active side per hinge for these ranges |
| Passive terms | Scalar damping .05 and armature .01; dry-friction DOFs count 0; no need to invent frictionloss for this asset |
| Motors | 29 scalar stateless unit-gear torque motors; native ctrlrange clamps; no joint actuator-force clamp enabled |
| Contact law | Robot collision class priority 1, condim 6; all possible pairs mix to condim 6. Preserve sliding, torsion and rolling coefficients |
| Filtering | 4 explicit body exclusions; no explicit pairs, equality constraints or tendons |
| Sensors declared | pelvis gyro, velocimeter, accelerometer and root subtree angular momentum; the policy reads its observation from state/kinematics, not all four sensor outputs |

The four exclusions are left elbow/left wrist pitch, right elbow/right wrist pitch, pelvis/right hip roll, and pelvis/left hip roll. Support compiled `exclude_signature` in the generic pair filter; do not ignore it or reject the unchanged asset solely for containing exclusions.

Motor ranges are asymmetric between left and right ankles and must stay so: left ankle pitch/roll are ±50, right ankle pitch/roll ±25. Hip pitch/yaw are ±88, hip roll/knee ±139, waist yaw ±88, waist roll/pitch ±25. Shoulder pitch/roll/yaw, elbow and wrist roll are ±25; wrist pitch/yaw are ±5. Native motor clamp behavior is already in the T2 scalar actuator path. The deployment mesh scene's separate `jnt_actfrclimited` requirement is not present in this selected primitive consumer.

A six-dimensional pyramidal contact has **10 nonnegative edge forces**, not six independent unilateral coordinates and not only the four sliding edges. Condim 4 has six edges; condim 3 has four; condim 1 has one. Never lower the asset's condim to fit the first friction implementation.

### Static pairs and capacity

All 30 non-world bodies have exactly one free/scalar joint, so there are no non-world welded bodies to ambiguously count in this asset. XML body topology plus native default masks gives 506 pairs after same-body/parent filtering; the four body exclusions remove four geom pairs, leaving 502. The parent's compiled inventory independently confirms these numbers.

| Pair type | Eligible pairs | Max contacts per pair | Static slots |
|---|---:|---:|---:|
| capsule-capsule | 409 | 2 | 818 |
| plane-capsule | 31 | 2 | 62 |
| sphere-capsule | 59 | 1 | 59 |
| plane-sphere | 2 | 1 | 2 |
| sphere-sphere | 1 | 1 | 1 |
| Total | 502 | — | 942 |

Thus the fully authored contact-edge table would have 9420 rows. This is a **potential** count, not an observed simultaneous-contact count and not a license to allocate/factor a 9420-square dense matrix every step.

Parent-selected T4 capacity direction: G=128 remains; static P=1024 and S=2048; detected-contact capacity 128 and active EFC capacity 512. Keep all static candidates in device geometry, then deterministically compact detected contacts and actual active constraint rows. Limits and dry friction share the EFC capacity. For this asset 29 ordinary active limit sides plus 48 condim-6 contacts would be 509 rows; that arithmetic is a capacity example, not a contact-count prediction. If real physics exceeds either runtime capacity, reject the entire attempted batch without publication and report required counts. Counts exactly at the configured capacity succeed.

## 3. Native joint row semantics

DOF dry friction and viscous damping are distinct. Existing damping belongs to smooth dynamics and Euler's implicit damping matrix. Dry friction becomes a bounded force in the shared constraint solve.

For each positive compiled `dof_frictionloss[d]`, create one row:

- J=e_d, position=0, margin=0, velocity=qvel[d].
- Lower/upper force bounds are -loss/+loss.
- Use that DOF's native `dof_solref`, `dof_solimp`, and `dof_invweight0`.
- K=0; aref=-B*velocity. R follows the native impedance/diagonal-approximation rule; D=1/R.
- Preserve the free-root six native coordinates if they have positive compiled frictionloss. A force on angular native coordinates is local generalized torque; do not reinterpret it as a world-axis torque.

Do not implement dry friction by subtracting loss*sign(velocity). That misses sticking, bounds and coupling with contact/limits.

For every limited hinge or slide, author both sides using native qpos/dof addresses:

- Lower: dist=q-low, J=+e_d.
- Upper: dist=high-q, J=-e_d.
- A side is active iff dist < margin; force bounds are [0,+infinity).
- Use `jnt_solref`, `jnt_solimp`, `dof_invweight0[d]`, row velocity Jv, and aref=-B*Jv-K*imp*(dist-margin).
- CPU 3.3.7 iterates lower then upper, and can retain both if margins overlap. Warp 0.0.2 takes only the nearer side. Preserve CPU's two-side semantics for a general narrow-range/large-margin input; do not silently adopt Warp's narrower behavior.

Existing T1/T2 geometry and integration reject entry/candidate q outside their limit-free safe range using vectors slots 7/8. T4 must explicitly select constraint-supported limit mode and disable those old *limit-domain* rejections for admitted joints, retaining all finite/state validity guards. Old T0/T1/T2 default behavior remains unchanged. Never clamp or repair q/v to enforce a soft native limit.

## 4. Common contact and joint solver

Keep accepted manifold generation and full current body COM Jacobians. At a contact point, form relative translational and angular Jacobians for both bodies. Sliding uses the two frame tangent projections of the complete point velocity; torsion uses the normal projection of relative angular velocity; rolling uses the two tangent angular projections. Angular contact rows are pure torque directions and must not get an extra position lever arm.

For pyramidal dimension d>1 and each k=1..d-1, append Jn+mu[k-1]*Jk followed by Jn-mu[k-1]*Jk. Native friction coefficients are lower-clamped by mjMINMU. Decode the solved edges back to a six-component contact-frame wrench: normal is the sum of all edges, and each friction component is mu[k-1]*(f_plus-f_minus). Publish the same wrench used for generalized force accumulation.

Preserve CPU parameter order. For pyramids, let tran be the sum of the two native translational body_invweight0 values, imp the contact impedance and mu0 the first sliding coefficient:

1. Rbase=max(mjMINVAL,(1-imp)/imp * tran*(1+mu0^2)).
2. R1=Rbase/max(mjMINVAL,impratio), mu_scaled=mu0*sqrt(R1/Rbase).
3. Every edge gets R=2*mu_scaled^2*Rbase and D=1/R.
4. Native final diagApprox is R*imp/(1-imp). It is not necessarily the initial geometric diagonal approximation.
5. Each edge retains normal solref K/B and penetration displacement; aref uses that edge's Jv. Do not use a normal-only aref for all edges, or solreffriction on pyramidal edges.

The local Warp implementation moves a scale factor before the R clamp. That can change small-R behavior, so native CPU order wins. This source-based distinction is reviewed independently in the T4 friction research lane.

All admitted rows solve one mixed-bound convex QP:

`min_f .5 f^T (J M^-1 J^T + diag(R)) f + (J a0 - aref)^T f`, with `lo <= f <= hi`, `a0=M^-1(applied+passive-bias)`.

Contact edges and limits have lo=0; dry friction has finite symmetric bounds. Geometry-inactive rows are absent from the active solve. Numerically zero Jacobian rows are not removed when they are structurally valid native rows: R and bounds still define their force.

Extend the bounded active-set algorithm to lower/free/upper states. KKT checks use g>=0 at a lower bound, g<=0 at an upper bound, and g=0 in the interior, together with primal bound error. Consume the declared budget for every face solve, including a blocked move. Publish the actual normalized KKT and applied tolerance; no native-Newton/warmstart equivalence claim.

Avoid active-row-square factorizations. For current free rows F and bound rows B, solve the equivalent nv-dimensional system:

`(M + J_F^T R_F^-1 J_F) a = rhs + J_B^T f_B + J_F^T R_F^-1 aref_F`

and recover `f_F=R_F^-1(aref_F-J_F a)`. This is the same physical QP face, not an independent per-contact response. The factorization is at most 35x35 in the current floating domain, while row storage remains bounded by E. Recover complete gradients from J and generalized acceleration; a dense dual matrix may be a bounded diagnostic product, not required resident factor scratch.

After successful solve, one generalized constraint vector J^T f augments applied load. The existing Euler solve still uses M+hD for velocity integration; physical QP uses M. State, force, accepted time, counters and action history publish together. A capacity, nonfinite, factorization or convergence failure preserves the previous bank.

## 5. Proposed static/runtime ABI

These names/layouts are proposals for the parent to freeze before implementation; the parent owns shared admission, schemas and resident wiring.

Static additions:

| Array | Proposed fields |
|---|---|
| friction_ids[F] | native dof |
| friction_law[F,9] | loss, invweight0, solref[2], solimp[5] |
| limit_ids[L,3] | native joint, qpos address, dof address |
| limit_law[L,11] | low, high, margin, invweight0, solref[2], solimp[5] |
| pair_condim[P] | full mixed native dimension |
| pair_friction[P,5] | full mixed friction coefficients, retained from T3 |
| contact_row_map[C,2] | first compact EFC row, row count; -1/0 when not active |

Runtime proposal:

- `efc_ids[W,E,5]`: kind, native owner id, side/edge, contact-slot id, dof id. Use -1 for inapplicable fields. This separates a joint constraint from a contact.
- `efc_data[W,E,12]`: position, margin, velocity, aref, R, D, impedance, lower bound, upper bound, force, gradient, active flag.
- `efc_J[W,E,nv]`, `efc_count[W]`, compact-row-to-source mapping and contact-frame wrench[W,C,6].
- Preserve per-row scalar parameters needed for audit (K/B or their frozen source binding), phase-local raw input state and full dynamics receipts.
- Keep raw geometry candidates and present/inactive contact records. An inactive detected contact still has an auditable current J/frame/position even though it emits no active EFC rows.
- Keep `normal_rows[W,S,32]` only as a compatibility/geometry view where useful; the general EFC table is the authoritative joint/friction/contact force ledger.

Deterministic order: dry-friction DOF ascending, limited-joint ascending with lower before upper, then contact slot and native edge order. Compact on device. Geometry ordinal is local to the backend's current step; retain accepted cross-precision matching semantics rather than pretending it is a persistent material identity.

Suggested decomposition has three constraint stages, **not a claim that the entire engine becomes three launches**:

1. Row construction: accepted geometry/current body J plus q/v, native joint laws and pair laws -> compact EFC/J/map/counts. Check contact/EFC overflow before state publication.
2. Mixed-bound solve: physical M/rhs plus compact rows -> forces, gradients, KKT diagnostics and one qfrc_constraint.
3. Decode/publication preparation: contact wrench6 and joint forces plus total applied -> existing Euler integration and atomic candidate checks. This can be fused with a compatible existing solve/decode stage after the ABI is fixed.

Existing body geometry, mechanics and optional extended primitive manifold launches remain. No per-step host linear algebra, CPU contact solve or download/reupload of q/v is introduced.

## 6. G1 consumer and validation entrance

Reuse [g1_policy_reference.py](../../examples/g1_policy_reference.py) only for its original policy protocol; its default mesh scene must be overridden explicitly with the selected unchanged primitive file for a same-asset CPU reference. The existing [g1_connected_policy.py](../../examples/g1_connected_policy.py) and [g1_resident_policy.py](../../examples/g1_resident_policy.py) are older Phymeo dynamics/contact consumers, not proof of the new MuJoCo-TYX constraint path.

Required inputs remain the research archive's `g1_primitive_actuated.xml`, `deploy.yaml`, `dance1_subject2.npz`, `policy.onnx` and external ONNX data. The original policy has input obs[1,154], output actions[1,29], control_dt=.02; native physics timestep is .002, giving ten physics steps per policy update. CPU ONNX inference is acceptable and must be disclosed separately from device physics.

Initial q/v must come from the original motion's frame zero: world root-origin position, WXYZ orientation, original scalar positions, original world root-origin linear velocity, world root angular velocity transformed into native local coordinates, and original scalar velocities. Only the existing initial quaternion normalization is permitted. Do not start with zero velocities, shift the policy root to the floor, pin a foot, or alter geometry/mass/damping to pass.

Build observations from the same accepted state: reference positions/velocities, torso-relative orientation, local base angular velocity, joint offsets/velocities and previous accepted action. Compute targets with original offsets/scales and per-physics-step PD torque with original kp/kd, then native motor control clamps. These externally generated torques are inputs; they must not be replaced by target-q writes or state replay. Rejected physics attempts do not advance policy action history or motion-frame time.

The four declared sensors do not authorize fabricated outputs. The 154-vector can be obtained from existing state/body kinematics without an accelerometer/subtree-momentum port. Mark unimplemented native sensor queries unsupported until independently migrated.

### Predeclared development/test progression

1. **Generic row fixtures:** hinge and slide, lower and upper, zero/equality boundary and overlapping margins, both limit sides simultaneously, dry-friction stick/positive slip/negative slip, nonzero free-root loss, stateless actuator saturation, and no-contact mixed constraints.
2. **Coupled fixtures:** contact+limit, contact+dry friction, and all three together on floating/branched systems; multiple contacts, unequal inertias, all five friction components, condim1/3/4/6. Compare original native J/R/aref and same-input M/rhs; separately solve small independently authored bounded QPs.
3. **Continuations:** enter/leave a limit, stick-slip reversal, persistent support with sliding/torsion/rolling, contact unload/reentry, mixed rows appearing/disappearing, two worlds with distinct active rows, count==capacity and count>capacity, budget/factor/nonfinite transactional failures.
4. **Unchanged G1 admission and snapshots:** bind the original asset and compile inventory, preserve all502pairs/4excludes, compare first input M/bias/passive/applied and all actual rows/wrench at original frame-zero q/v. Do not begin by deleting candidate pairs to fit capacity.
5. **G1 short native-reference/interpret/CUDA continuity:** same asset, same original observation/PD path and accepted-time history. Then sustained support and the full intended dance window, with fall/load/motion criteria declared before execution. Previous full CPU dance on the mesh scene is historical context, not the oracle for this primitive scene.
6. **Integration:** adjacent T0/T1/T2/T3 behavior and fresh outside-source wheel consumers, only for new source. Parent schedules all execution in its unique serial slot.

This research does not establish a G1 support or dance pass. A small numeric discrepancy may be accepted under the predeclared engineering gates; missing torque channels, dropped pairs/rows, false capacity success, state repair or a different collision asset may not.

## 7. Implementation split and exit criteria

Parent: static model/dispatch, complete pair filtering/exclusions, capacity separation, old-limit-guard mode, resident buffers/wiring, public capability/receipt schema and NOTICE. Source lane after a new explicit grant: new typed constraint-row kernel and its low-level parameter functions; old accepted geometry and mechanics are reused. Solver lane: mixed-bound active set with the nv-dimensional free-face solve and full KKT/transaction semantics. Independent test lane: native same-input oracles, generic matrix above and unchanged G1 consumer.

Research is complete when the parent can freeze concrete fields/capacities and independent tests without guessing G1 semantics. Implementation is complete only after generic row/combined dynamics cases, actual interpret/CUDA consumers, capacity/error behavior and the selected G1 stage pass. A failed G1 window is an identified integration failure to diagnose from retained rows/forces, not permission to keep changing its physical model.

Open scope remains explicit: elliptic cone/SOCP, ball-joint limits, tendon/equality constraints, joint-wide actuator force clamps, physical mesh/cylinder collision, native sensors and broader actuation are not delivered by this proposal. The selected primitive G1 needs none of the mesh or joint-wide clamp extensions, but it does require full condim6, joint limits, excludes and larger candidate handling.
