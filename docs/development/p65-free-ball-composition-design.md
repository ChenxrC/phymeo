# FREE root + BALL composition: Batch B / R2

Status: AUTHORING_AUTHORIZED, all new code/tests UNEXECUTED. Baseline c45a01740d131ed6bebfabe2285169e67f296bd7. Parent remains the sole execution owner.

## 1. Finite public behavior

The first slice accepts one leading native FREE root and BALL/HINGE/SLIDE descendants, with real direct JOINT stateless motors and external generalized/body-COM loads. It preserves the original public compile -> MuJoCoRuntime(implementation="tyx") -> control -> step -> integration_snapshot/checkpoint/clone/reset workflow. No synthetic three-hinge BALL, fixed root substitution or CPU dynamics fallback.

Limits: nv <= 35, native bodies <= 128, 1 <= BALL count <= 9, nq = nv + 1 + BALL count <= 45, nu <= 32 and complete JOINT moment nnz <= 96. Root is a world child and its only joint; every dynamic descendant belongs to that root. Each BALL is its body's only joint. All original finite/normal-or-zero FP32, positive inertia, Euler, option and resource gates remain.

This slice rejects activation, every joint limit/dry-friction row, BALL/FREE springs, potential contact pairs, multiple FREE roots, JOINTINPARENT/tendon transmissions and BALL affine bias. Scalar fixed-gain NONE/AFFINE laws retain their original native meaning. BALL uses fixed gain and BIAS_NONE. Shared targets, finite zero/oblique/negative gear, disabled groups and native control/force clamps remain meaningful.

No-BALL T2 models retain their old metadata/geometry/integration route and admitted scope, including their previously accepted contact/activation wrappers. This new finite capability does not inherit those combinations.

## 2. Actual source gaps and primary evidence

At baseline:
- _model.py:94-104 routes any BALL to the fixed BALL compiler; that compiler rejects FREE.
- _floating_model.py:33-42 assumes only scalar descendants and nq=nv+1; its joint_ids are DOF pseudo-rows and body ranges start at DOF addresses.
- _floating_kernels.py:317 uses q[i+1] for scalar springs; :446,496,544 assume nq=nv+1 and one scalar per nonroot DOF. Only the root quaternion is integrated.
- _tree_model.py:228-265 already separates native joint rows, qpos/DOF addresses and three ancestor columns. _tree_kernels.py:158-252 contains the accepted child-BALL frame/velocity transport.
- _joint_transmission.py:50-52 hard-bounds the fixed-tree CSR to nv8/nq10. Public typed transmission has the corresponding fixed-schema bound.
- _normal_device.py:236-264 and _constraint_device.py:297-325 finally call the original floating integrator; their existing scalar calls must receive the new explicit legacy-tail arguments.

Fixed primary references (facts, not claims of this slice passing):
- [MuJoCo 3.3.7 kinematics and DOF axes](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_smooth.c#L50-L135): native FREE position/quaternion; child BALL rotation at the native anchor. Its DOF construction at lines 209-228 uses world translation and child-frame angular axes; lines 1825-1837 treat BALL's three velocity columns together.
- [MuJoCo 3.3.7 native integration](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_support.c#L498-L536): integrate every joint using its own qpos and DOF addresses.
- [MuJoCo quaternion update](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_util_spatial.c#L214-L224): normalize input and right-multiply the local angular increment.
- Local Newton 1.7.0 solver_mujoco.py:7250-7362 maps FREE and BALL to native joints and preserves their distinct coordinate widths. Local file: C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/.venv-newton-1.7.0/Lib/site-packages/newton/_src/solvers/mujoco/solver_mujoco.py. This is a mapping reference, not a Newton speed or whole-solver parity claim.
- Fixed mujoco-warp 0.0.2 smooth.py is a secondary implementation reference; MuJoCo 3.3.7 CPU fields and actual calls are the independent runtime oracle. Source lineage remains Apache-2.0; existing LICENSE/NOTICE remain.

## 3. Native metadata and reusable mechanics

TyxFloatingModel.native_joint_layout defaults false. Only FREE+BALL selects true.

For true:
- joint_ids[njnt,4] = (kind, native_qadr, native_dofadr, native_body), kinds SLIDE=0/HINGE=1/BALL=2/FREE=3.
- body_joint_range uses native joint ordinals, with root handled by the existing root branch.
- dof_ids[nv,3] = (rotational, scalar_qadr_or_-1, native_joint). FREE owns six columns, BALL three, scalar one. All three BALL axes are the columns of the same final child rotation.
- ancestors preserves every native physical body, world/fixed markers and all root/descendant columns. No coordinate aggregation or same-position identity merging.
- vectors has independent qpos width max(8,nq); damping/armature/spring arrays remain DOF-indexed. Full qpos0 is bound.
- Root linear qvel is world-frame body-origin velocity; angular qvel is local. Child BALL velocity is local angular velocity. Anchored transport retains parent angular cross terms and COM offsets.
- Full M and body J include root/child/branch cross terms. Physical M computes published acceleration; M+hD computes Euler velocity update. Existing factorization, finite checks and transaction remain.

For false, old joint_ids/ranges stay unchanged. dof_ids is added as an equivalent read table; scalar qadr still equals its former i+1. No extra kernel or alternate force law.

## 4. Exact device ABI and parent seams

geometry signature remains unchanged. Add child kind2 behavior only; root branch unchanged.

mechanics keeps its argument order except:
- its existing joints argument receives dof_ids;
- immediately after act_values add the same fixed-tree five arguments:
  transmission_enabled, transmission_offsets, transmission_dofs, transmission_moment, transmission_fields.
Flag zero retains the scalar arithmetic order. Flag one sums the complete CSR once in actor order and records actual clamped control/raw force/final force.

solve retains all original arguments through status, then appends:
native_joint_layout, nq, njnt, joints.
Flag zero preserves original root/scalar update. Flag one copies all nq, keeps the original FREE update and integrates each child native joint once. It never uses quaternion qpos as a spring length or motor length.

Floating allocation/download uses the existing fixed-tree transmission buffer names and joint_transmission_fields[W,nu,3]. Three launches remain. No activation bank for the new domain.

compile_joint_transmission gets keyword capacity=(8,10), accepting only exact builtin tuple/int values (8,10) or (35,45). Floating structural admission must complete before the latter is supplied. nu32/nnz96/complete moment FP32 checks do not change.

Parent-owned seams:
- _model.py: FREE+BALL selects compile_floating_model before fixed-BALL dispatch; no fallback retry.
- runtime capability labels: free-ball-no-contact-v1 / free-ball-joint-motor-no-contact-v1.
- public transmission uses existing typed class with new schema native-free-ball-joint-motor-transmission-v1 (nv35); old schema retains nv8. Original entry control/dynamics/force bindings and checkpoint exact schema validation remain.
- normal/constraint kind3 callers pass dof_ids, five mechanics parameters and four solve tail parameters. Existing no-BALL routes set false; new BALL contact remains rejected.
- Initial/restore quaternion checks cover every native FREE/BALL qadr; FP64 input unit gate and FP32 scratch unit gate stay unchanged. Root clock hi/comp and accepted bank lifecycle remain.

## 5. Ownership

Author MOD only: mujoco_tyx/_floating_model.py, _floating_kernels.py, _floating_device.py, _joint_transmission.py.
Author NEW: tests/test_mujoco_tyx_free_ball.py, tests/test_mujoco_tyx_free_ball_transactions.py, tests/installed/check_free_ball.py, this design and p65-free-ball-composition-report.md.
Parent owns _model/runtime/public receipt/checkpoint/exports and normal/constraint callsites. No other product file, dependency, pin or old test changes by this author.

## 6. Frozen cases and evidence

All continuous cases use h=.001, sparse world keys (3,29), actual nonzero root and child velocities, eccentric COM, anisotropic inertia and nonidentity quaternions. Controls have fixed forward/reverse/unload phases (ticks 0-7,8-15,16-23); no runtime tuning. Fixture source is authored before product changes.

| Case | Distinct property |
| --- | --- |
| passive | FREE+BALL, no motors, real generalized/body loads |
| oblique | one BALL, complete signed 3-column motor |
| shared | shared BALL target, cancelling/zero columns retained |
| mixed | FREE+BALL+HINGE+SLIDE, native qadr offsets, scalar affine law |
| branches | two independent BALL branches, full off-diagonal M |
| nested | BALL-HINGE-BALL-SLIDE chain, all ancestor columns |
| clamps-disabled | native control/force bounds, disabled group and zero gear |
| sign | independent ± quaternion representatives with original physics |
| capacity | 9 BALL +2 scalar = nv35/nq45, 32 actors; bounded two-step case, not scale qualification |
| transaction | full checkpoint bytes, clone and true next step, partial reset; late status/readback and invalid last-world control; wrong saved schema rejected |

Independent same-entry native CPU forward/fullM/jac/body inertial fields plus analytic complete JOINT moment/force/power; continuous CPU data advances from its own original state with the same stored FP32 control, never follows TYX state. CPU mj_step is forbidden only around actual TYX step. Every raw accepted receipt is saved before comparisons, first failures preserve entry/native/control and returned state.

Original floating gates are reused without enlargement: position/R/COM rtol3e-5/atol4e-5; J 2e-4/5e-5; M 3e-4/5e-5; bias/passive/applied/actuator 4e-4/4e-4; physical/integration acceleration 2e-3/4e-3; continuous q1e-4/v1e-3/time2e-6 absolute. Complete transmission retains original motor force/power gate 3e-4/2e-4. No exact FP64-vs-FP32 trajectory demand and no lowered mass/changed gate after failure.

Before-device negatives: activation, contact potential (even initially separate), any limit/dry row, BALL spring/same-body, multiple roots, FREE target, BALL affine bias, nu33/nv36, incomplete/tiny y/z gear, invalid CSR capacity. Invalid quaternion initial/restore and malformed control are transactional negatives.

Declared capacity is separate from measured scope: W2, eight 24-step structural cases and one two-step capacity case; public installed consumer uses two complementary cases plus lifecycle, not a whole new matrix. Maximum W is not qualified here.

## 7. Completion and stopping

Tests-first -> source -> parent review -> one serial static/interpret/CUDA queue -> narrowly affected legacy wrappers -> new wheel consumer. No independent author execution or rerun of unchanged accepted groups. Record source/product/reference/environment failures separately. Two investigations of one hypothesis without progress produce a minimal saved counterexample and stop that hypothesis; do not tune inputs, tolerances or solver budgets. Activation/contact combinations remain explicitly pending after this slice, as does heterogeneous multi-model scheduling.

