# R1: direct native JOINT activation composition

Status: AUTHORING_ONLY; no import, collection, test, numerical probe, lint or build run by this lane.

## Frozen scope and ownership

Fixed-root BALL/HINGE/SLIDE with direct JOINT transmission, NONE/FILTEREXACT,
fixed gain, BALL BIAS_NONE; original scalar NONE/AFFINE bias remains.
Reuse the existing tree, normal-condim1 and pyramidal-condim3 mechanics/contact
solvers, physical M, damped Euler solve, quaternion integration, double banks,
and common batch publication. No FREE, BALL affine bias, limits, dry friction,
tendons, activation plugins, new contact primitives or changed tolerances.

Author MOD only _tree_model.py, _tree_device.py, _actuator_model.py,
_actuator_device.py, _actuator_kernels.py under src/phymeo/backends/mujoco_tyx/.
Author NEW tests/test_mujoco_tyx_ball_activation.py,
tests/test_mujoco_tyx_ball_activation_transactions.py,
tests/installed/check_ball_activation.py and this design/report.
Parent owns _model.py, runtime, public records, checkpoint and exports.

## Actual ABI, force law and shared publication

Append to existing activation update after status:
transmission_enabled:i32, transmission_offsets:i32[1],
transmission_dofs:i32[1], transmission_moment:f32[1],
transmission_fields:f32[3]. Flag zero retains the old scalar arithmetic/order.
Flag one visits the same immutable native-address CSR used by direct motors:
one scalar or all three BALL columns, including zero gears and shared targets,
in native actor order. It sums each force once into the original actuator and
applied vectors. BALL qpos is never used as an actuator length.

FILTEREXACT keeps ctrl clamp, raw act_dot, disabled integration-zero, exact-filter
candidate, act clamp, actearly selection, gain/bias, then force clamp. Disabled
raw act_dot may be nonzero; disabled raw force is zero before positive lower force
clamping. tau_eff=max(mjMINVAL,tau). All original finite and FP32 gates remain.
No new physics stage: existing activation adds one launch to each original route
(tree 4, normal 6, pyramidal 8 launches for these primitive fixtures).

Parent adds MuJoCoTyxJointActivationStepReceipt in mujoco_joint_transmission.py,
inheriting MuJoCoTyxActivationStepReceipt and adding transmission.
Actuation schema is native-joint-filterexact-actuation-v1 only for a CSR plan;
original scalar schemas do not change. The two records bind the same actual
FP32 clamped_control/raw_force/scalar_force/generalized_force and original ctrl.
Checkpoint v2 preserves actual act banks, accepts the combined type explicitly,
and rejects either record independently altered. Original stateless bytes remain.

Expected capabilities:
- fixed-ball-joint-motor-filterexact-no-contact-v1
- fixed-ball-joint-motor-normal-contact-filterexact-v1
- fixed-ball-joint-motor-pyramidal-contact-filterexact-v1

## Fixed cases, independent oracles and stop rule

W=(7,41), h=.001, 24 continuous steps, nonzero initial v and FP32 act.
Commands: 8 forward, 8 reverse, 8 unloaded steps; original qfrc/xfrc removed only
for the last eight. No intermediate state copy from the native reference.

Eight sources: late and early eccentric no-contact BALL; original MC1 normal;
MC2 pyramidal; MC3 mixed H/B/S and scalar affine bias; MC4 shared negative/zero
BALL gears with mixed NONE/FILTEREXACT; MC5 two moving BALL branches plus slides;
MC6 disabled BALL with positive force lower clamp. Original contact geometry,
mass, damping, h and entry configurations remain. FILTEREXACT tau=.004 except
shared zero actor tau=.007; early cases explicitly selected in source.
Each filter has ctrlrange [-.45,.45], actrange [-.3,.3], initial act
(-1)^address*(.34+.01*worldslot+.005*address). Native disabled integration
still applies the original subsequent act clamp.

Per actual entry, independent native mj_forward/step supplies full M/J/bias,
passive/applied/actuator/constraint force, contact and EFC fields. Independently
evaluate native address CSR, stable expm1 filter and scalar/three-axis power.
Keep an independent continuous CPU reference. CPU forward/step is forbidden only
inside actual TYX step; independent oracles run outside that guard.

Original gates: act rtol 3e-5/atol 3e-6; actuator rtol 3e-4/atol 2e-4;
contact force rtol/atol 3e-3; contact acceleration rtol 3e-3/atol 1e-2;
contact q/v absolute 2e-4/3e-3; no-contact acceleration 5e-4/8e-4 and q/v
5e-5/3e-4; time 2e-6. Full geometry/M/J/bias gates copied unchanged from MC.
No tolerance, source iterations, capacity, mass or control tuning to pass.

Transactions cover all three routes' checkpoint/clone/partial reset successors;
normal/pyramid status, full readback, last-world actuation/transmission publication
and candidate-verified restore failures; independent single-record checkpoint
tampering must retain complete accepted state and match an uninterrupted clone.
Pre-allocation negatives keep FREE/limits/dry/springs/unsupported cone/dimension,
BALL affine bias/non-JOINT target/non-FILTEREXACT dynamics/capacity.

Installed consumer is self-contained: no tests/helpers/checkout imports.
It runs three 24-step paths and one eight-call real lifecycle, records provenance,
full inputs/native and actual receipts, and preserves primary plus cleanup errors.

Success requires parent serial interpret/CUDA source qualification, meaningful
affected compatibility, then fresh wheel public consumers. Authoring is not PASS.
A failed unchanged hypothesis after two diagnosed revisions becomes a concrete
remaining issue; do not change gates or start unbounded trajectory tuning.

## Primary fixed semantics

MuJoCo 3.3.7 Apache-2.0:
[actuation and Euler activation](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c),
[native JOINT transmission](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_smooth.c).
The existing accepted local _joint_transmission.py, scalar activation stage and
BALL-MC native tests are implementation evidence, not the independent oracle.
