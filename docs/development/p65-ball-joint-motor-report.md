# BALL-M1 direct native JOINT motor transmission

Status: S0 TESTS_AUTHORED_UNEXECUTED. Parent alone owns execution and shared routing/publication.

## Frozen interface

`TyxTreeModel.joint_transmission: TyxJointTransmission | None` defaults to None.
The immutable static object has:
- `ids: int32[nu,6]`: actor, native joint, native joint type, qpos address, DOF address, width.
- `gear: float64[nu,6]`: original native gear, including all zero coefficients.
- `row_offsets: int32[nu+1]`, `dof_indices: int32[nnz]`, `moment: float32[nnz]`.
- Scalar rows contain one column; BALL rows contain all three native tangent columns.
  Actor order is unchanged; zero entries remain; nu<=32 and nnz<=96.

Mechanics inserts these five parameters immediately after act_values and before controls:
`transmission_enabled:i32, transmission_offsets:i32[1], transmission_dofs:i32[1], transmission_moment:f32[1], transmission_fields:f32[3]`.
The fixed-tree flag-zero callers alias existing parents/gravity/body_force buffers, allocate
nothing new, do not access aliases and retain their old arithmetic and three launches.
Floating kernels and solve signatures do not change.

New actual download `joint_transmission_fields: float32[W,nu,3]` contains clamped
control, raw scalar force, final scalar force. The existing actuator result is the complete
generalized force from this same mechanics invocation. Parent public receipt `.transmission`
uses schema `native-joint-motor-transmission-v1`, the five static arrays plus
`input_control, clamped_control, raw_force, scalar_force, generalized_force`.
Original input_control is explicitly bound to the base receipt control. No new activation state.

## Fixed qualification and gates

Predeclared main file: 11 cases x interpret/CUDA = 22 continuous nodes, each sparse worlds
(3,29), h=.001, 24 actual steps with drive/reversal/unload. Cases are x/y/z, oblique,
shared target, two ball branches, mixed scalar joints, clamps, disabled/zero gear,
quaternion sign covariance, and 32 actors. These cover the ten research groups; the
axis group is three explicit cases, lifecycle is in the transaction file.

Each step binds full native M/J/b/passive/applied, physical and Euler acceleration,
same-entry native successor and an independently advanced native trajectory. An independent
native-address CSR and scalar force law checks gear scatter and power. Native actor moment
and velocity are also read. No use of quaternion qpos as actuator length.
Original gates: force rtol3e-4/atol2e-4; q atol5e-5; v atol3e-4; time2e-6.
M/J and acceleration gates are the existing BALL table imported unchanged.
Explicit actual step forbids CPU mj_step/mj_forward. No extra TYX kernel.

Ten pre-device negatives: activation, affine BALL bias, JOINTINPARENT, nonzero gear tail,
possible contact, limit, dry friction, BALL spring, actor33, FREE.
One static node checks complete 32x3 CSR, owned raw gear and immutable arrays.
Transaction file: 8 nodes across both modes: checkpoint bytes/clone/CAS/partial reset
and actual successors; last-world status and late-readback faults after real computation;
bad last control preserves every world. Failed candidates cannot replace old receipt.
Total authored enumeration: 41 nodes, not an observed collection/result.

## Source scope and semantics

Only new joint-transmission module and the three authorized fixed-tree model/kernel/device
files change. Parent owns runtime/model/public receipt/wrapper ABI. Existing scalar-only
and scalar-actuated BALL-A/C/ROB2b paths retain their existing scope.
New direct BALL target: JOINT/NONE/FIXED/BIAS_NONE, na0; scalar NONE affine-bias may coexist.
Disabled raw force is zero before native force clamping, including positive clamp ranges.
Capacity and domain checks precede table/device allocations.

The installed consumer is a later authoring deliverable in the same allowed file list.
No import, collection, test, numerical probe, formatter, lint, build or install has run in this
lane. Tests-first is source ordering, not a fabricated RED result.
