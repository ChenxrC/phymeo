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

## Source implementation checkpoint

The four authorized product files now implement the same-stage CSR scatter. M/J/bias,
physical-M and Euler solves, quaternion integration and accepted-bank commit are unchanged.
The legacy actor block is preserved with only an enclosing flag-zero branch.

New device storage is four tracked buffers:
4*(nu+1) + 8*nnz + 12*W*nu bytes, excluding the existing mechanics storage.
This is exact declared capacity, not measured live memory. Only the actual three-column
field adds download bytes (12*W*nu) per step; static tables stay source-bound on HOST.
No new kernel or activation state is introduced. Buffer owners use the existing resource ledger.

Two formerly unsupported declarations are explicitly promoted by this scope: BALL1
ROTOR plus direct motor gear(1,2,3), and BALL-A's stateless source with its original
direct hinge motor retargeted to BALL. The NEW file preserves each exact XML/default
entry and checks two native steps in both modes (4 extra nodes). Parent alone retires
the corresponding old negative parameters. BALL-C's contact+BALL-motor negative remains.
Updated authored enumeration: 45 nodes; not a collection or execution result.

Parent text review found that the initial new helper only rejected FP32 overflow.
The existing scalar representation contract also rejects every nonzero coefficient whose
FP32 conversion is subnormal/zero. The helper now calls the same fp32 gate on every
complete native moment row before allocating final tables; the local import avoids a module
initialization cycle. Fixed y/z tiny-gear negatives (1e-45 with x=1) exercise the previously
uncovered columns before device allocation. No numerical gate changes. Authored count is now47.
