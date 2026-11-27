# Original M16 forces and scalar constraints on TYX CUDA

The instantaneous chain now generates noncontact forces, free acceleration,
polynomial joint equality, DOF dry friction and scalar limits on CUDA, and feeds
them to the existing contact solver. It no longer uploads captured native
`smooth`, `smooth_low` or scalar constraint rows as production inputs.
This is a fixed-state component, not an integration or learning release.

`compile_smooth_model` compiles original static fields. `SmoothWorkspace`
consumes a current `RigidResult`, controls, generalized applied forces, body-COM
world wrenches and equality activation. Its device API downloads only one status
word per world for admission. Full output readback is explicitly diagnostic.
NumPy is used for static compilation, host convenience uploads and independent
verification; no native forward or CPU state solve supplies the production path.

## Supported semantics and ownership

- Scalar-joint fixed-gain, none/affine-bias actuators without activation.
  Controls are clamped first, then actuator forces, then the transmitted sum at
  each joint. Negative gears and multiple actuators per joint are supported.
- Scalar joint springs, DOF damping, generalized applied forces and world-frame
  force/torque at body COM. World-body wrench has no effect but must be finite.
- The physical mass solve uses a FP32 Cholesky preconditioner and compensated
  residual refinement, retaining high/low acceleration. It does not add an
  implicit damping diagonal or claim IMPLICITFAST support.
- Joint polynomial equality (including a single joint against a constant),
  per-world activation, DOF dry friction and both sides of scalar limits retain
  native row order, inverse-weight convention, reference and impedance laws.
- Flexible/plugin/tendon/activation/fluid/gravity-compensation and unsupported
  actuator/equality domains are rejected. Simulation flags must remain default.
  This does not relax the existing whole-model compiler's integration gates.
- State inputs have exact schemas. Invalid inputs expire the previous result;
  borrowed buffers cannot be overwritten or closed. Smooth outputs own their
  mass and rows, so upstream workspace reuse cannot silently change their epoch.
  Rigid qpos and qvel are also owned snapshots.

## Precision defect found by composition

Initial per-component and 24 targeted checks passed. The new composed result
nevertheless failed independent original force balance in all eight states:
the maximum normalized residual was 4.5139073e-5 against 2e-6. The acceleration
comparison alone would not have caught the problem.

An explicit decomposition found physical mass rounding multiplied by large
free acceleration contributed up to 9.6375057e-5 generalized-force defect;
bias/force rounding contributed up to 5.2137937e-6. These eight original reset
states have zero velocity. This was not a solver convergence failure.

The fix retains compensated FP32 low lanes through original static mass,
inertia, axes, COM Jacobians, physical mass and gravity projection, then consumes
them in the free solve. All worlds use the same path. It does not inject native
results or change a physical threshold. Velocity-dependent RNE body wrenches
remain FP32; moving-tree tests cover nonzero velocities, but these eight static
states do not establish full moving-M16 or long-run numerical accuracy.
The coupled contact solver still consumes FP32 M/J/rows and compensated free
acceleration; passing the bounded check does not imply bitwise native equivalence.

## Actual verification

42 checks passed with CUDA required: 25 new smooth/constraint cases and 17
affected rigid cases, including nonzero controls/passive/external force, combined
clamps, negative gear, polynomial derivatives, activation, both limit directions,
direct and time-constant solref, low-lane precision, resource/ownership and invalid
states. Earlier overlapping test counts are not added. Ruff/format checks pass.

The same eight original M16 saved states (17 bodies, 15 DOFs, 1,939 geometries,
1,930 meshes) were independently checked against native FP64 forward outputs:

| Check | Result |
| --- | --- |
| All force/free-acceleration/scalar component criteria | Passed, 8/8 |
| Free acceleration maximum absolute error | 7.0031092e-11 |
| Refined mass solve maximum normalized residual | 5.1701358e-15 |
| Scalar rows | 3/world; IDs, kinds and J match |
| Contact set | All 4,154 match; no additions/missing pairs |
| Contact point differences above 5 micrometres | 0 |
| Coupled iterations | 9–10, all converged |
| Original FP64 force balance maximum | 2.2809947e-7, below 2e-6 |
| Constrained acceleration maximum absolute error | 3.7641360e-5 |

Generalized acceleration has mixed coordinate units. The inherited componentwise
acceptance `.002 + 5e-4 * abs(native)` also passes; the initial local plan recorded
a looser `.005` constant, so final verification explicitly retains the stricter
inherited criterion. Force-balance tolerance is unchanged. Failed first results
and the decomposition remain in the external evidence bundle.

The native contact archive serves only as an independent comparator. Current
qpos/qvel/mocap/control/applied state enters the new GPU chain. The unchanged box
kernel uses a hash-verified compiled cache from the parent package; old cache
locks are preserved. No new AV was observed, and no old AV root-cause fix is claimed.

No time integration, policy samples, PPO updates or training checkpoint changes
occurred. Next work is original IMPLICITFAST and transactional state integration,
followed by resident PD, observations/rewards, termination/local reset and rollout
interop with the existing model/Adam/RNG lineage. Component success does not
authorize an old runner or establish 32/256-environment learning qualification.

Source semantics: MuJoCo 3.3.7
[engine_forward.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c),
[engine_passive.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_passive.c),
[engine_core_constraint.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_constraint.c).
See the package NOTICE for attribution.
