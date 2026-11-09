# H1 bounded 29-joint resident capacity and passive actuators

## Parent execution, 2026-11-09

The seven focused cases passed in both TYX interpret (50.79 s) and CUDA
(8.52 s). They include full 35-coordinate dynamics for an actuated 29-joint
tree, N=32 storage/bounds/restore, N=33 rejection, and actuator identity.
Two adjacent ordinary/affine floating response cases additionally passed per
backend after the shared gain-width change. Exact-file Ruff checks passed;
the parent formatter changed five author files. Evidence is in
`artifacts/p65-h1-20261109-01`, tasks 09–12 and 16–17.

This accepts the bounded device mechanics and declarations exercised here;
it does not establish G1 balance, device foot friction, unbounded capacity or
large-environment throughput. Nonzero affine actuator terms remain unsupported.

Actual G1 validation subsequently passed in tasks 27–28: frames 0 and 200,
29 revolute joints / 30 bodies / 35 velocities, each advanced independently by
one physical step in both interpret and CUDA. Full M/b/continuous acceleration,
q/v, all-body FK/twists and kinetic energy matched HOST within the predefined
FP32 comparison gates. On CUDA the maximum configuration error was `6.36e-8`
and velocity error `5.48e-7`; maximum continuous-acceleration error was
`2.17e-4`, within its relative-plus-absolute gate. This is implementation
agreement, backed by the separate HOST/native mass/bias checks; it is not an
independent native trajectory or contact validation.
The following author handoff is historical.

**AUTHORING_ONLY / UNEXECUTED.** This is an implementation/test handoff at base `666dd66184869ff36401551abcd0904e7564128a`, branch `codex/h1-device-capacity`. No project import, collection, numerical calculation, pytest, lint, formatter, compile, build, install or profile was run. Only source inspection, file IO and Git checks were performed. The parent owns the single execution queue and the HOST actuator model integration.

## Scope and storage contract

The checked maximum is **32 physical scalar joints**, with no humanoid name or geometry special case. Ordinary fixed trees require `1 <= N <= 32`; ordinary free-root trees allow `0 <= N <= 32`, with `B=N+1`, `D=N+6`, `Q=N+7`. Both affine declarations require `1 <= K < N <= 32`. Tree group bindings use `[0,N)`; floating body bindings use `[0,N+1)`, including group 32 when N=32. Both device current-pose bounds kernels and floating packet/diagnostic readback gates use the same cap.

Existing Jacobian, mass, Cholesky, inverse, state and FK arrays already derive their dimensions from each batch's actual N/B/D. They do not use a fixed 22/23-DOF scratch bank. The fixed-tree property width **16 is a data layout, not a joint capacity**, and remains unchanged. Floating properties now have width 18: the original 16 columns followed by physical-joint armature and passive damping, stored on child body rows 1..N. Floating motor gains have width 4: `(kp,kd,effort_cap,passive_damping)`. Root rows carry no actuator. The floating-affine gain bank also has width 4 with passive column zero.

The ordinary floating model copy retains the HOST `joint_armature` and `joint_damping` immutable FP64 arrays. Compiled FP32 arrays independently reject unsupported/nonrepresentable values through the existing normal-or-zero conversion gate. Nonzero actuator arrays extend the original physical model token, so checkpoint/material topology cannot alias models that have the same geometry and different actuators. Explicit all-zero arrays retain the original token exactly.

## Dynamics and diagnostic meaning

For the physical M, the kernel adds each declared armature to its joint diagonal. Physical kinetic energy includes `0.5 * armature * joint_velocity**2`. The physical continuous bias includes `D * v_old`, where D has zero root coordinates and the declared joint damping diagonal.

The actual velocity increment uses the same HOST contract:

```text
(M + h D) delta_v = h (applied - physical_bias)
v_new = v_old + delta_v
```

The bounded PD iteration uses the motor block of the inverse of this full coupled matrix; it does not damp the free root independently or replace the root Schur response. The equation residual includes `h D delta_v` and the step matrix row norm, preserving the existing acceptance threshold `2e-5`. The published diagnostic mass remains physical M, and diagnostic acceleration remains `solve(M, applied-physical_bias)`. A separate reuse of the Cholesky scratch after fixing the candidate velocity computes this continuous acceleration when passive damping is present. The mobility bank retains the implicit step inverse for the actual drive solve. Armature/damping zero paths skip the extra arithmetic and preserve the previous token and dynamics path.

Floating affine models with any nonzero armature/damping **explicitly reject during declaration**, before device allocation. General reduced actuator matrices need `R.T A R` and `R.T D R`, which can be dense and cannot be represented by the ordinary motor diagonal. Zero-actuator affine models remain supported; this handoff does not silently discard physical actuator terms.

## Authored acceptance nodes

The new test file is [test_resident_humanoid_capacity.py](../../tests/scale/test_resident_humanoid_capacity.py). The following is a handwritten inventory, not pytest collection or a pass count.

| Node | Difficulty | Independent expectation and behavior |
|---|---|---|
| `test_h1_fixed17_binary_tree_mass_and_fk` | medium | A branched 17-joint prismatic tree. The complete incidence matrix gives `M=J.T J`; checks full mass/acceleration, physical velocity and every body's FK against the authored ancestor sum. |
| `test_h1_floating_capacity_step_and_material_restore[g29-actuated]` | high | 29 physical prismatic branches with nonzero armature and unequal damping. Complete 35x35 block mass formula, continuous acceleration versus implicit velocity update, energy, N/B/D/Q, last-body material bounds, stale observation rejection and selected-world restore isolation. |
| `test_h1_floating_capacity_step_and_material_restore[bound32-zero]` | medium | Hard maximum N=32, B=33, D=38, Q=39; actual step, last-body bounds and selected restore. Zero actuator case exercises compatibility. |
| `test_h1_joint33_rejected_before_runtime[fixed]` / `[floating]` | low | Public declaration/compile gate rejects N=33 before runtime allocation, retaining actual error text. |
| `test_h1_actuator_copy_token_and_checkpoint_binding[joint_armature]` / `[joint_damping]` | medium | HOST-only declarations check immutable copy/FP32 packing, explicit-zero token compatibility, changed actuator topology, and cross-model checkpoint rejection. |

Device cases save the complete declared model, authored expected quantities, actual accepted state, diagnostic, FK, bounds and resource statistics directly to gzip JSON. Array dtype/shape/raw bytes and values are retained. Broadphase owner and registration are process-local opaque objects; the serialization explicitly records that their process identity cannot be restored, while live assertions check stale epoch ownership. These are device mechanics and material-bounds tests, not foot support, narrowphase or full G1 dance acceptance.

## Parent integration and execution

Required adjacent integration, deliberately outside this lane:

- Merge the HOST `FloatingRobotModel` keyword-only actuator fields, physical mass/bias and implicit damping helper from the actuator lane before running the actuated or token cases.
- Raise shared `_resident_contact_bridge.py` Site group gate to 32 (owned by parent).
- Keep runtime/contact response metrics consistent with physical M versus actual implicit step response. The parent owns `tyx_world_batch.py`, shared runtime and H1 policy/foot support integration.

After merge and a fresh execution guard, the parent may start with the HOST identity/33 rejection nodes, then fixed17, actuated29 and boundary32. Do not invoke any command concurrently with another project's import/test/compile activity. Example full focused command (replace basetemp with a newly absent absolute path):

```text
python -m pytest tests/scale/test_resident_humanoid_capacity.py --require-tyx --tyx-mode=interpret -q -x --tb=short -p no:cacheprovider --basetemp=<fresh absolute path>
```

CUDA uses the same exact test file with `--tyx-mode=cuda`, in the same sole serial queue after the interpreted result is known. The parent must retain first lint/format/test failures and final source bindings. The author has not run any of these commands. Existing small ordinary/floating/zero-affine cases should be selected by the parent to verify the altered shared gain width without repeating already accepted unchanged groups unnecessarily.

## Changed paths and limits

Modified backend modules: `_resident_tree_world`, `_resident_tree_binding`, `_resident_affine_tree`, `_resident_floating_tree`, `_resident_floating_storage`, `_resident_floating_binding`, `_resident_floating_contact`, `_resident_floating_affine_tree`, `_resident_floating_affine_storage`, `_tyx_resident_tree_bounds`, `_tyx_resident_floating_bounds`, `_tyx_resident_tree_kernels` (capacity documentation only), `_tyx_resident_floating_kernels`. Two new files are this report and the focused test file. No shared runtime, export, pin, native dependency or previous test was edited.

No result in this document establishes compiled TYX support or a performance guarantee. The device loop/storage increase, actuator kernels and negative gates need parent serial validation. Dense mechanics remain bounded O(D^3); this task does not claim a scalable sparse solver, stable locomotion or sustained foot support. Physical rotor inertia here is the declared scalar reflected joint model, not a new independent motor-body inertia model.
