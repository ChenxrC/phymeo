# ROB2b: compose native activation with free roots and constraints

Status: proposed executable contract, authoring only. Baseline `a18305cf716cf67cbfdd745df5e91ac730bba93e`; parent executes every project command. Deadline: 2026-11-12 07:20 Beijing, no new long run after 06:20. ROB2a is already accepted; its history is not reopened.

## Mechanism and unchanged boundaries

Reuse the actual ROB2a `_actuator_kernels.update` without changing its law or arithmetic. Preserve NONE/FILTEREXACT, fixed gain, NONE/AFFINE bias, native control/activation/force clamp order, actearly and disabled-group semantics. All actuator targets must explicitly be native HINGE or SLIDE joints; reject FREE/BALL targets even if only the first gear component is nonzero. Keep 32 actuator / 32 activation limits and the original scalar JOINT transmission restrictions. No tendon/site/closed-loop, new cone, time integration or solver is introduced.

The current `_model.compile_model` already routes `na>0` to the dedicated activation compiler. That compiler will select the original mechanical/contact plan by source structure, in this order:

1. Original `constraint_pairs` / `needs_constraints`: existing T4 (limits, dry friction, pyramidal contact, excludes or larger admitted candidate layout).
2. Otherwise original `possible_pairs`: existing T3 condim1 geometry/normal solver.
3. Otherwise one free root: original T2.
4. Otherwise original T1.

Each chosen compiler keeps every previous admission gate and capacity, adding only an explicit `allow_activation` keyword to forward the NONE/FILTEREXACT admission. No failed lowering is retried through another route. Fixed T1 without contacts keeps its existing activation plan class. Other combinations use a frozen activation wrapper with `base` forwarding; all body/joint/geometry/row data remain the original plan. Native activation address coverage and source parameter binding remain complete and immutable.

The sequence for every new combination is:

    accepted q/v/act -> original geometry -> original mechanics(nu=0)
      -> one common device activation stage -> original contact/constraint stages
      -> original Euler solve/integration -> existing all-world publication

The stage adds its full generalized force to the already assembled original external load, before the solver constructs its RHS. The physical M, complete J, bias, passive forces, constraints, root angular convention and original force/torque channels are not replaced. Root generalized actuator entries remain zero for joint-only actuators; root motion still responds through full root/link coupling. No per-stage publication or HOST activation update occurs.

A shared `_actuator_device.py` will centralize validation/allocation/launch/download. It uses the existing kernel and device counters. All four device routes call it exactly once when activation exists and not at all for arithmetic/allocations when na=0. New launch counts are T2=4, T3=6 (7 with extended geometry), T4=8 (9 with extended geometry); prior stateless counts remain unchanged. T1 remains four. Buffers are still bounded by admitted native dimensions and owned by the original runtime resource stack.

## Shared state and parent seam

Existing runtime pack/build, input-act binding, step publication, integration_snapshot and checkpoint v2 are generic in `plan.activation`; they require no new state schema or integration formula. Normal/constraint device constructors must accept and forward the existing `activation=` keyword, and floating download must return the same actual fields as T1. All act/q/v/time buffers share the original active index and single publication.

Parent patch only: make runtime capability domain describe the selected T1/T2/T3/T4 combination rather than unconditionally saying fixed T1. Preserve the old fixed-T1 domain string, checkpoint v2 and all original solver/contact metadata. `_model.py` and `mujoco_checkpoint.py` are expected unchanged; any discovered need is reported before another shared edit. Parent's current observation/training work remains exclusive.

Checkpoint, clone and selected reset must use accepted native activation bytes, not recompute from controls. Failure after real candidate work or late restore replacement must retain all q/v/act/time/held-control/last-receipt channels. Operational counters remain monotonic and are not inferred from restored ticks. New combinations retain the original per-step complete dynamics/contact/constraint and actuator evidence.

## Exact file ownership requested

Author modified files (8):

- `src/phymeo/backends/mujoco_tyx/_actuator_model.py`
- `src/phymeo/backends/mujoco_tyx/_floating_model.py`
- `src/phymeo/backends/mujoco_tyx/_normal_model.py`
- `src/phymeo/backends/mujoco_tyx/_constraint_model.py`
- `src/phymeo/backends/mujoco_tyx/_tree_device.py`
- `src/phymeo/backends/mujoco_tyx/_floating_device.py`
- `src/phymeo/backends/mujoco_tyx/_normal_device.py`
- `src/phymeo/backends/mujoco_tyx/_constraint_device.py`

Author NEW files (4): `_actuator_device.py` in the same package; `tests/test_mujoco_tyx_activation_composed.py`; this design; `docs/development/p65-rob2b-actuator-composition-report.md`.

No `_tree_kernels.py`, `_floating_kernels.py`, `_actuator_kernels.py`, geometry or solver modifications. No old results rewritten. Parent subsequently authorized the exact compatibility update to `tests/test_mujoco_tyx_activation.py`: retire only the old free-root and scalar-limit unsupported parameters and their dead fixture branches; all remaining negative/positive gates remain unchanged. This adds one old-test MOD to the author allowlist. Parent alone applies a reviewed ignored-artifact runtime capability patch; no direct child edits to `_mujoco_tyx_runtime.py`, `_model.py` or `mujoco_checkpoint.py`.

## Predeclared independent cases

All sources are self-contained native MJCF with Euler h=0.001, original Newton solver options/budget where constraints exist, nonzero initial velocity and activation, actual raw held controls and external loads. Tests are authored before any execution. Both interpret/CUDA run the same cases. No fixture-name branch enters product code.

| Case | Fixed scope | Required evidence |
|---|---|---|
| B01 free driven tree | One free root + R/P links, off-center inertia/nonidentity pose, 64 steps positive/reverse/unload | nq=nv+1, full root/link M and J, actual root response, mixed NONE/FILTEREXACT, negative/shared gear; both actearly values |
| B02 normal support | Fixed scalar vertical slider with sphere/plane condim1, initial closed soft contact, 96 steps with loading then unloading | T3 route, nonzero normal support, changing real actuator load and decay during zero-control tail, full normal J/force and step energy diagnostics; no forced nonpenetration claim for native soft contact |
| B03 floating support + driven link | One free root, a scalar driven link, sphere/plane contact, 64 steps | T4 condim3 and condim6 variants, full COM wrench / generalized constraint force, actual base/link motion, actuator force included before the contact RHS |
| B04 native joint constraints | Fixed R/P, finite scalar limit initially active plus dry friction, 64 steps including reversal | T4 with no possible contact pairs, nonzero limit/friction rows, full source row identity and current force; distinct clamp ranges and disabled-group positive force-range corner |
| B05 mixed free constraints | Free root + limited/frictional driven scalar link and condim3 contact, 64 steps | Simultaneous contact/limit/dry rows at an authored entry, complete current M/J and RHS; no dropping inactive/static slots |
| B06 dimension bound | Free root + 29 scalar links, 29 FILTEREXACT actors, W1, 4 steps | nv35/nq36/na29 address and allocation bounds, nonzero per-link input, same-entry native full-state/force check; not a long-life capacity claim |
| B07 transactions | Sparse W2 on B01 and B05; 4 real steps, checkpoint -> successor -> restore -> successor, clone and partial reset | Exact same-backend accepted bytes and successor; input-act corruption and last-world downloaded status failure retain the whole publication; late replacement readback failure retains destination, actual retry succeeds |
| B08 admission / compatibility | FREE/BALL actuator target; existing unsupported dynamics/capacity; one na0 control per newly touched device route | Reject invalid source before device allocation. Existing na0 behavior/launch count and T1 activation short compatibility are checked narrowly, not by rerunning unrelated old suites |

B02 uses the already qualified native sphere/plane scales (r=.1, entry center=.099, solref=.02/1, solimp=.8/.9/.002/.5/2, mass1) and an actual filtered scalar actuator. B03/B05 use the same simple sphere/plane geometry, not new manifold fixtures. B04 starts outside one declared scalar soft limit to exercise an actual row; initial-state repair is forbidden. Exact authored parameters/control schedules and enumeration belong to the tests-first commit and are then frozen before execution. Failure is retained rather than adapting geometry/gains/tau or reducing the requested matrix.

## Oracles, gates and saved result

Independent CPU MuJoCo **3.3.7** compiles the original XML. Same-entry forward/step uses actual accepted q/v/act and FP32-stored raw controls. A second native trajectory initializes once and never follows the device after that. Independent analytic exponential/gear/force computation reads original native source fields, not product lowered arrays or producer helpers. Entry forces and endpoint observations remain labelled by phase. Constraint/contact J, R, aref, forces and source identities must be compared; a small product KKT value alone cannot prove the same physical problem.

Reuse ROB2a activation rtol3e-5/atol3e-6 and actuator-force rtol3e-4/atol2e-4. Reuse original T2/T3/T4 gates: M rtol3e-4/atol5e-5, bias/passive/applied 4e-4/4e-4, geometry 3e-5/4e-5, J 2e-4/5e-5; normal/constraint force 3e-3/3e-3, constrained acceleration 3e-3/1e-2, EFC row values 5e-4/4e-4. Free no-contact continuous q/v absolute1e-4/1e-3; constrained short windows2e-4/3e-3; time2e-6. Preserve original normalized solver KKT gate=max(native tolerance,2e-5) and source iterations unchanged. No new precision demand and no relaxed physical gate.

Save full source/initial/held packet, every actual entry and accepted receipt, native/analytic references, full contact/EFC bindings and all before/after/failure/retry evidence. Independent native computations run only in the parent's serial test slot. No producer rerun merely to obtain saved counters. Constructor/source negatives are identified as assertions, not invented physical trajectories.

## Risks, bounded order and stop rule

Primary implementation risks are qpos-vs-DOF addresses after the free quaternion; accepting a FREE actuator as a scalar gear; double actuator addition in nested device routes; activation candidate leaking on late constraint rejection; and wrapper delegation losing a geometry/constraint flag. Each has an explicit case or structural source assertion. Transport-only refactor must retain original T1 raw bank and launch semantics.

Implement B01/route/device first, then B02/B04, then B03/B05, transaction/capacity qualification. First functional run is a small cross-route smoke, then the complete authored matrix under the same gates. No G1 long replay. If a new combination exposes an existing solver boundary, retain the concrete input and stop that combination; do not rewrite or tune the solver in this grant. Two revisions with no new diagnostic information trigger a concrete handoff, not further parameter search. Whole-ROB2b acceptance requires both backend matrices and the parent's final integrated verification, not this document or a successful constructor.

Fixed behavioral sources: [MuJoCo 3.3.7 forward/actuation/advance](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c), [actuator computation](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html), and the already reviewed local T1/T2/T3/T4 source. ROB2a law and Apache-2.0 provenance remain unchanged.
