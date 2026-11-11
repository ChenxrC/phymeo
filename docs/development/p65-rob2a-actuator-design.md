# ROB2a: connected first-order actuator state

Status: DESIGN_FROZEN_FOR_AUTHORING; implementation and validation pending. Baseline: 847b8ee359320d260e3d36c13cd0530cde7a28b9. No project execution was used for this design.

## Decision and scope

Implement a real MuJoCo-TYX T1 vertical slice: native `filterexact` activation, native scalar JOINT transmission, existing fixed gain / none-or-affine bias, actual device force assembly, accepted integration state and exact continuation. Existing stateless paths remain unchanged. This is not a HOST control-filter wrapper and does not copy the mass, bias, Jacobian or integrator implementation.

Admission is fixed-root 1..8 scalar hinge/slide DOFs, at most 128 native bodies, 1..32 actuators and 1..32 activation coordinates. Every actuator is NONE or FILTEREXACT; each stateful actuator has exactly one native activation address. Preserve mixed native actuator order, negative/zero gears, and multiple actuators acting on one joint. Source limits are explicit finite control/activation/force intervals. Unbounded channels use masks, not invented physical limits. This slice rejects free/ball joints, contacts or possible collision pairs, joint limits, dry friction, equality/tendon/plugin actuators, joint-wide summed force clamps and other dynamics/gain/transmission modes before device allocation. The earlier admitted stateless domains/capacities are not reduced or widened.

Ball-joint support is a successor: native qpos is a WXYZ unit quaternion (four entries), velocity is a three-component tangent angular velocity; it cannot be added to the resident scalar-per-link layout by calling a quaternion three scalar joints. T2/T3/T4 activation reuse is also a successor, not inferred from T1 acceptance.

## Existing support and sources

- `core/robot.py` accepts only revolute/prismatic links; resident PD controls and delay histories are not independent actuator activation states.
- `_tree_model.py` and `_floating_model.py` explicitly reject `model.na` and non-NONE dynamics. Existing scalar JOINT gear, fixed gain, affine force bias, control/force clamps and disabled-group rules already exist.
- `MuJoCoInitialState.act` and `MuJoCoState.act` are public. The checkpoint contains an act mirror, but its authoritative device banks currently contain only q, v and compensated clock. Merely retaining that mirror does not implement activation dynamics.
- Newton's experimental general [Actuator API](https://newton-physics.github.io/newton/latest/concepts/actuators.html), inspected 2026-11-11, lists transmission and actuator dynamics as limitations. Its [SolverMuJoCo integration](https://newton-physics.github.io/newton/latest/solvers/mujoco.html) separately exposes native MuJoCo actuator attributes. These are different capability boundaries, not a claim that Newton lacks native activation.
- Behavioral reference is fixed [MuJoCo 3.3.7 computation](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html), [general actuator XML](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#actuator-general), and [`engine_forward.c`](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c), specifically `mj_fwdActuation`, `nextActivation`, `mj_advance`. MuJoCo and Newton are Apache-2.0; existing package LICENSE/NOTICE remain, new source carries matching attribution. Latest Newton documentation is context, not a version-pinned numerical oracle.

## Exact model and state semantics

Native compiled fields determine identity and parameters: actuator index, joint ID, qpos/dof address, activation address/count, scalar gear, dynamics type, time constant, actearly, disabled group, clamp masks/ranges, gain and three bias coefficients. Arrays are immutable and source-bound; activation addresses are a complete unique cover of `[0, na)`, not assumed equal to actuator indices. Admit finite positive filter time constants; `tau_eff=max(mjMINVAL,tau)` preserves the native lower bound.

For each actor, clamp the held raw control to the source control range when enabled, obtaining u. For a FILTEREXACT actor at accepted entry activation a:

    adot = (u-a)/tau_eff
    proposed = a + (u-a)*(1-exp(-h/tau_eff))
    next = clamp_activation(proposed)
    force_input = next if actearly else a

A disabled stateful actuator stores `clamp_activation(a)` (native advancement uses zero derivative), while its raw derivative remains the native diagnostic. Disabled force generation starts at zero. Force clamping still occurs afterwards, including disabled zero, matching the fixed source's existing nonzero force-range behavior. No activation is integrated twice for actearly. NONE actors use u directly and own no activation coordinate.

For a scalar joint transmission, length = gear*q and velocity = gear*v; scalar raw force is `gain*force_input+b0+b1*length+b2*velocity` (bias zero for NONE). Apply the native force clamp, then accumulate `gear*force` into the corresponding generalized actuator force. No control-range clamp is applied to activation. Both negative gear and shared-DOF accumulation are required cases. Original user qfrc/xfrc inputs remain separately identified.

The new device route is exactly:

1. Original geometry kernel from accepted q/v.
2. Original mechanics kernel with `nu=0` for this new route only: complete M, bias, passive and external loads, zero actuator accumulator.
3. One new actuator kernel: next activation, derivative, clamped input, selected force input, raw/final scalar force and generalized actuator force; add the latter to the already assembled external load.
4. Original solve/integrate kernel. Same old physical and finite-state gates.

The old NONE-only route retains its original three launches and arithmetic. No `_tree_kernels.py` modification is needed. New buffers are bounded by admitted nu/na and included in existing allocation/transfer counters. Nonfinite actuator intermediates mark the actual attempt failed; no hidden clamp, CPU fallback, or partial world publication.

## Public API and publication

Construction stays `compile_mujoco_xml(...)` then `MuJoCoRuntime(..., implementation='tyx', mode=..., initial_states={key: MuJoCoInitialState(q,v,act=...)})`. Existing `set_controls`, `step`, `snapshot`, `integration_snapshot`, reset, clone and checkpoint APIs remain the entry points. New public immutable `MuJoCoTyxActuation` records source identity and actual input/output act, derivative, clamped control, force input, raw/final actuator forces and full generalized force. A new `MuJoCoTyxActivationStepReceipt` subclasses the existing receipt and carries that record. Old stateless receipt schema/type is unchanged.

The activation device buffers use the same accepted/candidate bank index as q/v/time. Runtime checks the actual input activation against the accepted mirror, validates the complete candidate and receipt, then performs the existing single publication. Rejection retains accepted q/v/time/act, held controls, ticks and old receipt. reset uses the authored initial activation only for selected worlds; clone and restore use actual accepted bytes. Candidate activation cannot leak into observation, checkpoint or retry.

Stateful checkpoints have explicit schema v2, raw activation bank and activation-layout binding. Stateless v1 remains byte/schema compatible. Restoration validates all identity, shapes, bytes and CAS before replacement allocation/commit; restored stateful receipt type is explicitly whitelisted. Native integration-state act and explicit mirror must equal the actual FP32 bank promoted to FP64. The parent-owned integration snapshot must include accepted act and the same complete receipt, without endpoint `mj_forward` generating actuator state.

## Exclusive implementation files

Author owns NEW:

- `src/phymeo/backends/mujoco_tyx/_actuator_model.py`
- `src/phymeo/backends/mujoco_tyx/_actuator_kernels.py`
- `src/phymeo/backends/mujoco_actuation.py`
- `tests/test_mujoco_tyx_activation.py` (tests first)
- `docs/development/p65-rob2a-actuator-report.md`

Author may narrowly modify `_tree_model.py` (explicit stateful admission keyword) and `_tree_device.py` (optional act banks, fourth stage, complete actual download). No other existing product/test file changes.

Parent alone applies reviewed patches for `mujoco_tyx/_model.py` (explicit stateful T1 route), `_mujoco_tyx_runtime.py` (pack/build/input binding/candidate act/receipt), `mujoco_checkpoint.py` (stateful bank/schema/restore). Author supplies a concrete patch in ignored `artifacts/p65-rob2a-authoring/`; never edits these shared files. Parent controls any exports and new integration-snapshot compatibility. This division does not authorize project execution in the author lane.

## Predeclared cases and acceptance

Both interpret and CUDA use the same authored cases and native MuJoCo 3.3.7 independently compiled source. Same-input oracle receives actual accepted entry q/v/act and raw stored control; independent continuous native trajectory is not overwritten from device state. An analytic exponential/gear/force oracle is also required; product helpers cannot be the oracle.

1. One hinge, nonzero q/v/act, explicit nonzero external load: 32 continuous held step/reversal/unload steps; actearly false/true and h/tau both below and above one. First impulse must distinguish early from ordinary actuation.
2. One slide with negative gear; distinct control, activation and force ranges. Test each clamp stage and disabled-group zero followed by positive force-range clamp. Initial activation outside actrange tests native advancement clamp rather than construction repair.
3. Coupled R/P tree with mixed NONE/FILTEREXACT native order and two actuators sharing a joint; nonidentity inertias, body wrench, nonzero speed and multiple time constants. Save complete entry force/M/b/diagnostics and 32-step continuous trajectory.
4. Sparse W1/W2/W8 batches: independent controls, selected reset, nonzero activation checkpoint bytes roundtrip, real successor, restore and identical successor, clone continuation. Wrong act length/order/source/CAS and unsupported model features reject without partial mutation.
5. Late candidate-download/publication failure after real activation computation: accepted whole batch and old receipt remain unchanged; same request retry agrees with an independent uninterrupted copy. Restore failure after replacement bank upload/readback retains the destination.
6. One existing stateless T1/A1 focused compatibility selection; no rerun of unrelated accepted suites. An installed public smoke is part of the parent's next common wheel if this slice is accepted.

Reuse current T1 same-input M/b/J/force and trajectory gates (`tests/test_mujoco_tyx_tree.py`), not a tightened FP64 target. Activation and scalar-force checks use FP32 engineering gates declared before first execution; exact same-backend clone/restore successors and input/output bindings are byte comparisons. Every requested physical gate, raw diagnostic and first failure is retained. No fixture/tau/gain changes to obtain a pass, no G1 requirement, no implicit expansion to contacts or floating trees.

Success is a connected, executed two-backend slice with real act/force/motion co-publication, bounded resource accounting and transactional continuation. Authoring or HOST parameter tests alone are not activation acceptance. After two unsuccessful diagnosed revisions, record the minimal concrete failing input/seam and return to parent rather than expanding the model or chasing unlimited precision.
