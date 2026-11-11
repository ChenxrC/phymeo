# ROB2b: activation composed with floating and constrained dynamics

Status: **CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**. This report is an authoring handoff, not a test result. The author ran no project import, test collection, numerical probe, pytest, lint, formatter, compiler, build, installation or native execution. Parent owns the only execution slot.

## Delivered behavior and exact boundary

The existing public `compile_mujoco_xml` / `MuJoCoRuntime` / `MuJoCoInitialState(act=...)` route now composes the already accepted NONE/FILTEREXACT scalar-JOINT law with the original free-root T2, normal-contact T3, and mixed-constraint T4 plans. The normal/constraint admission, capacities, geometry and numerical solvers are unchanged. Each source follows one structural route; there is no catch-and-retry fallback.

The original T1 activation model type stays intact. Other admitted plans are wrapped by an immutable `TyxActivationModel(base, activation)` whose missing attributes delegate to the full base, including `kind`, `root_body`, normal/constraint flags and all geometry/constraint tables. Each base compiler gets one explicit `allow_activation` keyword with a false default. The free compiler accepts FILTEREXACT only under that keyword; all other options and transmission checks remain. The activation compiler additionally checks scalar JOINT transmission and actual HINGE/SLIDE target type before indexing qpos/DOF addresses. A FREE/BALL target does not become supported because its first gear component is nonzero.

Capacity remains nu/na at most32, checked before ids/values allocation. Native activation addresses remain a complete, unique `[0, na)` cover. No ball, tendon, multi-free-root, equality/closed-loop, muscle, general gain, summed joint actuator clamp, new collision family or G1 trajectory claim is added.

## One state stage, original mechanics

New `_actuator_device.py` moves accepted T1 validation, bank allocation, actual kernel invocation and actual readback into common transport functions. It does not implement a new integrator, actuation law or publication mechanism. `_actuator_kernels.py` is byte-unchanged.

For stateful plans only: original mechanics runs with `nu=0`, retaining full M/J/bias/passive/external-force assembly and clearing the actuator accumulator; the one activation kernel adds actual scalar efforts through source DOF/gear into the full generalized RHS. Original normal/constraint kernels then see the same full RHS before the original solve/integration. Floating root DOFs receive no artificial actuator entries; their response comes through full mechanical coupling. The free/root angular coordinate convention is unchanged.

The same accepted bank index selects q/v/clock/act. Failure leaves the accepted bank, mirrors, controls and prior complete receipt unchanged. The common helper never publishes. Normal/constraint downloads delegate to their original physical base download and include activation arrays once. Existing runtime input-act binding and checkpoint v2 bytes/source identity are reused. Stateless sources still allocate no activation bank, launch no extra kernel and use checkpoint v1. Original T1 remains four stateful / three stateless kernels. New stateful T2/T3/T4 counts are four / six-or-seven / eight-or-nine respectively, including the existing optional extended-geometry stage.

The sole parent patch is `artifacts/p65-rob2b-authoring-20261112/runtime-capabilities.patch` (SHA256 `d92cd61e2c5d3e6b9e96df3a31786d9759876ec95a839101964d3920fc49f307`). It changes only the activation capability domain string for the selected T2/T3/T4 route, preserving T1's existing string. No child edit to `_mujoco_tyx_runtime.py`, `_model.py` or `mujoco_checkpoint.py`; it does not overlap the parent's clone-failure cleanup or integration observation work.

## Tests authored before execution

The new `tests/test_mujoco_tyx_activation_composed.py` has **22 predeclared nodes per backend /44 total**, with original h=.001, source Newton128/tolerance1e-10, no tuned source parameters or output-dependent case selection:

- Eight continuous paths: free R/P (actearly false/true,64 steps each); fixed normal support with filtered loading/reversal/zero-control decay (96); free contact condim3/6 (64 each); fixed zero-contact limit+dry friction with normal/disabled group variants (64 each); simultaneous free contact+limit+dry friction (64).
- One free-root+29 scalar driven joints/29 activation coordinates (nv35/nq36), four steps, bounded capacity qualification only.
- Two sparse-world checkpoint/clone/restore/partial-reset cases, for free and mixed routes. The test advances the real saved state before restoration and compares actual successors, not reconstructed initial states.
- Six late-failure/retry cases: actual input-act tamper, final world status refusal and late restore readback failure on each of those routes. Prior whole-batch state/checkpoint bytes remain; a subsequent real retry matches an uninterrupted clone.
- Two pre-allocation nonscalar target negatives: actual FREE and BALL JOINT actuator targets.
- Three original stateless-route controls, preserving T2/T3/T4 bank schema and three/five/seven launches.

Only two obsolete scope-negatives were removed from the prior activation test parameter list: a free root with a legal scalar driven descendant and a scalar soft joint limit. Both are now deliberate positive domains. Their now-unreachable fixture construction branches were removed; the remaining filter/integrator/ball/capacity negatives and all prior positive cases/oracles/gates are unchanged. This is an explicit admission update before execution, not suppression of a failed regression.

The tests reuse only the earlier independent analytic source-field interpreter and evidence encoder; no product helper is an oracle. The new native oracle explicitly assigns actual entry **act**, q/v and FP32-stored raw controls before `mj_forward`. It independently reconstructs full body COM Jacobians, M, applied/bias/passive/smooth acceleration, contact relative Jacobians, and EFC J/R/aref/force from native MuJoCo3.3.7. Continuous native paths initialize once and are never reseeded from device output. Entry quantities and successors are saved with phase context. Both-side force and COM moment are independently checked. Native `mj_forward`/`mj_step` are temporarily forbidden during actual TYX `step` plus `integration_snapshot`.

All gates are inherited from the design, not chosen after output: act/derivative rtol3e-5 atol3e-6; scalar/generalized effort3e-4/2e-4; M3e-4/5e-5; bias/passive/applied4e-4/4e-4; geometry3e-5/4e-5; J2e-4/5e-5; smooth acceleration2e-3/4e-3; constrained acceleration3e-3/1e-2; constraint force3e-3/3e-3; EFC R/aref5e-4/4e-4. Continuous q/v absolute free1e-4/1e-3 and constrained2e-4/3e-3, clock2e-6. Original normalized KKT/source iteration gates remain in the product. No new energy certification is claimed by this composition slice.

Each completed path saves full source/initial state, every actual entry/control/native/analytic/receipt/output and native continuous successor. Pending failures retain their available entry/oracle and original exception. Transaction files retain checkpoint bytes, previous and retry states, counters. Invalid-source nodes prove rejection by assertions; they do not fabricate trajectories.

## Checkpoints and handoff

- `e38084e9`: short design/allowlist.
- `969bb4d8`: tests first and exact old-negative retirement.
- `677debf2`: nine product files (eight modified, one new helper).
- `437cd67d`: correct the stateless T3 compatibility selection to the already declared free-root+condim1 contact source; a one-slide plane/sphere source belongs to original T0. No physics source or gate changed. Also removed an unused test import.

Final exact file manifest and SHA/blob binding are in the ignored `artifacts/p65-rob2b-authoring-20261112/handoff.json`. That manifest is file IO, not numerical validation. The only remaining untracked source-tree draft is the pre-existing `docs/development/mujoco-tyx-t1-review.md`, untouched and unstaged.

Parent pending sequence: review/import these commits and the capability patch; first static checks on this exact scope; cross-route smoke; both complete 22-node backend selections; narrow accepted T1 helper and remaining admission negatives; then integrated/package checks if the time budget permits. No command has been executed in this author lane. If a composed source reaches a genuine existing solver boundary, preserve the failing source/receipt and stop that combination rather than rewriting a solver or relaxing gates in this grant.
