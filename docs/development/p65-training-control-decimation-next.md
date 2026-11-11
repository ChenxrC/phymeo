# MuJoCo-TYX control decimation: bounded next slice

Status: RESEARCH_ONLY / NOT_IMPLEMENTED / NOT_EXECUTED. 2026-11-12.

This proposes one policy decision driving several existing physical steps. It is a training-interface feature, not another J learning intervention. Original J/R results, tasks, rewards, seeds and learning gates remain historical and unchanged. Parent approval of an implementation contract is still required.

## 1. Source basis and actual gap

Current child snapshot: `5e1bc41140f64278610a6a79f7e0cbe83f867740`. Source was read as text; no project execution accompanied this research.

| Source | What it establishes |
| --- | --- |
| `src/phymeo/training/mujoco_tyx_env.py`, constructor and `step` | The binding currently states decimation 1. One action submission precedes one physical step, one task export and one episode update. The same method automatically resets done worlds. Calling this method D times is therefore incorrect action-repeat wiring. |
| `src/phymeo/backends/mujoco_device_controls.py`, `_Packet.inputs`; `_mujoco_tyx_runtime.py`, `_pack_controls`/`step` | An accepted packet checks engine/device owner, control epoch and canonical control ownership. Ordinary physical revision advancement does not invalidate it. Holding one packet across several public physical steps is already supported by this ownership contract. Reset/restore invalidates it and must occur after the control window. |
| `src/phymeo/training/rsl_mujoco_tyx.py`, constructor/`step` | RSL receives one transition per Env call, including final observation and timeout. Its current config reports physical dt and decimation 1. |
| `src/phymeo/training/mujoco_tyx_checkpoint.py`, `_current_binding`, `_empty`, `update` | N3 saves only at a completed update with empty storage and no pending upstream transition. Env state and configuration bindings participate in restore. |
| Newton 1.7 installed `examples/robot/example_robot_policy.py`, lines 229–236, 345–365, 422–428 | The deployment example computes one target and executes decimation 4; each simulation call may itself have solver substeps. This supports separating policy and physical rates, but does not establish PPO reward, timeout or transaction semantics. |

Fixed local Newton root: `C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/.venv-newton-1.7.0/Lib/site-packages/newton/`. Robot-policy file SHA256: `3c86030a3db7978b05edfaf6e6be3a745942f53e152f9f99ff033af964416693`. Use actual solver calls to describe physical elapsed time; the example's display `sim_time` increment is not a training-clock contract. Newton's [1.7 Isaac Lab integration page](https://newton-physics.github.io/newton/1.7.0/lab/isaac-lab.html) identifies the training integration, not an alternative reward definition.

Isaac Lab v2.3.2 [DirectRLEnv](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/envs/direct_rl_env.py#L298-L374) processes an action, applies it during each physical substep, then computes reward/done and resets. Its policy step duration is physical dt times decimation. That is a suitable first-slice terminal timing model; task reward details remain task-defined.

The [manager-based environment](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/envs/manager_based_rl_env.py#L137-L216) computes reward after the substep loop using policy step dt. Its [RewardManager](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/managers/reward_manager.py#L117-L145) multiplies each term by that dt. This is an endpoint rate convention, not a sum of independently evaluated substep rewards. Phymeo's existing quadratic reward is already a discrete per-transition objective, so adding a dt multiplier silently would change its meaning.

Fixed RSL 5.5.2 source: `C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/.venv-training-cuda313/Lib/site-packages/rsl_rl/algorithms/ppo.py`, SHA256 `125673c31542dc9e8d85c997d2dc1c6928bbb541dc904484484b719c7c73c572`. Its [upstream PPO](https://github.com/leggedrobotics/rsl_rl/blob/v5.5.2/rsl_rl/algorithms/ppo.py) stores one reward/done per call, adds timeout bootstrap using the entry transition value, and resets recurrent state on done. Gamma/lambda operate per transition. No new PPO implementation is needed.

## 2. Proposed first public contract

Proposed optional `control_window=MuJoCoTyxControlWindowSpec(physics_steps=D)` on `MuJoCoTyxVectorEnv`; proposed initial domain is integer D in 1..8, excluding bool. This ceiling bounds evidence and fault testing, not a numerical limit claimed for the engine. Default omission preserves the existing D=1 behavior and old checkpoint binding. Explicit D=1 must be behaviorally equivalent. An opt-in specification is immutable and hashed.

The only initial semantics are:

- `action_hold = native_channels`: evaluate the existing `bias + scale*tanh(raw_action)` once, submit all three channels once, and hold ctrl/qfrc/world-COM xfrc through D physical steps. Native actuator dynamics, damping, geometry and constraints still evolve every physical step. No callback recomputes a custom PD controller inside this first slice.
- `reward = endpoint`: one unchanged task reward from the accepted D-th endpoint, with no hidden multiplication by D or dt, no intermediate sum and no hidden discounted substep rewards. Episode return is the sum of these policy rewards. This does not claim time-integrated cost invariance between different D values.
- `done = endpoint`: evaluate task termination at the final accepted endpoint. The runtime's physical admission, constraint and numerical failure gates still apply to **every** substep. Intermediate task-bound crossings are not latched by this mode; the API/doc must say so. First-crossing termination and freezing only some worlds require an additional design because current batch stepping has no per-world active mask.
- `horizons = policy_steps`: increment each episode length once per completed control window. Termination takes priority over timeout as today. A horizon H describes H*D*h nominal seconds; callers requesting an equivalent physical duration must explicitly choose an appropriate integer H before execution. There is no silently shortened final window.

`h` remains the original native physical timestep. Publish `physics_dt_s=h`, `policy_dt_s=D*h`, both frequencies, and `decimation=D`. For clock checks preserve the existing stored FP32 h and compensated two-word clock. Compare device progression to the D actual stored-h advances; do not mistake nominal-versus-stored h drift for a failed step.

Changing D changes policy sampling, episode duration and discounting per second. Keep configured RSL gamma/lambda unchanged and bind them with the rest of the learner config. Do not silently use gamma**D or claim an unchanged continuous-time objective. Matching discount per second would be a separately declared configuration change, not needed for this feature qualification.

## 3. One control-window transaction

Use the existing exclusive Env ownership and lock, with an explicit in-progress control-window marker. Do not call `Env.step` recursively. The small internal sequence is:

1. Validate world order, input frame/revision, producer stream and raw actions. Allocate owned request/evidence storage before action submission where possible.
2. Map and submit one complete action packet. Retain its receipt and immutable requested action/native channels.
3. For substep indices 0..D-1, call public `runtime.step(h)`. Immediately record the returned world acceptance and available published step receipts before later snapshot/measurement can fail. On success capture the selected full/integration state mode for that substep. Do not export a new policy observation, update episode metadata or reset between these calls.
4. After all D steps, export one task frame, reward and termination; preserve pre-reset final observation/state. Compute timeout and stage episode metadata once.
5. Reset the endpoint done subset once using existing reset/N5 APIs; export post-reset observation. N5 consumes exactly the samples belonging to the actual committed reset, including a later task-export failure.
6. Publish one learner transition and one immutable control-window record. Previous action and RNN handling occur once at this boundary. Clear in-progress state only as part of success or explicit failure bookkeeping.

The record must include declared D, attempted/completed counts, ordinal per substep, entry/exit revision, per-world tick and raw two-word clock, requested action, the one action receipt, native-channel identity, acceptance map and complete per-substep public physics receipts/states. Endpoint transition fields stay endpoint fields; do not replace a substep's force history with the final force. D is bounded, so this is an owned bounded tuple rather than an unbounded history.

A new error should attach the same accepted-prefix record, failed substep index and failure phase. Existing `state_accepted: bool` alone cannot distinguish zero, one or D committed physical substeps. A snapshot failure after an accepted physical return must preserve that acceptance even if full diagnostic state is unavailable; mark the exact missing evidence rather than manufacture it.

| Failure point | Required observable result |
| --- | --- |
| Action preflight, before submission | No action/physics/episode/reset change; original frame remains usable. |
| Submitted action, first physical step rejected | Held action is already committed, zero accepted physical substeps; no learner transition/reset; Env requires recovery. |
| Substep k rejected after k-1 successes | Retain exactly that physical prefix and held control. No whole-window rollback, no suffix step and no automatic reset. Poison Env/adapter/session. |
| D physics steps accepted, endpoint export/bookkeeping fails | Preserve all D receipts and actual physics state; no successful learner transition. |
| Endpoint reset commits, following export fails | Report committed reset and N5 stream consumption, preserve pre-reset final state and prefix, require recovery. Do not rewind its sample index. |
| Late upstream/RSL failure | Existing session failure semantics apply; do not re-run the action window to reconstruct evidence. |

Recovery remains explicit all-world reset or valid N3 composition restore. No checkpoint is supported inside a control window. Public save/update boundaries must reject in-progress or poisoned owners, including same-thread reentrant calls. Native held controls and double clocks remain covered by the physical capsule.

## 4. N3 and RSL integration

The optional control-window binding must include D, h, action hold, endpoint reward/done, horizon units and record schema. Cross-D or cross-rule restores fail before physics CAS. Default old D=1 checkpoints remain readable; new opt-in state cannot be silently interpreted as old mode. There is no new in-flight substep cursor in an update-boundary checkpoint.

Env sequence and episode length count policy transitions; runtime ticks count physical steps. Preserve all old model/device/world/task/action/N5 bindings. N3 continues saving physics/controls, episode state, previous action, RNG, optimizer and actual RNN hidden state after an empty-rollout update boundary. On restore, the next complete control window and next random reset must reproduce, while monotonic operational revisions/counters need not rewind.

RSL config must expose actual D and policy dt. The adapter produces one reward/done and one storage entry per window. Timeout remains the pinned upstream entry-value bootstrap; final observation remains an explicitly saved diagnostic/consumer field, not a new critic bootstrap implementation. Recurrent state is reset once for endpoint done worlds; non-done hidden state is retained. Do not turn physical substeps into extra optimizer samples or alter the original learner algorithm.

## 5. Proposed files and ownership

No runtime, physics kernel, action packet or model/pin change is currently indicated by the held-control source contract.

| Scope | Proposed owner/change |
| --- | --- |
| `src/phymeo/training/mujoco_tyx_control_window.py` NEW | Author: small immutable specification/record and schema validation; no task names or physics implementation. |
| `src/phymeo/training/mujoco_tyx_env.py` MOD | Parent shared seam: refactor only raw physical advancement from existing episode publication; add bounded prefix/failure evidence and stable optional binding. |
| `src/phymeo/training/rsl_mujoco_tyx.py` MOD | Parent shared seam: actual rate metadata and optional window evidence propagation. |
| `src/phymeo/training/mujoco_tyx_checkpoint.py` | Prefer existing generic Env binding; parent changes only if explicit in-progress/recovery checks cannot be supplied by Env. No serialization redesign. |
| `tests/training/test_mujoco_tyx_control_window.py` NEW | Author: independent schedule, failure prefix, endpoint/reset/RNN/restore cases. |
| `examples/mujoco_tyx_control_window.py` NEW | Author: short public consumer; no training-budget sweep, no imports from tests. |

First implement/test the immutable record and D=1 compatibility, then D>1 stepping and short CUDA qualification. Only after those pass attach the focused N3/RNN continuation case. Optional endpoint-rate integration, per-substep reward integration, early-done active masks, custom low-level controllers and asynchronous rollout are separate work; none is needed to release this useful first slice.

## 6. Bounded cases before implementation outputs

These are proposed fixed cases for the implementation contract, not executed tests or node counts. Reuse accepted task XML/action mappings without altering masses, geometry, limits or tolerances.

| Case | Fixed scope and independent checks |
| --- | --- |
| Joint control | Existing J, four sparse worlds, D=1/2/4, eight policy windows each. Nonzero q/v and three native input channels where admitted. Compare with one independently submitted HOST control followed by exactly D public runtime steps, retaining the native J recurrence check. Check D physical ticks, one policy length increment and one action submission per window. D=1 preserves existing endpoint fields. |
| Continuous support | Existing S, four worlds, D=4, eight policy windows. Assert actual contact/finite q/v and complete force evidence at every physical step, not only at the endpoint. Compare held-input reference. Predeclare endpoint termination/timeout subsets via task bounds/horizons; capture final observation before reset and N5 sample consumption exactly once. A synthetic intermediate-cross/endpoint-inside sequence explicitly verifies the selected endpoint semantics without claiming continuous safety. |
| Recurrent composition and faults | Existing J, four worlds, D=2, fixed GRU seed, rollout length 8. Save after one completed update, continue one rollout/update, restore, repeat and compare action/obs/reward/done/physics/Adam/RNN/RNG plus the next N5 subset sample. Inject substep-2 rejection and late endpoint-export/reset-export failures in focused tests; assert complete prefix and no further physical step. Wrong D/rule and mid-window save must reject. |

Use the existing numerical gates, with exact comparisons only where the old identical-input contract already warrants them. No extra long learning run is required. Tests must distinguish counter scopes: physical step calls, policy transitions, action submissions, task exports, per-world reset samples and optimizer updates. Resettable device diagnostic counters are not automatically cumulative totals.

Success means these cases demonstrate held-control evolution, correct endpoint semantics, exact failure-prefix accounting and restoration under the declared domain. It does not mean J/R learned, G1 improved, all rewards are time-integrated, or arbitrary controllers can be decimated safely.

Stop and report a minimal counterexample if held controls cannot survive normal physical advancement, any intermediate reset reuses an old action, an accepted substep disappears from failure evidence, or restore mixes different rates. If early per-world stopping is required by a consumer, stop the endpoint-only claim and design the missing active-mask capability instead of hiding it. Two diagnosis rounds without narrowing a failure should produce that counterexample, not a larger parameter/budget search.

## 7. Scale value and measurement limits

The source-supported opportunity is fewer policy calls, action ingress operations and policy task exports per equal number of physical steps. Native physical solving and the current HOST physics validation/readback still occur on every step. A D-times speedup is not implied, and the same action being held defines different closed-loop behavior from D=1.

After functional qualification, a bounded measurement may compare equal physical-step counts and the same predetermined held-native-input schedule. Report actual complete wall time through evidence close, policy-call/physical-step counts, both frequencies, observer mode and copied bytes whose ownership is measurable. Keep setup and final hashing explicit. Do not call optimizer-only time end-to-end, infer unavailable internal synchronization counts as zero, or claim performance against Newton without an equivalent measured comparison.

No implementation, runtime qualification, training, numerical probe, lint or test was performed for this document. The current N5 validation remains a separate parent-owned execution.

## 8. N6 authorized authoring handoff (2026-11-12)

Status: CODE_READY_TESTS_PENDING, AUTHORING_ONLY. Parent subsequently approved the endpoint first slice. Child delivery adds only `src/phymeo/training/mujoco_tyx_control_window.py`, `tests/training/test_mujoco_tyx_control_window.py`, `examples/mujoco_tyx_control_window.py`, and this appended section. Parent exclusively implements Env/RSL shared seams and performs every project execution. No implementation result is claimed by the research sections above.

Frozen API:

- `MuJoCoTyxControlWindowSpec(physics_steps=1)`: builtin integer 1..8, immutable `physics_steps` and `sha256`, fresh `binding(h)` including physics/policy dt and endpoint semantics.
- `MuJoCoTyxSubstepRecord(ordinal, entry_revision, exit_revision, accepted_worlds, states, diagnostics_complete, error=None)`: 0-based ordinal, complete public states/last-step evidence, owned recursively read-only NumPy arrays/dataclasses/mappings. `states=None` is legal for an accepted step followed by failed diagnostic capture.
- `MuJoCoTyxControlWindowRecord(spec_sha256, physics_steps, attempted, completed, substeps, action_receipt, phase, failed_substep=None)`: `completed` counts actual nonempty all-world successful returns, independently of later diagnostic availability. No retry or synthetic physical evidence is produced by these passive record classes.

Clarifications superseding the proposal where necessary: raw two-word clocks remain preserved at existing public checkpoint/N3 save boundaries; each substep records public state time/tick and complete receipt rather than serializing another large capsule. Default omission preserves the original Transition class and binding. Opt-in returns the parent-owned `MuJoCoTyxWindowTransition`, derived from the original Transition and adding `control_window`. Error gets an optional control-window record. Requested action/native channels remain in the existing transition/request evidence, without a second hidden device readback.

Recovery is the original protocol: an Env poisoned by an accepted partial window needs explicit all-world reset before becoming usable. A poisoned N3 session is not revived by that Env reset; discard that failed session, construct a healthy owner/session at an empty update boundary and restore a valid saved checkpoint. No new direct poisoned-session restore path is implemented or promised. The example saves/restores only healthy completed-update sessions. The parent's step-only busy guard must not block a normal N3 update already holding its surrounding training boundary.

Tests were authored first in `aeb1efc7`, before module `9ca88491`; example and endpoint crossing coverage followed in `d060edec`. Manual declared collection count is **38 nodes: 21 HOST schema/ownership negatives and 17 CUDA integration nodes**; this is a source enumeration, not observed collection/pass. CUDA includes four J/S window cells, default D1 compatibility, endpoint terminated-over-timeout, four late-failure stages, invalid action, four same-thread reentrancy operations, one complete GRU continuation/cross-D negative, and one intermediate-task-bound crossing. The latter uses unchanged J damping/mass/h, zero action and native v bound `0.03998`: the original positive v is above it after one substep and below it after four; this explicitly exercises endpoint-only semantics without changing physics.

Public example entry points are `run_case(case, D, output)` and `continuation(output)`. The four rollout cells each take eight policy windows, with all substep states compared against a separate CUDA runtime receiving identical stored inputs through the HOST control route. J retains an independent ordinary Euler recurrence and both tasks independently check native-feature endpoint reward/termination. S checks actual positive contact force on every world/substep. GRU fixes seed 6511, W4, D2, rollout length 8: warm update, save, A update, healthy restore, B update; complete learner/Adam/RNN/RNG/physical/window equality plus the next N5 subset sample and cross-D pre-CAS rejection. No 128-update learning or performance experiment is added.

Saved files are `<case>-d<D>.jsonl` and corresponding `-summary.json`, or `J-d2-gru.jsonl`, `J-d2-gru-summary.json` and `J-d2-gru-bundle/{manifest.json,physics.bin,training.json}`. Each rollout stores request before the Env call and complete accepted transition before independent comparison. The failed-window record preserves the actual prefix; parent semantics determine how unavailable final diagnostics are marked. Successful snapshot/receipt differences are not normalized away. The continuation comparison excludes only monotonic source/destination publication revisions, task/action epoch and resource counters; full physical time/q/v/control, per-step force and policy/optimizer state remain compared.

Suggested parent-only focused execution is the full new `tests/training/test_mujoco_tyx_control_window.py` split into non-CUDA and qualified CUDA selectors. The same four rollout/one GRU examples are already invoked by those nodes; do not execute them a second time merely to obtain another green result. Optional source-outside-wheel consumer qualification is a separate parent decision. Author performed text reads, manual review and exact Git path checks only: no import, collection, tests, numerical probe, lint, formatting, compilation or installation.
