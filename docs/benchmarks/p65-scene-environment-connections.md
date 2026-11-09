# HOST scenes connected to the vector environment

## Parent execution — 2026-11-10

This connection is now exercised on parent source `31058805`. The complete HOST connection suite passed **72 tests** in1.61s: environment publication protocol, real loaded HOST scenes, common policy scheduling, and saved-publication parser semantics. The exact combined TYX suite passed **12 tests per backend** (interpret1.64s; actual CUDA7.13s), including real contact loads, the same borrowed plan across two reset episodes, actual all/partial rejection, policy holding and adjacent vector-env compatibility. No skip/deselection or fallback was used in these commands.

Evidence: `artifacts/p65-connections-20261110-0800-01/08-connections-host`, `09-connections-interpret`, `10-connections-cuda`. Each command retains argv/environment, before/after source snapshots/hashes, stdout/stderr and raw exit. Actual numerical assertions are in the tests; parser protocol fixtures do not claim physical results. The subsequent G1 common-connection and optional learner runs are separate acceptance items.

The first exact16-file lint found one import-order issue and four E402 diagnostics in the optional-dependency test. The latter now have local explained annotations after intentional dependency admission. Fourteen files needed formatting. Final lint/format passed; 15/16 raw ASTs remained equal, and all16 retained the same non-import AST and contiguous scoped import bindings. This qualification does not claim import execution order was unchanged. Original failures remain in08a/08b, with repairs/proof/final results in08c–08g.

The author handoff below is historical.

Status: **AUTHORING_ONLY / UNEXECUTED**. This lane is based on clean `d8d6f7f` in `codex/h1-foot-support`; it reads the parent `5a84d38` public scene APIs and the N2 stepper failure contract. Only three new files belong to this checkpoint. No import, collection, numerical computation, lint, formatter, build or test was run by the author. The parent owns the sole execution queue and final acceptance.

## Public connection

`SceneBatchBackend(world_keys, factory)` owns one original public HOST RobotScene or MultiRobotScene per logical world. The factory receives that logical key and transfers a fresh scene at its original initial publication. Physical owner IDs inside the scenes remain unchanged. A task may retain public asset/shape context populated by its factory; it must not separately advance or close the backend-owned scene.

```python
from phymeo.training.resident_env import ResidentVectorEnv
from phymeo.training.scene_backend import SceneBatchBackend, SceneControl

# factory(world) uses the original ContactSceneBuilder.build(backend="host").
# task.controls maps each policy action to SceneControl({robot_id: RobotCommand(...)})
# or FloatingRobotCommand for an actual floating-root scene.
with SceneBatchBackend((7, 19), factory) as backend:
    env = ResidentVectorEnv(backend, task, (7, 19), action_size=1, dt=1/1024, horizon=100)
    observation = env.reset()
    transition = env.step(actions)
```

The environment borrows the backend. A policy runner borrows that same environment and controls its cadence; this backend never adds a second integration step. Actions in `SceneControl(actions, external=...)` are actual original robot command mappings. Bare rigid-only ContactScene has no matching observe() record and is explicitly outside this first backend; it needs its own observation adapter. No G1-specific path, reference motion, control gain or asset name belongs to this adapter.

| Method | Ownership and physical behavior |
|---|---|
| `snapshot(keys=None)` | Read-only logical-world mapping of the original `scene.observe()` records, with original entry-load/commit clocks and physical channels. It does not convert them into invented resident states. |
| `set_controls(mapping)` | Requires exactly all logical worlds and one immutable `SceneControl` per world. It stages values without advancing or querying physics. |
| `step(h)` | Calls each original public scene once in declared key order, holding staged actions. Success returns an exact all-world boolean map. The first failure stops further worlds. |
| `reset(keys=None)` | Builds every selected replacement first. Constructor/validation failure closes only new objects and leaves prior scenes and staged controls intact. Success swaps selected owners, closes the old ones and requires a fresh full controls map before another step. |
| `close()` / context manager | Closes owned scenes, including retained cleanup failures. Repeated successful close is idempotent. The environment does not own the backend lifetime. |

`SceneControl(actions, external=None)` requires a robot actions mapping and stores detached immutable value descriptions: numeric arrays retain their exact dtype, shape and bytes; public dataclass fields and mappings retain their original values. Its properties materialize fresh public commands/inputs. Mutating an input mapping, an original `CoupledInputs`, or a returned external value cannot alter the staged packet. Original scene command validation still runs when the scene is stepped; this layer does not duplicate dynamics, solve mass matrices or write q/v.

## Failure and reset semantics

A failure raises `training.resident_stepper.PhysicsStepFailure` from the original exception, with a complete `accepted_worlds` map and `acceptance_known`. Worlds accepted before that failure retain their actual state; worlds not attempted are False. On a scene exception the adapter compares public observation publication time/counter before and after the call. An unchanged publication is a known rejection. A new publication is retained as accepted even when an outer callback raises after the real scene returned. Unreadable or inconsistent timing is explicitly unknown and must conservatively poison the environment.

N2 `ResidentVectorEnv` validates the complete mask. A known all-False rejection produces no reward, done or episode transition and permits retry. Partial acceptance, known accepted publication followed by failure, and unknown acceptance require a full environment reset; physical accepted states are not rolled back. Environment counters do not advance on that failed transition.

A factory alias to an existing owner is rejected without closing that owner. A duplicate newly prepared object is closed once. A reset that has installed new scenes but cannot close an old scene reports that cleanup failure and retains ownership for a later close; it does not claim the new scenes were rolled back. Factories remain responsible for cleaning allocations they have not yet returned when their own construction raises. Reset atomicity covers backend-owned scene objects; arbitrary external dictionary writes made by a factory are not rolled back. Task callbacks should keep immutable asset/shape metadata rather than assume a staging-time external scene reference was committed after a different factory failed.

## Authored real physical cases

The tests create original `ContactSceneBuilder` fixed-root vertical sliders carrying finite spheres against finite box floors. The 2 kg, 3 kg and optional 1.5 kg variants have different initial q/v and skin overlaps. Gravity, mass, motor effort, normal spring, reciprocal floor load and actual motion are computed by the existing scene. The one-step oracle independently evaluates `N = k*depth`, `v_next = v + h*((motor+N)/m-g)` and `q_next = q + h*v_next`. Contact stiffness 1000 N/m, zero damping/friction and h=1/1024 are declared input parameters, not fitted outputs. No old foot-support group is repeated.

The eight intended test nodes in `tests/test_scene_vector_env.py` are:

1. `test_controls_are_owned_and_only_real_scene_step_advances`: exact command/external snapshots, no staging advance, invalid coverage rejection, original observation type, accepted bool map.
2. `test_real_loaded_worlds_feed_actions_observation_reward_and_selective_done`: two independent real loaded scenes through task controls/observation/reward/termination; independent first-step mechanical oracle, final observation and selective automatic reset.
3. `test_selected_reset_preserves_other_physics_and_requires_new_controls`: selected scene rebuild, other world's actual q/v/time unchanged, explicit command renewal and selection checks.
4. `test_partial_rejection_poison_has_complete_mask_without_reward_or_episode`: real invalid joint target in the second of three scenes, first world's accepted motion retained, third not attempted, original causal chain, no reward/counter transition and mandatory full reset.
5. `test_all_rejected_worlds_are_retryable_without_false_transition`: first scene rejects before publication; every scene remains at its initial state, no reward and no poisoning, followed by a real successful retry.
6. `test_scene_error_after_publication_retains_that_world_as_accepted`: a wrapper raises after an actual original scene step; the accepted motion is retained in the failure mask.
7. `test_factory_failure_and_alias_cannot_destroy_prior_live_worlds`: a replacement factory failure cleans a new prepared scene, preserves both old live scenes, rejects aliasing and then completes a real reset.
8. `test_constructor_failure_cleans_created_scenes_and_close_owns_current_scenes`: failed initial batch construction cleans prior created owners; normal close owns the actual scenes and remains idempotent.

These are authored nodes, not collected or passed tests. Each case writes original URDF sources and declared physical parameters, all task/control and snapshot events, every dataclass field of original physical readings, numeric dtype/shape/data, vector transitions, full failure masks, and factory/close ownership events. Expected exceptions retain the accepted physical prefix and attempted request.

## Validation and limits

Parent integration order: N2 stepper/error contract, this backend, then policy runner/task connections. This new module imports `PhysicsStepFailure`; it must not be imported on the authoring branch before the N2 module is integrated. The parent should first run exact-file static checks, preserve initial findings and any style-only change, then one focused HOST test-file run with a fresh absolute basetemp. No separate pre-collection/probe is required.

This is sequential HOST scene orchestration with NumPy observations. It is not zero-copy GPU interop, device scene batching, GPU collision/robot support, large-scale throughput validation or a G1 dance result. The unchanged public scene remains responsible for atomic physical publication. Randomized factories, seed/checkpoint protocols, cross-world physical coupling and asynchronous stepping are later interfaces. No numerical or cross-backend result is claimed in this checkpoint.