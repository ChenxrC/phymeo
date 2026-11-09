# N2 vector environments: physical acceptance and borrowed contact steps

**AUTHORING_ONLY / UNEXECUTED.** This handoff is based on clean author branch `codex/h1-device-capacity` at `73c48f37d2d3d9e73a8effc69188545369348cfd`. The parent plan is [the 08:00 robot/environment connections plan](../superpowers/plans/2026-11-10-0800-robot-environment-connections.md). That plan and the current HOST actuator/support integrations are read from the parent; no parent merge, reset, old WIP edit or project execution occurred here.

The original environment called `batch.step(dt)` and discarded its returned per-world flags. The batch can preserve one rejected world's physical state while another world advances. Advancing the reward/episode clock for both would manufacture a transition. This change consumes the actual map before any post-step observation or task evaluation and introduces a public, reusable failure protocol.

## Public protocol and physical ownership

```python
from phymeo.training import (
    PhysicsStepFailure, ResidentContactStepper, ResidentVectorEnv, VectorEnvStepError,
)

stepper = ResidentContactStepper(batch, contact_plan)
env = ResidentVectorEnv(batch, task, world_keys, action_size=A, dt=h,
                        horizon=H, stepper=stepper)
```

The new keyword `stepper=None` preserves the ordinary default `batch.step`. A selected callable accepts `stepper(h)` and returns exactly one boolean for every environment world. `ResidentContactStepper(batch, plan)` validates the actual `TyxWorldBatch`/`ResidentContactPlan` types and owner, and the environment rejects pairing that stepper with a different batch. Each call uses the existing public `batch.step_contacts(plan,h)` transaction. The wrapper owns no runtime, plan or close operation. Closing the original batch/plan invalidates calls normally; creating or resetting an environment does not close them.

The backend duck protocol remains:

- `snapshot(keys=None)` returns the actual current accepted state mapping;
- `set_controls(mapping)` validates/stages controls without physical advancement;
- `step(h)` returns the complete world-key boolean acceptance map;
- `reset(keys)` restores/recreates the selected original worlds.

Task callbacks remain `controls(actions,keys)`, `observe(states,keys)` and `evaluate(before,after,actions,keys)`. No robot-specific observation, force law, motion reference or q/v injection is introduced. The HOST scene adapter can implement the same backend protocol without pretending to be a device batch.

A backend that raises after known physical publication must use:

```python
raise PhysicsStepFailure("original failure", accepted_worlds=complete_bool_map) from cause
```

`accepted_worlds` is an owned immutable mapping. `acceptance_known=True` by default; `physics_accepted` is `any(accepted_worlds.values())`. If publication timing cannot be determined, explicitly use `acceptance_known=False`; then `physics_accepted=True` conservatively means advancement cannot be excluded. It does not assert that all worlds accepted. Ordinary exceptions during the physical call are treated the same conservative way, with an empty unknown mask. Missing/extra keys, non-mapping results and non-boolean values are invalid and require full reset. Builtin bool and NumPy bool scalars are admitted; numeric 0/1 are not silently cast.

## Failure and transition semantics

| Actual physical report | Env result | Reward/done/episode | Recovery |
|---|---|---|---|
| Complete all-True normal return | Original `VectorStep` success | Original final observation and selective autoreset | Continue |
| Complete all-False return or explicit known failure | `VectorEnvStepError`, `physics_accepted=False`, known immutable mask | No callback/transition or counter advance | Retry permitted; no poison |
| Some True and some False | `VectorEnvStepError`, `physics_accepted=True`, exact known mask | No callback/transition or counter advance | Poison; reset all worlds |
| Explicit failure after all worlds accepted | Same known all-True failure provenance | No reward/done/episode transition | Poison; reset all worlds |
| Invalid map, unknown exception or uncertain publication | `VectorEnvStepError`, unknown acceptance, conservative physical flag | No reward/done/episode transition | Poison; reset all worlds |
| Observation/reward/autoreset/result failure after all-True physics | `VectorEnvStepError`, known all-True physical mask | Counters commit only after constructing the whole result | Poison; reset all worlds |

The error mask reports **physical publication**, not a fictitious rollback. On partial acceptance, already advanced worlds remain advanced until explicit recovery. A rejected physical step can still leave the new controls staged/held, as defined by the backend; no all-False error promises to undo a prior control submission. Full recovery does not rewind an accepted transition by editing q/v. Post-accept task failures and uncertainty keep the existing explicit reset requirement.

## Authored validation

All tests below are **uncollected and unexecuted**. Eighteen small protocol cases are handwritten from public failure categories, plus five real TYX cases. The original `tests/scale/test_resident_vector_env.py` is unchanged and remains the compatibility selection.

[test_resident_env_acceptance.py](../../tests/test_resident_env_acceptance.py) uses only logical publication clocks, never a synthetic numerical dynamics oracle. It covers returned/raised all-False, returned/raised partial acceptance, invalid/missing/extra/non-bool maps, unknown and incomplete exceptions, late all-True failure, post-accept task failure, immutable owned masks, and explicit/default stepper dispatch. It checks that no task evaluation or counters are fabricated for failed transitions, partial state is retained, and reset/retry behavior is exact.

[test_resident_env_contacts.py](../../tests/scale/test_resident_env_contacts.py) exercises real backend execution:

| Node | Independent expected quantity |
|---|---|
| `test_env_contact_stepper_advances_real_load_and_preserves_final_observation[fixed-box]` | A 2 kg one-joint box has initial depth1/32 and stiffness64, giving 2 N. Thus `a=1`, `v=h`, `q=h^2`. The contact stepper must produce the response; an ordinary zero-load step would fail. Termination preserves this final observation and then resets the world. |
| Same node `[floating-offset-sphere]` | A real free root has mass2, principal inertia1/2 and a sphere at local x1/4. Its 2 N upward load gives `tau_y=-1/2`, `a_z=1`, `alpha_y=-1`; compare all six velocities and quaternion drift. Horizon truncation preserves the accepted observation, then resets. |
| `test_real_two_world_rejection_mask_never_emits_a_training_transition[partial]` | Two real mass1/2 free spheres. Finite FP32-max force times inverse mass2 makes only world7 reject; world4 under1 N advances with `v=2h`, `x=2h^2`. Verify exact mask, no reward/counters, real partial state retained, mandatory full reset and valid retry. |
| Same node `[all-false]` | Both receive the same finite overflowing load. Both reject through the real kernel; no transition, no poison, and a zero-load retry accepts without reset. |
| `test_contact_stepper_checks_both_owner_edges_and_borrows_open_resources` | Actual plans reject a different batch or different environment owner, reject an invalid plan type, preserve both batches, and fail after the borrowed plan is closed without closing the batch. |

H is exactly1/128. Analytic contact constants and negative-force inputs are declared before execution; no product HOST FK/M/force result sets expectations and no numerical kernel is mocked. Existing contact laws, mass/limits, thresholds and kernel acceptance remain unchanged. Each real force/mask case directly saves its full model, controls, actual before/after/final/reset state, contact receipt, diagnostics and statistics as gzip JSON, including a first-failure record. Arrays retain dtype/shape/raw bytes and values. The owner test has no physical step and no numerical result claim.

## Parent-only execution and limits

The parent is the sole project execution owner; the running G1 long window must finish before import, lint, format, collection, pytest, probes, compile/build or profile from this lane. The author ran none of those operations. Only source reads, file IO, Git diff checks and a small commit were used.

Suggested order after source review/import, with fresh guard and new absolute basetemps:

```text
python -m pytest tests/test_resident_env_acceptance.py -q -x --tb=short -p no:cacheprovider
python -m pytest tests/scale/test_resident_env_contacts.py tests/scale/test_resident_vector_env.py --require-tyx --tyx-mode=interpret -q -x --tb=short -p no:cacheprovider --basetemp=<fresh absolute path>
```

CUDA uses the same real/compatibility files with `--tyx-mode=cuda` and a different fresh output root. Keep first static/test failures and tested-source bindings. The HOST scene adapter and policy scheduler have independent authors and require parent integration on this public protocol. This patch does not add PPO, randomized resets, zero-copy GPU tensors, larger-scale throughput, device narrowphase/friction, resolved-tilt pressure distribution, or a new humanoid physical success claim.
## Reset failure follow-up

Read-only connection review found that a failed reset observation could leave the environment retryable after physical worlds had already reset. `reset()` now marks the environment reset-required if either the backend reset or its snapshot/task observation raises. Original exceptions and their causes are re-raised unchanged; episode counters remain at their pre-request values. No physical rollback is claimed. A full successful reset is required before stepping and is the only operation that clears this uncertainty.

The added `test_reset_publication_then_readback_failure_poison_keeps_counters_and_cause` first accepts a step, resets only world4, then fails the observation. It checks the retained physical clocks `{4:0,7:1}`, unchanged counters, original exception/cause, blocked step/partial-reset, and successful full-reset recovery. This is a single additional pure publication-protocol regression, still **UNEXECUTED**. It changes no contact law, stepper, HOST adapter, policy runner, optional RSL module or learning configuration.
## Open-plan second episode follow-up

Each of the two analytic contact positive cases now performs a second `env.step` after automatic reset while retaining the same open batch, environment, `ResidentContactStepper` and borrowed contact plan. Neither case rebuilds/closes its plan between episodes. The second call must again produce the independently declared 2 N response (and -1/2 Nm on the eccentric free root), `final_observation=(h^2,h)`, the original quaternion/velocity expectations, episode ID2 and a fresh 0->1 physical receipt. The retained first transition/receipt must remain unchanged; batch statistics must show two actual steps. Both complete receipts and reset states are saved. These are extensions of the same two authored nodes, still **UNEXECUTED**, with no plant/law/oracle/cap change and no additional test count claim.
