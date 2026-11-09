# N6 bounded W256 connection holdout

**AUTHORING_ONLY / UNEXECUTED.** Exact two new files: this report and [test_resident_connection_w256.py](../../tests/scale/test_resident_connection_w256.py). No core, old test, fixture, shape/site/candidate capacity, plant or physical tolerance was changed. The parent owns all project execution; this author performed only source inspection, file IO and Git checks while the parent runs the G1 common-entry window. No import/collection, numerical probe, pytest, lint/formatter, build or installation was run.

## Capacity check before authoring

Current [TyxWorldBatch](../../src/phymeo/backends/tyx_world_batch.py) rejects world counts beyond its signed-i32 interface and allocates legacy state/control/property arrays from the actual registered count. W256 is inside that original count domain. The two selected existing families use the original world-parallel translating-sphere and scalar-revolute kernels; no per-world HOST dynamics substitute is introduced.

The requested **one-plan W256 loaded box/floor case is not supported by the current shape table**. [CollisionModel](../../src/phymeo/constraints/collision_shapes.py) has `MAX_SHAPES=256`, while256 independent worlds with a moving box and a world-owned floor need512 shape registrations. Their4096 vertex addresses fit `MAX_VERTICES=8192`;256 expected dynamic/static candidates fit the existing default4096 candidate limit, and four box sites/world fit the original per-world128 site default. Those later limits do not override the earlier256-shape gate.

A single floor key cannot be relabeled as belonging to256 independent worlds. Two separate128-world plans also cannot be silently turned into one physical W256 contact step: calling the current `step_contacts` twice would advance the whole batch twice and leave a different subset unloaded in each call. This author does not add paging, combine plans, raise caps or run a smaller loaded case to claim W256 support.

## Authored positive: mixed physical motion through the public connection

`test_connection_w256_mixed_motion_policy_hold_and_independent_resets` has exactly256 registered worlds:128 existing `FreeSphere` instances and128 existing `RevoluteRotor` instances. Keys are `1003+7*i`, deliberately distinct from row indices. All inputs are declared before execution:

| Item | Declaration |
|---|---|
| Physics window | Eight physical substeps at h=1/128;2048 world-steps if all accept |
| Sphere | Actual mass2kg, radius1/16m, zero gravity, declared nonzero initial position/velocity |
| Rotor | Real one-revolute `DrivenRobotProgram`, mass2kg, COM at pivot, axis z, principal inertia(.5,.5,.5), zero gravity, no stops/delay, zero KP/KD, original8Nm effort cap |
| Connection | `TyxWorldBatch` → `ResidentVectorEnv` → `PolicyRunner` with integer decimation3 |
| Horizons | 3/5/7/9 steps, repeated independently across both families; completion uses normal timeout autoreset |
| Control | One declared NumPy linear state-feedback callback; held three-component actions, with only the scalar torque channel used by rotors |
| Independent oracle | Sphere `a=F/2`; rotor `alpha=tau/(1/2)`; `v+=h*a`, `q+=h*v`, independent held-action clocks and original-state reset recurrence |
| Existing comparison budgets | rtol5e-5, atol5e-6; no budget fitting after observing outputs |

The reference trajectory never calls a product HOST mass, FK or dynamics oracle. Its own q/v, accepted-step clocks, horizon counters, held actions and reset state evolve separately from the actual device output. Every frame checks the physical state, input action, due mask, reward, final/pre-autoreset observation, current observation, termination/timeout masks, episode ID/length and runner-local clocks. Some worlds reset twice, some once and some never inside the window. Policy calls must actually be skipped while actions are held; independent reset phases must generate mixed due masks. Actual input forces/torques must change over the window, and kinetic energy is checked from the declared masses/inertia.

After the eight steps, a sparse explicit reset of indices0/127/255 verifies unchanged full checkpoints, held actions and clocks for every unselected world. Selected worlds return to their real initial state and request a fresh action. This adds no physical step. It tests both existing motion families, not a general six-DOF free rigid body or an unimplemented contact family.

The positive record saves all actual full checkpoints, commands, load-time/final states, complete `VectorStep`/`PolicyRunnerStep` records, policy inputs/proposals, independent expectations, energy summaries, reset states and resource statistics as lossless gzip JSON. Arrays include dtype/shape/raw bytes and values. The first failure retains the attempted frame, current published state and scheduling snapshot. A fixed64MiB decoded-record bound rejects oversized evidence rather than dropping fields.

These are real TYX physical kernels with a NumPy/HOST observation/policy/control boundary. There is no contact geometry in the positive case. No pure GPU end-to-end, PPO learning, humanoid GPU dance, W256 contact feedback or throughput result is claimed. Counters/transfer statistics are descriptive, not a performance acceptance gate.

## Authored negative: retain the current loaded registration boundary

`test_connection_w256_contact_registration_rejects_existing_shape_capacity` constructs exactly the512 valid box/floor shape declarations and checks the original `CollisionModel` shape-count rejection. No batch, plan, device runtime or physical step is created. It preserves every shape/key, the actual error, original shape/vertex caps and explicit `physical_steps=0`. This is a HOST declaration negative, never a loaded W256 positive.

## Parent serial selection

The two authored nodes remain uncollected/unexecuted. The parent may run the declaration negative once under its HOST slot, then select **only the physical positive** for each backend if time remains. Do not rerun W32 or repeat an already accepted selector.

```text
python -m pytest tests/scale/test_resident_connection_w256.py::test_connection_w256_contact_registration_rejects_existing_shape_capacity -q -x --tb=short -p no:cacheprovider --basetemp=<new absolute path>
python -m pytest tests/scale/test_resident_connection_w256.py::test_connection_w256_mixed_motion_policy_hold_and_independent_resets --require-tyx --tyx-mode=interpret -q -x --tb=short -p no:cacheprovider --basetemp=<different new absolute path>
```

CUDA uses the same exact positive selector with `--tyx-mode=cuda` and a different fresh evidence root. Parent source review, first exact-file lint/format checks and diagnosed source binding precede execution. A successful run would establish bounded mixed-motion connection at W256 only; loaded-scale support needs a separate collision capacity/ownership/one-step-plan design and subsequent tests.
