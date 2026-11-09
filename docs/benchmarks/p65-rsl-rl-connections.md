# N4 optional RSL-RL connection and two physical learning targets

## Parent execution — 2026-11-10

Subsequent N9 also passed directly from the final installed wheel: sixteen real HOST contact scenes, original fixed100×24 budget, overall cost reduction64.8555%, .5kg/1kg groups44.1993%/84.1158%, actual public contact loads and fresh-world native checkpoint replay. See [contact PPO acceptance](p65-contact-ppo-connections.md). This extends the learning workflow to actual environment interaction; CPU/HOST copy and non-G1-training limits remain.

The complete optional adapter file ran with actual dependencies: **10 passed**, no skip. Both predeclared real HOST tasks passed their unchanged20% learning gate with the original seed20261110 and100 iterations: prismatic cost .4059548378→.0620736182 (84.7092% reduction), revolute cost .7493789196→.0074049751 (99.0119%). Actor weights changed. Native RSL checkpoint save/load reproduced192 evaluation steps on fresh original worlds, including actions/observations/costs and accepted q/v/final masks. RAW_EXIT0.

Actual resolved versions are **Torch2.15.0 CPU, TensorDict0.15.5, RSL-RL5.5.2, NumPy2.4.6**. The initial Torch2.10 installation was superseded by dependency resolution; it is not the accepted final environment. Full freeze and pip check are retained; no broken requirements. The separate TYX environment/pin and NumPy-only core remain unchanged. Evidence is `artifacts/p65-connections-20261110-0800-01/17–19`, `41-training-freeze`, `42-training-pip-check`, and `ppo-outcome.json`.

This is CPU policy/HOST physics with copies, no contact in the two learning tasks, no G1 training and no GPU training or differentiation claim. The author-only statements below are historical; the frozen experiment and limitations remain.

**AUTHORING_ONLY / UNEXECUTED.** This conditional work follows N2 at `be835102e613f3bd978c41c5f80adcd8d9bf8711`. No package installation, project import, Torch calculation, pytest, collection, lint/format, compilation or training was executed. The parent retains the single execution slot while the original G1 long window runs. Four new files only; base exports, core physics, dependencies/pins and existing tests are unchanged.

## Research before implementation

The official [release page](https://github.com/leggedrobotics/rsl_rl/releases/tag/v5.5.2) confirms v5.5.2 at commit `1156331`; this is a released version, not merely main's version string. The [tagged package metadata](https://github.com/leggedrobotics/rsl_rl/blob/v5.5.2/pyproject.toml) declares Torch>=2.6 and TensorDict>=.7. The adapter requires exactly `rsl-rl-lib==5.5.2`; optional dependencies belong in the parent's isolated CPU environment, never the base TYX environment.

The public [VecEnv contract](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/env/vec_env.py) defines TensorDict observation groups and requires returned observation buffers to remain stable until rollout storage consumes them. The checked [tagged runner](https://github.com/leggedrobotics/rsl_rl/blob/v5.5.2/rsl_rl/runners/on_policy_runner.py) constructs PPO from those observations, exposes save/load and deterministic inference, and has an optional clock-only initial episode randomization. The [configuration guide](https://github.com/leggedrobotics/rsl_rl/blob/main/docs/guide/configuration.rst) describes actor/critic group selection; [tagged PPO](https://github.com/leggedrobotics/rsl_rl/blob/v5.5.2/rsl_rl/algorithms/ppo.py) consumes `time_outs` for time-limit bootstrapping. VecEnv/guide main pages supplement the checked release runner/model/algorithm sources; their URLs are distinguished deliberately. No upstream optimizer or PPO implementation is copied into Phymeo.

## Optional adapter

Explicit import only:

```python
from phymeo.training.rsl_rl import RslRlVecEnv
adapter = RslRlVecEnv(resident_vector_env, device="cpu")
```

The adapter borrows the existing `ResidentVectorEnv`; it owns no physical backend and has no close operation. Actions are detached and copied to HOST NumPy before entering the existing step. Observations/rewards/masks become newly owned Torch buffers on the selected tensor device. Every observation TensorDict has batch dimension W and a `policy` group; actor and critic can both select it. This remains a synchronous HOST-copy bridge, including when a user explicitly requests CUDA tensors. It is not device-resident simulation/tensor interoperability.

Feature width is fixed from the initial observation. Every subsequent frame and final observation must retain that width and fit finite float32. Previous TensorDict frames are never reused or overwritten. The returned extras contain `time_outs`, `terminated`, `truncated`, `final_observation`, `final_mask`, `episode_id` and `episode_length`. Horizon metadata and counters are copied read-only from the same package's environment metadata; the adapter does not edit them. Episode-length getter tensors are detached copies. Clock-only `init_at_random_ep_len=True` is explicitly rejected rather than silently changing bookkeeping while physical worlds stay unchanged; the example passes False.

N2 physical rejection exceptions propagate unchanged, including immutable accepted-world masks and unknown-publication flags. A conversion failure after a successful env transition retains that transition on its error, marks the adapter reset-required and reports all-True physical provenance. This does not pretend to undo either physical or environment publication. Reset the whole adapter after such failure. Use one consumer, and reset through the adapter while it is being used; external physical steps/resets cannot be synchronized implicitly.

## Fixed learning experiment, declared before execution

[train_scene_ppo.py](../../examples/train_scene_ppo.py) delegates all learning to the actual RSL-RL `OnPolicyRunner`/PPO. Both tasks use original public `ContactSceneBuilder`→HOST `RobotScene`→`SceneBatchBackend`→`ResidentVectorEnv`→optional VecEnv adapter. No q/v replacement, trajectory playback, artificial dynamics backend or custom optimizer is used.

| Item | Frozen declaration |
|---|---|
| Tasks | One massive prismatic joint reaches fixed goals; one massive revolute joint tracks fixed velocities |
| Physical plants | 1 kg body, explicit principal inertia(.2,.2,.2), sphere collider radius1, zero gravity, real fixed base; original URDF loading and joint dynamics |
| Goals | Prismatic [-.25,-.15,.15,.25]m; revolute[-.75,-.5,.5,.75]rad/s; each repeated in registration order across16 worlds |
| Action/effort | Finite scalar action clipped to[-1,1] in the declared task; prismatic scale2N, revolute scale1Nm; physical motor cap2 |
| Initial state and reset | q=v=0; fresh original scenes from factory; no randomized physical reset |
| Budget | Seed20261110 for each task; N16;100 PPO iterations;24 env steps/iteration;h=1/32s;episode96steps |
| PPO | Two32-unit ELU layers; Gaussian std.4; Adam default with learning rate.0003;4 epochs/4 minibatches; fixed schedule; gamma.99/lambda.95; clip.2;entropy.01; no observation normalizer |
| Evaluation | Same deterministic mean policy before/after training; each192physical steps over the same original16 worlds, with normal horizon resets |
| Cost | Squared normalized position/velocity error; prismatic .05v² term; both .001*actual_effort²; identical before/after |
| Success target | Actor parameters actually changed and mean deterministic evaluation cost drops by at least20% on **each** task |
| Checkpoint | Native runner save/load includes actor/critic/optimizer/iteration and source/plan infos; fresh-world loaded evaluation must reproduce actions, observations and costs |

The generous joint/velocity limits are declared with the plants, not tuned after failures: under the declared maximum effort and96-step episode the simple free joint remains inside them. The tasks have no environment contact and therefore establish actuator-learning connectivity only, not learning manipulation, foot support or collision behavior. N2/N3 separately cover loaded interaction.

Both tasks always use the same fixed seed/budget. A missed20% gate is recorded as `learning_target_missed` with exit1; there is no seed search, budget extension, mass/threshold change or second learning attempt. A physical/API error stops the run and preserves its last request/accepted context and causal traceback. Completion, reproducibility and actual learning are separate checks. Checkpoint evaluation starts fresh physical episodes; mid-episode environment/rollout/RNG resume is not claimed.

## CLI, outputs and parent execution

Only after the parent finishes N1 and validates N2/N3, and only if the time budget permits, install the optional requirements in an isolated CPU environment. No dependency installation was performed by this author. The local logger is explicitly TensorBoard; no W&B/Neptune/network logging is enabled.

```text
python -m pytest tests/test_rsl_rl_adapter.py -q -x --tb=short -p no:cacheprovider --basetemp=<fresh absolute path>
python -B examples/train_scene_ppo.py --output-dir <new absolute directory>
```

The example refuses an existing output directory. `declared_plan.json` precedes the learning steps; `summary.json` records real result/status and dependency/source versions. Each task retains its exact URDF, local TensorBoard/checkpoint files, native `trained.pt` with SHA, and `complete_result.json.gz` with all deterministic before/after/reloaded evaluation frames/actions/costs, pre-autoreset accepted joint q/v and terminal observations/masks, plus failure context. The compressed result preserves array dtype/shape/raw bytes and explicit nonfinite words when reporting failure. It does not pretend to store every exploratory training frame or an exact training resume state.

[test_rsl_rl_adapter.py](../../tests/test_rsl_rl_adapter.py) uses logical publication clocks strictly for adapter semantics: fresh buffer lifetime, terminal observations/timeouts, invalid tensor actions, mask propagation, post-physics conversion failures, clock-randomization rejection, and actual runner construction/save/load. These unit fixtures are not the physical learning result. Missing optional dependencies skip the optional test file explicitly; that is not acceptance. The real two-task learning result exists only after the separate example completes its original gates.

Uncollected author inventory:10 optional unit instances and two fixed real learning runs. Parent should retain the first lint/format/test or learning failure, then diagnose API/style issues without changing seeds/plants/rewards/budgets to chase a result. This author checkpoint contains no learning pass, GPU training claim, W256/W1024 measurement or universal Newton parity claim.

## Optional installation boundary

The package declares `phymeo[training]` with Torch>=2.6,<3, TensorDict>=0.7,<1 and RSL-RL5.5.2. Core dependencies remain NumPy only; importing base `phymeo.training` does not import Torch/RSL-RL. The parent may first install a CPU-only Torch wheel into an isolated `.venv-training`, then the optional adapter dependencies; this does not change the TYX environment or pin. For this bounded experiment, Torch2.10.0 CPU is an available official Windows build, not a latest-version claim: https://pytorch.org/get-started/previous-versions/ . The actual resolved versions and install receipt must be saved before any acceptance claim.

Official RSL5.5.2 runner and PPO interfaces were reread from the exact tag. The native runner owns rollout/update/save/load; Phymeo supplies fresh observations, original dones and timeout metadata. Its timeout bootstrap follows the external algorithm; merely preserving `final_observation` does not establish custom terminal-value bootstrapping or recurrent-state resume. Sources: https://github.com/leggedrobotics/rsl_rl/blob/v5.5.2/rsl_rl/runners/on_policy_runner.py and https://github.com/leggedrobotics/rsl_rl/blob/v5.5.2/rsl_rl/algorithms/ppo.py . This optional module remains UNEXECUTED at authoring time.
