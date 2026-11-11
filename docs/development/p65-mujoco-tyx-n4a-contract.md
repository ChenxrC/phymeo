# N4a: learning effectiveness and bounded end-to-end scale

Status: PRE_EXECUTION_CONTRACT / AUTHORING_ONLY. Base ff478eaa6ee570562b229491cb25c95e17783740.
No project import, collection, numerical probe, training, test, formatter, build or
profile has been run for this contract. Parent owns the sole execution queue.
This is new N4 evidence, not a rerun of accepted N1/N2/N3.

## 1. Source basis and scope

- [Continuous plan N4](../superpowers/plans/2026-11-11-newton-parity-continuous.md#n1n4真实训练与规模):
  three task classes, three seeds, then W1/8/32/64/128/256; learning and throughput are separate.
- Existing public examples: `examples/train_mujoco_tyx_ppo.py`,
  `examples/mujoco_tyx_cuda_policy.py`, `examples/restore_mujoco_tyx_ppo.py`.
  N2 has small actions and very short horizons suited to protocol qualification;
  extending those runs alone is not a justified task-learning benchmark.
- `training/mujoco_tyx_env.py` supplies native affine observations, quadratic
  rewards, bounds termination, pre-reset final state, subset reset and complete
  HOST physics diagnostics. `rsl_mujoco_tyx.py::ppo_config` and N3's public
  `MuJoCoTyxPpoSession.update` supply original RSL5.5.2 optimization.
- Pinned RSL source is the parent's qualified `.venv-training-cuda313/.../rsl_rl`:
  `algorithms/ppo.py::construct_algorithm/act/process_env_step/update`,
  `models/mlp_model.py` and the original rollout storage. No PPO copy is proposed.
- Accepted T1 source `tests/test_mujoco_tyx_tree.py::FORK_XML` supplies the
  branching topology/inertia reference. The new example will contain its own
  explicit model, never import a test helper.
- Old HOST contact PPO demonstrated a separate learning benchmark, but its
  force law, timing, CPU learner and measured improvement are not inherited here.

N4a changes task definitions and adds optional measurement/evaluation consumers.
It does not modify the old physical kernels, constraints, wrapper transactions,
N3 checkpoint schema, old fixtures, dependency pins or core engine task semantics.
The new models are first-class declarations, not modifications to an accepted
fixture followed by reuse of its pass label.

## 2. Frozen tasks

All models use Euler, FP32 actual TYX dynamics, the existing full publication
checks, zero restitution and the admitted native actuator path. Policy action
is `ctrl = action_scale * tanh(raw)`, zero bias; unselected qfrc/xfrc channels are
explicitly zero. No state clamp, controller inside the engine, extra PD, target
teleportation or per-case fallback is allowed.

### J — independent joint regulation

Two independent world-root bodies, each mass1kg, diagonal inertia(.02,.025,.03)
kg m², one x-axis slide, armature .01kg and damping .1N s/m. Body origins
(-.3,0,0) and (.3,0,0); visualization-only boxes halfsize(.05,.04,.03), collision
masks0. Gravity0; no stiffness or joint limits; h=.002s. Two gear1 native motors
with ctrl/force ranges[-2,2]N. Native targets q=(0,0), v=(0,0).
Observation q/.2 and v/1; cost is mean over joints of
`(q/.2)^2 + .01*(v/1)^2`. Bounds |q|<1m, |v|<8m/s.

Static target is an exact zero-force equilibrium. Positive scalar effective
mass1.01 and damping give a directly actuated local regulation task; for example
the analysis-only feedback -2q-.5v is below .25N throughout the initial grid,
well within the declared2N authority. This is reachability reasoning, not a
trained policy, control baseline implementation or an executed stability result.

### S — continuous finite contact support

Reuse the original N1 support-slide XML bytes without changing its physics:
2kg vertical slider, original nonprincipal full inertia(.08,.10,.12,.002,-.001,.003),
zero root damping/armature; moving box halfsize(.12,.08,.06) at z=.109, finite
platform halfsize(.24,.18,.05) at z=0, gravity-9.81, h=.001,
condim1, solref(.02,1), solimp(.8,.9,.002,.5,2), original128 iterations and tolerance.
Only the new task's policy mapping has scale20N; original motor remains unchanged.
Target q=.00075m and v=0. Observation (q-target)/.0005 and v/.05.
Cost `((q-target)/.0005)^2 + .01*(v/.05)^2`.
Bounds q in[-.002,.004]m, v in[-.5,.5]m/s.

**Pre-execution correction:** the initial proposal q=-.004m was rejected by
source-based reachability review before any numerical execution. Native BB face
distance is half the geometric face separation; for this aligned pair it is
(q-.001)/2. The original off-diagonal inertia makes the body inertial frame
different from its body frame, so the general compiler branch gives
body_invweight0=(trace(J M^-1 Jᵀ))/3=1/6. Four active rows share J=1. At rest,
K=1/(.9²*.02²), R=(1-d)/(6d), and each f=K*d*(-native_dist)/R.
At q=-.004, d=.9 and each f=375N: 1500N contact would require about-1480N motor,
outside20N. That proposal is not retained as the current target.

At the frozen q=.00075, geometric separation=-.00025m and native_dist=-.000125m.
The original impedance curve gives d=205/256 and R=17/410. Thus the static
four-row sum is `4*K*(205/256)*.000125/(17/410)`, about29.8N; motor equilibrium
19.62N minus this force is about-10.2N, strictly within[-20,20].
The .25mm geometric overlap is small relative to the .12m moving-box thickness.
These are source-algebra calculations, not measured pressure or a passed support
window. Exact FP32/native mapping and real contact are checked first.

Primary fixed upstream sources:
[MuJoCo3.3.7 body_simple construction](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/user/user_model.cc#L2438-L2540),
[inverse-weight definition](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_setconst.c#L117-L145).
Existing product `_normal_geometry.py` implements the corresponding K/R/impedance;
`_normal_extended_geometry.py` preserves the native BB face half-depth convention.

### R — shared-ancestor branching robot regulation

Three hinges: shoulder z axis/ref .03, left x axis/ref-.04, right y axis/ref .06.
Use the exact original FORK body geometry and inertia: shoulder mass2 at(0,0,.3);
left mass.8 at(.3,.18,.02), right mass1.5 at(-.2,-.14,.1); each body COM
(.04,-.03,.02), fullinertia(.02,.025,.03,.001,-.0005,.0008), joint armature.04,
damping.12, visualization-only box halfsize(.04,.03,.05). Explicit N4 changes
from the original source fixture: gravity0 and three gear1 motors with
ctrl/force ranges[-2,2]Nm. h=.001. Targets are native qpos refs and zero velocity.
Observation (q-ref)/.25 and v/1. Cost is mean over joints of
`((q-ref)/.25)^2 + .01*(v/1)^2`. Bounds |q-ref|<1rad, |v|<8rad/s.

Full current coupled M/J/bias remains in the existing solver. Zero velocity at
the reference is a zero-force equilibrium under g0; positive inertias and
armature make the joint mass matrix positive definite. Direct actuation covers
all three coordinates. An analysis-only local -2(q-ref)-.5v request is below
.35Nm on the initial grid, inside2Nm. This establishes local task authority,
not global convergence, contact locomotion or G1 competence.

## 3. Frozen training and held-out grids

Each task uses W8 for learning and evaluation. Define four sign pairs in fixed
order (+,+),(+,-),(-,+),(-,-), followed by the same signs at the second magnitude.
Every world has nonzero velocity; world keys are 5+36*i. No sampled initial-state
distribution is introduced in this first slice.

| Task | Training eight templates | Held-out eight templates |
|---|---|---|
| J | for magnitudes a=.06 then .10: q=(s*a,-s*a/2), v=(t*.04,-t*.02) | a=.04 then .08: q=(s*a,-s*a/2), v=(t*.03,-t*.015) |
| S | a=.00010 then .00015: q=target+s*a, v=t*(.002 for first magnitude, .004 for second) | a=.00005 then .000175: q=target+s*a, v=t*(.003 for first magnitude, .005 for second) |
| R | a=.10 then .15: q=ref+s*(a,-.8a,.6a), v=t*(.05,-.04,.03) | a=.075 then .125: q=ref+s*(a,-.8a,.6a), v=t*(.04,-.03,.02) |

The pair (s,t) traverses the four declared signs. Held-out q/v tuples are
disjoint from training; tests check this directly before runtime construction.
No target, reward weight, seed, model parameter or grid is fitted to output.

Seeds: J(14011,14012,14013); S(14111,14112,14113); R(14211,14212,14213).
Per seed:128 updates×32 rollout steps×8 worlds=32768 effective transitions.
Episode horizon256 for all eight worlds, decimation1. Use the original fixed N2
32×32 tanh actor/critic, Gaussian std.1, learned logstd, FP32, lr.0003, one epoch
and one minibatch/update, clip.2, gamma.99, lambda.95, entropy.01, Adam,
no observation normalization/AMP/compile/distributed/RND. No hyperparameter
search, curriculum, extra epochs or longer budget after poor outcomes.

Before training, save the untrained actor and deterministic evaluation. After
update128, evaluate the final actor, not the best checkpoint selected afterward.
Each evaluation constructs the original model in eight held-out worlds and
advances all256 steps without early reset. A wrapper timeout on step256 is
allowed only after recording the pre-reset transition; an earlier terminated,
physical rejection or post-publication failure makes the evaluation FAIL,
not a shorter low-cost episode. Cost is computed from all2048 actual pre-reset
states with the declared native-coordinate formula; device reward is checked
against that independent stored-state calculation using existing N1 reward
comparison tolerances. No CPU forward/step supplies policy observations.

Also run one deterministic zero-raw-action evaluation per task on the same held-out
grid. This is diagnostic only and shared across that task's three seeds;
it adds no new learning acceptance gate. Report whether learning also beats zero
action, but never relabel a poorer learned policy as better than this baseline.

## 4. Acceptance and failure classification

For each task, let B_s and L_s be complete untrained/final mean held-out cost
for each of its three seeds. B_s must be finite and strictly positive; zero
baseline yields BASELINE_DEGENERATE, with no division or automatic pass.
All physical/finite/termination gates must pass in all evaluations and updates.
Learning gate: every L_s/B_s<=1.05; at least two ratios<=.90; median ratio<=.80.
Report all three ratios/absolute costs, no best-seed-only result. This is bounded
learning on these tasks, not broad robot mastery.

S additionally consumes actual device `state.last_step.contacts` for the original
platform/moving pair (not endpoint CPU observer contacts). Record present/active
rows, individual and summed normal_force, contact positions, native distance,
actual geometric separation q-.001, input/output times and matching state/step.
For each held-out world, steps65..256 must have four positive-force contact rows
on at least95% of steps; mean total normal force must be between .5mg and3mg.
Peak force and maximum geometric overlap are reported without inventing a new
solver tolerance. These declared application gates apply to both learned and
untrained evaluation; zero-action remains diagnostic. No final-phase support
sample is extrapolated to the whole window.

Before learning, one fixed32-step W8 task preflight per task runs zero raw actions,
retains all states/receipts and must remain finite/accepted. S must show actual
positive device contact in every world somewhere in this short prefix. It does
not certify the complete support gate. A failed preflight stops that task for
model/ABI diagnosis; input/target/budget cannot be altered to chase its output.

Independent tasks may continue after another task fails. Any failure preserves
the complete accepted prefix, requested actions, source/config and actual error
stage. At most two targeted diagnoses of the same blocker; if neither narrows
cause, report the minimal case and stop that task. Poor learning after the fixed
budget is a learning result, not a reason to add updates, seeds or rewards.

## 5. Separate scale protocol and accounting

Only tasks with a completed functional preflight are eligible; learning outcome
is reported separately and cannot turn a scale run into learning success.
Matrix is each of J/S/R × W(1,8,32,64,128,256), using that task's first fixed seed.
World templates cycle the eight training states, with explicit keys/order.
Each cell creates a fresh learner/physical batch, performs one32-step warmup
update, then exactly three32-step measured updates. No replayed output or missing
worlds count toward transitions. Runtime/task caps remain unchanged; unsupported
capacity or allocation is an explicit cell failure, never a smaller actual W.

For this scale-only protocol use horizon64+8*(row%8), declared separately from
learning horizon256, so the 128-step warm+measured prefix exercises actual
subset resets and includes their cost. Measured transitions are96*W per cell.
The warmup update is reported separately; startup/compilation/allocation are not
hidden inside a claimed steady-state speedup. No minimum throughput or required
monotonic scaling is invented.

Primary timer uses wall perf_counter and explicit CUDA completion at the outer
update boundaries. It includes policy forward, action submission/audit, physical
solve/full diagnostics, task copy, episode/reset work, returns, backward/Adam,
and the public session's own complete transition retention. Full evidence
serialization/compression/write time is separately measured and reported both
excluded (compute+mandatory diagnostics throughput) and included (delivered
evidence throughput). Optimizer-only speed is a labeled submetric, never total.

Every cell records cold startup, warmup, each full update, actual accepted
transitions, terminal/reset counts, complete counter deltas and current source/
dependency/device/seed/model hashes. Optional bounded observer samples the fixed
existing warmup update only, with observer overhead explicitly included. The three
formal measured updates remain unprofiled; no extra profile update is run.

Existing exact counters cover runtime uploaded/downloaded/static bytes,
control D2D/raw audit/status bytes, task D2D/status/explicit-readback bytes,
published kernels and named synchronization sites. Wrapper additionally exposes
done-mask bytes and action finite scalar reads. These categories may overlap
at ownership boundaries: report them by source instead of summing duplicate
counters. Torch-owned model/rollout memory is measured by Torch peak allocated/
reserved counters; TYX allocations are separately reported from runtime counters
and CUDA free-memory before/after (not called a precise TYX-only peak).

Existing counters do NOT fully count internal Torch synchronizations, all
implicit blocking tensor copies, or CPU MuJoCo validation duration. The bounded
source-aware observer may count visible calls/costs to public runtime
set_controls_from_device/step/snapshot/reset, workspace refresh/to_torch and
selected Torch/TYX synchronization APIs. Unobservable internal GPU sync remains
UNKNOWN, never0. Inclusive call costs must not be summed as disjoint stages.
Complete public step timing and HOST diagnostic cost retain their actual boundary;
no private physics bank access, solver edit or skipped validation is authorized.
A later production mode or Newton comparison needs a separately matched contract.

## 6. Frozen implementation ownership

Parent authorized exactly these six NEW paths:

1. `src/phymeo/training/mujoco_tyx_evaluation.py`: optional reusable deterministic
   evaluation, cost/complete-window validation and bounded timing/counter helper;
   accepts task callbacks/specs, no task names or XML in the engine.
2. `examples/mujoco_tyx_learning_tasks.py`: immutable model/task/seed/grid registry
   and public runtime/env/session construction, independent of tests.
3. `examples/evaluate_mujoco_tyx_learning.py`: explicit CLI
   --phase preflight|learn|scale|summarize --task J|S|R --seed INT --worlds INT --output NEWDIR;
   validates each fixed matrix cell and preserves per-task failures.
4. `tests/training/test_mujoco_tyx_evaluation.py`: frozen declaration/grid,
   cost/ratio/window refusal, accounting coverage and actual first-slice entry
   tests; parent chooses focused serial execution, not old complete N2/N3 reruns.
5. This contract: `docs/development/p65-mujoco-tyx-n4a-contract.md`.
6. `docs/development/p65-mujoco-tyx-n4a-report.md`: precise authoring and handoff scope.

No old solver, wrapper, session, exports or RSL implementation change is proposed.
The fixed three-seed `summarize` phase reads bound saved summaries/evidence only,
rejects missing/duplicate/changed input bindings, and performs no physical replay.
The zero-action diagnostic is run once by the first predeclared seed per task;
other seeds reference it through that final three-seed aggregation. Preflight is
an explicit one-time phase, never automatically repeated inside learn or scale.
New training code,
whole-grid performance, learning quality and W256 remain UNEXECUTED/UNQUALIFIED
until the original frozen runs and evidence checks actually finish.
