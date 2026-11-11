# N4d: one target-mean initialization ablation

Status: PREDECLARED / AUTHORING_ONLY. This protocol is frozen before any
ablation training. It does not replace the original J failure or alter S/R.

## Identity, factor and budget

Only J seeds 14011, 14012, 14013; original task XML, world order, train/held-out
templates, h=.002, action mapping, 32x32 tanh actor/critic, Gaussian learned
log-std .1, lr .0003 and the full original PPO configuration remain unchanged.
Each run remains 128 updates x32 rollout steps x8 worlds, with original
256-step untrained/final held-out evaluations. Final means update128, never a
selected intermediate checkpoint. No new seed, gain search or extra update.

The one factor is the actual initial actor's final mean Linear bias:
b_new = FP32(b_old - mu_old(observation_target)).
Every other actor tensor, critic tensor, Gaussian std parameter and fresh Adam
state is identical. The bias remains an ordinary trainable parameter afterward.
There is no per-step mean subtraction, fixed PD feedback, physical model edit
or object/time special case. Initial/final policy hashes and all old results
remain separately identified.

The unchanged three-seed learning gate applies to the variant's own untrained
baseline: every ratio<=1.05, at least two<=.90, median<=.80. Both absolute costs
and ratios are reported alongside original untrained/final/zero costs. A changed
baseline alone cannot establish superiority. The original J classification stays
FIXED_BUDGET_NOT_LEARNED. Weak or contradictory results stop this factor; no
budget or threshold is adjusted afterward. S's passed support result is reused,
not rerun or forced to have zero equilibrium control.

## Public operation and computation boundary

NEW policy_initialization.center_initial_mean accepts an explicit TensorDict
reference observation and actual actor, critic, optimizer plus expected whole
actor/critic hash. Supported actors are exactly pinned RSL5.5.2 MLPModel with
Identity observation normalization, state-independent GaussianDistribution and
an actual final Linear mean head. Only finite FP32 bounded layouts and a fresh
Adam owning the unchanged actor/critic parameters are admitted. Arbitrary class
loading, RNN, normalizer updates, state-dependent standard deviation and trained
optimizer state are rejected.

Full validation and independent candidate inference precede the unique real
bias copy. Candidate mutation never touches the actor. Caller RNG and temporary
model modes are restored before publication. A late copy/completion/hash failure
rolls the real bias back; rollback failure is explicit and cannot be called an
atomic refusal. No physics/optimizer step is part of this operation. Complete
before/after state tensors, hashes, original mean, candidate mean/residual and
actual published result are recorded. Near-zero residual is diagnostic, not a
new numerical acceptance threshold or repeated centering loop.

The wrapper obtains the reference observation from the original actual TYX
feature kernel using owned q=q_target/v=0 native FP32 arrays and the unchanged
task scale/bias. It reuses N4c's accepted independent feature-launch helper and
requires its spec SHA to equal the live environment task spec. It does not
hand-fill a zero observation, reset the physics to target, create a substitute
physical runtime or run CPU forward. Generated native inputs, kernel outputs,
feature parameters and source hashes are retained. Generic initialization knows
no J identity; the example owns this fixed experiment.

All Python/NumPy/Torch CPU/all CUDA RNG are preserved; original IEEE operation
configuration is used through the accepted N4c execution context. No allocator,
arbitrary caller Generator or concurrent mutation rollback is claimed.

## Shared hook proposed to parent

The author will provide an ignored unified patch, not edit the shared original
learning entry. run_learning gains optional keyword initializer=None. When
non-None, call initializer(session,env,name,seed) immediately after session
construction, before manifest/initial_policy/hash. The initializer receipt is
added only on that branch to manifest, trace (after initial_policy) and summary.
With None the old numerical path and emitted record set/fields remain unchanged.

NEW example calls original run_learning exactly once per fixed seed and reuses
original summarize_task/learning_outcome. It binds the previously accepted J
three-seed anchor and three original summary/publication identities before
starting. It does not reread old complete trajectories or repeat old audits.
Ordinary seed failure retains its real prefix and permits the other independently
declared seeds; absent/incomplete runs prevent a successful aggregate.

## Frozen files and tests

Only NEW:
- src/phymeo/training/policy_initialization.py
- tests/training/test_policy_initialization.py
- examples/evaluate_mujoco_tyx_initialization_ablation.py
- docs/development/p65-n4d-initialization-protocol.md

Original tasks/evaluation/learner/engine are not edited by this author. The
shared optional-hook patch is handed to the parent separately.

Tests are written before implementation. They exercise real upstream MLP/Adam
on CPU and CUDA, exact only-bias mutation, complete critic/std/optimizer/RNG/mode
preservation, malformed input/class/state/hash refusal, prepublication and late
publication failure/rollback, and fixed three-seed wrapper/original-hook
composition using short synthetic orchestration. One real TYX target-feature
case verifies native input/spec/feature provenance without advancing physics.
Tests do not execute the 128-update experiment themselves; the parent owns that
single fixed execution and all static/collection/test/build commands.

No implementation or learning result is claimed by this preregistration.
