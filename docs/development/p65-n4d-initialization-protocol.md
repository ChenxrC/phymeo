# N4d: one target-mean initialization ablation

Status: CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED.
This protocol was frozen before any
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
bias copy. Candidate mutation never touches the actor. Caller RNG/computation flags are restored before publication; original model
modes are unchanged. A late copy/completion/hash failure
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

## Final authoring handoff (2026-11-12)

All four new files are authored, not executed by this author. Product commits
are d5bb38ba (protocol), 184edc83 (tests first), e8691f9b (generic operation),
01f847fe (fixed wrapper), d7d5856c (metadata/hook cases), a5157b67 (CUDA rollback
and owned receipt evidence), and 2453f013 (failed-receipt classification).
The parent has text-reviewed these files and applied the separate shared hook
patch; source review is not a test or learning pass.

The exact ignored hook artifact is
artifacts/p65-n4d-authoring-20261112/initializer-hook.patch, SHA256
5ca094c32c8283a955bae46cbf7538d175a79324536fbf2fb9d449d02baf52dd.
It contains 11 insertions and one deletion in the parent's original learning
entry. The child has not modified that shared file. Hook-dependent tests need
that parent integration before execution.

The original state_dict evidence is copied through owned immutable bytes before
publication, rather than retaining Tensor aliases. mean_after is the independent
candidate's actual deterministic inference; after publication the complete
actor/critic hash must match that candidate. No original Gaussian distribution
cache is populated or replaced. All original caller RNG/computation flags are
restored before the unique real copy. Successful rollback is distinguished from
prepublication refusal. A real copy/synchronization failure followed by rollback
failure propagates both failures and never reports atomic refusal.

Manual authored enumeration is 28 nodes: 24 HOST and four CUDA. This count is
not a collection result. It covers the actual pinned MLP/Adam, ten admission
negatives, candidate/copy/verification failures on each backend, owned receipt
history, fixed-three-run orchestration, five original-metadata bindings, three
shared-hook paths and one actual CUDA TYX target-feature case. The parent alone
may run qualified static checks and the following focused commands:

```text
<qualified-training-python> -m pytest tests/training/test_policy_initialization.py -m "not cuda" -q -x
<qualified-training-python> -m pytest tests/training/test_policy_initialization.py -m cuda -q -x
<qualified-training-python> examples/evaluate_mujoco_tyx_initialization_ablation.py --evidence-root artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01 --output <fresh-N4d-output>
```

The training command calls the original learning entry for exactly the three
fixed seeds, retains all actual prefixes/failures and calls the original gate.
Neither these tests nor this wrapper authorize a fourth seed, larger budget,
changed target, repeated centering or an alternative learning threshold.

## Historical-control interpretation boundary

The original N4a J control is preserved historical evidence with its original
source hashes. The present complete package also includes subsequently qualified
ROB2b/lifecycle and other shared changes. The task/configuration/seed/initial
policy bindings are strict, and the three new runs will use one frozen product
source snapshot, but this is not a simultaneous same-package unmodified control.
No claim of uniquely isolated causal attribution across those package revisions
is made, and the old control is not rerun for this slice. The experiment is one
fixed-budget test of the initialization hypothesis. Absolute old/new untrained,
final and zero-action costs, original/new source bindings and all contradictory
outcomes remain visible. A successful variant gate would not retroactively turn
the original J failure into a success or establish that initialization was the
only weak-learning cause.

Current result: UNEXECUTED by author. No new inference, project import,
collection, numerical probe, lint/format, test, training or build was run here.
Final four-path SHA256/Git-blob manifest is handed off separately after this
report commit. Parent serial validation and actual three-seed outcomes remain
pending.


## Parent serial qualification (02:31)

Author UNEXECUTED above is historical. First HOST run89 retained an RSL layout failure: the pinned MLP reuses the same Tanh object, so children() removes repeated applications. Sequential iteration now validates the actual forward registration order without relaxing the Linear/Tanh whitelist. Run93 passed 24 HOST cases and run90 passed 4 CUDA cases. Independent text review also identified frozen mean bias admission; the head must remain trainable, and a new frozen-bias refusal plus the actual trainable positive passed run96 (2 cases). Runs91/92 and94/95 are exact-scope static successes. The CUDA model publication and all existing cases retain their earlier qualified source bindings; run96 is the focused admission delta, not a claim that all cases were rerun on that delta.

The original failure is preserved. No physics or optimization was run by these tests except the declared unit test optimizer step. The fixed three-seed physical learning experiment follows separately; its outcome is still pending.
