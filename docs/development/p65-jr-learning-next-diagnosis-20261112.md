# J/R weak learning: one bounded credit-assignment diagnostic

Status: RESEARCH_ONLY / NOT_IMPLEMENTED / NOT_EXECUTED.
2026-11-12. This proposal changes no model, objective, initialization, seed set,
training budget, acceptance gate or already accepted source. Parent remains the
only project execution owner. The new experiment needs a separate implementation
and execution grant.

## Decision

Do one **frozen-policy 32-step rollout and upstream backward, stopping before
Adam mutates anything**, on J14011, R14211 and the existing positive control
S14111. Use the original final actor and critic of each first declared seed,
not the centered N4d variant. This is one instrumentation protocol with three
fixed controls, not three new learning experiments. It fills the specific
missing value/advantage/gradient evidence. Do not repeat mean centering,
rerun128updates, tune gamma/lambda/lr/std, change decimation, or search seeds.

J's reference controller has now demonstrated reachability through the actual
engine, so another J motion/force experiment is lower priority than observing
what PPO credits. R has no corresponding reachability qualification: its new
measurements are comparative descriptions, not proof that R's target is reachable
or that its weakness has the same cause as J.

## Why this follows from actual evidence

- [N4a](p65-mujoco-tyx-n4a-report.md) completed nine128x32x8 runs. Original median
  learned/untrained ratios were J0.9439251168, S0.7359448237, R0.9182687886.
  S passed the original three-seed gate; J/R did not. Every learned R seed
  remained worse than zero action; only J14013 slightly beat zero.
- [N4b](p65-mujoco-tyx-n4b-diagnosis.md) bound complete requests, native controls,
  reward/cost, worlds and clocks. J/R error was predominantly position error;
  persistent same-sign mean forces favoured one initial-state group and harmed
  its opposite. No learned J/R coordinate reached the declared95%-of-action-limit
  bin. All nine Adam states reached128; parameters and post-clip gradients were
  nonzero. Thus neither silent no-update nor observed control saturation is
  supported as the explanation.
- N4c's actual fixed-nine summary has status
  PASS_FIXED_POLICY_RESPONSE_ONLY, with nine bound initial/final policy pairs.
  Its target/even/odd/cross-joint response results describe offset and feedback
  asymmetry. Those are fixed-input secants, not stability or causality proofs.
  The original observations already contain centered/scaled position and velocity.
  Adding the same normalization again is not a missing feature.
- [N4d](p65-n4d-initialization-protocol.md) tested one-time target-mean centering
  on all three original J seeds, unchanged128updates. Ratios were
  .977842/.993057/1.028220; the median .993057 failed the unchanged gate and the
  third seed worsened. Centering alone is insufficient in this budget. This
  factor is closed, not a reason to recenter after every rollout.
- [N4e](p65-n4e-next-diagnostic.md) run131 used the original J held-out8x256 window
  and public action/physics path. Cost .023813799676380473 was obtained with
  maximum terminal |q|3.63e-8 and |v|8.74e-8, with .5751821994781494 force margin.
  This establishes that that fixed J window is achievable without exceeding the
  original action/time limits. The old policy/zero controls are historical
  different-source evidence, not a simultaneous causal control.
- N7's real multi-substep training/restore connections establish integration
  correctness, not learned control quality; do not import N7 decimation/reset
  settings into this diagnosis.

The weak-learning cause remains unproved. In particular, nonzero combined
post-clip norms do not reveal actor versus critic, pre-clip size, directions,
value calibration, or how advantages relate to different initial worlds.

## Actual source facts that constrain the diagnosis

The unchanged [task declarations](../../examples/mujoco_tyx_learning_tasks.py)
use J h=.002, R/S h=.001, horizon256, W8 and the original8 training templates.
The public task kernel already centers/scales q/v. Native force is
action_scale*tanh(raw Gaussian action): J/R scale2, S scale20.
Reward is the original negative normalized quadratic endpoint cost, with
velocity weight.01. No control cost or hidden target substitution is proposed.

The [adapter configuration](../../src/phymeo/training/rsl_mujoco_tyx.py) specifies
RSL5.5.2 MLP[32,32]/tanh, Identity observation normalizers, gamma.99, lambda.95,
one epoch and one minibatch, fixed Adam lr.0003, Gaussian learned log std
(initial.1), entropy coefficient.01, and separate actor/critic clipping at1.
The [session](../../src/phymeo/training/mujoco_tyx_checkpoint.py) invokes
act -> public adapter.step -> process_env_step32times, then compute_returns and
the original upstream update. It does not expose optimizer-free capture today.

Primary local source actually read:
- .venv-training-cuda313/Lib/site-packages/rsl_rl/algorithms/ppo.py:
  act/process_env_step/compute_returns/update, especially129–191 and193–313.
- .venv-training-cuda313/Lib/site-packages/rsl_rl/storage/rollout_storage.py:
  mini_batch_generator225–259.
- .venv-training-cuda313/Lib/site-packages/rsl_rl/modules/distribution.py:
  GaussianDistribution132–238; independent trainable log_std_param.
- .venv-training-cuda313/Lib/site-packages/rsl_rl/models/mlp_model.py:
  forward and public distribution/log-prob properties.

GAE uses delta=r+gamma*not_done*Vnext-V and recursively gamma*lambda=.9405;
the final rollout observation is bootstrapped. Its geometric weighting scale
1/(1-.9405) is about16.8 physics steps, whereas an episode has256. J position
responds to force at orderh^2. These facts make short-horizon credit and critic
bootstrap worth measuring; they do not prove either is the problem.

There is also an important non-hypothesis: one epoch/one minibatch means the
first loss evaluation occurs before any update to the sampled policy. With
the same frozen actor/Identity normalizer, log-prob ratios should remain near1
and old/new KL near0 up to computation effects. Large clipping fractions here
would first be a binding/computation problem, not evidence to tune PPO clip.
Separate gradient norm clipping can still be active.

## Existing records versus genuinely missing fields

| Already saved and reusable | Not saved by N4a; new bounded capture required |
|---|---|
| original manifest/model/dependency/config/source hashes; initial/final actor+critic; final Adam and next_iteration128 | actual intermediate actor/critic snapshots or exact historical terminal RNG |
| all accepted physical transitions, requested actions, three native channels, reward, done/timeouts, full receipts and partial failure prefixes | per-transition critic values, last bootstrap value, stored bootstrapped rewards, raw TD residuals, raw GAE, normalized advantages |
| original untrained/final/zero evaluation costs and N4b's world/joint/time decomposition | sampled old log-prob/mean/std and recomputed minibatch log-prob, KL and clipping inputs |
| aggregate loss and real post-clip parameter gradient norms; complete final parameter changes | separately owned pre-clip actor/critic gradients, mean-network versus log-std contributions and per-world cancellation |

The writer in evaluate_mujoco_tyx_learning.py stores final_policy_and_adam with
actor/critic/optimizer/next_iteration only. It does **not** store the global
RNG at that boundary. Therefore this cannot be called exact continuation of
the old final training rollout. Do not reconstruct missing RNG or old per-update
gradients from scalar summaries. Read each required old stream once with the
existing bounded decoder and bind its original manifest/publication/hashes.

## Fixed first diagnostic

Predeclare before any new payload inference or physical run:

1. Exactly original final J14011, R14211, S14111; W8 in original stable order
   (5,41,77,113,149,185,221,257), original training templates and native h.
   No N4d-centered weights, held-out replacement, random reset, control window,
   architecture changes or fourth policy. Preserve original source identities
   and explicitly bind the newly frozen package separately.
2. Construct fresh diagnostic-owned actor/critic and environment; load and verify
   the complete saved weights/Identity normalizers. After all construction and
   loading, set the explicit diagnostic sampling seed to that original task seed
   (14011/14211/14111). Record caller Python/NumPy/Torch CPU/all-CUDA RNG before
   and after, and restore every caller RNG/computation flag in finally. This is
   a newly declared reproducible sample stream, not recovered historical RNG.
3. Exactly32 accepted policy steps per task, D1, all three native input channels
   from the original mapping. Total768 world-transitions across the three tasks.
   No warmup, preliminary control loop, hidden evaluation, timestep changes or
   automatic retry on a different state. On failure retain the exact prefix and
   failing request; no shortened statistic can claim a complete rollout.
4. Record the actual upstream values, stored rewards and dones while calling
   the original act/process_env_step/compute_returns. These original32-step
   windows do not reach horizon256; timeout-specific behavior is out of scope
   and keeps its earlier acceptance, not a new timeout claim.
5. Invoke the original upstream PPO.update **on the diagnostic-owned learner**.
   Use ordinary parameter autograd hooks to copy pre-clip gradients without
   returning replacement gradients; capture the actual batch and distribution
   quantities. An optimizer step-pre-hook records post-clip gradients then raises
   a dedicated stop sentinel before Adam mutates parameters/state. No substitute
   PPO, surrogate backward, optimizer.step no-op or private runtime bank.
   The implementation must verify the installed optimizer hook API as text
   before freezing code; if no safe pre-mutation boundary exists, stop design.
6. Verify zero completed optimizer steps, exact actor/critic/Adam bytes unchanged,
   expected single upstream backward, and complete data/parameter/RNG bindings.
   Dispose this intentionally interrupted learner/session; it is not a healthy
   update-boundary checkpoint. All32 physical transitions remain accepted and
   recorded: optimizer suppression is not physical rollback.

The upstream minibatch shuffles256 samples. Capture the actual yielded batch
and join to retained (world,tick) through a unique exact tuple of observation,
sampled action and old log-prob; require a complete bijection, refusing duplicate
ambiguity rather than inventing indices. Do not call another randperm or another
stochastic actor forward for measurement. Additional diagnostic arithmetic uses
only captured tensors, never another physical path or policy sample.

## Predeclared measurements and mutually exclusive next decisions

Keep raw tensors; summaries must link to their identity/shape/hash and state
whether FP32-upstream or independent saved-tensor arithmetic.

- Independently reconstruct TD and unnormalized GAE from actual values/rewards,
  including last bootstrap; verify them against upstream returns-values and
  the original normalization equation (including the actual std convention).
- Linearly decompose raw GAE into raw-reward and critic/bootstrapping terms,
  using the same dones/gamma/lambda; record norms and full signed vectors.
  Also separate between-world mean and within-world variation. Do not infer
  causal advantage from a large scalar value loss.
- Record exact old/new log probabilities, ratios, KL and clipping indicators.
  Save per-world/joint score terms A*(a-mu)/sigma^2 and their aggregate/cancellation.
  For J only, compare their signs with negative position error and the N4e
  restoring-direction interpretation. R torque/error signs remain descriptive
  because the coupled dynamics invalidate an independent-joint sign oracle.
  S's nonzero support mean is legitimate, not an offset to center away.
- Record actor mean-network, actor log-std and critic pre-clip norms separately;
  post-clip norms and actual factors; complete gradient arrays, not only norms.
  No claim of gradient direction follows from an unsigned norm.

Apply these ordered branches, per task, so results cannot be relabelled after
seeing them. They are diagnostic routing rules, not new learning pass gates:

A. Incomplete binding, nonfinite values, wrong log-prob replay, failed TD/GAE
identity, or unexpected mutation/optimizer call: INSTRUMENTATION_OR_BINDING_FAILURE.
Stop physical diagnosis and minimize the exact counterexample; no training.

B. Valid A checks, but mean-network gradient is exactly zero (while required
graph/critic evidence is present): NO_MEAN_CREDIT_IN_THIS_BATCH.
Inspect actual score/advantage path; do not call all prior128updates gradient-free.

C. Nonzero mean gradient and critic-component GAE norm exceeds raw-reward
component norm: CRITIC_DOMINANT_BATCH. Prioritize one separately preregistered
critic/bootstrap-calibration study. Record whether signed J score aggregates
change between raw-reward and complete advantages; dominance alone does not
prove wrong credit or authorize gamma/lambda changes.

D. Otherwise, actual actor pre-clip norm exceeds the original threshold1:
ACTOR_CLIPPING_ACTIVE_BATCH. Quantify the recorded factor and contribution
split; clipping occurrence alone does not justify a new threshold or lr.

E. None of A–D: NO_DISCRIMINATING_CAUSE_IN_THIS_CAPTURE. Retain finite useful
gradients and the unresolved sampling/longer-horizon question; stop this
diagnostic rather than adding seeds, more rollouts or new model variants.

S is captured once with identical instrumentation as an accepted-task positive
comparison; no new32-step learning/sustained-support gate is invented. The old
three-seed J/R classification remains NOT_LEARNED in every branch. Agreement
among these three first-seed observations cannot establish a population cause.

## Implementation ownership and stop boundary

If approved, one training author can implement an optional diagnostic helper,
one public example and synthetic tests in NEW files. Parent owns any necessary
session observation seam and all actual execution; no backend/physics edits,
old evaluation rewrites, installed RSL modification or new solver. Prefer a
bounded original-method observer and diagnostic-owned learner; never reuse a
poisoned live training owner. Synthetic tests must cover wrong source/shape,
missing bootstrap, permutation/duplicate ambiguity, incomplete prefix,
nonfinite gradients, unexpected optimizer entry and all-hook/RNG cleanup.

Success is complete, independently bound diagnostic evidence with768 actual
transitions, zero optimizer steps and a justified A–E classification. It is
not improved learning. Failure does not trigger a second sample stream. Two
nonproductive instrumentation rounds stop with a minimal example. Any actual
follow-on learning experiment requires a new single-factor preregistration
using these results; neither larger budgets nor bias centering are preapproved.
