# Credit diagnostic: decision after the fixed three-batch capture

Status: **DOCS_ONLY_NEXT_DECISION**. Parent run301 completed the fixed diagnostic;
this author read its complete summary and fixed local implementation text only.
No trace was decompressed, no new arithmetic script/inference/rollout/backward
was run, and no optimizer setting or learning acceptance changed.

## What the actual result establishes

The authoritative summary is
`artifacts/p65-parity-until-0720-20261112-01/301-n8-fixed-credit-cuda/result/summary.json`
in the parent integration tree, SHA256
`28172b1f4c319f9bb476609959fcb12d29a58213f681266765165f00ffff0365`.
Its status is PASS_FIXED_CREDIT_DIAGNOSTIC_ONLY, RAW_EXIT0:768 accepted world
steps, zero new optimizer updates, one blocked Adam entry per task, all original
Adam steps128→128, and exact caller restoration. Parent separately reported the
30 HOST checks passed; that is not another physical or learning result.

| Fixed task/seed | Actor norm before → after clip | Actual norm factor | Critic-component / reward-component GAE norms | Frozen branch |
|---|---|---|---|---|
| J14011 | 1.9410654089544308 → 0.9999994480104082 | 0.5151807061200814 | 1.5118929705138195 / 19.648909274618166 | D_ACTOR_CLIPPING_ACTIVE |
| R14211 | 2.4977136418268255 → 0.9999996424994468 | 0.4003660090385894 | 1.5913445880334451 / 31.15074068589555 | D_ACTOR_CLIPPING_ACTIVE |
| S14111 | 0.17927137365425766 → 0.1792713736542577 | 1.0000000000000002 | 2.4112851295784408 / 10.056343562974973 | E_INCONCLUSIVE |

S's last-bit norm-factor excess is a ratio of independently accumulated norms,
not evidence that clipping amplified its gradient. All mean-network gradients
are nonzero. All three critic gradients were also clipped: actual factors
J0.22047981744352035, R0.11919835202764903, S0.4591832748195081.

All three have ratio-clipping count0 and reported mean KL0. This is the first
actual minibatch recomputation with the same parameters used for this fresh
rollout, before any update; near-unit ratios are expected. It does not establish
that PPO clipping was inactive during the historical128 updates, or that the
policy cannot change. The saved full score/gradient evidence is meaningful;
near-zero mean surrogate or KL at this point is not absence of a gradient.

The frozen branch D says only that this batch's actor gradient was clipped.
The GAE comparison does not prove correct credit, but does rule out the specific
predeclared *norm dominance* branch C in these three captures. S is one batch
from a task with prior learning success, not a matched causal control for J/R.
J/R remain NOT_LEARNED; this result does not reopen rejected bias centering.

## Why a gradient factor is not an Adam step factor

The installed fixed primary sources read were:

- `C:/Users/38347/AppData/Roaming/Python/Python313/site-packages/torch/nn/utils/clip_grad.py`,
  especially the norm factor and foreach multiplication around lines159–179;
  SHA256 `3b37b88692672b2a006e8829c6874bc8e04d73faace5aeb95829bb7c9354b159`.
- The same installed Torch root's `optim/adam.py`: first/second moments around
  lines454–475, bias correction/denominator/addcdiv around528–545, corresponding
  foreach path around696 onward, and default dispatch around935 onward;
  SHA256 `1adac65df82cfc583c02eb88e9aa55126af2d27ea485230481403db38857b15c`.
- Parent `.venv-training-cuda313/Lib/site-packages/rsl_rl/algorithms/ppo.py`:
  optimizer parameter order is actor then critic with identity deduplication;
  original backward is followed by separate actor/critic norm clipping and the
  same Adam step. No normalizer update occurs before the intercepted step.

Torch version.py is2.12.0+cu126. `clip_grad_norm_` uses
`c=min(1,max_norm/(total_norm+1e-6))` and applies one scalar to each parameter in
that actor or critic set. Apart from FP32 multiplication rounding, this preserves
the direction of the concatenated gradient within that set. It does not select
joints or suppress only the mean head. Actor and critic have different factors.

For the admitted ordinary Adam branch, with the *actual saved group parameters*
(not assumed defaults), a hypothetical next update using input gradient z is:

```text
m129(z) = beta1*m128 + (1-beta1)*z
v129(z) = beta2*v128 + (1-beta2)*z*z
delta(z) = -lr * [m129(z)/(1-beta1^129)]
                / [sqrt(v129(z)/(1-beta2^129)) + eps]
```

These are elementwise expressions. The original stored groups must verify
weight_decay0, maximizeFalse, amsgradFalse and the other branch flags before
using them. The implementation may choose foreach when its arguments are None;
this algebra is not a claim of bitwise equivalence to an executed GPU update.

A common positive rescaling of *the entire gradient history*, with consistently
rescaled moments and eps0, would cancel between Adam's numerator and denominator.
That is not this situation: m128/v128 already exist and stay unchanged when only
this next input is scaled. The historical numerator contribution can align with
or oppose the current gradient, and the denominator mixes historical squares
with the current squared gradient. Consequently the proposed parameter change
may shrink much less or differently than c, and its direction can change too.
Eps further breaks scale invariance where the corrected second-moment root is
small. Neither “Adam cancels clipping” nor “factor0.4 means a60% smaller update”
is justified from the summary alone. The actor mean/log-std group norm split
also cannot show per-parameter history alignment.

## Smallest next discriminator: only the existing saved data

A separately authorized saved-only pass can bind the run301 gzip's already
published compressed/logical hashes and consume it once. It needs no new model,
policy call, environment, optimizer object, random number or training update.
Only the following three fixed task records are used; there is no grid or search.

1. Bind each captured pre/post gradient name to the complete loaded actor/critic
   parameter tensor and saved Adam ID/moments/group. Use the fixed upstream
   registration/actor-then-critic order, exact names/dtypes/shapes and the existing
   distinct-parameter contract. Do not match repeated shapes heuristically. If
   the recorded schema plus fixed source cannot establish a unique ID→name map,
   stop with that exact missing binding; do not instantiate/step an optimizer to
   guess. Check all original steps128 and before/after learner digest equality.
2. In owned FP64 arrays, evaluate the two algebraic candidates above using
   **the actual saved post-clip gradient** and **the actual saved pre-clip gradient**,
   with identical saved m128/v128/eps/lr/betas. Do not rescale old moments, zero
   momentum, choose another threshold, write any parameter or call Adam.step.
   Save both complete candidate arrays, clearly labelled virtual step129, with
   optimizer_steps_executed0. This is not the already accepted next physical step.
3. Report, separately for actor mean, actor log-std and critic: candidate norms,
   their norm ratio, cosine/difference norm, and `-g_pre dot delta` (a local linear
   objective prediction, not measured reward improvement). Zero norm gives an
   explicit undefined angle/ratio, never a favorable substitute. Also retain
   the fixed numerator terms beta1*m128 and (1-beta1)*z and denominator terms
   beta2*v128, (1-beta2)*z² and eps, so the relative history/current/eps influence
   can be inspected without adding more counterfactual runs.

These quantities distinguish a raw-gradient attenuation observation from an
actual algebraic Adam attenuation/rotation under the saved history. They still
cannot establish that clipping caused weak long-horizon learning. No numerical
values for these proposed quantities have been computed in this document.

## Decision and stopping conditions

Do **not** change max_grad_norm, lr, seeds, target, horizon or update budget now.
Do not automatically promote branch D to a clipping-ablation grant. First finish
binding and the bounded saved-only discriminator if parent authorizes it.

- Missing/ambiguous state binding, unsupported Adam flags or a mismatch with the
  saved pre/post gradient relation: stop that analysis with a minimal schema or
  arithmetic counterexample, rather than collecting a second rollout.
- If the two virtual updates do not support the proposed “clipping materially
  suppresses a useful update” story, close that hypothesis for this capture.
  Report the continuous values and their precision scope; do not invent a
  post-hoc learning threshold or search another seed for a stronger effect.
- Even if a discrepancy is clear, its usefulness is only a local surrogate
  prediction. Before any new single-factor learning experiment, preregister the
  precise factor, a quantitative predicted effect, one fixed treatment/control
  protocol, source/config/RNG binding and the unchanged original three-seed
  learning gate. All outcomes must be kept. The existing first-seed final-policy
  capture alone does not authorize choosing a new clip value or adding epochs.
- No defensible prediction after the saved pass means E/inconclusive and a stop
  to this causal line, not more credit captures. Continue independent framework
  functionality. S remains a limited contextual comparison; its one un-clipped
  actor batch cannot validate a general no-clipping policy.

The parent owns any later saved-only implementation/execution and the decision
whether evidence warrants a newly frozen experiment. This document authoring
changes only this file. No experiment beyond run301 is claimed or authorized.
