# N4e: one bounded reachability diagnostic before another learning change

Status: PROPOSED_ONLY / UNEXECUTED. This document grants no execution or new
training. The parent owns the serial execution slot. No N4a/N4b/N4c/N4d source,
model, seed, reward, learner configuration or acceptance gate is changed.

## Evidence that narrows the next question

The closed run97 summary reports all three original J seeds completed, with
`FIXED_BUDGET_NOT_LEARNED` unchanged after one-time initial mean centering:

| Seed | Centered initial cost | Final cost | Final / initial |
|---|---:|---:|---:|
| 14011 | 0.07249307744997935 | 0.07088676235331695 | 0.9778418139611915 |
| 14012 | 0.0648595588645807 | 0.06440924883216323 | 0.9930571524028144 |
| 14013 | 0.06064320512386087 | 0.06235456357377333 | 1.0282201187489528 |

Median ratio is 0.9930571524028144; zero-action cost is 0.063695137045106.
Only seed14013's final policy beats zero, while that policy worsens relative to
its centered initialization. These are copied summary values, not a new
calculation or re-audit. The input is parent
`artifacts/p65-parity-until-0720-20261112-01/97-n4d-fixed-three/ablation/summary.json`.
The original J failure and N4d historical-source comparison boundary remain.

[N4b](p65-mujoco-tyx-n4b-diagnosis.md) already establishes complete input/cost
bindings, changed parameters, nonzero post-clipping gradients, and predominantly
position cost. N4c additionally found bias, wrong-sign own-axis response and
cross-axis response. Centering alone was therefore a testable sufficient-cause
hypothesis, not an established explanation. Run97 rejects its sufficiency under
the fixed budget; stop this factor rather than retrying centering or another seed.
S remains the accepted learning positive control. R is coupled and remains open.

## Recommended execution slice: J reference, one window only

Use the original J XML, public VectorEnv/action route, world order, eight held-out
q/v templates, h=.002, horizon256 and original task observations/reward/bounds.
One W8 x 256 window is 2048 world transitions. There is no actor, optimizer,
training seed sweep, warmup, extra settling or repeated trial. The reference is
an explicit diagnostic caller controller, never a condition inside the engine.

The two native slide axes are independent: mass1, armature.01, damping.1,
gravity0, no contact/spring/limits, motor gear1 and force/control range[-2,2].
The current [tree solve](../../src/phymeo/backends/mujoco_tyx/_tree_kernels.py)
uses physical M for reported acceleration and M+hD for Euler integration. For
each axis, with e=q-target and x=(e,v), its exact-arithmetic discrete recurrence is

```
a = M / (M + h*D), b = h / (M + h*D), M=1.01, D=.1
v_next = a*v + b*u
e_next = e + h*v_next
A = [[1,h*a],[0,a]], B = [h*b,b]
L=128, N=256, G = sum(A**j @ B for j=0..127)
[A**L @ G, G] @ [u_first,u_second] = -(A**N @ x_initial)
```

Here `A**j` denotes matrix power, never elementwise exponentiation. This is one
two-by-two solve per native axis and start, with a fixed halfway
switch. No gain, switching time, target or effort search is permitted. Use actual
stored FP32 entry q/v and timestep as the starting data; report the original
native mass/armature/damping and their stored representations separately. The
FP64 recurrence is an independent arithmetic reference, not an assertion that
FP32 Cholesky/integration executes exactly these operations or reaches exact zero.

Before any physical step, validate the complete source/protocol binding and
compute the entire reference schedule. Every force must be finite and strictly
inside (-2,2), and every predicted state inside the original bounds. A singular
solve, out-of-range force or bound violation ends the diagnostic as
REFERENCE_INFEASIBLE; do not clip, shift the switch or pick another controller.
Save the solve residual/conditioning and all proposed commands. This source-based
construction is not yet evidence that these numerical checks pass.

Preserve the original mapping `ctrl=2*tanh(raw)`: submit FP32
`raw=atanh(u/2)` through the public training wrapper, with the unchanged zero
qfrc/xfrc channels. Do not submit u/2 as raw, bypass tanh, or replace the stored
control with its ideal target. Record raw and actual stored ctrl every step;
independently advance the recurrence using those actual stored controls. Preserve
the usual accepted receipts, pre-reset endpoint, all 256 q/v/reward/done records,
source hashes, world order and event/failure prefix. Horizon auto-reset must not
erase step256's terminal state. A rejected or premature-terminated window has no
shortened mean cost and cannot pass.

Report the original all256-step cost, per-world endpoint q/v error, trajectory
prediction errors and actual force margin. As a distinct diagnostic check, reuse
[N1's existing absolute engineering comparisons](../../tests/training/test_mujoco_tyx_cuda_policy.py)
Q_ATOL=2e-4 and V_ATOL=3e-3 for
actual-versus-reference and actual terminal-versus-target; do not rewrite any
learning or physical tolerance. Save both residuals even if one check fails.
The original J three-seed gate is never evaluated on this reference controller.
Use existing zero/initial/final costs as historical comparisons, with their
source bindings; do not rerun the zero baseline or claim a contemporaneous control.

## Decision and stop rules

- Full original physical gates, predicted/actual agreement and endpoint checks
  pass: these eight J starts have a concrete bounded-action route to the target
  within the original time window. Report whole-window cost too: endpoint
  reachability alone does not prove the reference minimizes the learning cost.
  This removes simple actuator/time insufficiency for this set; it does not
  establish PPO failure, a root cause, a learning pass, or R reachability.
- Schedule is infeasible: this particular fixed two-pulse witness failed. It
  does not prove the target unreachable. Stop and report exact inputs/residuals;
  no second controller search in this slice.
- Schedule is feasible but actual trajectory differs: prioritize the first
  stored-control/dynamics/clock mismatch as an integration counterexample.
  Do not train around it or relax the comparisons.
- Actual/reference agree but endpoint or original task checks fail: retain the
  complete result and inspect the analytic construction; do not truncate cost.

## What would justify a later optimization diagnostic

Only after reachability is demonstrated should a separate authorization freeze
one small learner-instrumentation experiment. The current
[PPO configuration](../../src/phymeo/training/rsl_mujoco_tyx.py) uses 32-step
rollouts, gamma=.99, lambda=.95, one epoch/minibatch, fixed Adam lr=.0003, a .1
initial Gaussian std, and separate actor/critic norm1 clipping. RSL5.5.2 computes
GAE backwards, bootstraps at the rollout boundary, then normalizes advantages.
The saved combined post-clip norm cannot identify actor/critic dominance or
pre-clip signal; the old traces lack per-transition values/advantages/log-probs.

A useful later unit is one unchanged 32-step rollout from an exactly bound saved
policy, with zero optimizer steps: retain requested/mapped actions, raw rewards,
values/bootstrap, unnormalized TD/GAE and normalized advantages, plus separately
measured actor/critic pre-clip gradients from the actual upstream objective.
Its policy/RNG identity must be frozen before execution, not selected for a good
result. No new learning-rate/seed/budget search follows automatically. This is a
missing-evidence proposal, not part of the one-window execution requested above.
Inertial position response begins at order h^2 while velocity is also penalized;
that makes credit assignment worth measuring, but does not prove it is broken.

## Minimal implementation boundary if approved

NEW `examples/diagnose_mujoco_tyx_reachability.py` and
`tests/training/test_mujoco_tyx_reachability.py`; extend only this result document.
Reuse the existing task registry/public environment, not a copied engine or PPO.
Tests should cover the two-pulse algebra/zero and sign symmetry, inverse-tanh
mapping, infeasible refusal without clipping, complete256-step/pre-reset records,
and a second-step injected failure retaining its prefix and cost=None. Parent
then executes exactly the one fixed J CUDA window; no old full trace is reopened.

This proposal was written from source and already-computed summary metadata.
No import, numerical evaluation, policy inference, physical step, test or training
was performed, and no result is predicted as passed.

## Implemented authoring handoff (execution still pending)

Parent approved exactly two NEW Python files and this append. Baseline
`81500dc8` contains the independently qualified compact mirrors; this work does
not modify those runtime/checkpoint files. Tests-first commit `07b8320e`,
additional metadata/precision cases `ddd59f8d`, and CLI implementation `1cc1e4bf`
are pure authoring. No project import, collection, numerical calculation, test,
lint, formatter, build, policy inference or physical step was run by this author.

The CLI reuses `evaluate_mujoco_tyx_learning.py` and its original task registry,
not copied task XML or an alternative environment. It never constructs a learner.
Its three required arguments are `--evidence-root`, `--ablation-root`, `--output`;
output must be fresh. N4a's accepted J anchor plus all three original summaries,
publications and manifests are checked by the existing bounded metadata reader.
Run97's summary is fixed to SHA256
`107a2fbfd55b889b2e174f4900db9ab767547575e88450342febe58089e5b193`;
its three complete summaries/publications/manifests and shared protocol are bound.
Original and centered zero costs remain separate historical values. No old trace
is opened, no old source files are overwritten, and no old result is reclassified.

`make_schedule(q,v)` accepts actual stored FP32-representable 8x2 coordinates,
solves the single fixed two-pulse system, checks strict force and full-trajectory
bounds, and publishes owned immutable arrays including both ideal and FP32-raw
HOST-tanh preflight paths. `run_window` receives each actual public transition,
records it before checking, binds all three native channels to the action and
physical receipts, and advances the independent FP64 recurrence with the actual
stored control. Source dynamics can use the original kind0 precomputed inverse
and Euler operators; the reference deliberately does not copy their FP32 operation
sequence. The native source parameters, stored representations and both clocks
are saved so this arithmetic boundary stays explicit.

The final q/v and cost use `transition256.states`, not the already-reset live
state. All eight final timeouts and resets are checked. A step, publication,
recording or independent-check failure retains requests and every available
accepted transition, with `cost=None`; no shortened score is published. If a
transition record itself cannot be written, the failure detail retains that
received transition for the final failure record. Public counters are captured
before cleanup; complete package/example source hashes must match before/after.
The successful summary explicitly leaves learning classification unchanged.

The manual, uncollected test count is **26 HOST nodes**, with **zero CUDA physical
pytest nodes**. They comprise protocol1; independent recurrence/symmetry/mapping3;
infeasibility5; bounded/hash/duplicate-JSON metadata1; full-window and failure
publication12; complete historical source/zero binding4. Synthetic steps use the
actual public publication dataclasses and an independently written scalar Euler
loop. These are not actual TYX physics results. The only proposed CUDA physical
execution is the one CLI window below.

```text
<qualified-training-python> -m pytest tests/training/test_mujoco_tyx_reachability.py -q -x
<qualified-training-python> examples/diagnose_mujoco_tyx_reachability.py --evidence-root <parent>/artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01 --ablation-root <parent>/artifacts/p65-parity-until-0720-20261112-01/97-n4d-fixed-three/ablation --output <fresh-N4e-directory>
```

The complete stream contains protocol/history, native entry/model metadata,
reference schedule, each request/transition/independent result, completion,
owner-exit and final source bindings; failure adds its actual phase and available
transaction flags. `manifest.json`, `summary.json` plus hash publication (success),
or `failure.json` (failure), bind the compressed stream. This is not a performance
benchmark or a Newton comparison. Numerical feasibility, endpoint success and
cost improvement remain **UNKNOWN until the parent executes the fixed window**.

| Authored Python path | SHA256 | Git blob |
|---|---|---|
| examples/diagnose_mujoco_tyx_reachability.py | 9d80edca63930ae84737df8d270b2639c2d83add8f49e19fe3570ba5ac0255c0 | be6d7fb7305e51e113ddeff7ce08a3b9fe191ca8 |
| tests/training/test_mujoco_tyx_reachability.py | 9b72a327866e0e191ecc097d1a2b12cc2dec280bf91298be7878a4b7fbfdcd3a | 7f14aece48e703e0667e3a803ca3cec52633eae9 |

CODE_READY_TESTS_PENDING. AUTHORING_SLOT_RELEASED after the final documentation
commit; parent retains sole execution ownership and may perform focused static
checks and the declared tests/window. No follow-on optimizer experiment is started.


## Parent actual qualification, 03:07

Tested51b1d77e. Runs125/126 retained the first lint pass and two-file format failure;128 formatter and129/130 final static checks passed. Run127 complete HOST suite:26 passed in1.71s. Run131 actual CUDA window: original8 held-out states×256 steps=2,048 world-transitions, RAW_EXIT=0, PASS_FIXED_REFERENCE_DIAGNOSTIC. No learner or optimizer ran.

Mean normalized cost0.023813799676380473; maximum final |q|3.6288483329371957e-8 and |v|8.736260070918433e-8. Independent FP64 recurrence versus actual published trajectory errors were at most2.7403840176796132e-8(q) and7.853073543534483e-8(v); maximum reward error1.1039317016070527e-8. Stored force remained0.5751821994781494 inside the original bound. All eight final timeout/reset publications and original pre-reset transitions were checked. The complete774-record stream is2,497,923 compressed bytes,21,330,862 logical bytes.

This establishes that this fixed J window is reachable through the actual public force/step/publication chain. The historical zero cost0.063695137045106 and policy costs are explicitly cross-source historical controls, not contemporaneous policy trials. It does not identify PPO's cause, prove optimality, validate R learning or change the failed N4d classification. No follow-on J training is authorized by this result alone.
