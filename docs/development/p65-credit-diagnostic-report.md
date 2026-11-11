# Fixed credit diagnostic authoring report

Status: **CODE_READY_TESTS_PENDING / UNEXECUTED**. This report describes the
new diagnostic, not an execution result. Parent owns every import, static check,
HOST test and CUDA execution. Original J/R learning remains NOT_LEARNED; the
accepted S learning result and completed N7 connection tests are unchanged.

The exact declaration is [the accepted single-diagnostic protocol](p65-jr-learning-next-diagnosis-20261112.md)
and the parent plan's 06:20 authoring grant. This slice has only three NEW files:

- `examples/diagnose_mujoco_tyx_credit.py`
- `tests/training/test_mujoco_tyx_credit_diagnostic.py`
- this report.

No backend, task, solver, learner, upstream implementation or old evidence is
modified. The final product wheel remains its earlier frozen snapshot; this
optional diagnostic example does not justify another package build.

## Fixed input and execution boundary

Exactly original N4a final J14011, R14211 and S14111 are selected, in that order.
Each original three-seed summary has its predeclared SHA checked. The original
bounded decoder and complete `read_policy_pair` binder each consume the selected
trace once. A temporary observer of the module's existing `read_records` only
copies the unique final Adam payload, yielding every original decoded record
unchanged. Missing/duplicate final records, incomplete binding and decoder
failure reject. The exact original callable is restored in `finally`; there is
no second full trace pass and no persistent decoder cache.

Actor, critic, Identity normalizer and all Adam state/group tensors are loaded
and checked. Saved Adam step128 must exist for every parameter. Config is exactly
the original MLP `[32,32]`, one epoch/minibatch, fixed learning rate .0003,
gamma .99, lambda .95, clip1, no mixed precision, no adaptive KL/RND/symmetry.
The complete config and current pinned dependencies are checked rather than
assuming equivalent defaults.

The original public task factory creates W8, stable IDs5+36i and the original
training initial templates (not held-out templates). D1 and original h are
unchanged. After construction/loading, the task's original seed starts a **new
diagnostic** Python/NumPy/Torch stream. Historical final RNG was not saved and
is not reconstructed. Caller Python, NumPy, Torch CPU/all CUDA RNG and effective
IEEE/TF32 settings are restored through the already qualified response context;
the previous CUDA device is restored too. RSL construction's global
`Distribution._validate_args` change is separately saved and restored using the
public setter. The read of the class default is explicit because Torch has no
public getter for that default; no temporary distribution is fabricated.

Per task, original PPO.act/process_env_step are called32 times, through the
public RSL/Env path. Each request is saved before stepping; each actual complete
transition, next observation, reward/done/extras and all native receipts are
saved immediately after acceptance, before later diagnostic assertions. Actual
raw Gaussian action, old mean/std/log-prob/value, mapped native channels and
published force/motion remain distinguishable. Full actual transition payloads
preserve the physical input/source/clock/owner evidence. Early rejection/done
retains the available prefix and ends this fixed diagnostic, with no retry.

The original compute_returns executes once under inference_mode. Its one real
critic bootstrap is observed, not evaluated again. Independent FP64 TD/GAE,
raw-reward and critic-component recurrences and sample-standard-deviation
normalization are saved. Diagnostic arithmetic comparisons allow FP32 versus
FP64 accumulation: returns rtol2e-5/atol2e-6, normalized advantages and replay
log-prob rtol2e-5/atol2e-5. These are instrumentation identity bounds, not changes
to physical or learning acceptance. Raw complete arrays remain available.

## One actual backward, zero optimizer updates

The original PPO.update supplies its one original random256-sample batch.
Its observation/action/old-log-prob tuple must match the stored rollout by an
exact unique full-row bijection; duplicate identities fail instead of guessing
indices. Values, returns, normalized advantages and both old distribution
parameters then match under that actual permutation. Each batch row records
its real world and tick. No additional stochastic actor call or bootstrap is
introduced. The actual update's replay log-prob, distribution and critic values
are observed and retained.

Parameter hooks copy the real pre-clip gradients and return None. Original
actor/critic clipping runs. A public Adam `register_step_pre_hook` copies the
post-clip gradients and raises a dedicated sentinel **before Adam.step's body**.
The exact actor/critic/Adam state digest is checked both around backward and
around the complete rollout/backward operation. Every saved step128 remains128;
there is one blocked optimizer entry and zero completed new optimizer steps.
All hooks/method replacements are removed, including on failure. Unexpected
return, missing/duplicate gradients, nonfinite gradients or any parameter/Adam
mutation reject. Cleanup attempts every registered handle and preserves errors.

This does not claim the whole learner is unchanged: gradients, distribution
cache, RNG, storage and upstream temporary state have changed. The diagnostic
owner is discarded. Normalizer updates lie after the interrupted optimizer
loop and are not reached. No live training session or checkpoint is resumed
from this deliberately interrupted state.

The report saves complete actual gradient arrays; norms split mean network,
log standard deviation and critic, before/after clipping. Independent score
terms retain perworld/perjoint signed and absolute sums and cancellation,
reward/critic contributions, between-world means and within-world variation.
J restoring-direction projection is diagnostic; R remains coupled and S's
nonzero equilibrium mean is legitimate. Ordered A–E routing is fixed in the
protocol and is not another learning pass gate or proof of a unique cause.

## Predeclared tests and parent-only commands

Manual source count: **30 HOST synthetic nodes + 1 CUDA fixed-three node**.
The HOST nodes cover independent GAE/done/normalization and invalid layouts,
exact batch bijection/ambiguity, five routing branches, observer restoration,
single-pass binder failure/missing/duplicate Adam, actual Torch backward with
nonempty Adam step128/momenta, pre-backward failure, mutation, unexpected return,
nonfinite gradients and all-hook cleanup; caller RNG/precision/distribution
restoration has success and injected-failure nodes. Optional Torch is imported
inside relevant tests, never eagerly by the new example.

Parent first executes focused static and the HOST selection, with the qualified
training profile and its sole serial slot:

```text
pytest tests/training/test_mujoco_tyx_credit_diagnostic.py -m "not cuda"
```

Then choose **one** of the actual CUDA node or this CLI, not both:

```text
python examples/diagnose_mujoco_tyx_credit.py --evidence-root artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01 --output <fresh-output>
```

The actual node calls the same fixed CLI function once. Expected success is
768 accepted world-transitions, three complete credit captures, zero new Adam
steps and exact128→128 state. It is not a new training run. Output is one
streaming `credit.jsonl.gz` plus `summary.json` with logical/compressed identity;
first failure and every already written prefix are retained. Current and
historical source bindings remain separate. Summary PASS is emitted only after
all three captures and owner cleanup; finalization errors cannot become PASS.

At authoring handoff no command above, project import, numerical calculation,
collection, static checker, formatter, test or GPU action has been executed by
this author. Only source reading, text authoring and explicit Git checks were
used. Parent execution outcomes must be appended later and cannot be inferred
from these proposed checks.

Parent pre-execution review added complete owned caller before/after RNG,
effective precision and Distribution-default receipts at the actual restoration
boundary, with an exact digest gate even on failure. The after-seed rollout RNG
is separately labelled. The same captured arrays also supply original .2 ratio
clipping masks/count/fraction and actual actor/critic post/pre norm factors;
zero pre-norm reports factor unavailable rather than dividing by zero. No
additional inference, sampling, physics or optimizer call is introduced.
