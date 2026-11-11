# N4b: saved learning diagnosis

Status: CODE_READY_TESTS_PENDING / ANALYSIS_UNEXECUTED / AUTHORING_SLOT_RELEASED.
This author only reads source and small JSON metadata, authors code/tests/docs,
and uses Git. No imports, collection, numerical probes, pytest, static tools,
training, policy forward, physical replay or diagnostic execution were performed.
The parent owns all execution. Base847b8ee359320d260e3d36c13cd0530cde7a28b9;
the preceding N4a learning source remainsa4c4e2bc and is not modified.

## Question and fixed input

N4a completed all original128x32x8 updates for each of nine seeds. S passed its
fixed learning gate; J and R did not. Those classifications remain unchanged.
This slice separates three concrete explanations using existing evidence:

1. Is weak improvement shared across initial states or concentrated in particular
   coordinates/times? Decompose the original native cost into q and velocity
   terms per world/joint and each fixed quarter of the full evaluation window.
   Compare learned against its same-seed untrained actor and the common first-seed
   zero-action trajectory. The latter is diagnostic, not a replacement gate.
2. Does the learned policy use the declared actuator authority differently?
   Save raw-action and actual-ctrl RMS/mean/max, the predeclared >=95% of action
   scale bin, and actual ctrl times endpoint position error/velocity. These
   signed endpoint products are correlations, not energy or proof of a correct
   feedback direction. Native ctrl/qfrc/xfrc must match the saved transition.
3. Do actual optimizer updates and timeout boundaries explain a mismatch?
   Retain every128 update's costs, loss, Adam step, parameter hashes and actual
   post-clip gradient norms; retain reset/episode lengths and final actor/critic
   tensor changes separately. Do not infer zero gradient from near-zero scalar
   surrogate loss. S retains actual last-step four-site contact forces as the
   existing positive control.

The single CLI accepts the original artifact root and a fresh output directory.
It consumes only these exact predeclared anchors, with SHA checked before opening
any gzip payload:

| Task | Seeds | Original three-seed result directory | SHA256 of summary.json |
|---|---|---|---|
| J |14011/14012/14013|50-n4-summary-j/result|50020100fe49b7d6220f766281fcb72c159a810a2d09af2e048661fdeb2e4153|
| S |14111/14112/14113|51-n4-summary-s/result|60a27613183f0ec96b92e288694dc02a005b62a6388d9765fc7c52f08c6e128d|
| R |14211/14212/14213|52-n4-summary-r/result|bdbf77e984eb0ddbd27f8364110b7aedcdaef7a79b15c5bdab1af86339a2d808|

The anchors identify run40 J14011 and run41's other eight complete runs. Original
run31 is excluded and retained as a failed attempt with its known update7 payload
gap. No reconstruction is applied to that missing evidence. No other run or new
seed may silently substitute for a selected identity.

## Source and evidence facts

The original evaluation writer stores every record as ordinal/kind/typed payload.
Arrays include dtype/shape/base64/raw SHA; mappings retain typed world IDs, and
dataclasses retain type names and all fields. Each complete run binds manifest,
summary publication hash, compressed and logical bytes/SHA and record count.
The successful stream order is initial_policy; complete untrained window; one
zero window only for the first seed;128 update/update_checks pairs; complete
learned window; final_policy_and_adam; completion.

The actual existing optimization_evidence in rsl_mujoco_tyx.py records
finite_gradient, optimizer_step, parameter_count and gradient_norms. It runs
after upstream RSL5.5.2 clips actor/critic parameter gradients and performs Adam.
Therefore these are real **post-clip** parameter norms in saved parameter order.
The diagnosis reports their combined norm and nonzero counts, while preserving
the originals. Parameter-to-actor/critic attribution of those norms is UNKNOWN;
the initial/final saved model tensors do independently identify their owner.

Pre-clip norms, gradient vectors/directions, per-transition value predictions,
returns, advantages, policy log probabilities and KL were not saved by N4a.
They remain UNKNOWN. The scalar value loss cannot reconstruct those arrays or
establish a bootstrap error. The original adapter hands time_outs to upstream
RSL and uses its already-qualified timeout convention; this analysis does not
rerun or alter it. No PPO implementation, reward, target, horizon or learner
configuration changes are made.

## Public code and bounded processing

Exactly four NEW paths:

- src/phymeo/training/learning_diagnostics.py: optional NumPy/stdlib data decoder,
  streaming reader, phase accumulator and diagnose_seed. Does not import Torch,
  MuJoCo, training examples, policy constructors or simulator code.
- examples/diagnose_mujoco_tyx_learning.py: fixed nine-identity CLI, comparisons,
  source/anchor binding and atomic report publication.
- tests/training/test_learning_diagnostics.py: independently constructed synthetic
  typed evidence, known arithmetic, corruption/identity/completeness negatives.
- This report.

decode_evidence does not import or instantiate a declared dataclass, use pickle,
or evaluate source. Hard bounds:64MiB per JSONL record,32MiB per numeric/bytes
array,96 nesting depth,2million typed nodes per record,1GiB compressed and4GiB
logical per stream,10000 records. Source metadata showed the largest original
logical stream at1,804,933,575 bytes, so the bounds do not require increasing any
physical or training capacity. Only one record (at most one complete32-transition
update) is decoded at a time. All nine traces are read sequentially; no complete
plain copies or whole-trace lists are created.

Each run checks expected task/seed/directory, manifest/publication/summary,
complete update/phase ordering, every decoded array's bytes/SHA/layout, full
logical/compressed stream identity, world/episode/done sequence and actual
state/native-channel binding. Independent saved costs and summary reductions
are compared to the same source formula. The existing device reward check keeps
rtol2e-5/atol2e-6. A separate FP64 reduction-consistency check uses rtol2e-12,
atol2e-14 only for saved diagnostic aggregates; it is not a new physics or
learning acceptance threshold.

Source/config/dependencies must agree across all nine runs. The fixed anchors
also bind each task's complete protocol and exactly one complete zero window.
The reusable per-run API allows a smaller explicit OfflineBudget for synthetic
tests; the public N4 CLI always uses the original128/32/256/256 budget and8 worlds.

Output status PASS_SAVED_DIAGNOSIS_ONLY means complete input analysis, not learned
behavior. All metric data and all three original formal-gate outcomes are preserved.
The report is written to a temporary file, read back for byte integrity, then
published once all nine streams pass. An ordinary exception produces failure.json
and no successful summary. There is no successful partial analysis, no mutation
of evidence and no automatic corrective run. Interruption still propagates.

## Tests and parent execution

Tests were authored first at406238c0; later synthetic completeness negatives and
final source binding requirements extend those declarations. Current manual
count is33 HOST nodes, uncollected and unexecuted: data-only decoder; malformed
dtype/base64/hash/shape; ambiguous JSON; gzip/logical/count/ordinal/truncation;
known q/v/action decomposition; world/control/reward/done/missing-window rejection;
real-vs-missing gradient norm semantics; separate actor/critic tensor changes;
two complete synthetic runs; changed manifest/publication/source identity;
bound-but-missing/extra-phase/request mismatch; actual-step support consumption.
No actual CUDA, learner or physics qualification is claimed by these tests.

Parent serial commands, after static review:

    pytest tests/training/test_learning_diagnostics.py -q -x --tb=short -p no:cacheprovider
    python examples/diagnose_mujoco_tyx_learning.py --evidence-root C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01 --output NEW_OUTPUT_DIRECTORY

This is one diagnostic run for all nine fixed identities, not nine re-training
commands. It must not import or execute the old learning CLI. The parent may
then use the observed per-world/time/term differences to preregister one bounded
follow-up protocol. No learning change is selected before these actual results.

## Frozen authoring checkpoint

Tests-first406238c03761e2af4974670f16037e9f7fb6674e; implementation
bda2f82736aba0e99b7a22614906ce0483e55fb0. The three Python files below are
UNEXECUTED working-byte bindings; Git normalizes text blobs separately.

| Path | SHA256 | Git blob |
|---|---|---|
| src/phymeo/training/learning_diagnostics.py |8257737f01e26a66892448f4fef3188315eeceae7b28f13dfb12e9dbf72ea693|936bb2cb1ce1e5ab0cc6675b5eb29e8f07d22b72|
| examples/diagnose_mujoco_tyx_learning.py |586f9ff5c3ad801d36fe6b6515d72b35f87e263e1d15863c348e223c103bdc99|5792110eb21108632fe7870a8411f0f0dfeda02e|
| tests/training/test_learning_diagnostics.py |f414c74ea72f6d93123199552763921add25496669aea030fe8a4fea9fa5f6a0|79f27560473fca60484136ea736a6210ebb64ba8|

The first staged Git whitespace check rejected one extra blank line at the
tests-first file's EOF; it was removed before that commit. This was text
verification, not an executed test failure. No old source/test/contract was
changed, and no complete original trace was read by the author.
