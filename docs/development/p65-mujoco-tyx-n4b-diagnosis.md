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
final source binding requirements extend those declarations. Original manual
count was33 HOST nodes, then uncollected and unexecuted: data-only decoder; malformed
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

## R1/R2 schema review repair (2026-11-12)

The parent imported and formatted the original four files at
fdba57d22b2788fe6a387fe861a93f6b89ff5f6d. Its original 33 HOST tests passed
in 1.35 seconds; the initial import-order/format failures and subsequent static
passes are retained by the parent. That result predates this repair. The fixed
nine real traces have not been analyzed. No old result is assigned to the new
source below.

Independent text review found two missing bindings. The original accumulator
accepted any last_step dictionary with the correct backend string, and the
training reader did not consume the already-saved step_requests. The repair is
limited to this module, its synthetic tests and this report. The CLI and parent
formatting remain unchanged; the N4a source, physics, objectives, horizons,
seeds, budgets and acceptance gates are unchanged.

R1 now requires the actual saved MuJoCoTyxStepReceipt and MuJoCoControl type
identities. Each receipt must name its world and current state tick, publish
output_time equal to state.time and preserve the declared h. Its input_time
and previous tick bind to the transition's actual MuJoCoTyxTaskFrame, and its
contact_time equals input_time. The two saved times must be finite and advance.
The check deliberately does not reconstruct time as tick*h: device integration
uses compensated FP32 clock words, whereas the receipt h is the original public
step size. All three stored_control channels must exactly equal their saved
native_channels and thus the already-bound state control arrays. Synthetic
receipts now carry the real dataclass field schema instead of fixture.Step;
these remain independent synthetic evidence, not native execution results.

R2 checks one complete request per rollout transition, the exact four writer
fields, ordinal order, accepted status, and transition_index. Each finite,
shape-checked requested_action is supplied to the existing transition action
binding. Missing, reordered or changed requests fail before successful analysis
publication. The writer schema was read directly from
mujoco_tyx_evaluation.measure_update; no new field was invented.

Tests were committed first at 32d0bc0f, followed by implementation 3182a3f7.
There are 14 new manually enumerated negative cases: receipt type/world/tick,
input/output clock, h, each of three stored channels; then missing/order/action/
status/index for training requests. The current total is 47 HOST nodes,
UNEXECUTED by this author and pending the parent's serial validation. Existing
33 nodes and all original synthetic trace integrity checks remain. No complete
original trace was read, no project module imported, and no test, static tool,
numerical script or training run executed by this author.

Current repaired code/test working bytes (the earlier table remains the original
historical authoring snapshot):

| Path | SHA256 | Git blob |
|---|---|---|
| src/phymeo/training/learning_diagnostics.py |23e49408e1c533a29975bae2f0384ae60916b0b830cc1ffa7f2d8d43060e4dd5|fc6dd874b892a3c6e1532a1f8e2cb0179d825d1e|
| tests/training/test_learning_diagnostics.py |bfeef7b5ebefa33d7f0dcf1aa8551b78b38156f53e8330e5efe8d933d43301a7|7f6b35d711b97ffea31f283fe30e0475241ce824|
| examples/diagnose_mujoco_tyx_learning.py (unchanged from fdba57d2) |a2647bc865eed19c3561be542adc6768c8dccdfd77aed58f4dbddf0d5b076c45|99810abc160b13005b650ef5251a3adff9a7623c|

Repair handoff status: CODE_READY_TESTS_PENDING / ANALYSIS_UNEXECUTED /
AUTHORING_SLOT_RELEASED. The parent retains the sole execution slot.

## Actual nine-trace diagnosis and bounded follow-up (2026-11-12)

This section supersedes the preceding pending-execution status; the earlier
checkpoints remain historical. Parent run37 used source15704211 and completed
PASS_SAVED_DIAGNOSIS_ONLY, runs=9, RAW_EXIT0, from00:10:19 to00:13:27+08.
Its final.json records unchanged before/after source hashes. Parent48/49 Ruff
and format checks returned0; parent50 completed all47 HOST tests in1.10s with
RAW_EXIT0. These are the parent's executions, not author executions.

The authoritative diagnostic is
`artifacts/p65-parity-n4b-rob2a-col-stop-20261111-01/37-diag-fixed-nine/result/summary.json`
in the parent checkout, with the adjacent final.json, stdout.log and RAW_EXIT.txt.
This author read bounded projections of that completed JSON and those small
receipts only. No original gzip, policy forward, numerical script, retraining
or physics replay was run for this interpretation. The nine original N4a
128x32x8 runs and their three bound learning summaries remain the inputs.
The analysis passing does not change the learning outcome: S passes, J/R remain
FIXED_BUDGET_NOT_LEARNED. Installation/package qualification is owned separately
by the parent and is not asserted here.

### Observed learning and zero-action comparison

Costs below are copied from the diagnostic, rounded only for display. Each
row's untrained actor uses the same seed and fixed held-out states as its learned
actor. Zero is the one original first-seed diagnostic shared within that task;
null zero fields in later runs do not mean those runs created another zero trace.

| Task/seed | Untrained cost | Learned cost | Learned minus zero |
|---|---:|---:|---:|
| J14011 |0.0798635333|0.0740519219|+0.0103567849|
| J14012 |0.0651763097|0.0647360449|+0.0010409079|
| J14013 |0.0669394627|0.0631858401|-0.0005092969|
| S14111 |0.0369562417|0.0271977548|-0.0011449393|
| S14112 |0.0299416978|0.0238532838|-0.0044894103|
| S14113 |0.0271564944|0.0177158286|-0.0106268656|
| R14211 |0.1456878237|0.1452686509|+0.0316434204|
| R14212 |0.1936347145|0.1285030184|+0.0148777879|
| R14213 |0.1317695218|0.1209998391|+0.0073746086|

Original median learned/untrained ratios remain J0.9439251168,
S0.7359448237 and R0.9182687886. R14212 improves substantially over its own
untrained actor but still loses to zero; that distinction prevents calling a
favourable initialization-relative ratio a competent controller. Only J14013
slightly beats zero in task mean. Every S seed beats zero separately in all
eight saved worlds, not only after averaging them.

### Where J/R differ: coordinates, time and actual controls

These are descriptions of saved trajectories, not proofs of an optimizer cause.
World order is5,41,77,113,149,185,221,257. Full per-world/per-joint arrays remain
in the diagnostic; the examples below use fixed named seed/world identities so
the quoted contrasts are reviewable. Each q cost entry is the original squared
normalized position error, and each v entry already includes the original0.01
weight; the task cost then averages over coordinates and worlds.

- J is dominated by its position terms. For J14011/world257, q terms are
  (0.26512665,0.06025492), whereas v terms are
  (0.000138685,0.0000235332). Its saved native q-error RMS is
  (0.1029809,0.0490938)m, with velocity RMS(0.1177646,0.0485110)m/s.
  This world's cost grows from0.107237 in the first64-step quarter to0.238979
  in the last. The small mean improvement therefore does not imply convergence
  of all retained initial states.
- J14011's j0 mean ctrl is negative in all eight worlds, including both signs
  of initial position error. Its learned-minus-zero per-world costs are
  negative for worlds5/41/149/185 and positive for77/113/221/257.
  J14012 has positive j0 mean ctrl in all eight worlds and reverses those four
  positive/negative groups. J14013 is mixed and retains j1 mean ctrl negative
  in every world. This is visible state-group asymmetry, not evidence that the
  action sign or joint mapping in the engine is reversed.
- R shows the same need to inspect branches separately. R14211/world149 has
  q terms(0.35325748,0.31089453,0.17761971), versus
  v terms(0.000742158,0.00153248,0.00110661). Its native q-error RMS is
  (0.1485887,0.1393948,0.1053624)rad. Its quarter cost rises from0.176363 to
  0.430084. For R14212/world257 the largest q term is j1=0.36325869
  (j0=0.25522629,j2=0.05732761); R14213/world149 also has j1 largest at
  0.33700935. A single scalar loss would conceal this shift between branches.
- R14211's root-joint mean ctrl is positive for every world, while R14213's
  is negative for every world. Corresponding endpoint ctrl*error products and
  restoring-direction fractions differ between opposite initial groups.
  These endpoint products are correlations; they are not integrated physical
  work and cannot prove that an individual torque is wrong for a coupled tree.
- All learned J/R per-world/per-joint fractions in the predeclared >=95% action
  scale bin are0. The actual learned ctrl RMS values are well below the2-unit
  bound (for example J14011/world257=(0.310379,0.140781),
  R14211/world149=(0.586503,0.184033,0.182328)). The saved evidence does not
  support saturation as the limiting mechanism in these evaluation windows.
  This does not establish ideal action authority for every unvisited state.

The source already exports centered, scaled q errors and scaled v using the
original q_scale/v_scale; actor and critic use that public task observation.
A proposal to simply add that same normalization would duplicate existing work.
The9trace pass also binds requested actions, native channels, stored controls,
state/world/tick and saved clocks, and verifies the original reward formula.
That excludes those specific evidence mismatches; it does not establish that
all learning-protocol choices are effective.

### Updates happened; remaining unknowns

All nine runs reach saved Adam step128. The saved actor tensors change in all
1284 J scalars,1186 S scalars and1382 R scalars. Corresponding critic tensor
counts are1249,1185 and1313. These counts cover all saved model-state tensors,
not an inferred parameter-only subset. Relative actor L2 changes are
J=(0.0172466,0.00799560,0.0142780),
S=(0.0234978,0.0177171,0.0336046),
R=(0.0121762,0.0303421,0.0232114).
The saved final-update combined post-clip norms are nonzero:
J=(1.1388783,1.4142132,1.4142130),
S=(1.0933258,0.6569128,1.0205169),
R=(1.4142132,1.4142132,1.4142130).
Thus near-zero scalar surrogate loss cannot be used to claim no gradient,
no optimizer step, or an unchanged actor. These combined norms pool the saved
parameter order; they do not identify actor versus critic contributions.

Pre-clip norms, gradient directions, per-transition values/returns/advantages,
log probabilities and KL remain UNKNOWN. Neither the relatively larger critic
change nor first/last value-loss differences prove an actor-credit problem:
those losses come from different rollout states and episode phases. The weak
J/R gains may involve learned offset versus useful feedback, sampling/credit
assignment or optimizer configuration; this analysis distinguishes none of
those as an established cause.

S remains an actual loaded-contact positive control. In every learned S world,
all four sites have positive normal force for all192 tail samples. Per-world
tail means span the recorded ranges19.9359..20.2357N for seed14111,
21.2160..21.4371N for14112, and24.2165..24.2488N for14113;
peak forces are22.9244,23.3532,25.2539N respectively. These are actual receipt
forces and satisfy the existing S gate; a different mean support force is not
by itself an extra reward or energy result. No S gate or tolerance is changed.

### Proposed next protocol: one bounded saved-policy response test

This is a proposal for parent approval, not newly authorized implementation or
an executed result. Before any further training, distinguish a policy dominated
by its offset from useful state feedback using the already-saved initial/final
MLPs for **all nine original identities**, with S retained as a positive control.
No new policy, reward, target, seed, training horizon, budget or learning gate is
selected from these results.

1. Read the two saved model records and original public config/task/action
   bindings with the existing bounded decoder. Verify the original whole-policy
   hash (actor plus critic) against each bound summary, then retain the actor
   state. Load the actual pinned MLP class; preserve caller RNG and run
   deterministic actor inference. No optimizer, environment or physics step
   is involved. Reject missing fields rather than reconstructing an old policy.
2. Freeze the input grid before reading outputs: the original eight held-out
   initial observations; the target with v=0; and for every coordinate j,
   q=q_target +/-q_scale[j]/4 with all other errors and v zero, then
   v_j=+/-v_scale[j]/4 with all q errors zero. Use the same existing public
   scaling/bias and action mapping. These are bounded counterfactual policy
   observations, not new physical trajectories or proofs of state reachability.
3. Save every raw deterministic action and mapped ctrl. Separately report the
   target offset, paired even component, paired odd response and full cross-joint
   response for both checkpoints. Keep each seed separate and preserve zero
   denominators explicitly; do not choose a best seed or add a new success
   threshold. For uncoupled J at v=0, this directly tests whether opposite
   position errors receive opposite restoring commands. R's full coupled
   response is descriptive, not a per-joint sign proof; S need not have zero
   equilibrium ctrl.
4. Completion means exact model/config/input identity, finite complete outputs
   for the fixed grid and a discriminating report. If the saved mapping differs
   from the public mapping, stop and fix that concrete defect first. If a dominant
   even offset is supported, use that evidence to preregister one initialization/
   feedback ablation at the unchanged original learning budget and all three
   seeds. If the probe does not distinguish the hypothesis, stop this probe;
   do not expand the grid or repeat training. A separate bounded instrumentation
   proposal would then have to capture the missing actor-gradient/advantage/KL
   evidence before selecting a learning change.

This next step is deliberately one fixed inference experiment rather than an
open-ended search. It cannot upgrade J/R to learned, prove controller stability,
or replace the existing physical and learning acceptance gates. This report
change is docs-only; all code/tests and source hashes above remain unchanged.
