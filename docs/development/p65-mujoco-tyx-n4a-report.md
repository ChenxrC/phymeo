# MuJoCo-TYX N4a learning results and partial qualification

Status: **LEARNING_PROTOCOLS_COMPLETE / PARTIAL_LEARNING_QUALIFICATION /
SCALE_MATRIX_PASSED / INSTALLED_HOST_EVALUATION_PASSED / AUTHORING_SLOT_RELEASED**.
The parent executed validation; this author only read the closed reports and
updated this document. The current results below supersede historical pending
labels, without assigning new results to earlier source snapshots.

## Actual learning results

The repaired learning source is `a4c4e2bc3ce7636c798b4f869e442b28aa91ba20`.
Each declared seed completed the original **128 updates × 32 steps × 8 worlds**:
32,768 training transitions per run, nine completed runs and **294,912 training
transitions** in total. Each untrained and learned held-out evaluation covers all
256 steps in eight worlds. No extra update, seed, reward change or budget increase
was used to obtain these results. The failed original run31 is excluded from
these completed-run totals and remains retained as described below.

The values below are copied from the actual three-seed summaries. Lower cost is
better; ratio is learned cost divided by that seed's untrained deterministic
actor cost. A completed training run alone does not pass the learning gate.

| Task | Seed | Untrained cost | Learned cost | Ratio | Better than zero action? |
|---|---:|---:|---:|---:|---|
| J: joint control |14011|0.07986353326689878|0.07405192191321378|0.9272307257648748|No|
| J |14012|0.06517630966803778|0.06473604490144086|0.9932450184915452|No|
| J |14013|0.06693946266999315|0.06318584011928716|0.9439251167997704|Yes, slightly|
| S: continuous contact support |14111|0.03695624173391796|0.027197754806520428|0.7359448236739582|Yes|
| S |14112|0.029941697820910537|0.023853283820822663|0.7966576900046103|Yes|
| S |14113|0.027156494447525728|0.017715828556652435|0.6523606568912856|Yes|
| R: branching robot |14211|0.1456878236563657|0.14526865089763472|0.9971228016987906|No|
| R |14212|0.19363471446867714|0.1285030183783298|0.6636362634196814|No|
| R |14213|0.13176952176094292|0.12099983911820782|0.9182687885725689|No|

The unchanged formal gate requires all three ratios <=1.05, at least two <=0.90,
and median <=0.80. The parent executed the complete source/seed bindings and gate:

| Task | Median ratio | Formal result | Summary command exit |
|---|---:|---|---:|
| J |0.9439251167997704|FIXED_BUDGET_NOT_LEARNED|1|
| S |0.7359448236739582|PASS_FIXED_LEARNING_GATE|0|
| R |0.9182687885725689|FIXED_BUDGET_NOT_LEARNED|1|

J/R exit1 is the declared learning-quality refusal after complete training, not
an incomplete rollout or failed optimizer invocation. S passes this fixed task
and budget, including its actual step-contact support checks; this does not
qualify general locomotion, G1 dance or arbitrary robot learning.

The independent zero-action diagnostics all completed 2,048 world steps. Their
costs are J `0.063695137045106`, S `0.028342694125254343`, and R
`0.11362523048658707`. S's three learned actors all improve on zero action; J only
seed14013 does, by the saved cost difference `-0.0005092969258188446`; none of R's
three actors does. Zero action remains diagnostic and was not substituted for
the frozen untrained-policy acceptance baseline.

Source evidence under `artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01`:

- `39-n4-repair-host`: eight affected HOST checks passed, 48 deselected, 1.21s.
- `40-n4-learning-j14011-fixed`: the one fresh original-seed restart passed,
  210.47s; `41-n4-learning-remaining`: the remaining eight runs passed, 1448.97s.
- [J summary](../../artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01/50-n4-summary-j/result/summary.json),
  [S summary](../../artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01/51-n4-summary-s/result/summary.json),
  [R summary](../../artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01/52-n4-summary-r/result/summary.json)
  retain all per-seed costs, ratios, zero comparisons and bindings.

### Prior learning-report publication state (historical)

Scale qualification and the new wheel remained **PENDING** at that report update.
The parent's root-only scale timing correction is source `06a19790`, not the
child's learning source `a4c4e2bc`. Parent commands43–48 completed its static and
one new HOST timing check; run49 scale was still active and had no final result
claimed here. That correction measures real wall time from the formal measured
loop through final gzip close, rather than summing partial timings; final file
hashing/summary publication remain separate. Historical timing descriptions below
are not proof of complete scale throughput. No scale capacity or new installed
wheel claim is inferred from the nine learning runs.

## Actual scale matrix results

Parent run49 completed **18 passed in 530.94s / RAW_EXIT0** on tested root source
`06a197900ad300f11092aa5e926e0aaed3221b2f`. This is separate from the nine learning
runs at `a4c4e2bc3ce7636c798b4f869e442b28aa91ba20`; the root CLI includes the
actual-loop-to-trace-close timing correction. All 18 closed cell summaries report
PASS_N4A_SCALE_CELL. The dependency records identify MuJoCo3.3.7, RSL5.5.2,
TensorDict0.15.5, Torch2.12.0+cu126, NumPy2.4.6 and tyx-compute0.1.0a0.

Each cell ran its one fixed 32-step warmup, followed by three complete unprofiled
32-step updates. The measured numerator is 96W accepted world transitions, not
optimizer steps or the warmup transitions. The full-trace wall interval starts
before the first formal update and ends after gzip close. It includes complete
public updates, CUDA completion, HOST physics diagnostics, measurement/accounting
work, independent transition checks, lossless trace encoding/compression/writes
and environment close. Cold construction and warmup are outside this interval;
final compressed-file hashing and summary publication are also outside it and
are separately reported. The 530.94s pytest total is not any cell's denominator.

The following values are copied from each summary's timing fields. Update/diagnostic
throughput uses only compute_and_mandatory_diagnostics_s and is shown as a narrower
subtotal; full-trace throughput uses measured_loop_to_trace_close_s directly. The
latter is not reconstructed by adding partial timers.

| Task | Worlds | Measured transitions | Full-trace wall s | Full-trace transitions/s | Update/diagnostic transitions/s |
|---|---:|---:|---:|---:|---:|
| J |1|96|0.6402476001530886|149.94199115630514|176.67477109047164|
| J |8|768|1.8714569001458585|410.3754673378496|490.3483107425137|
| J |32|3072|5.123485600110143|599.591809125795|765.3766714420666|
| J |64|6144|9.357303000055254|656.5994496452365|837.0687617641521|
| J |128|12288|17.266282099764794|711.676082262515|933.8728708816335|
| J |256|24576|35.446579799987376|693.3250016975898|914.78655990966|
| S |1|96|1.651651999913156|58.123624107891786|69.14050927307541|
| S |8|768|6.729629199951887|114.12218670316794|148.5901407696375|
| S |32|3072|16.257074700202793|188.963885363809|313.80081265574034|
| S |64|6144|28.300352700054646|217.0997678056548|403.58661592006035|
| S |128|12288|53.957941599655896|227.7329274561942|461.16099164630043|
| S |256|24576|112.1638683998026|219.10799217801633|454.1496770854403|
| R |1|96|0.7665200000628829|125.24135050895536|147.9607541044334|
| R |8|768|2.592518000397831|296.2370945475201|382.2772486314242|
| R |32|3072|6.363001000136137|482.79106037139934|689.5318913170034|
| R |64|6144|11.567475599702448|531.1444097758065|776.3148168682994|
| R |128|12288|21.641610000282526|567.7950947198283|847.4255064052694|
| R |256|24576|43.84875479992479|560.4720159588694|841.8363364801237|

Evidence root: [closed scale run49](../../artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01/49-n4-scale/final.json).
For each table row, its complete summary is under
`49-n4-scale/basetemp/test_fixed_scale_cell_{W}_{task}_0/{task}{W}/summary.json`
relative to artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01.
The original summary also preserves
three per-update records, warmup observation, counter snapshots/accounting, actual
reset counts, memory readings, dependencies, source bindings and trace identity.
This document update read those JSON records and raw completion files only; it
did not rerun a cell, recompute physics or execute a saved-evidence audit.

Measurement limits remain explicit:

- Runtime/action/task HOST and device transfers and named synchronization sites
  keep their own recorded scopes. Complete current HOST action/physics audits
  remain in the path; these are not zero-HOST or device-kernel-only timings.
- The bounded observer runs only in the existing warmup. Its inclusive call times
  overlap and cannot be added into an exclusive cost breakdown. Formal updates
  have no profiler. Internal Torch synchronizations and exclusive CPU physics
  validation time are UNKNOWN, not zero.
- Device-owner counters can restart on resets. Their signed before/after snapshot
  differences are not cumulative work; total device work across resets and an
  unexposed device generation are UNKNOWN. Public revision is not generation.
- Torch peak allocated/reserved memory and CUDA free-memory net changes retain
  their original scope. TYX-only peak memory is UNKNOWN.
- This is one fixed bounded measurement per task/world cell, with three formal
  updates. It is not a repeated statistical performance comparison. W256 passes
  capacity for these J/S/R tasks; its measured full-trace throughput is lower than
  W128 in all three tasks. This does not identify a unique bottleneck by itself.

These are the declared two-joint, finite-support and three-joint branching tasks,
not 256 G1 humanoids. No same-protocol Newton performance baseline was run, so this
result makes no faster-than-Newton claim. It does not change the J/R fixed-budget
learning refusals or expand the already-defined model domain. At the scale-author
handoff the wheel was still pending; the subsequent parent result follows.

## Parent installed qualification

Parent commands80/81 built and installed a fresh source-external wheel. Command82
compared all494 package products byte-for-byte between tracked source, wheel and
installed target, with no issues. Package source is `7964d1b`; later root changes
are CLI/tests/docs and do not alter packaged members. Wheel SHA256 is
`aca8054872902a050865c4c5b0ac9b083435dd1cc7743d959fe5f115699ce1e5`
(1552449 bytes). Nothing was published remotely.

Command88 completed **PASS_INSTALLED_EVALUATION_HOST / RAW_EXIT0** outside every
checkout. It checks the public QuadraticCost/native_cost result (0.65625), missing
world refusal, positive/negative three-seed gates, installed module provenance
and HOST import without Torch. This is installed API qualification, not a new
learning run. The same wheel also passed ROB1d interpret/CUDA consumers (86/87),
whose mechanical scope is documented separately.

The first ROB consumer launch83 rejected the Windows virtualenv-root sys.path
entry before any physical scenario. Bootstrap v2 removes only the exact known
editable source and interpreter-root entries, retaining dependency site-packages
and all consumer provenance checks. Neither product nor consumer was changed.
The original failure remains. Canonical complete installed outputs and scripts
are in `artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01/installed-evidence`;
the archive receipt binds all199 files, including failure evidence, without
decompression or physical reruns.

## Historical authoring scope and handoff

The original handoff status was CODE_READY_TESTS_PENDING / UNEXECUTED. This author
performed only source/test/doc authoring, text/file reads and Git operations;
all actual project execution above belongs to the parent.

Base: `ff478eaa6ee570562b229491cb25c95e17783740`.
Branch: `codex/p65-n4a-learning-scale-20261111` in the reused isolated
`C:/Users/38347/.codex/worktrees/p65-h1-support/phymeo`.
Contract: [frozen N4a protocol](p65-mujoco-tyx-n4a-contract.md).

## Scope and source ownership

Exactly six NEW files relative to the base:

- `src/phymeo/training/mujoco_tyx_evaluation.py`: optional generic objective,
  complete-window checks, full evidence encoding, upstream update observation,
  measurement and three-seed gate. No task XML or engine task switch.
- `examples/mujoco_tyx_learning_tasks.py`: frozen J/S/R models, independent grids,
  seeds, budgets and public runtime/environment construction.
- `examples/evaluate_mujoco_tyx_learning.py`: explicit preflight/learn/scale/
  summarize entry. No import from tests and no hidden preflight repeat.
- `tests/training/test_mujoco_tyx_evaluation.py`: declarations and new scope only.
- The contract above and this handoff report.

Old N1/N2/N3, physics kernels, runtime, upstream PPO, exports and pins are unchanged.
The session is original `MuJoCoTyxPpoSession` with 32 rollout steps and original
RSL5.5.2 configuration; the code does not implement an optimizer or PPO variant.

## Public entry and fixed execution units

`run_preflight(task, NEWDIR)` advances the fixed W8 zero-action training-grid
prefix for32 steps. S reads actual step contact forces and requires a positive
contact in every world somewhere in the prefix. This is not the full support
window gate. Parent executes this once per task, by pytest **or** CLI.

`run_learning(task, seed, NEWDIR)` constructs one W8 learner, evaluates its
untrained deterministic mean actor on8 disjoint held-out states, performs exactly
128×32 steps, and evaluates the final actor on the same held-out grid. Each
held-out window is256 complete steps; early termination/refusal is failure,
never a lower cost from truncation. The first declared seed also performs the
one zero-action diagnostic per task in a separate environment with RNG restored.
A diagnostic ordinary exception records FAIL_DIAGNOSTIC/cost=None and its available
full prefix; it adds no learning gate. Interrupt/SystemExit still propagate.

`run_scale(task, first_seed, worlds, NEWDIR)` supports only the declared
W1/8/32/64/128/256 matrix. It creates a fresh learner, performs one32-step warmup
with bounded source-aware observer, then exactly three unprofiled32-step updates.
Only the96W measured transitions enter throughput. Variable declared horizons
64+8*(row%8) exercise actual subset resets; learning horizons remain256.

`summarize_task(task, [three seed directories], NEWDIR)` is saved-value-only.
It binds manifest, summary publication SHA and complete compressed evidence SHA,
requires the exact distinct seeds/config/source/dependencies/budgets and applies
all three fixed ratios. It does not replay physics or infer quality from a lucky
seed. Successful zero-action comparisons are reported separately; failed zero
has no cost/ratio. A completed seed is not a three-seed learning success.

The CLI exposes `--phase preflight|learn|scale|summarize`, `--task J|S|R`,
`--seed`, `--worlds`, `--inputs DIR DIR DIR` and `--output NEWDIR`, with phase
argument validation. Each run refuses an existing output directory.

## State, force and failure evidence

Device observations and native action submission use the accepted public N2/C2
interfaces. Complete current HOST physics validation remains. Independent cost
uses the actual pre-reset native q/v in every returned transition. Device reward
is checked against this FP64 algebra at the existing N1 rtol2e-5/atol2e-6 gate.
This objective is a task metric, not physical energy.

S force inspection uses `state.last_step.contacts`, with backend
`mujoco-tyx-cuda`, original platform/moving geometry pair, full force rows and
matching input/output times. Endpoint CPU observer contacts are retained in the
complete state evidence but never substituted for the integrated step force.
The source retains native box face half-distance and separately reports the
geometric q-.001 separation; no thickness or force-law change was made.

A temporary observer wraps the public adapter.step during every measured/learning
update. Each request gets an owned CUDA clone, the original public step is called
once, and each accepted transition is retained. On a mid-rollout or optimizer
failure, `UpdateCaptureError` carries the whole accepted prefix plus failing
request and public error state to the writer. The original method is restored
in finally. No private physics bank is read and no failed update is rolled back.
On success, requests reference the complete session transitions by same-record
ordinal rather than duplicating full physical payloads. The clone/observer cost
is inside the full update timer. Preflight/evaluation save each request before
its step and the complete returned transition afterward.

Each fresh run writes complete gzip JSONL using typed arrays with shape/dtype,
base64 bytes and SHA; mapping keys retain their types. No full plain duplicate
is created. It retains all transition fields, actions, dynamics, contact/constraint
rows, rewards/done/final observations, parameter hashes per update, Adam steps,
finite losses/gradients and final actor/critic/Adam state. No restore or checkpoint
qualification is repeated. Failures keep their actual phase and full available
prefix. This file encoding is evidence, not a deserializer or new checkpoint API.

Successful output files: `manifest.json`, `trace.jsonl.gz`, `summary.json`,
`publication.json`. Failure retains `failure.json` and the closed partial stream.
A summarize result has its own `summary.json`, bound to the three source runs.

## Historical timing and accounting (scale correction above takes precedence)

Primary measured wall time includes the full public session.update and final
CUDA completion: actual policy rollout, request copies/audit, physics and full
HOST diagnostics, task publication, reset bookkeeping, returns, backward/Adam,
finite gradient evidence and transition/request retention. The profiler exists
only for the already-budgeted warmup; no extra update is inserted.

Post-update independent audits and lossless serialization/compression/write are
separately timed. Delivered throughput includes the measured window's complete
trace writes and final gzip close; warmup trace work remains outside that numerator.
Final compressed-file hashing and summary publication are separately reported
outside training/trace throughput. `publication.json` records summary encode/write
seconds and summary SHA; it is not called optimizer or simulation time.

Counters retain their original owner categories: runtime upload/download,
action raw-bit audit/D2D/status, task D2D/status, wrapper done-mask/finite reads,
kernels and named synchronization sites. Categories may overlap and are not
summed into an invented total. Torch peak allocated/reserved memory and CUDA
free-memory net change are reported; the latter is not TYX-only peak allocation.
Unobserved internal Torch synchronizations and exclusive CPU physics-validation
cost remain explicitly UNKNOWN. No zero-HOST or GPU-PPO-throughput claim is made.

## Historical test declaration and execution handoff

At the original authoring checkpoint, the manual declaration count was
**50 nodes, uncollected**:

| Group | Nodes | Purpose |
|---|---:|---|
| HOST contract/objective/ratio/window/counter and evidence |20| Exact original S XML literal via ast.literal_eval, disjoint grids, no truncated evaluation, full bit encoding, second-step failure prefix/entry restoration, saved summary binding |
| First preflight |3| One fixed32-step actual CUDA task prefix each |
| Fixed learning |9| Three tasks × three seeds, original128 updates |
| Scale |18| Three tasks × six world counts |

Only existing `cuda`/`tyx` markers are used; no unregistered slow marker. Optional
Torch/RSL/CUDA imports are lazy for HOST declaration checks. Parent must explicitly
qualify CUDA and cannot count a skip/fallback as N4a execution. The learning tests
assert completed fixed protocols; task success still requires three-seed summarize.

Suggested serial selection, not executed by this author:

```text
pytest tests/training/test_mujoco_tyx_evaluation.py -m "not cuda" -q -x --tb=short
pytest tests/training/test_mujoco_tyx_evaluation.py::test_fixed_first_preflight -q -x --tb=short
```

Thereafter use each fixed learning/scale node **or** equivalent CLI exactly once;
do not execute both. Example phases (the qualified interpreter/PYTHONPATH belong
to the parent execution receipt):

```text
examples/evaluate_mujoco_tyx_learning.py --phase learn --task J --seed 14011 --output NEWDIR
examples/evaluate_mujoco_tyx_learning.py --phase scale --task J --seed 14011 --worlds 256 --output NEWDIR
examples/evaluate_mujoco_tyx_learning.py --phase summarize --task J --inputs J14011 J14012 J14013 --output NEWDIR
```

Run preflight before eligible learning/scale cells. Do not rerun old complete
N2/N3. Poor quality after128 updates remains a fixed-budget result, not permission
to add epochs/rewards/seeds. W256 capacity, learning improvement, throughput and
all new runtime outcomes were unqualified at that historical authoring checkpoint;
actual learning results and remaining qualifications are recorded above.

## Authoring checkpoints and bindings

- `82839958`: first complete pre-execution contract.
- `c7539c25`: tests-first declarations.
- `b2a77bee`: parent review findings, exact XML, failure-prefix and binding tests.
- `c4bd7bc7`: implementation and final zero-diagnostic scope.

Working-byte hashes at the frozen implementation checkpoint (Git blobs are
newline-normalized separately). The final manifest includes this report too:

| Path | SHA256 | Git blob |
|---|---|---|
| src/phymeo/training/mujoco_tyx_evaluation.py | bca4a9c9bcddac67416058f166d125841ff8c58ab96d693716dff50d583596ea | 89b9975161dd470fda262fec32ca8f8d89aefae2 |
| examples/mujoco_tyx_learning_tasks.py | e01c9e91adc7d3aae35da1eef848b271c55e087ef5f38e60066cc15a8086bd9a | 9de86777ca512e39bda9941e46519d1dee494449 |
| examples/evaluate_mujoco_tyx_learning.py | 0c0e9e00809063f9925b73a7465c596f9b4bb6e5d4dd4ecc967796dd4c6318fd | 0de269bf38540e9f5bc2e433612a98b1f787eb08 |
| tests/training/test_mujoco_tyx_evaluation.py | 4bdfa8437b66ac6f90e16e99aee85cd4b74a4cab4fd9dd7b493faafbff9d5646 | 7e37d7c075e61149ff793c190f07da0819dce70a |
| docs/development/p65-mujoco-tyx-n4a-contract.md | ced1fb6d5a65645d7861bb35116139dd11267fa785835dac6cea09cc67699ec2 | d385b26a16657a039223bef7c2743076adcfeaf1 |

Only text/manual reasoning, exact file reads/hashes and small Git commits were
used. A PowerShell hash-list command initially had a parser error before any
read; the corrected file-I/O command produced the bindings above. No product
failure or verification result is inferred from authoring checks.

## Retained run31 failure and historical repair handoff

Parent tested source5a3957a6: HOST20 and the three fixed CUDA preflights passed.
Original J14011 learning run31 failed at upstream-update-7 after its first
all-world256-step reset. The update returned before `counter_delta` rejected
completed_kernels decreasing on the replacement device. The outer failure writer
then tried counters after runtime close and recorded a secondary NoneType error.
Baseline and zero-action evaluations had completed; the original run is still
FAIL_N4A_RETAINED and its full update7 payload is missing. This repair does not
backfill it or call that attempt a completed fixed-budget learning run.

Pure-author repair base is5a3957a6. Tests-first5ba721bb adds six HOST nodes;
the repair authoring checkpoint had manual total56 (HOST26, CUDA30), then
uncollected; actual affected validation is recorded above.
The three already-passed physical preflights are not a rerun requirement.

Implementation changes stay in the original NEW helper/CLI/tests and two docs:

- `measurement_counter_snapshot` uses public env/runtime/frame ownership and
  transaction identities. No private device bank or generation identifier.
- `counter_accounting` strictly enumerates5 env fields,10 runtime-extra fields,
  seven task fields, and six device base plus two optional transport fields.
  The first three scopes have strict monotonic differences under the same public
  owners. Current-device fields instead preserve before/after and signed
  snapshot differences with cumulative work/absolute generation UNKNOWN.
- Real transition ticks distinguish reset calls from reset worlds and bind the
  latter to env.resets. A revision/epoch delta is never called device generation.
- `measure_update` retains its32-step attempt list through every post-update
  measurement exception, with the actual returned record when one exists.
  Final CUDA completion, counter-read and finite-audit evidence paths are covered.
- `_observed_environment` captures public exit metadata before resource owners
  close. `_output` only uses that saved metadata afterward.

No numerical/policy/physical parameter or budget was changed. The tasks registry
is byte-unchanged. The author ran no import, collection, tests, arithmetic probe,
lint, formatter or physical execution. The parent subsequently completed the focused affected HOST checks
and the single fresh restart of the original J14011 full fixed budget (runs39/40 above). Historical
hashes above remain the original implementation checkpoint, not this repair.
