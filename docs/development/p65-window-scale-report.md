# N7 control-window and random-reset scale qualification

Status: AUTHORING_ONLY / CODE_READY_TESTS_PENDING. This is a fixed connection and
resource experiment, not a learning-quality experiment. Parent alone executes.
Only this report, `examples/mujoco_tyx_window_scale.py` and
`tests/training/test_mujoco_tyx_window_scale.py` are new; old products/tasks stay frozen.

## Pre-execution protocol

| Task | Worlds | D | Learner | Policy windows / mode | Physical worldsteps / mode |
|---|---:|---:|---|---:|---:|
| J | 8 | 2 | original GRU | 24 | 384 |
| S | 128 | 4 | original MLP | 24 | 12288 |
| S | 256 | 4 | original MLP | 24 | 24576 |

Every mode runs one 8-step warmup update followed by two measured 8-step updates.
Full then integration each has an independent factory/ExitStack and no retained
learner/native owner between modes. All source task XML, h, action/reward mapping,
PPO config and hidden widths stay unchanged. Seed6511, namespace N7, stable world
IDs `7+13*i`; anchors repeat original eight training templates by `i%8`.
N5 q/v bounds are J `[-.1,.1]`/`[-.04,.04]`, S `[.00065,.00085]`/`[-.002,.002]`.
Horizons cycle3/5/7/11. N6 uses endpoint reward/done and one action submission per
D physical steps. All CPU/NumPy/Torch global RNGs are reseeded identically at
mode entry; N3 captures actual CPU/currentCUDA and explicit empty named-generator
registry; N5's independent reset indices are included. No new reward/learning gate.

Compare every update's complete common transitions, all substep states/receipts,
held channels, control/source/task/layout bindings, actual reset receipts/index
ledger, RSL return values/masks, learner/Adam/normalizers/GRU hidden/RNG. Only opaque
runtime/workspace owners and explicitly full-only endpoint observation fields
are excluded, with both raw records retained. Publication revisions/epochs are
not removed between equal fresh-mode executions.

All accepted physics must have correct membership/tick/time/control binding.
Independent ordinary J Euler and S 2kg force-to-motion recurrence uses actual
stored controls/actual published contact force. S check is not a second contact
law solver. Endpoint feature/reward/termination/timeout is recomputed from native
q/v, with terminated priority. Complete attempted requests, reset snapshots and
accepted substep prefix survive late failures; no repeat step repairs evidence.

Measurement reuses the accepted read-only native factory/copy observation and
public runtime method wrappers. Caps derive from this fixed24×D/W matrix and
worst-case every-policy-window reset; they do not relax physics resources.
Record process commit/RSS, actual construction/copy counts, all counters with
resettable scope, full update+final synchronization time and measured-loop through
gzip close. Diagnostic wrapper/owned evidence costs are included and disclosed;
no exclusive HOST percentage, unobserved synchronization zero or Newton speed claim.

J8 must qualify before S128; S128 before S256. On common-state mismatch, physical
refusal or resource failure stop expansion, retain complete failure and do not
lower W/change gates. Parent will run each fixed cell once; no implicit preflight
or old suite rerun. 06:20 is the parent stop-new-long-task boundary.
## Implementation handoff (unexecuted)

The new entrypoint supports one fixed cell via `--task J --worlds 8 --output ABS`
or `--all --output FRESH_ABS` for the frozen J8→S128→S256 sequence; each pair runs
full then integration. It dynamically loads only the original optional examples,
never tests, and does not copy PPO. The N5 W4 example constructor cannot accept
expanded keys, so the new factory uses its exact public spec/reset formulas with
the explicitly registered expanded keys and original templates. This is not an
implicit extension of the old constructor's interface.

Source reuse: `mujoco_tyx_learning_tasks.py` for all task inputs and original
`seed_all`, `mujoco_tyx_evaluation.measure_update` for actual update/request capture,
`compare_mujoco_tyx_observation_training.py` for stable binding/digest/process
memory and accepted native factory/copy observer. The observer's per-mode maximum
constructor count is `4*W*(24+2)+128`, copy count `2*(24*D+24+2)*W+128`, and event
count320. These are conservative observation bounds covering the declared policy
calls and worst-case reset of every world; they do not change engine capacities.
The original observer already permits96 physical calls per instance, exactly
S/D4's24×4; each mode gets a fresh instance. Its private runtime-layout census is
not invoked. Native references are not retained across the mode function boundary.

A second public adapter observer captures RSL outputs, actual reset receipt and
reset ledger for every call. On an error after physical/adapter acceptance, the
accepted transition is recorded before any fallible evidence copy. Every upstream
failure includes both this complete prefix (including N6 error.control_window)
and the original measure_update prefix/completed record. No extra step is called.
Each successful update saves the complete raw record plus common-view signature;
full current training_state includes actual Adam, normalizers, Python/NumPy/
TorchCPU/currentCUDA RNG, N5 indices, and real GRU non-state_dict hidden state.
Named custom RNG registry is empty, not silently claimed to enumerate callers.

The pair removes only opaque task/runtime owners and the known full-only state
view fields. Publication revision/epoch, all time/tick, source/model/layout/task,
full step receipts, future transition fields, all action/control, substep and RSL
output fields remain compared. Raw full and integration records remain intact.
Support load count is observed; every S substep checks complete force/motion even
when unloaded, without adding an unregistered always-loaded success criterion.
Task termination is rejected by the original evaluation gate; timeouts reset via
N5 and continue with a new policy action, never continue a stale action window.

Hand-counted test identities: **21 HOST + 3 fixed CUDA cells = 24**. HOST nodes
cover fixed matrix/counts, unsupported cells, complete comparison binding/force/
motion/revision/action differences, late public and post-acceptance evidence
failure prefixes, owned RSL output copies and rejection of retained owners.
CUDA nodes are the actual three pairs, not an extra hidden preflight; parent must
select J first then S128/S256 once each. Same source qualification and actual
execution results remain pending. No import, collection, numerical computation,
formatter/lint, tests or build were performed by the author.
Observer schema note: the reused observer's `step` identifies the public adapter
policy-window ordinal here, so the `physics_steps` list has D entries carrying
that ordinal. Output explicitly records this meaning and D; physical work is the
complete list length plus original Env and substep receipts, never the largest
ordinal alone. RSL returned mapping/tensor values are owned and their original
outer type/device/batch metadata is saved. Reset cleanup errors, native constructor/
copy failures and rejected physical batches are qualification failures.
Authoring source SHA256: `08827938bcf33507d54178aec091cd707dbcb529915155c47facc452095bf65e`.
Authoring test SHA256: `490c8e11fac9484589226fde773cd2d0289ca5e555168d8c32753d307fc7d82d`.
Only text review and Git diff checking were performed; static and actual CUDA qualification remain parent-owned and pending.
