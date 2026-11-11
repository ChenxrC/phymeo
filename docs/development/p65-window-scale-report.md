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