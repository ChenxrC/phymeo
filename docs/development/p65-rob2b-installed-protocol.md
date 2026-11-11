# ROB2b installed activation composition consumer

Status: **CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**. This protocol adds one standalone public consumer and this document only. Parent builds/installs one integrated wheel and owns every actual run. The author has not imported, collected, executed, linted, formatted, compiled or numerically probed this consumer.

## Invocation and source isolation

Copy `tests/installed/check_activation_composition.py` outside every checkout, execute with an outside-source cwd and the existing optional dependencies:

```text
python check_activation_composition.py --mode interpret --installed-root C:/.../fresh-installed-site --output C:/.../new-interpret-result
python check_activation_composition.py --mode cuda --installed-root C:/.../fresh-installed-site --output C:/.../new-cuda-result
```

The root argument is the actual installation directory containing `phymeo/__init__.py`, not the package directory. Both paths must be absolute and output must be fresh. The consumer never edits `sys.path`. It rejects explicit checkout-root or checkout/src entries; cwd and script must be outside source-checkout ancestry. Existing venv/site-packages dependency paths are permitted, even when the environment happens to reside in a checkout. It does not mistake an arbitrary sys.prefix directory for project source.

Before construction and after all lazy imports, every loaded `phymeo.*` module with a file is required to resolve under the declared installation root. Full paths and file SHA256 values are recorded. The source XML, consumer bytes, checkpoint bytes and complete logical/compressed trace are independently hashed. Parent's whole-wheel parity check remains separate; this consumer does not re-audit every package file or import repository tests/examples/helpers.

## One fixed mixed path

The self-contained XML preserves the already qualified ROB2b B05 dimensions/physical parameters: one free root, a coupled hinge/slide descendant, offset inertias, sphere/plane condim3, scalar soft limit and dry friction, mixed NONE/FILTEREXACT actors, negative gear and shared-joint accumulation. `nq/nv/nu/na=9/8/4/3`. Native Euler h=.001, Newton128/tolerance1e-10 and pyramidal contact are unchanged. Sparse worlds `(7,91)` start with nonzero velocities/activation and a genuinely active initial limit and soft contact. No free body is fixed or artificially corrected.

There are16 main batch calls: four positive-control, four reverse-control and eight exactly zero-requested-control steps. Nonzero generalized and body-wrench external channels remain explicit; zero control does not mean forced activation zero. Both worlds must actually exhibit positive contact support, nonzero limit force and nonzero dry-friction force. No event is inferred from a constructor or capability label alone.

After those16 calls, the consumer saves a byte-serialized public checkpoint, advances a real successor, advances a public clone from the saved accepted state, restores the original by public CAS, and advances the restored successor. Clone and restored full integration state/receipt must be byte-equivalent to that first successor under the same held control. It then resets only world91, proves world7 retained, checks the selected initial activation/tick/empty prior receipt, and advances both worlds again. The independent native lane restores its own checkpoint, not the device's state; only the explicitly reset world restarts from its declared initial state.

Exactly **19 main batch calls +1 clone batch call =40 accepted world-steps per backend**. Counters must agree with actual calls despite restored/reset tick values. No failure or hidden reset is allowed. Runtime and clone close in `finally`; no additional physics call is used for warmup, qualification or evidence generation.

## Independent force/state evidence

Only public Phymeo construction, controls, step, observations, checkpoint/restore, clone and reset are used. No private storage, lowered model, bank or solver is inspected. Native MuJoCo3.3.7 is independently compiled from the same inline XML. Each native same-input oracle receives the actual entry q/v/**act** and stored FP32 control, qfrc/xfrc before forward/step. A separate native trajectory is initialized once and is never overwritten with device output during ordinary continuation.

The independent analytic exponential/gear/force law is embedded in this file. Full native body positions/rotations/COM, body Jacobians, M/bias/passive/applied/smooth acceleration, actuator forces/act_dot, constraint force, contact geometry/frame/full relative J/wrench and EFC identity/J/R/aref/force are saved and compared. Each contact also checks equal/opposite force and full COM moments on both owners. The original activation, force, M/J and T2/T3/T4 tolerances are retained exactly; constrained continuous q/v remain absolute2e-4/3e-3 and time2e-6. Solver iteration/KKT law is not changed or independently replaced here.

The capability domain must be `tree-pyramidal-mixed-filterexact-v1` with checkpoint v2 and an actual `MuJoCoTyxActivationStepReceipt`. Actual input/output activation, complete raw control, full generalized actor effort and first six unactuated root entries bind the same receipt. Each actual step must use eight existing composed device launches. Native `mj_step`/`mj_forward` are temporarily forbidden during Phymeo `step` and `integration_snapshot`; native oracles and explicit full endpoint observation are outside that guard. Full/integration common q/v/act/ctrl/time/tick/held control and entire receipt must match exactly.

## Saved artifacts and failure semantics

`activation-composition.jsonl.gz` preserves typed complete records: initial provenance, source/declaration, initial observations,20 full attempt/step pairs, checkpoint bytes/binding/native state, partial-reset before/after, final state/counters, resource closure and summary. Arrays retain dtype, shape and raw bytes; dataclasses retain every field. Ordinary Python floats preserve their binary representation. No field is omitted merely to make comparisons pass.

`accepted.checkpoint` is the actual public saved byte capsule. `summary.json` contains status, true call/world-step counters, all exercised constraint classes, final module provenance and trace compressed/logical size/SHA. Exceptions retain original type/message/traceback and a failure event, and are re-raised. A consumer PASS only qualifies this installed mixed combination and continuation; it does not claim all MuJoCo, all robot joints/transmissions, multiple free roots, full source matrices, training, G1 or performance parity.
