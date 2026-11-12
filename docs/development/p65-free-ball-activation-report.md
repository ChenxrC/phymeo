# R3-A FREE+BALL activation: source author handoff

Status: CODE_READY_TESTS_PENDING / AUTHORING_ONLY. The parent grant is
f1f8da56, last section of [the design](p65-free-ball-activation-next-design.md).
This lane performed no project import, collection, numerical probe, pytest,
lint/formatter, compile, build/install, profile or native execution.

Main tests-first checkpoint: ff73ee91c722c28970c4796b309be14f1a560f07 in the
independent source lane, UNEXECUTED. The two-source checkpoint follows it:
b294bf291d4e0c35664bd5e0acbd36ed2791d29a. This report is a separate commit.

## Exact source delivery

Only two existing product paths changed, under `src/phymeo/backends/mujoco_tyx/`:

- `_floating_model.py`: FREE+BALL permits NONE/FILTEREXACT only with explicit
  `allow_activation=True`; all other dynamics remain rejected. The existing
  default admits only stateless NONE. The structural activation coordinate cap
  is 32, while the common activation compiler retains complete address coverage,
  range/time-constant/finite parameter validation before device allocation.
- `_actuator_model.py`: distinguish FREE+BALL before the fixed BALL branch.
  First compile the complete narrow floating structure with explicit contact and
  activation permission. Only then route complete declared pairs to the original
  condim1 normal or mixed condim1/3 pyramidal base, passing `allow_activation=True`.
  No pairs retains the validated floating base. Generic `needs_constraints`
  cannot bypass the new branch's limits/dry/excludes/shape/static-capacity gates.

The fixed BALL branch, no-BALL fixed/scalar-FREE branches and stateless callers
retain their original route and parameter semantics. The plan wrapper delegates
all original base fields and adds the original activation layout; it does not
copy or replace mechanical state. No new layout, kernel or device-call ABI.

The source delta is 29 insertions and three removals. No geometry, M/J/bias,
passive force, implicit damping, q/v/quaternion integration, CSR accumulation,
filter arithmetic, contact law, solver budget or allocation machinery changed.
Numerical correctness of the new composition still needs the parent's actual
qualification; inspection alone is not a passing result.

## Separate owners and runtime obligations

The parent owns `_model.py`, `_mujoco_tyx_runtime.py`, `mujoco_actuation.py`,
`mujoco_joint_transmission.py` and `mujoco_checkpoint.py`. These were not edited
or copied into this author tree. In particular, this tree's old combined receipt
still rejects the new free schema until those parent seams are integrated.

Required public pair: `native-free-ball-joint-filterexact-actuation-v1` plus
`native-free-ball-joint-motor-transmission-v1`, same existing combined receipt
class, original same-attempt FP32 field/control/accepted-force binding, checkpoint
v2 complete act bank and exact plan-schema validation. Original fixed/scalar
schemas stay unchanged. Capability labels must identify free stateful no-contact,
normal and pyramidal domains rather than be overwritten by stateless labels.

The original source author owns the main tests and self-contained installed
consumer; the original tests author owns transactions. Their tests-first and
subsequent fixed-source commits are separate deliverables, not author-run results
in this report. No old test was modified here. The parent retires only exact
old activation-negative sources that receive a corresponding positive successor.
These include original R2, R3-N and R3-P `activation` sources plus the fourth
original R1 `tests/test_mujoco_tyx_ball_activation.py` `NEGATIVES['free']`.
That last source wraps the complete original `r5-mixed` worldbody, including its
box support, under a new FREE root; it is not a FREE-target actuator. Preserve
that exact XML/default entry for two actual steps, without manufacturing nonzero
contact. R1 affine-ball, parent-transmission and non-FILTEREXACT negatives remain.

## Qualification pending and limits

The frozen three families/six continuous sources retain two sparse worlds and
24 real calls, nonzero q/v/act/controls, reverse and unload. Independent same-entry
native and separately continuous native references must check act, original full
M/J/bias/forces, all relevant contact/EFC rows, CSR power and joint movement.
Keep R1 act/force gates and the stricter original FREE+BALL q/v gates; do not
substitute the fixed BALL trajectory tolerances. Complete actual return/failure
payloads and source-external module provenance remain required.

Late last-world publication/readback/restore failure, independent actuation or
transmission record corruption, fixed/free schema swaps, checkpoint/clone/partial
reset with real successors, and focused old route compatibility remain pending.
Both interpret and CUDA and a fresh product wheel are required for acceptance.

Scope remains one FREE root, BALL/R/P, direct JOINT NONE/FILTEREXACT, fixed gain,
BALL BIAS_NONE (original scalar affine bias allowed), nv35/nq45/body128,
nu32/na32/CSR96, primitive implicit pair16/rawslot32 and condim1 or pyramidal3.
No FREE actuator, multiple free roots, limits/dry, BALL spring/affine bias,
tendon/plugin, new shape/flag, larger world capacity, training or G1 long-window
qualification. No D4/D6 steady or strict precision result is changed.

Original branch `codex/p65-d6-installed-20261113` and old untracked
`docs/development/mujoco-tyx-t1-review.md` are preserved. Git diff checking is the
only authored-source check here; it is not a project/static/physics execution.
