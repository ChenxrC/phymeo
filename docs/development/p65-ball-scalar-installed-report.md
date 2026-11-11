# BALL-A independent installed consumer

Status: **CODE_READY_TESTS_PENDING / AUTHORING_ONLY**. No import, collection,
Python execution, numerical probe, lint, formatter, build or installation has
been run by this author. The parent owns the only execution slot. This report
is a pre-execution contract, not a pass or an extension of the accepted domain.

## Exact delivery and source basis

Only two NEW files are authored:

- `tests/installed/check_ball_scalar_actuation.py`
- this report, `docs/development/p65-ball-scalar-installed-report.md`

The product seam is parent `3b9072e4`, frozen in the BALL-A section of
[the 07:20 plan](../superpowers/plans/2026-11-12-newton-parity-until-0720.md).
The consumer is standalone: it imports no checkout test/example and uses public
`compile_mujoco_xml`, `MuJoCoRuntime`, full/integration observations, held controls,
clone, exact checkpoint serialization/restore and partial reset. Inline geometry
is the original HINGE_BALL_SLIDE declaration. Its four scalar actors are the
predeclared BALL-A filter-early case: one NONE direct motor and three FILTEREXACT
actors, negative gear, shared hinge transmission, affine slide bias,
control/activation/force clamping and one early-actuation channel. The ball has
no actuator; original scalar stiffness is retained, not ball stiffness.

Dimensions: nq=6, nv=5, nu=4, na=3; stable worlds `(7,91)`; h=.001. Nonidentity ball
quaternions, nonzero original q/v, activation `.34/.35` and start time `.125`
are explicit. The original scalar qpos/DOF address mismatch is asserted.
No asset download or external fixture discovery is performed.

## Fixed path and checks

1. Both worlds advance 24 uninterrupted steps: 8 loading, 8 reversal, 8 unloading.
   Original FP32 three-channel requests are saved. Unloading means zero requested
   ctrl/qfrc/xfrc, not artificially zero filter memory or joint force.
2. A nonzero reverse command is held before checkpoint. Save actual accepted state,
   original v2 capsule bytes and continuous native state. A and restored B advance
   once, and an independent public clone advances once; complete integration state
   and full receipts are compared exactly. A stale CAS restore is rejected without
   changing capsule bytes.
3. Reset only world91 to its original q/v/act/time; retain world7's full state,
   controls and last receipt exactly. Advance the mixed batch twice, checking both
   real successors against appropriate continuous native references.

Expected physical work: main runtime 28 batch steps, clone 1; W2 yields **58
worldsteps per mode**. Native reference steps are separate and not counted as
TYX work. Restore does not rewind diagnostic counters. All acceptance maps and
four real stage launches per step are asserted; no rejected physical step may
be treated as success.

Every attempted step is journaled with full input state, all requested controls,
same-entry native fields, continuous native output and independent scalar oracle
before TYX execution. Every returned step saves complete full/integration
publication before assertions, including M, complete COM J, body geometry, bias,
passive/applied force, smooth and integration acceleration, act/act_dot, scalar
and generalized actuator forces, q/v, control binding and receipt identities.
All seven scalar activation channels are checked independently using the exact
exponential filter, clipping, affine force and native gear/address mapping;
all ball tangent columns of actuator force must be zero. No product helper is
used to construct an oracle. Full native M/J/bias are obtained from fixed CPU
MuJoCo3.3.7 at the actual stored entry; a distinct native trajectory continues
without overwriting its state from TYX.

Original BALL1 gates are retained: q5e-5, v3e-4, time2e-6, each dynamics field's
original TOL; activation rtol3e-5/atol3e-6 and force rtol3e-4/atol2e-4. This is an
engineering comparison, not bitwise CPU/TYX equivalence. Exact equality is used
only for same-backend lifecycle and publication binding.

## Source isolation, failure and ownership

The CLI requires absolute installed-root/output paths and a fresh output folder;
consumer, cwd and installed root must be outside any checkout. The module-origin
inventory hashes every loaded `phymeo.*` under the expected wheel installation,
before and after use. Actual TYX step is guarded against CPU `mj_forward`,
`mj_step`, `mj_step1` and `mj_step2`; independent reference calls occur outside
that guard. Detached full observation outside step may use the supported native
observer. No private device banks, solver internals or runtime monkeypatches are
used.

Attempt records precede step. A failing step/readback keeps the original traceback;
optional failure-state capture errors are separate. Both runtime owners receive
a close attempt even after another close fails; successful cleanup cannot be
reported when a close error occurs. Original physics errors are retained alongside
cleanup diagnostics. The gzip journal is closed in a finally path and logical /
compressed hashes are saved. This consumer does not inject new physics faults;
the author's BALL-A suite owns those negative controls.

## Parent-only next execution

Copy this one file to a fresh directory outside the checkout and run once per
mode with the newly installed wheel environment, for example:

```text
python check_ball_scalar_actuation.py --mode interpret --expected-install-root <absolute-wheel-site> --output <fresh-absolute-output>
python check_ball_scalar_actuation.py --mode cuda --expected-install-root <absolute-wheel-site> --output <fresh-absolute-output>
```

Outputs: `ball-scalar-actuation.jsonl.gz`, `accepted-checkpoint.bin`, `summary.json`,
plus explicit failure/cleanup records if needed. No old source suite is imported
or implicitly rerun. Product parameters, tolerance, pin and capacity are unchanged.
This does not qualify ball-target motors, free/contact/limit/dry/ball-spring
composition, JOINTINPARENT, G1, training, or any Newton speed comparison.
Consumer author SHA256: `8907edc12180d1bfdeaeea8163e96c9c5035a0b0c336943130e16cdd7d3489df`.

## Outer finalization review correction (still unexecuted)

The parent identified that outer provenance/journal finalization could replace an
already recorded primary exception. The consumer now retains the original
exception object and traceback, independently attempts failure-file/journal
publication, final provenance, summary journal, journal close/hash, disk summary
and stdout summary, and collects every finalization exception by stage. It raises
the original primary error when it is alone, or a BaseExceptionGroup containing
it and all finalization errors. The in-memory summary retains every traceback;
if an output channel itself fails, the exception group remains the authoritative
failure report rather than claiming that unavailable file was written. A summary
already written before a later stdout failure does not imply command success:
the command must also return RAW_EXIT0. No close is skipped because another
finalization stage failed. Model, oracle, gates and all physical call counts are
unchanged. This remains text-only authoring; the parent owns static/runtime checks.
Final consumer author SHA256: `1898c8e3aa887ec0b69153a1afb387742666bd667f15ba9f2e6cec10f5d9842a`.
