# P6.5 B1 signed-dihedral bending checkpoint and local verification

Current status: **B1_LOCAL_TESTS_PASSED** on source
`8b6d935034fb5596af086ffdc6e8f11ddf1d6e8a`: 98 collected, 98 passed,
RAW_EXIT=0; Ruff lint and format checks passed. This is local host B1 evidence,
not native/composite physics-stage acceptance.

Historical checkpoint `ee6679182dae00be5faca0beb1dc753702066fa3` was correctly
reported **CODE_READY_TESTS_PENDING**: no tests, numerical imports, compilation,
lint/format, build, installation or CI had run at that checkpoint. The human
approved parallel coding and deferred serial execution. The later authorized
run below is first post-code verification, not historical TDD RED/GREEN.

Worktree: `C:/Users/38347/.codex/worktrees/p65-bending-module/phymeo`.
Branch: `codex/p65-bending-module`.
Frozen contract base: `1ccab18635a3b89c09a0c2e3f735bca2f823e65d`.
Ownership is limited to `constraints/bending.py`, `constraints/bending_trace.py`,
`tests/test_bending.py`, `tests/test_bending_trace.py` and this document. No shared
exports, distance law, runtime, native backend or material ledger are modified.

## Angle, derivative and units

For the ordered hinge `(a,b,c,d)`, the faces are `(a,b,c)` and `(b,a,d)`:

- `e=b-a`, `u=c-a`, `v=d-b`;
- `nL=e cross u`, `nR=(-e) cross v`;
- `theta=atan2(e_hat dot(nL_hat cross nR_hat), nL_hat dot nR_hat)`;
- `C=theta-rest_angle`, with no wrapping or rest-angle replacement.

The opposite-vertex gradients of this convention are
`gc=-|e|*nL/|nL|^2` and `gd=-|e|*nR/|nR|^2`.
Let `tc=(c-a) dot e/|e|^2`, `td=(d-a) dot e/|e|^2`.
The endpoint gradients are `ga=(tc-1)*gc+(td-1)*gd` and
`gb=-tc*gc-td*gd`. Rotation about the edge gives the negative signs at both
opposite vertices. The endpoint coefficients are obtained by differentiating
the oriented edge and the vertex projections; they enforce the translation
identity `sum(g)=0` and rotational identity `sum(x cross g)=0` in exact arithmetic.
Production uses these analytic expressions, with each face scaled before
cross products and divisions. Finite differences exist only in the tests.

A flat unit hinge has `ga=(0,0,2)`, `gb=0`, `gc=gd=(0,0,-1)`.
With `d=(0,0,-1)` its angle is `+pi/2` and compliance 2 gives energy `pi^2/16`.
Swapping `c,d` alone or `a,b` alone negates the angle. The complete consistent
face exchange `(a,b,c,d)->(b,a,d,c)` preserves it: both edge axis and normal
order reverse. The parent explicitly confirmed this convention.

Coordinates are metres; angle is radians (dimensionless for dimensional
analysis). Compliance has units `1/(N*m)=s^2/(kg*m^2)`; therefore
`E=C^2/(2*compliance)` is Joules. The caller supplies
`alpha=compliance/h^2`, with units `1/(kg*m^2)`. The gradient has units `1/m`,
so `sum(w*|g|^2)` has the same units as alpha. Lambda and delta-lambda have
units `kg*m^2`; `dx=w*delta_lambda*g` is metres. This multiplier is not a
world impulse and is not automatically an angular momentum.

## Supported discrete domain and explicit limits

Every configuration owns read-only FP64 rest/compliance arrays and canonical
int64 indices. Each row contains four distinct nonnegative indices. Duplicate
hinges are rejected using the unordered edge and unordered pair of opposite
vertices, independent of either orientation. Positive compliance is mandatory;
hard zero-compliance bending is outside B1. Mutable working arrays must be
finite FP64 NumPy arrays. Masses are nonnegative; alpha is strictly positive.
The implementation accepts disjoint strided publication targets and rejects
read-only, self-overlapping or input-aliasing mutable storage.

For each face, scale is the maximum absolute component of the two edges from
its first oriented vertex. The norm of their scaled cross product must exceed
`1e-10`, matching the explicit dimensionless offset-port admissibility rule.
There is no example-specific tolerance, clamping, normal fallback or normal
repair. Overflow, invalid arithmetic, division by zero and underflow in the
actual evaluation/update are rejected as nonrepresentable. Scaled geometry
avoids unscaled area overflow, but B1 does not promise every extreme input with
a mathematically finite result is evaluable through its FP64 intermediates.

**Parent branch ruling:** B1 certifies only each discrete projection's recorded
pre/post domain: both theta values are strictly in `(-pi,pi)` and both angular
errors satisfy `abs(C)<pi`. Additionally `abs(theta_after-theta_before)>=pi`
rejects an explicit atan2 jump. A returned trace's `branch_status="smooth"`
means exactly these checks. It does not prove the full affine correction segment
avoids degeneration or crosses the branch cut and returns. Such continuous-path
proof, if required by B2 or accepted step/CCD integration, remains a separate gate.

`bending_energy` and `bending_residual` apply the same evaluation domain. The
residual returns `max(abs(C+alpha*lambda)/max(abs(rest_angle),reference_angle))`
and `max(abs(C))` in radians, with positive finite `reference_angle` (default 1).
The existing normalized-error helper preserves positive residual underflow.
An empty constraint set has zero energy/residual and an empty trace tuple.
All-fixed soft hinges still update lambda and record a complete projection;
their positions remain unchanged. No fixed-node or hard-law fallback is added.

Invalid shapes/types/configuration are `ValueError`/`TypeError`. Unsupported
geometry/branch or numerical evaluation is `SolverFailure`. Input validation
and all local numerical work precede caller-state publication.

## Sweep and trace ownership

The sweep evaluates hinges in declared row order on private candidate copies.
It computes `dlambda=(-C-alpha*lambda)/(sum(w*|g|^2)+alpha)` and then applies
`dx=w*dlambda*g`. Every local pre/post state and gradient is owned by an
immutable `BendingProjectionTrace`. The record includes element/node identity,
angles, errors, old/new multiplier, delta, supplied alpha, actual denominator
and branch status. Trace order is complete element order; a future integrator
adds the sweep index. A later hinge failure discards all private updates and
publishes neither positions nor multipliers nor a partial returned trace tuple.
The final writes target validated disjoint mutable storage; concurrent external
mutation of these arrays is not supported.

Records describe FP64 operations and actual stored endpoints. They are not
exact-real-angle certificates, energy-acceptance flags, force/torque proofs,
continuous geometry proofs or a second time integrator. In particular,
`Fraction(float(atan2(...)))` would not certify an ideal exact dihedral.

## Authored coverage and original deferred state

`tests/test_bending.py` covers literal flat/right-angle values, all three
orientation permutations, rigid translation/proper rotation/scaling, independent
centered angle derivatives, translation and rotational identities, a two-hinge
shared-node energy derivative, constitutive residual normalization, all-fixed
soft behavior, owned configuration, duplicate/invalid inputs, dimensionless
face conditioning, antipodal/degenerate/extreme geometry and empty sets.

`tests/test_bending_trace.py` covers an independent centered-gradient two-hinge
reference sweep, full ordered trace replay, late second-hinge geometry/arithmetic
rollback, explicit branch jump, pre/post angular-error rejection, trace ownership
across later sweeps, malformed records, invalid input transactionality, supported
strided targets and rejected overlapping targets. These are independent local
hinge/sheet fixtures, not handoff-specific cloth/contact scenarios.

At the initial `ee667918` checkpoint every authored test was **UNEXECUTED**.
The parent subsequently granted the sole B1 local execution slot, recorded in
integration queue commit `92daf10`, after the human deferred remote CI to a
separate machine. No threshold, unit, branch rule or independent oracle was
changed during local verification. The original deferred history remains true;
it must not be relabeled as a prior failing/passing TDD cycle.

Native B2 bending, a composite distance/bending material path ledger, full
force/moment/power and NUM005 accounting, backend parity, coupled contact/grasp,
mesh/time-step/duration/repeat/perturbation/failure acceptance and folded-cloth
integration remain open. The existing distance-only ledger is unchanged and
must not accept these records until the separate composite protocol is proved.

## Authorized first local verification, 2026-10-30

Preflight confirmed a clean B1 worktree at `ee667918`. The approved delivery
policy `50a5eeb469390cee032fef6d7ee6e0150ad539f3` differed from the frozen base
only in workflows/documentation, with no underlying production/test changes.
It was merged into this lane as `9fd3f55f0ff4d9951ff71c018bfa31326e4a3c50`.
The initial restricted-shell Git/CIM reads could not access the real worktree
and process inventory; elevated read-only commands established the actual clean
state. No repository repair was performed. An unrelated transient Python PID
20620 disappeared naturally before validation. A fresh inventory showed zero
matching numerical/build processes; no process was killed or restarted.

Environment: existing delivery `.venv-py311`, CPython 3.11.15, NumPy 2.4.6,
pytest 9.1.1, pluggy 1.7.0 and Ruff 0.17.2 on Windows. No installation occurred.
Backend: host NumPy FP64 only. `PYTHONPATH` explicitly selected the B1 worktree's
`src`; an assertion checked the actual `phymeo.constraints.bending.__file__`.
The package, bending module and trace module all resolved to this worktree.
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` were each 1;
`PYTHONDONTWRITEBYTECODE=1` was set for the import probe and pytest.

The following commands were executed serially from the worktree named above:

```powershell
$b1Python = 'C:\Users\38347\Documents\ChatGPT\phymeo\.venv-py311\Scripts\python.exe'
$b1Files = @(
    'src/phymeo/constraints/bending.py',
    'src/phymeo/constraints/bending_trace.py',
    'tests/test_bending.py',
    'tests/test_bending_trace.py'
)
& $b1Python -m ruff check @b1Files
& $b1Python -m ruff format --check @b1Files
& $b1Python -m ruff format @b1Files
& $b1Python -m ruff check @b1Files
& $b1Python -m ruff format --check @b1Files
$env:PYTHONPATH = 'C:\Users\38347\.codex\worktrees\p65-bending-module\phymeo\src'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $b1Python -m pytest tests/test_bending.py tests/test_bending_trace.py -vv --basetemp 'C:\Users\38347\.codex\worktrees\p65-bending-module\phymeo\artifacts\p65-b1-local-20261030-141144\pytest-first-basetemp' -p no:cacheprovider
```

Each command's stdout/stderr was redirected to its separate artifact log, and
`$LASTEXITCODE` was captured immediately to the matching `.exit` file. Results:

| Evidence log | Result | RAW_EXIT |
| --- | --- | --- |
| `environment.log` | Actual child import path asserted; environment recorded | 0 |
| `01-ruff-check.log` | All checks passed | 0 |
| `02-ruff-format-check.log` | Four files would be reformatted | 1 |
| `03-ruff-format.log` | Four files formatted | 0 |
| `04-ruff-check.log` | All checks passed | 0 |
| `05-ruff-format-check.log` | Four files already formatted | 0 |
| `06-pytest-first.log` | 98 collected; 98 passed in 0.60 s | 0 |

The first and only failed check was formatting; the logged diff changes layout
and final newlines. It was applied only to the four allowlisted Python files,
then committed as `8b6d935034fb5596af086ffdc6e8f11ddf1d6e8a` before pytest.
There were **no numerical test failures**, no solver correction, no tolerance
change and no oracle change. The two-file group passed on its first numerical
execution, so no redundant test rerun or wider old-suite run was performed.

Actual parametrized coverage was 56 passing items in `test_bending.py` and 42
in `test_bending_trace.py` (28 authored test functions, 98 collected items).
The complete log retains every nodeid. Covered behaviors are the mathematical,
state-ownership, branch, degeneracy, storage and transaction cases listed above;
line/branch coverage instrumentation was not run and no percentage is claimed.

Evidence directory (new, preserved, Git-ignored):
`C:/Users/38347/.codex/worktrees/p65-bending-module/phymeo/artifacts/p65-b1-local-20261030-141144`.
The pytest basetemp was resolved, checked to remain inside that evidence
directory and verified absent before its first use. No old basetemp was deleted
or reused. `initial-source-test-git-manifest.txt` records the merged pre-format
source/test tree; `verified-source-test-git-manifest.txt` records the complete
tracked source/test/config Git blobs at the tested commit.
`pytest-first-manifest.json` contains the exact executable, arguments, source
commit, working directory, import path, backend, basetemp check and disk hashes.

Tested file SHA256 values:

| File | SHA256 |
| --- | --- |
| `src/phymeo/constraints/bending.py` | `3425E035A7B52279D94EA66F3B4D13669BFA45BB9B2C63E2E25557ABB8CA33CC` |
| `src/phymeo/constraints/bending_trace.py` | `AD102923C3985A134D338441A66AE6BA28ABD599A5347772FC3E5C3097FCA13C` |
| `tests/test_bending.py` | `3C5219D7D791671F397819C89C0D386A6E05C851462BAC08952F3BBAA4032369` |
| `tests/test_bending_trace.py` | `5EBBA8D96512EDCAE3E7641A68C828075241A06B7BDAB46F7D22437D3FD2D536` |
| `pyproject.toml` | `9DE728C631C14DBAFE50BEEE0E9BCF5A2C07E03E8A40C57BBA073D85225A1D14` |
| `tests/conftest.py` | `979DC68D6D47B0016AE1F2E2765592C0914E773352FA246EFD8953617AF20614` |

Raw pytest log SHA256:
`7F873446865B51920B314544D2B8AFBEF45AB68F705B20BAAEE6E981519EA4D4`.
Pytest run-manifest SHA256:
`3FEB7200F4E6E96DD88CA29B03E2D0FF854AD899A6B5E54F3C8B488187C58948`.
Full tracked source/test Git-manifest SHA256:
`C28ACD559948595C07A57708408F529A1390A5C42C588456589521099F767D58`.

Only this report changes after the passing candidate. The release receipt
records the final report commit, unchanged tested-file hashes, clean Git state
and final numerical-process inventory. No remote CI, native/backend test,
profiling, build, wheel smoke, dependency installation or old complete suite was
run by this lane. Local B1 success does not close B2/native/composite material,
continuous-path certification, S2/G2 or H2/P6.5 integration acceptance.
