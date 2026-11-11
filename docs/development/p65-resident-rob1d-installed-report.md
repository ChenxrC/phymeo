# ROB1d installed resident participant consumer

Status: **QUALIFIED_INSTALLED_INTERPRET_AND_CUDA / EXECUTION_SLOT_RELEASED**.

The two bounded public installed cases passed on both backends on 2026-11-11.
This qualifies the installed integration slice described below; ROB1c's two R03
numerical failures remain open. The author lane only read bounded metadata and
updated this report. It did not execute a project command, read the large traces,
repeat the package audit or rerun any accepted case.

## Actual installed qualification

Parent package source: 7964d1b192917821a919d8a6a065be5a6ac2bc32.
The parent reports no subsequent product-source change in this batch.

The tested consumer is the parent-formatted version at 29d74d79:
SHA256 b4d5e3ab68ad829a4c90b91dd0fb70f13b88b2fbee0e9505cba4765a0e77de3c.
Its original author version and pending commands remain in the historical
section below; that older hash is not the tested consumer hash.

Parent evidence root:
artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01.

| Actual command | Result | Meaning |
| --- | --- | --- |
| 80-build | RAW_EXIT 0 | New batch wheel built once. |
| 81-install | RAW_EXIT 0 | New wheel installed into the fresh target. |
| 82-wheel-parity | RAW_EXIT 0; BYTE_IDENTICAL; 494 products; 0 issues | Parent source, wheel members and installed product bytes agree. |
| 83-rob-installed-interpret | RAW_EXIT 1 before physics | Original environment preflight rejected the Windows sys.prefix virtualenv-root path as a checkout entry. Retained first failure. |
| 86-rob-installed-interpret-v2 | RAW_EXIT 0; PASS | Actual source-external interpret integration completed. |
| 87-rob-installed-cuda-v2 | RAW_EXIT 0; PASS | Actual source-external CUDA integration completed. |

Run 83's rejected path was
C:/Users/38347/Documents/ChatGPT/phymeo/.venv. It was not a physical rejection or
numerical failure. The parent changed only the external bootstrap: v2 removes
the explicitly known main/src entry and sys.prefix root entry, retains existing
dependency site-packages, and leaves the consumer's source-path guard and
all-module provenance checks intact. Neither product nor consumer changed for
this correction. The actual v2 bootstrap SHA256 is
d1f6cc0909dc06ccabd3e5c8d43907b98a6fff2871b935ca496251be4eda41ce.
Both final command receipts bind unchanged bootstrap/consumer before-after
hashes. Run 83 and its original stderr remain evidence rather than being
relabelled as a passing run.

Actual execution cwd and installation were outside source:
C:/Users/38347/AppData/Local/Temp/phymeo-parity-n4a-rob1d-wheel-20261111-01,
with the installation under its site/ directory. Final module provenance lists
297 loaded Phymeo modules per backend, all under that expected installation.

| Saved successful result | interpret-v2 | cuda-v2 |
| --- | ---: | ---: |
| Main-scene physical step calls | 13 | 13 |
| Accepted logical-world steps | 21 | 21 |
| Complete typed trace events | 53 | 53 |
| Logical trace bytes | 19,286,317 | 19,287,592 |
| Compressed trace bytes | 731,975 | 733,214 |
| Closed main-scene device-owned bytes (arm / mixed) | 0 / 0 | 0 / 0 |

The output summaries are in the outside-source results/rob-interpret-v2 and
results/rob-cuda-v2 directories. The full typed stream includes the separate
foreign-owner negative scene's close check. Each PASS requires the public
contact checks, full participant receipts, actual checkpoint successors,
partial reset, foreign-owner rejection and resource checks described in the
historical contract.

The count is cumulative real work, not the final restored clock: R02 makes five
calls in one world; the mixed case makes 3+2+2+1=8 calls in two worlds. Restore
and reset change selected state/ticks but do not roll back cumulative accepted
work. Thus each backend has 13 calls and 5+16=21 logical-world steps even though
the two mixed worlds finish at ticks 1 and 6.

The existing parent package-parity summary SHA256 is
0520f4d2135109677db495ffa092345b41320cc8fc22fd342433c6080df78e17.
Saved trace identities, copied from the actual successful summaries:

- interpret logical SHA256: eeb8f460596e7f1104237362ef70cf65882ca427a8546c74b62901024a64ef55;
  compressed SHA256: 7908b86a2446e3d21e8efb04a8e072bb41ec9032b4ad4951b138d7f59321e2cc.
- CUDA logical SHA256: d66134b413513a0fe31cf0ec1738c2a922a3e1a3b3a32fc9cefc355ef8147dc7;
  compressed SHA256: 6bf6885a320550f320888fe5a9c6ab7f357a5c367a8fc410fad7cfe21036e43d.

Canonical evidence copying and the batch's final index remain parent-owned;
this report update neither duplicates the archives nor independently rehashes
the already-audited package or complete saved payload.

## Accepted boundary

The actual acceptance covers the declared R02 short window and the mixed
R04/R06 installed transaction slice, using fixed, affine, floating and
floating-affine mechanics. The mixed case uses two worlds, five participants
and eleven full physical bodies per world, within the original <=33-body limit.
Collision sampling remains HOST finite patches on one page; dynamics uses the
selected TYX interpret or CUDA backend. Complete double-sided COM wrenches,
full Jacobian load projection and the original diagnostic gates remain in force.

R03's two existing numerical failures are explicitly excluded. This is not a
claim of full ROB1c acceptance, arbitrary robot manipulation, larger participant
or body/page capacities, device-only collision sampling, CCD, a cross-tree
implicit KKT solve, or Newton functional/performance parity.

## Historical authoring record (retained)

The following record predates the successful parent executions above. Its
pending commands and original author hash describe that historical checkpoint,
not the current installed qualification.

Status: **CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**.

This authoring checkpoint adds an installed-package integration consumer, not a
new dynamics implementation or a claim that ROB1c is fully qualified. The parent
owns the one serial execution slot, wheel construction, installation and actual
interpret/CUDA execution. No import, collection, numerical calculation, probe,
pytest, Ruff, formatter, build, install or profile was run in this author lane.

## Scope and source

Parent baseline: ff478eaa6ee570562b229491cb25c95e17783740.

Branch: codex/p65-rob1d-installed-20261111.

The parent approved the established wheel-consumer naming convention:
[resident_participants.py](../../tests/wheel_consumers/resident_participants.py).
The initially suggested tests/reference_newton/scale path was withdrawn: this
consumer does not execute Newton and must not imply an upstream comparison.

Exactly two new tracked files belong to this handoff: the consumer above and
this report. The old untracked mujoco-tyx-t1-review.md is untouched. No product,
packaging, shared source, fixture, solver or tolerance changed.

The fixture declarations reproduce the qualified R02 and mixed R04/R06 input
values from the [ROB contract](../superpowers/plans/2026-11-11-resident-multi-participant.md)
and the parent-corrected ROB1c cases at the baseline. They are declared in this
consumer without importing tests, examples or reference engines. The two small
windows complement the existing physical suite instead of duplicating it.

## Predeclared cases

| Case | Actual mechanics | Window and checks |
| --- | --- | --- |
| R02-installed | Ordinary two-joint Rz/Px arm and eccentric free box; nonzero initial q/v, target/rate/feedforward and box external force | Five continuous h=1/1024 steps. First common entry has M01=-9/32, 2 N contact, arm COM moment +3/16 ez and box COM moment -1/8 ez. Each step checks both dynamic owners, weighted contact law, full COM wrench, full J-transpose-wrench projection, accepted-state binding and original actuator/mechanical diagnostic gates. |
| R04-R06-installed | In each of worlds 503 and 91, fixed, affine, floating and floating-affine two-prismatic participants plus a free box; five participants and eleven full physical bodies per world | Three continuous steps, actual checkpoint, two actual successors, restore using current CAS epoch, the same two successors, then reset only world 91 and continue both worlds once. Eight physical calls / sixteen accepted logical-world steps. Source control is nonzero; both worlds occupy identical coordinates and remain isolated. |

The mixed tree uses the original two-prismatic mimic relation q1=-q0, root mass
4 kg for floating participants and free tool, stiffness 64 N/m, zero damping,
zero friction and slip regularization 1/16 m/s. The R02 free box retains its
original nonzero incoming linear/angular velocity and 1/32 N transverse external
force. No motion is prescribed and no participant is replaced by a static owner.

Four separate actual storage families must each stage entry mechanics, solve and
finalize exactly once per mixed step. The ordinary arm case uses only its two
families. The total is thirteen main-scene physical calls and twenty-one accepted
logical-world steps; the summary also binds this count to actual public counters.

Both cases retain the single-page, <=33 full physical bodies/world, complete
HOST finite-patch sampling boundary. All allowed pairs are explicitly material
bound and all other same-world distinct-body pairs explicitly excluded. This is
common-entry contact loading with one solve per original mechanical block and
one world publication, not a cross-tree implicit KKT solve, CCD or device BVH.

## Public-only observations and independent checks

The script uses only public Phymeo constructors, bindings, scene methods,
snapshots, receipts and stats. It does not import private runtime modules, tests,
CPU integrators or producer functions. Private module names are inspected only
through sys.modules to record their installed file provenance.

The independent algebra checks use actual published geometry, velocities,
full Jacobians and declared contact laws. They reconstruct each site's opposite
forces, moment about each actual COM, aggregated full body wrenches and full-D
projection. Total pair force and total moment about the world origin are checked,
without incorrectly requiring the two COM moments alone to cancel. Site weights
enter once. All uploaded body-wrench FP32 bits are compared.

The existing mechanical residual threshold is 2e-5. The normalized PD assertion
uses the actual accepted independent q/v and authored drive/kp/kd/caps, including
both pre-position-rounding and published-position evaluations, with the original
dimensionless 5e-5 gate. Raw diagnostics are saved unchanged. Affine reaction
effort is projected through the declared full-to-independent map and must vanish
in independent coordinates. This consumer does not duplicate the exhaustive
independent integration oracle from the physical tests.

The real checkpoint contains noninitial state, held controls, history and all
participant identity. Restore reuses the same scene and reproduces both actual
successors exactly. Complete receipts remain saved; the comparison excludes only
world transaction owner/epoch/bank metadata from the receipt-level physical
projection. Full participant records, diagnostic arrays, sites, entry/accepted
frames, candidate pairs and geometry identity are compared exactly. No second
trajectory is overwritten with the first.

After partial reset, the unaffected world's state and retained receipt remain
unchanged; source controls are restored only for the reset world before both
worlds continue. A separately constructed real scene must reject the original
scene's checkpoint as foreign-owned, preserving its state, bank, epoch and
published objects. This is a public owner negative, not private fault injection.
Both main scenes and the negative scene close and require device_owned_bytes=0.

## Installation provenance and saved evidence

Before importing Phymeo, the runner requires its cwd, script and fresh output
directory to be outside every detected checkout. It never changes sys.path.
Checkout source/root entries are rejected. Existing virtualenv site-packages
inside a different checkout are explicitly allowed as dependencies; arbitrary
sys.prefix paths are not treated as installed package evidence.

The expected installation root is explicit. Every actually loaded phymeo or
phymeo.* file and package path must lie inside that root. Final provenance must
include resident_scene, participant layout/contact/transaction kernels, the
shared transaction service and TyxWorldBatch. Parent package byte-parity checks
remain the authority for source-to-wheel-to-target identity; this script does not
repeat that full audit or claim that a matching directory alone proves wheel
contents. A synthetic checkout/src path exercises rejection without importing or
adding that path to sys.path.

A fresh output directory contains:

- summary.json: status, mode, consumer hash, module origins, actual counts and
  close stats, plus full original exception/traceback on failure.
- resident-participants.jsonl.gz: complete typed input declarations, inline asset
  text/hash, public controls/bindings/laws, every attempt, actual step output
  before assertions, full receipts/diagnostics, per-step checks, checkpoint,
  restore/reset/owner-negative data and resource close records.
- Literal per-world URDF assets and normal runtime caches.

Arrays retain dtype, shape and complete raw bytes; scalar floats retain binary64
bits. Mapping keys, tuple shape and dataclass type/fields are preserved. Process
owner tokens are labelled as process-local identities, not portable state or
restorable serialization. The final summary records compressed and logical
stream sizes/SHA256. No fields are deleted to reduce output. A failure saves the
original traceback and raises; it cannot become a PASS or silently skip a case.

## Pending parent-only validation

The parent should first review this source and run focused static checks on the
one new Python file. Build the batch's final wheel once, install it into the
fresh target, and perform the batch's source/wheel/installation byte check once.

Copy the consumer to a fresh directory outside all checkouts; use that directory
as cwd. Supply the actual installed target and existing dependency site-packages
through the controlled environment, with no checkout source entry. Then run the
same consumer once per backend, serially, with distinct absent output paths:

~~~text
python -B <outside>/resident_participants.py --mode interpret --expected-install-root <new-target> --output <outside>/rob1d-interpret
python -B <outside>/resident_participants.py --mode cuda --expected-install-root <new-target> --output <outside>/rob1d-cuda
~~~

These are pending commands, not executed results. No skip, smaller world count,
fixture adjustment, new tolerance or CPU fallback is authorized by this report.
If the consumer exposes an integration defect, retain the first failure and
diagnose its actual layer before selecting any rerun.

## Boundary and handoff

R03's two existing numerical failures remain open and are not included in this
qualified installed slice. Full robot manipulation, implicit joint/contact
coupling, wider participant/body/page capacity and device-only contact sampling
do not gain new qualification from this smoke. No timing or Newton performance
comparison is claimed.

Author commits: 7e71d9c0 first declares evidence/provenance and force/publication
checks; 3d1e9101 completes public declarations, scenarios and recording.

Consumer author SHA256:
c795de081f42b758efb2f1fa52db2c95838aa5b310350ec32f62847dbd6e0335.

Only text review and git diff --check were performed by this lane. The parent
receives both new files for one serial integration/build/install validation.
