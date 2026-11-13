# M-B2 articulated/deformable authoring report

Current status: **SOURCE_AND_INSTALLED_VERIFIED / VALIDATED_ACCEPTED_SLOT_RELEASED**. The [parent acceptance](#parent-acceptance) below supersedes the historical authoring status. [Fixed cases](p65-articulated-deformable-cases-20261113.md) were committed before tests and the checker; the [product contract](p65-articulated-deformable-implementation-20261113.md) is unchanged in scope. The delivery hashes and authoring statements below describe the original unexecuted handoff, not the later formatted/tested snapshot.

## Delivery

- `b2752ddc`: original C/D inputs, controls, eight-step windows and independent gates.
- `129d320f`: tests-first source matrix, referring to the standalone checker.
- `08d36f0c`: complete standalone public consumer and independent reference.
- This report is the only subsequent tracked document. No old source, tests, helper, fixture, threshold or budget was edited; no parent branch was merged.

| New file | Working SHA256 |
| --- | --- |
| `tests/installed/check_deformable_articulated.py` | `0d91ecbdd4c8702003ee8a99e4c677b81fb0d0badd7a734108e0df62ab2189a1` |
| `tests/collision/test_deformable_articulated_contact.py` | `628e18e5ccf519076147b55afb5cbd2c4c99d1861ba11196728878ae6a9f5add` |
| `docs/development/p65-articulated-deformable-cases-20261113.md` | `5d8e2d640e95e46b2e29f9884e74b6e5ec32b2f1bc32c04900f2405cebc8889d` |

The consumer includes local copies of the existing independent `resident_floating_reference.py` transform-time-jet, Jacobian, mass, gyro/bias, bounded-PD and root-COM drift calculations, and the accepted M-B1 Fraction finite-geometry/material/serialization/cleanup helpers from base `9c7433d7`. It imports no repository test/example helper at runtime. Product geometry, mass and integration do not generate expected results. The source-only mixed test retains the original M-B1 A inputs verbatim and adds independent reference metadata.

## Predeclared source nodes

This is a manual enumeration, **not collection**: ten TYX nodes, each intended for interpret and CUDA, plus five HOST nodes run once; 25 executions total.

| Group | TYX nodes | Checks |
| --- | ---: | --- |
| C / D | 2 | Each W2, eight continuing steps, nonzero full-tree velocities and a bounded-control change after four steps; root and unloaded child retained |
| Reverse registration | 1 | Two steps in each declaration order, complete keyed states/history equality |
| Mixed old/new | 1 | Original A world plus articulated D world, two actual steps |
| Lifecycle | 1 | Warm/A/restore/B, stale query, selected-world reset and actual successor |
| Late faults | 3 | Actual query, complete publication, last-world wrong body slot; both owners/banks/history retained and real retry versus independent owner |
| Complete body memory | 1 | Seven physical body rows, exact response+query native budget and one byte less |
| Zero response | 1 | Explicit None laws; real material/tree advancement and all zero body contact rows, not a claim of empty geometry query |
| HOST layout and source | 5 once | Full addresses/offsets/immutability and forbidden tree-plus-participant world; wrong world/body/link or missing original source before response allocation |

Every real returned step is journaled before checking. Independent finite geometry determines active material addresses. The checker preserves complete per-body raw/delivered/uploaded force and COM moment, root–joint M/J/bias, bounded effort, generalized projected load and mass equation, same-entry material trace and fixed reaction, finite contact work, complete source body addresses including no-collider groups, and the independent continuous node/tree trajectory. The full tree angular momentum is compared with the independent finite integration; no exact angular-conservation claim is substituted for the declared split.

Installed execution per backend contains C and D, each eight steps in two worlds, followed by the four-step lifecycle in two worlds: **40 logical worldsteps and 40 participant steps**, with complete body arrays for every tree. The 3/4-body distinction is retained in each raw record. The consumer uses a fresh absolute output directory, checks all loaded Phymeo origins against the installed root, streams `deformable-articulated.jsonl.gz`, and records every owner close. A failed close retains its owner, is not retried, and cannot produce `closed=true`; primary and cleanup exceptions are preserved together.

## Parent validation entry points

After source review and the parent's fresh serial guard, use the new test file only for the first qualification. Suggested order is HOST source/layout nodes, C and D interpret, remaining new interpret nodes, then the corresponding CUDA nodes with HOST deselected. Any needed old M-B1 compatibility is a separate parent selection; this report does not request rerunning unchanged accepted suites.

```text
python -m pytest tests/collision/test_deformable_articulated_contact.py -q -x --tb=short
python <outside-copy>/check_deformable_articulated.py --mode interpret --expected-install-root <absolute-site> --output <fresh-absolute-output>
python <outside-copy>/check_deformable_articulated.py --mode cuda --expected-install-root <absolute-site> --output <fresh-absolute-output>
```

The parent supplies the repository's existing mode flags/environment and source snapshots. No command above has been run by this author. Only textual source/API inspection, file I/O, `git diff --check` and scoped commits were performed. One initial PowerShell text-copy command had a parser error before execution; it wrote no file, was corrected, and is not a product result.

Qualification remains bounded to the original ordinary floating R/P domain, sphere/capsule normal response and short continuous windows. Multiple articulated participants in one world, RR/friction/CCD/steady support/performance remain outside this handoff. A first failure is retained and classified; no input or threshold is tuned to force acceptance.

## Parent acceptance

Tested and installed snapshot: `6b4a639c3e3126fcc6b1048b97c7a24cf18497a1`. Product changes are limited to the original four layout/source/response/lowering files. Source author `e1410564` and test author `129d320f/08d36f0c` were imported without changing the fixed cases. The only post-authoring Python changes were formatter layout; no numerical implementation repair, input change or gate change was needed.

| Actual command | Result | Scope |
| --- | --- | --- |
| 01–05 product static | Ruff pass; first format found one tuple layout, repaired and checked | Exact four product files; only the reformatted file rechecked |
| 08–12 checker static | Ruff pass; first format found two files, repaired and checked | Exact new test and standalone consumer |
| 13 new interpret | 15 passed / 44.22 s | Ten TYX plus five HOST nodes, no skip |
| 14 new CUDA | 10 passed / 50.14 s | The ten new TYX nodes; HOST not rerun |
| 06 / 07 old single-body compatibility | 8 passed / 15.76 s; 8 passed / 16.44 s | Six selected existing selectors per backend |
| 15 / 16 old M-B1 compatibility | 12 passed / 36.51 s; 11 passed / 27.42 s | Affected mechanics/source/transactions/budget; layout HOST once, unchanged raw transpose test not rerun |
| 17 / 18 / 19 wheel | Build/install/parity RAW_EXIT=0 | 534 product members source/wheel/installed byte-identical, issues=0 |
| 20 / 21 outside installed consumer | PASS, RAW_EXIT=0 both | Each 40 worldsteps /40 participant steps, C/D continuous and lifecycle |

The source qualification is **25 new executions plus 39 affected compatibility executions**; these counts describe evidence, not overall engine completion. Each installed run has C 16 worldsteps, D 16 and lifecycle 8; every constructed owner closed once successfully, retained close failures=0. Each trace contains 147 complete event records. Interpret logical bytes 7,878,646 / compressed 845,879; CUDA 7,880,022 /843,533. No package consumer imports repository tests/examples or substitutes HOST current-state physics.

Wheel: 1,727,428 bytes, SHA256 `c8cdf8d182e120a8c687e9ded18408852135ff475ab57ce0f05b5d47b4e6d642`. Standalone tested consumer SHA256 `88b09238c54e4b688c1dded8824f91193fea0dff78a6a2e3e7342c427212e030`. Fresh source staging and full parity are retained; this wheel does not imply a fresh rerun of every historical capability.

Canonical evidence: `artifacts/p65-articulated-deformable-20261113-01`, including all raw exits/logs/guards/source snapshots and the original format failures. Installed compressed evidence, exact scripts, bootstrap receipts and wheel are retained under `installed-closed-archive`; original outside files remain. No gzip payload was rewritten or dropped. Source/layout first failures were formatting only; no functional failure occurred in this batch.

This acceptance adds actual nonroot group/FK/COM/source identity, complete B×6 load delivery and original full-tree J/M projection with node/tree atomic publication. Root and no-collider children are retained. It supports the fixed R/P chain and branch matrix and the stated ordinary-tree domain; it is not MuJoCo FREE＋BALL coupling, multiple articulated participants in one world, friction, CCD, steady support, a dense cloth benchmark or Newton performance parity. The new tree cases do not add a dedicated D6-composition qualification. [Public usage](p65-articulated-deformable-usage-20261113.md) explains the shared interface.
