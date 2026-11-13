# M-B2 articulated/deformable authoring report

Status: **CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**. This is an unexecuted authoring handoff, not source or installation acceptance. The parent owns the sole execution slot. [Fixed cases](p65-articulated-deformable-cases-20261113.md) were committed before tests and the checker; the [product contract](p65-articulated-deformable-implementation-20261113.md) remains unchanged.

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