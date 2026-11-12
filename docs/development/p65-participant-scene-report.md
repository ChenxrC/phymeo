# Asset-level participant scene assembly: author handoff

Status: **CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**. No product import, collection, test, numerical probe, lint, formatting, build, install or native execution occurred in this lane. `git diff --check` is only a whitespace check, not functional evidence. Parent owns all serial validation and public exports.

The [frozen contract](p65-participant-scene-contract.md) is implemented by a new opt-in module. An application now declares original assets, independent initial states, bounded controllers, full-body IDs, original-to-installed collider keys and complete pair materials. It does not construct ResidentParticipant or family bindings by hand. Existing one-articulation SceneDeclaration and compile_scene remain unchanged.

## Exact new files and reuse

1. `src/phymeo/builders/participant_scene_compiler.py`: ParticipantArticulation, ParticipantSceneDeclaration, CompiledParticipantScene, compile_participant_scene.
2. `tests/test_participant_scene_compiler.py`: structural and original multi-case physical contracts.
3. `tests/installed/check_participant_scene.py`: standalone public installed consumer.
4. `docs/development/p65-participant-scene-contract.md`: pre-execution declaration/identity/transaction contract.
5. This report.

No old product, test, export or solver file changed. The base is 48724ae1d8f2be527a62d4b86cf0a62a1e3d410a on codex/p65-scene-assembly-20261112. Tests-first commit cd19a197 preceded implementation caf67e01. e50cc260 added source restriction checks and the installed consumer. eaf3d089 addressed the independent cleanup review below.

The new compiler reuses ArticulationWorld and `_world_spec`/`_validated_control` in the existing scene compiler; all four original from_asset binding methods; ResidentParticipantLayout.bind_geometry; ResidentSceneBatch.prepare_contacts/set_controls/step_contacts/snapshot/restore/reset/close. Runtime step does no HOST FK or integration added by this layer. Fixed-root source geometry is evaluated once during static lowering through the original asset's collision_motion. Original fixed/floating coordinate conventions and affine graph are unchanged.

Complete source metadata, names, original collider/vertex IDs, revision, body installation, graph, initial state/control and material table enter a typed deterministic identity. The identity encoder is digest-only, with no deserialization, pickle or eval. Original infinite metadata limits remain explicit; NaN is rejected. Arrays are immutable bytes-backed snapshots from the existing admission. Initial commands and all static source/geometry/material/capacity checks complete before allocating the batch. Dynamic controls are all validated and converted before the single public set_controls call.

The wrapper adds no post-commit validation or duplicate physical state. Complete world restore and selected reset keep the original batch owner and expected_epoch rules. One rejected participant rejects its full world; another world can still advance. No implicit cross-tree KKT is introduced.

## Review correction and cleanup ownership

Independent source review found that an exception group alone was not an explicit recovery handle when both construction and public owner close failed. The final code preserves the identical primary and cleanup exception objects and attaches `resident_owner` (a strong reference to the original public ResidentSceneBatch) and `cleanup_state='unresolved'` to the BaseExceptionGroup. The test proves the owner is reachable, actual release has not happened, and close was attempted exactly once. This is a diagnostic ownership handle, not permission to blindly retry unknown destruction. Only a separately established safe cleanup state permits caller-directed handling. There is no new retry loop or change to the old ResidentSceneBatch cleanup implementation.

The standalone consumer also preserves the primary physical failure if capture fails, and attempts provenance, trace close and final summary independently without replacing an earlier exception. All finalization failures remain in the reported exception group.

## Frozen test matrix (manual count, not collected)

**38 nodes for one complete file invocation:** 19 structural HOST nodes and 19 marked TYX nodes. To avoid rerunning the same HOST subset, the parent can run the HOST subset once and the TYX subset once per interpret/CUDA mode. This is 57 planned node executions, not 57 passes.

- Seven W2 asset-level physics nodes: R01 40 steps; R02 40 steps at mu=0 and 1/4; R04 16 steps for fixed, affine, floating and floating_affine. Literal old scenarios and independent reference helpers are reused, never their old test runners. All start from named original Asset/LinkCollision and high-level commands. Independent same-entry complete M/J/b/external/motor/contact wrench/response and independent continuous trajectory use original tolerances.
- Reversed participant/world registration, complete-world save/A/restore/B and partial reset with real successor, four stale/foreign/incomplete restore cases, three actual late preparation/participant failure cases, invalid late control atomicity, explicit None versus empty geometry, and a locally loaded URDF with explicit root-static installation.
- HOST ownership/immutability and source metadata checks; six malformed installation cases; eight invalid scene/material/capacity cases; immutable source mask/exclusion enforcement; normal constructor cleanup and unresolved double failure.

The core complete-window tests save complete source declarations, identity maps, input controls, raw checkpoints, status/contact receipts, independent oracle results and first failure. Transaction cases retain their actual rejected attempts, untouched publications and successful retries. Expected validation exceptions are not described as physical passes.

Suggested parent commands, **not executed by the author**:

```text
python -m pytest tests/test_participant_scene_compiler.py -m "not tyx" -q
python -m pytest tests/test_participant_scene_compiler.py -m tyx --tyx-mode interpret -q
python -m pytest tests/test_participant_scene_compiler.py -m tyx --tyx-mode cuda -q
```

The installed consumer is copied outside the checkout and run with `--mode interpret|cuda --expected-install-root <absolute site-packages> --output <fresh absolute directory>`. It constructs two original massive free-body assets in sparse worlds 91/503, original R01 masses/radii/stiffness/h, nonzero entry velocity and existing external force channel values 0 or +/-1/32 N. Twelve real batch steps (24 world steps) cover controls, save/A/restore/B and selected reset; all physical statuses must accept. The independent spring/mass recurrence checks force, torque, M/J/b, q/v and continuous motion. Full gzip JSONL, checksum, module origins and final summary are saved. CPU integrator sentinels guard the explicit installed path. It does not import source tests/examples or claim a new solver/performance result.

## Scope and pending validation

This assembly supports only the existing four resident scalar tree families and sphere/capsule/unrounded convex hull contact domain with explicit HOST discrete-entry sampling. Existing geometry/physical resource ceilings and tolerances are unchanged. Static defaults must be moved into the pair table; missing pair declarations fail, explicit None excludes, and source masks/exclusions cannot be overridden.

R03 is excluded from the positive matrix because its earlier independent acceleration gates remain unresolved. This is not MuJoCo-TYX BALL import, new floating joint support, global CCD, device contact sampling, general soft-body assembly, performance qualification or all of P6.5 completion. Actual source validation, installed package qualification and acceptance remain entirely pending.

## Author byte manifest (before parent style or validation)

| Path | SHA256 | Bytes |
| --- | --- | ---: |
| docs/development/p65-participant-scene-contract.md | a7956ca7132e9330f2ee51c9fcfc95ad6edf3fd8147ce0b457f7494aa33576aa | 6909 |
| src/phymeo/builders/participant_scene_compiler.py | 57d84bc0e96aba2207a0a6f7886980ffed6b0cfcaedb8420bb5adef749f1c86a | 24505 |
| tests/test_participant_scene_compiler.py | 27fcada440d9efd7ac0d14d165b9e5f3b9fbdfcdabb75c000eeea3a039bc16b8 | 30272 |
| tests/installed/check_participant_scene.py | 19ace6ed45698d91c2825692a59dbaaae7aca7014e20bdaaf5dcbb7518a71433 | 16273 |

This report's own hash is handed off separately after its final write to avoid a recursive manifest.
