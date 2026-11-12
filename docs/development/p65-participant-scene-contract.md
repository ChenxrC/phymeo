# Participant scene assembly v1

Status: AUTHORING_ONLY / UNEXECUTED. This is the asset-level assembly lane of [the current batch](../superpowers/plans/2026-11-12-parity-m1-assembly.md). Existing SceneDeclaration/compile_scene semantics and default routing stay unchanged.

## Public declaration and lowering

New module `phymeo.builders.participant_scene_compiler` exposes:

- `ParticipantArticulation(key, asset, initial_state, *, kp, kd, effort_limit, body_ids, collider_keys, initial_control=None)`. `key` is the existing stable `ResidentParticipantKey(world, participant)`. Asset is the original fixed/floating ordinary/affine asset. `body_ids` labels every full physical body in native body order. `collider_keys` maps every original asset CollisionKey to its installed CollisionKey; source IDs need not resemble body indices. Source keys, names, vertex IDs, revisions, graph and authored geometry remain bound.
- `ParticipantSceneDeclaration(articulations, *, static_shapes=(), static_body_ids=(), materials_by_pair=None, candidate_capacity=4096, site_capacity=128, intra_participant_pairs=False, schema='phymeo-participant-scene-v1')`. All values are owned immutable snapshots. Canonical registration order is by logical key. Missing collider mappings, duplicate installed keys, dynamic/static body aliases, foreign worlds, changed source, unsupported assets and incomplete material declarations fail explicitly.
- `compile_participant_scene(declaration, *, mode='cuda', cache_root=None)` returns `CompiledParticipantScene`. It exposes `declaration`, `identity`, `layout`, `source_map`, `world_keys`, `participant_keys`, `state_epoch`, `last_accepted`, `last_status`, and public `snapshot`, `set_controls`, `step`, `reset`, `restore(expected_epoch=...)`, `stats`, `close`, context manager. `source_map[participant][original_collision_key]` is the installed key. No caller must first construct a ResidentParticipant or binding.

Assembly reuses `ArticulationWorld` for the existing asset/state/controller admission and deep snapshot, `scene_compiler._world_spec` for the four supported systems, `_validated_control` for command admission, and the four `Resident*TreeBinding.from_asset` methods for geometry provenance. Those are pure lowering helpers, not private physical storage. Full body ownership is validated by `ResidentParticipantLayout`; its public `bind_geometry` validates the complete model and source bindings before allocation. Collider shape/body identities are not deduced from original numeric IDs. A fixed-root collider must be installed on an explicitly registered static body; moving/floating colliders must agree with their actual physical group. Fixed-root geometry uses the original asset transform, not a dummy mechanical participant.

`materials_by_pair` uses canonical installed pairs. It must cover every same-world, distinct-body, mask-enabled potential pair after source exclusions, static-static filtering and the explicit intra-participant policy. A RobotContactMaterial enables that pair; explicit None excludes it. Missing entries are errors. Original asset exclusions cannot be overridden. Static geometry uses existing FiniteStaticShape with `material=None`; pair laws are the sole authority in this new API, so a non-None static default material is rejected with a migration message. Empty geometry needs an empty table. A finite static tool or explicitly excluded pair is never silently turned into a dynamic participant.

Example literals: `(world, participant)=(91,7),(91,29),(503,7),(503,29)`; full body IDs `100*participant+local_body`; original collider `(0,42+ordinal,71+ordinal)` can install at `(world,body_ids[group],10+ordinal)`. Controls remain RobotCommand/FloatingRobotCommand. Only validated owned commands are converted to the original system.inputs; an invalid late entry cannot partly update an earlier participant.

## Transaction and supported scope

Compilation stages all static declarations, source identity, geometry and pair coverage before constructing a public ResidentSceneBatch. Any later construction failure closes that owner while retaining the primary error and cleanup error. The compiled wrapper owns the batch/contact plan and uses its public methods only. Successful step is the batch's per-world boolean mapping. A rejected world cannot partially publish participants; independent worlds may still accept as already defined. Full world checkpoints and expected_epoch CAS retain original owner/source restrictions. Restore/reset invalidate exactly the existing selected-world contact publication. The wrapper adds no post-commit checks that could turn an accepted physical mutation into an apparent rollback.

Supported: existing four resident scalar tree families; sphere/capsule/unrounded convex hull; discrete-entry HOST contact sampling and common participant response; interpret/CUDA resident mechanics. Not supported: native MuJoCo BALL conversion, new implicit cross-tree KKT, CCD, device narrow phase, cloth/soft bodies or a new solver. Old SceneDeclaration still means one articulation per world. No global default switches.

## Frozen cases before implementation

`tests/test_participant_scene_compiler.py` reuses only original ROB1 literal scenarios and independent oracle helpers, never their test runners or production FK as an expected trajectory. Every R01/R02/R04 path now starts with actual assets, named LinkCollision and controller declaration.

- R01 two massive free bodies, nonzero entry velocities, original h=1/1024 and 40 steps, W2 sparse IDs.
- R02 articulated R/P arm and eccentric free box, original 40 steps, mu=0 and 1/4, original external box force and gains.
- R04 fixed, affine, floating and floating-affine trees against a free box, original 16 steps, complete full-coordinate mechanics/reaction checks. All use W2. Same-entry independent M/J/b/load/control/force/motion checks plus independent continuous trajectory; original tolerances remain.
- Registration/world order reversal, source and material immutable ownership, original noncanonical collider IDs, actual local URDF load with fixed-root static installation, explicit excluded/no-contact geometry.
- Complete-world save/A/restore/B, selected reset and true successor; stale/foreign/incomplete checkpoint rejection; late preparation exception with no publication; one participant rejection rolls back its complete world while another advances; invalid late control preserves all held controls. Construction cleanup is tested separately.

Every physical test saves complete declarations, source mappings, checkpoints, controls, contact receipts/status, independent calculations and first failure. Structural HOST checks and physical tests are separate. No numerical validation, import, collection or style execution has happened in the author lane; the parent owns the sole execution slot. New products and tests are not yet accepted.
