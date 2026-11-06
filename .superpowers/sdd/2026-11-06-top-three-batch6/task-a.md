# A2 author brief

Status: TEST_AUTHORING_ONLY. Parent sole execution owner; no execution assigned yet.
Parent approved plan: docs/superpowers/plans/2026-11-06-top-three-batch6.md (A2).
Reuse original robot author/tree. Start by clean-check and ordinary fast-forward to parent plan commit supplied in message; stop on WIP/divergence, never reset.

Only NEW:
- src/phymeo/runtime/_robot_robot_contacts.py
- tests/test_multi_robot_direct_contact.py
- examples/multi_robot_contact.py
- docs/benchmarks/p65-multi-robot-direct-contact.md

First author complete fixed A01-A06 test inputs/assertions and standalone public example skeleton, but do not create implementation module yet. Test first public build node must use only existing imports and fail at existing direct-pair rejection, not missing modules/fixture infrastructure. Save all original force/pose/COM/mimic oracles from plan. Existing tests cannot be rewritten. Do not run imports, collection, Python, pytest, numerics, lint, formatter, build or profile. No subagents.

Design uses actual colliders/physical owners, sphere-sphere/sphere-finite-capsule/sphere-OBB, ordinary/mimic fixed root HOST. Enabled tree pairs validate all collider combinations; excluded pair None needs no supported narrow phase. Pad replacement must not cause duplicate direct contacts. Real point velocities at common force application point and both COM moments, full J transpose, existing mimic projection. One pair sampler per unordered tree pair. Root force must be observable. Empty rigid scene is a first-class case.

Parent owns every existing file, specifically multi_robot_scene.py build/_sample/step. Event adapter unchanged unless parent identifies a necessary bug. New private adapter RobotRobotContacts(robots,pairs,model).evaluate(terms,poses) returns per-owner full generalized/load/root records and single contacts/details; author chooses small explicit immutable result and reports exact hook use before coding. Do not monkeypatch old scene from new module.

Cases: exact A01/A02 and A03 power; mimic full reaction/constraint work; 40-step physically driven window; finite capsule/OBB negatives; root/second-tree failure; coexisting rigid pair only once. Immutable prior observations/event/material addresses. Example should emit JSON with native JSON types, both real forces/joint histories, no state resets. Unsupported combinations include actual owner/shape names.

Commit tests-first locally, clean report exact source files, valid RED node name and anticipated original guard error. Wait for parent RED/implementation grant. No push/old-suite reruns/shared edits.
