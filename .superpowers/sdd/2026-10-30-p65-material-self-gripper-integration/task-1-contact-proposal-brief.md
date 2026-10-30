# B2 internal-contact accounting proposal brief

Mode: DOCUMENTATION_ONLY, original p65_bending_module in its existing clean worktree at final reportdc9be828. No execution slot. The R7 local validation and all original failures remain immutable; do not rerun any passed group.

Only write one NEW file: docs/development/p65-b2-contact-integration-proposal.md. Read the accepted parent tree by absolute path; do not merge merely to read, edit source/tests/prior reports, import/run numerical probes/lint/compile/build/install/profile, start helpers/reviewers or push. Commit the finished proposal and report clean/document-only release. Parent remains sole owner of all shared implementation seams.

## Decision to support

Prepare the next typed material/contact integration, preserving the accepted standalone B2 R7 representation and the existing contact gates. Parent has confirmed three distinct seams from current source:

1. DynamicRobotThickContactProgram currently owns/validates only DistanceConstraints or None. _thick_contact_workspace constructs MaterialPathLedger. Removing only that gate would leave the wrong typed path reader and energy semantics.
2. MaterialContactBalance currently evaluates distance elastic energy through _material_path_math.elastic, then combines actual robot frozen-M0 kinetic energy, cloth kinetic/gravity/elastic intervals and ORIGINAL external endpoint work. Internal held/terminal contact work is not an external energy credit.
3. CompositeMaterialEnergyLedger's standalone external receipts subtract held/terminal transfer work; internal_boundary rejects that external-purpose use. Composite workspace diagnostics always emit its standalone physical_audit against effective replay inputs. This report may remain a local diagnostic, but cannot substitute the original-input joint system audit or authorize a forged internal external receipt.

Read those actual modules plus CompositeMaterialPathLedger, RepresentedMaterialPathLedger, Composite/TyxComposite participant/program/diagnostic boundaries, terminal selected-pre binding, current thick-contact replay/transaction tests, and the parent PointParticipantProgram report/API. The latter now has host17/interpret4/CUDA4/boundedCPP1 local acceptance at721b5c1; do not retest it.

## Required contents

Compare two realistic narrow integration arrangements using existing public APIs: typed material/path/elastic audit dispatch in the existing parent contact boundary versus an additive typed composite-contact coordinator with explicit unchanged legacy path. Recommend one with exact ownership, call sites and file allowlist. Do not proliferate duplicate solvers or make a case-specific program.

Define material identity, representation version and schema ownership across no-material, distance-v1, composite-host and represented-native-v2. Specify original input versus effective replay input, selected-pre/full terminal binding, R/d/dpred/lambda identities, complete path/full physical-affine domain certificates, h/units/reference scales, raw/native/requested/mapped work and publication. Keep old distance schemas and v1 default/failure unchanged. No host-solver or world-state repair.

Derive the actual joint-system energy/angular equations including bending elastic intervals, conservative enclosure widths, original external forces/efforts, internal held/terminal transfers and offset-contact couples. Do not repurpose CompositeExternalImpulses as internal credit, silently treat effective inputs as original, cancel across time/domains/components, or assert arbitrary robot M0/M1 whole-step energy. Specify how standalone diagnostics can be exposed honestly without deciding the global gate.

Freeze candidate validation data by hand: at least two independent non-handoff success controls with active bending material and actual coupled contact/transfer, plus hostile controls that distinguish wrong schema/material/representation, forged external-credit acceptance, missing normal derivative/couple and changed selected-pre/terminal paths. Provide explicit coordinates, masses, material parameters/units, forces, h/windows/two rebuilds, independent exact solution or conservative derivable bound, and why both old physical and new proof budgets can be met. If a proposed case lacks a proof, mark it unresolved rather than tuning it after a probe.

Retain original contact tolerances1e-8/2e-5, energy/angular1e-8/1e-5 where that contract applies, old material limits and final h/mesh/perturbation/rebuild/fault gates. B2 standalone1e-3 cannot silently become the contact budget. Complete lawful incident fold, general robots and full folded grasp remain separate open gates, not implied by this integration.

List exact public signatures, parent-only shared paths, backward-compatible controls, affected focused host/interpret/CUDA/bounded-native selection and source/evidence/rollback plan. Do not ask the human for routine implementation permission. Return PROPOSAL_NOT_FROZEN with recommendations and genuine mathematical uncertainties for parent adjudication. No implementation or passing result is claimed from the proposal.
