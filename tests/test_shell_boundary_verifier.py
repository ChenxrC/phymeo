"""Independent reauthentication, complete coverage, and containment controls."""
from dataclasses import replace
from fractions import Fraction as F

import pytest
from test_shell_boundary import control

from phymeo.constraints.shell_boundary import BoundaryProofLimits
from phymeo.geometry._shell_intervals import Interval
from phymeo.geometry.shell_boundary import certify_shell_boundary
from phymeo.validation.shell_boundary import verify_shell_boundary


@pytest.mark.parametrize("mutation",("pair","same_count_pair","stratum","owner","tie",
                                     "basis","exact_point","reference","counter"))
def test_forged_receipts_cannot_supply_verifier_facts(mutation):
    ref,x,hint,_,_=control("C2")
    good=certify_shell_boundary(ref,x,witness_hint=hint)
    if mutation=="pair":
        bad=replace(good,all_pairs=good.all_pairs[:-1])
    elif mutation=="same_count_pair":
        bad=replace(good,all_pairs=(good.all_pairs[1],)+good.all_pairs[1:])
    elif mutation=="stratum":
        bad=replace(good,full_strata_and_fans=good.full_strata_and_fans[:-1])
    elif mutation=="owner":
        bad=replace(good,contacts=good.contacts[:-1])
    elif mutation=="tie":
        contact=replace(good.contacts[0],support_ties=())
        bad=replace(good,contacts=(contact,)+good.contacts[1:])
    elif mutation=="basis":
        bad=replace(good,basis=good.basis[:-1])
    elif mutation=="exact_point":
        first=good.basis[0]
        badpoint=((F(99),),)+first.exact_point[1:]
        bad=replace(good,basis=(replace(first,exact_point=badpoint),)+good.basis[1:])
    elif mutation=="reference":
        bad=replace(good,reference_id="0"*64)
    else:
        bad=replace(good,operations=0)
    assert not verify_shell_boundary(bad,ref,x,witness_hint=hint,limits=BoundaryProofLimits())


def test_verifier_does_not_call_or_trust_producer(monkeypatch):
    ref,x,hint,_,_=control("C2")
    good=certify_shell_boundary(ref,x,witness_hint=hint)
    def forbidden(*args,**kwargs):
        raise AssertionError("producer is not a verification oracle")
    monkeypatch.setattr("phymeo.geometry.shell_boundary.certify_shell_boundary",forbidden)
    assert verify_shell_boundary(good,ref,x,witness_hint=hint,limits=BoundaryProofLimits())
    shifted=x.copy()
    shifted[0,0]+=1/64
    assert not verify_shell_boundary(good,ref,shifted,witness_hint=hint,limits=BoundaryProofLimits())


def test_outward_enlargement_is_allowed_but_overlap_alone_is_not():
    ref,x,hint,_,_=control("C2")
    good=certify_shell_boundary(ref,x,witness_hint=hint)
    basis=good.basis[0]
    point=basis.point
    delta=F(1,2**46)
    intervals=tuple(Interval(v.lo-delta,v.hi+delta) for v in point.intervals)
    widened=replace(point,intervals=intervals,errors=tuple(v+delta for v in point.errors))
    candidate=replace(good,basis=(replace(basis,point=widened),)+good.basis[1:])
    assert verify_shell_boundary(candidate,ref,x,witness_hint=hint,limits=BoundaryProofLimits())
    r,y,h,_,_=control("C1")
    other=certify_shell_boundary(r,y,witness_hint=h)
    b=next(v for v in other.basis if v.vertex==(2,1))
    fake=tuple(Interval(F.from_float(float(q)),F.from_float(float(q))) for q in b.point.values)
    forged=replace(other,basis=tuple(replace(v,point=replace(v.point,intervals=fake))
                                   if v.vertex==b.vertex else v for v in other.basis))
    assert not verify_shell_boundary(forged,r,y,witness_hint=h,limits=BoundaryProofLimits())


def test_legal_ordering_of_untrusted_witness_is_not_producer_byte_oracle():
    ref,x,hint,_,_=control("C2")
    hint=replace(hint,clusters=hint.clusters[::-1],pairs=hint.pairs[::-1],
                 exterior=hint.exterior[::-1])
    certificate=certify_shell_boundary(ref,x,witness_hint=hint)
    assert verify_shell_boundary(certificate,ref,x,witness_hint=hint,limits=BoundaryProofLimits())
@pytest.mark.parametrize("mutation",("sphere","interior_links","face_disk","zero_set", "orientation",
                                     "ordinary_fan","internal_owner","coincident_ids","reset"))
def test_independent_topology_incidence_and_fan_evidence_is_not_an_oracle(mutation):
    ref,x,hint,_,_=control("C2")
    good=certify_shell_boundary(ref,x,witness_hint=hint)
    cluster=good.clusters[0]
    if mutation=="sphere":
        changed=replace(cluster,boundary_facets=(cluster.boundary_facets[1],)+cluster.boundary_facets[1:])
        bad=replace(good,clusters=(changed,)+good.clusters[1:])
    elif mutation=="interior_links":
        changed=replace(cluster,links=cluster.links[:-1])
        bad=replace(good,clusters=(changed,)+good.clusters[1:])
    elif mutation in ("face_disk","zero_set"):
        face=cluster.planes[0]
        face=replace(face,triangles=face.triangles[:-1]) if mutation=="face_disk" else replace(face,vertices=face.vertices[:-1])
        changed=replace(cluster,planes=(face,)+cluster.planes[1:])
        bad=replace(good,clusters=(changed,)+good.clusters[1:])
    elif mutation=="orientation":
        pair=good.cluster_pairs[0]
        changed=replace(pair,oriented_incidence=pair.oriented_incidence[:-1])
        bad=replace(good,cluster_pairs=(changed,))
    elif mutation=="ordinary_fan":
        # A noncontact boundary owner is still mandatory. No chosen-contact shortcut.
        index=next(i for i,f in enumerate(good.full_strata_and_fans)
                   if f.material_stratum==((0,1),))
        fans=list(good.full_strata_and_fans)
        fans[index]=replace(fans[index],occupied_cones=())
        bad=replace(good,full_strata_and_fans=tuple(fans))
    elif mutation=="internal_owner":
        index=next(i for i,f in enumerate(good.full_strata_and_fans)
                   if not f.true_boundary_facets)
        fans=list(good.full_strata_and_fans)
        fans[index]=replace(fans[index],true_boundary_facets=(good.topology.boundary_facets[0],))
        bad=replace(good,full_strata_and_fans=tuple(fans))
    elif mutation=="coincident_ids":
        contact=good.contacts[0]
        changed=replace(contact,owners=(contact.owners[0],contact.owners[0]))
        bad=replace(good,contacts=(changed,)+good.contacts[1:])
    else:
        bad=replace(good,checker_operations=0,operations=good.producer_operations)
    assert not verify_shell_boundary(bad,ref,x,witness_hint=hint,limits=BoundaryProofLimits())


def test_producer_high_level_helpers_are_unavailable_during_independent_audit(monkeypatch):
    ref,x,hint,_,_=control("C2")
    good=certify_shell_boundary(ref,x,witness_hint=hint)
    def forbidden(*args,**kwargs):
        raise AssertionError("independent engine must reconstruct its own high-level facts")
    for name in ("_inputs","_raw","_field","_reference","_cluster","_pair","_fans","_basis","_coverage"):
        monkeypatch.setattr("phymeo.geometry.shell_boundary."+name,forbidden)
    assert verify_shell_boundary(good,ref,x,witness_hint=hint,limits=BoundaryProofLimits())


def test_public_verification_receipt_owns_actual_fresh_counter_and_complete_claim():
    from dataclasses import FrozenInstanceError
    from phymeo.constraints.shell_boundary import BoundaryVerificationReceipt
    ref,x,hint,_,_=control("C2")
    good=certify_shell_boundary(ref,x,witness_hint=hint)
    first=verify_shell_boundary(good,ref,x,witness_hint=hint,limits=BoundaryProofLimits())
    assert type(first) is BoundaryVerificationReceipt and bool(first)
    assert first.verification_kind=="static"
    assert first.status=="weak_material_geometry_certified"
    assert first.precision=="ideal-from-stored-fp64"
    assert first.created_cells==0
    # On this rational static case both complete independent checks have the
    # same exact operations. The public value is not the two-pass aggregate.
    assert first.operations==good.checker_operations<good.operations
    assert 0<first.peak_fraction_bits<=4096
    assert first.original_input_binding==good.original_input_binding
    with pytest.raises(FrozenInstanceError): first.operations=0
    # Alter only the informational producer count claim. No hook, observation
    # callback or second verification just to measure a previous invocation.
    alternative=replace(good,operations=good.operations+1,checker_operations=good.checker_operations+1)
    second=verify_shell_boundary(alternative,ref,x,witness_hint=hint,limits=BoundaryProofLimits())
    assert second
    assert first.original_input_binding==second.original_input_binding
    assert first.verified_certificate_binding!=second.verified_certificate_binding
    assert first.operations==second.operations!=alternative.checker_operations
    assert verify_shell_boundary(replace(good,all_pairs=()),ref,x,witness_hint=hint,
                                 limits=BoundaryProofLimits()) is False
