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