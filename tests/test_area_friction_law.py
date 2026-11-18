"""HOST declaration contracts; no contact simulation or solver execution."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from phymeo.backends.area_friction_law import (
    LAW,
    PRECISION,
    SCHEMA,
    AreaFrictionPairLaw,
    capture_pair_laws,
    law_binding,
)
from phymeo.geometry.area_pressure import AreaPressureIdentity


def _law(a="block", b="plate", material_a="soft", material_b="rigid", mu=0.5, slip=0.001):
    return AreaFrictionPairLaw(AreaPressureIdentity(a, b, material_a, material_b), mu, slip)


def test_pair_law_preserves_constructor_and_frozen_owned_values():
    identity = AreaPressureIdentity("block", "plate", "soft", "rigid")
    law = AreaFrictionPairLaw(identity, np.float32(0.5), np.float64(0.001))
    assert law.identity == identity
    assert law.identity is not identity
    assert type(law.mu) is float and law.mu == 0.5
    assert type(law.slip_speed_m_s) is float and law.slip_speed_m_s == 0.001
    assert law.law == LAW == "final_normal_smooth_v1"
    with pytest.raises(FrozenInstanceError):
        law.mu = 0.25
    with pytest.raises(FrozenInstanceError):
        law.identity.compliant_material = "replacement"
    object.__setattr__(identity, "compliant_material", "caller-mutated")
    assert law.identity.compliant_material == "soft"
    assert _law(mu=0.0).mu == 0.0


def test_capture_rebuilds_every_record_and_full_identity_in_order():
    first = _law()
    second = _law("block", "wall", "soft", "stone", 0.25, 0.002)
    source = (first, second)
    owned = capture_pair_laws((("block", "plate"), ("block", "wall")), source)
    binding = law_binding(owned)
    assert type(owned) is tuple and owned == source and owned is not source
    assert all(a is not b and a.identity is not b.identity for a, b in zip(owned, source))
    assert tuple((x.identity.compliant_body, x.identity.rigid_body) for x in owned) == (
        ("block", "plate"),
        ("block", "wall"),
    )
    object.__setattr__(first, "mu", 0.75)
    object.__setattr__(first.identity, "compliant_material", "changed")
    object.__setattr__(second, "slip_speed_m_s", 0.004)
    assert owned[0].mu == 0.5
    assert owned[0].identity.compliant_material == "soft"
    assert owned[1].slip_speed_m_s == 0.002
    assert law_binding(owned) == binding
    with pytest.raises(TypeError):
        owned[0] = second


def test_empty_declaration_is_legacy_noop_but_laws_require_plain_tuple():
    class UnreadablePairs:
        def __iter__(self):
            raise AssertionError("normal-only capture must not inspect pairs")

    class TupleSubclass(tuple):
        pass

    assert capture_pair_laws(UnreadablePairs(), ()) == ()
    assert law_binding(()) is None
    for value in ([], None, {}, iter(()), TupleSubclass(), [_law()]):
        with pytest.raises(ValueError):
            capture_pair_laws((), value)
        with pytest.raises(ValueError):
            law_binding(value)


def test_constructor_rejects_bad_identity_and_nonphysical_parameters():
    identity = AreaPressureIdentity("block", "plate", "soft", "rigid")
    for mu in (-0.1, float("nan"), float("inf"), True, np.bool_(False), "0.5", None):
        with pytest.raises(ValueError):
            AreaFrictionPairLaw(identity, mu, 0.001)
    for slip in (0.0, -0.001, float("nan"), float("inf"), False, np.bool_(True), "0.001", None):
        with pytest.raises(ValueError):
            AreaFrictionPairLaw(identity, 0.5, slip)
    for invalid in (None, ("block", "plate", "soft", "rigid"), {}):
        with pytest.raises(ValueError):
            AreaFrictionPairLaw(invalid, 0.5, 0.001)
    for field, value in (
        ("compliant_material", ""),
        ("rigid_material", 3),
        ("rigid_body", "block"),
    ):
        invalid = AreaPressureIdentity("block", "plate", "soft", "rigid")
        object.__setattr__(invalid, field, value)
        with pytest.raises(ValueError):
            AreaFrictionPairLaw(invalid, 0.5, 0.001)


def test_capture_requires_complete_unique_ordered_body_pairs():
    first, second = _law(), _law("block", "wall")
    for pairs in (
        (),
        (("block", "plate"),),
        (("block", "wall"), ("block", "plate")),
        (("plate", "block"), ("block", "wall")),
        (("block", "plate"), ("other", "wall")),
        [("block", "plate"), ("block", "wall")],
        (["block", "plate"], ("block", "wall")),
        (("block", "plate", "extra"), ("block", "wall")),
        (("", "plate"), ("block", "wall")),
        (("block", 3), ("block", "wall")),
    ):
        with pytest.raises(ValueError):
            capture_pair_laws(pairs, (first, second))
    for laws in ((first, first), (first, _law("plate", "block"))):
        pairs = tuple((x.identity.compliant_body, x.identity.rigid_body) for x in laws)
        with pytest.raises(ValueError):
            capture_pair_laws(pairs, laws)
        with pytest.raises(ValueError):
            law_binding(laws)


def test_recapture_rejects_tampered_frozen_values_and_wrong_record_types():
    pairs = (("block", "plate"),)
    for field, value in (
        ("mu", float("nan")),
        ("mu", True),
        ("slip_speed_m_s", 0.0),
        ("identity", None),
    ):
        corrupted = _law()
        object.__setattr__(corrupted, field, value)
        with pytest.raises(ValueError):
            capture_pair_laws(pairs, (corrupted,))
        with pytest.raises(ValueError):
            law_binding((corrupted,))
    corrupted = _law()
    object.__setattr__(corrupted.identity, "rigid_material", "")
    with pytest.raises(ValueError):
        capture_pair_laws(pairs, (corrupted,))
    with pytest.raises(ValueError):
        law_binding((corrupted,))

    class Subclass(AreaFrictionPairLaw):
        pass

    for value in (None, {}, Subclass(_law().identity, 0.5, 0.001)):
        with pytest.raises(ValueError):
            capture_pair_laws(pairs, (value,))
        with pytest.raises(ValueError):
            law_binding((value,))


def test_binding_is_canonical_and_distinguishes_all_law_fields_and_legacy():
    original = _law()
    expected = (
        "final_normal_smooth_v1",
        "mujoco-tyx-area-friction-law-v1",
        "actual-packed-point-fp32",
        (("block", "plate", "soft", "rigid", 0.5, 0.001),),
    )
    assert SCHEMA == expected[1] and PRECISION == expected[2]
    assert law_binding((original,)) == expected
    assert law_binding((_law(),)) == expected
    assert law_binding(()) is None
    variants = (
        _law(a="block2"),
        _law(b="plate2"),
        _law(material_a="soft2"),
        _law(material_b="rigid2"),
        _law(mu=0.25),
        _law(slip=0.002),
    )
    assert all(law_binding((item,)) != expected for item in variants)
    assert law_binding((_law(mu=0.0),)) is not None
    second = _law("block", "wall")
    assert law_binding((original, second)) != law_binding((second, original))
    with pytest.raises(TypeError):
        expected[3][0][4] = 0.75


def test_sixteen_pair_boundary_and_no_partial_or_silent_truncation():
    laws = tuple(_law(f"body-{i}", f"support-{i}") for i in range(16))
    pairs = tuple((x.identity.compliant_body, x.identity.rigid_body) for x in laws)
    result = capture_pair_laws(pairs, laws)
    assert len(result) == 16
    assert len(law_binding(result)[3]) == 16
    extra = _law("body-16", "support-16")
    with pytest.raises(ValueError):
        capture_pair_laws(pairs + (("body-16", "support-16"),), laws + (extra,))
    with pytest.raises(ValueError):
        law_binding(laws + (extra,))
    object.__setattr__(laws[-1].identity, "compliant_material", "")
    with pytest.raises(ValueError):
        capture_pair_laws(pairs, laws)
    assert result[-1].identity.compliant_material == "soft"


def test_historical_export_is_same_class():
    from phymeo.backends.mujoco_area_friction import AreaFrictionPairLaw as HistoricalLaw

    assert HistoricalLaw is AreaFrictionPairLaw
