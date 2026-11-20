"""Owned area-friction declarations independent of runtime and model construction.

An empty tuple preserves the existing normal-only declaration. Nonempty laws
are explicit opt-in declarations for the actual packed-point FP32 interface;
they do not reinterpret a normal-only asset bundle or certify model admission.
"""

from dataclasses import dataclass

from ..geometry.area_pressure import AreaPressureIdentity, _identity
from ..solvers.area_friction import LAW, _number

SCHEMA = "mujoco-tyx-area-friction-law-v1"
GUIDE_SCHEMA = "mujoco-tyx-area-friction-guide-v1"
LIMIT_SCHEMA = "mujoco-tyx-area-friction-limit-v1"
PRECISION = "actual-packed-point-fp32"


def solver_schema(plan):
    if getattr(plan, "coupled_damped_contacts", False) and plan.area_friction_laws:
        return "mujoco-tyx-area-friction-coupled-contact-v1"
    if getattr(plan, "scalar_damped_contacts", False) and plan.area_friction_laws:
        return "mujoco-tyx-area-friction-scalar-contact-v1"
    if plan.nlimit:
        return LIMIT_SCHEMA
    return GUIDE_SCHEMA if plan.nfriction else "mujoco-tyx-area-friction-live-v1"


def _require(condition, message):
    if not condition:
        raise ValueError("MuJoCo area-friction adapter: " + message)


@dataclass(frozen=True)
class AreaFrictionPairLaw:
    identity: AreaPressureIdentity
    mu: float
    slip_speed_m_s: float

    def __post_init__(self):
        identity = _identity(self.identity)
        mu = _number(self.mu, "mu")
        slip = _number(self.slip_speed_m_s, "slip_speed_m_s")
        _require(mu >= 0 and slip > 0, "require mu>=0 and slip_speed_m_s>0")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "slip_speed_m_s", slip)

    @property
    def law(self):
        return LAW


def _owned_laws(laws):
    _require(
        type(laws) is tuple and len(laws) <= 16, "laws must be a tuple of at most 16 declarations"
    )
    owned, seen = [], set()
    for law in laws:
        _require(type(law) is AreaFrictionPairLaw, "expected AreaFrictionPairLaw")
        item = AreaFrictionPairLaw(law.identity, law.mu, law.slip_speed_m_s)
        pair = (item.identity.compliant_body, item.identity.rigid_body)
        unordered = frozenset(pair)
        _require(unordered not in seen, "duplicate or reversed area-friction pair")
        seen.add(unordered)
        owned.append(item)
    return tuple(owned)


def capture_pair_laws(pairs, laws):
    """Capture complete ordered declarations, returning fresh immutable values.

    Empty laws leave pair admission to the existing normal-only model path.
    Nonempty laws require one declaration for every ordered body pair; complete
    material identities remain owned even though ``pairs`` contains only bodies.
    This function does not resolve body names or admit a dynamics model.
    """
    owned = _owned_laws(laws)
    if not owned:
        return ()
    _require(
        type(pairs) is tuple and len(pairs) == len(owned), "laws must cover all ordered body pairs"
    )
    for pair, law in zip(pairs, owned, strict=True):
        _require(
            type(pair) is tuple
            and len(pair) == 2
            and all(type(name) is str and 1 <= len(name) <= 128 for name in pair),
            "area pair must contain two body names",
        )
        _require(
            pair == (law.identity.compliant_body, law.identity.rigid_body),
            "law identity differs from ordered area pair",
        )
    return owned


def law_binding(laws):
    """Return a canonical owned law key, or ``None`` for the legacy path.

    Revalidation is intentional: frozen public records are not trusted caches.
    Source, model layout, and live state stamps belong to their existing owners
    and are not inferred from this law-only key.
    """
    owned = _owned_laws(laws)
    if not owned:
        return None
    return (
        LAW,
        SCHEMA,
        PRECISION,
        tuple(
            (
                item.identity.compliant_body,
                item.identity.rigid_body,
                item.identity.compliant_material,
                item.identity.rigid_material,
                item.mu,
                item.slip_speed_m_s,
            )
            for item in owned
        ),
    )
