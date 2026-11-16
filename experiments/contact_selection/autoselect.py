"""Finite-domain research selector; never imports a physics solver or an oracle.

The qualification registry is an offline empirical record, not a proof over an
interval. Queries evaluate the chosen formula rather than replaying stored values.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass

SCHEMA = "phymeo-bounded-contact-selector-v1"
FORMULA_VERSION = "power-laws-and-hertz-v1"
PHYSICS = "static-frictionless-nonadhesive-paraboloid-linear-elastic-halfspace"
RADIUS_M = 0.02
MODULUS_PA = 1e6
DOMAIN_M = 0.16
CALIBRATION_DELTA_M = 100e-6
VALIDATED_DELTAS_M = (50e-6, 75e-6, 100e-6, 150e-6, 200e-6)
POWERS = {"linear": 1.0, "hertz_shaped": 1.5, "quadratic": 2.0}
ALGORITHMS = (*POWERS, "hertz_analytic")
PROFILES = {
    "force": ("force",),
    "constitutive": ("force", "tangent", "energy"),
    "scalars": ("force", "tangent", "energy", "area", "peak"),
}
SUPPORTS = {
    **{a: PROFILES["constitutive"] for a in POWERS},
    "hertz_analytic": PROFILES["scalars"],
}


def number(value, name, *, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name}: finite real required")
    if (value <= 0) if positive else (value < 0):
        raise ValueError(f"{name}: invalid sign")
    return float(value)


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def valid_sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class Calibration:
    force_n: float
    source_sha256: str
    delta_m: float = CALIBRATION_DELTA_M

    def __post_init__(self):
        number(self.force_n, "force_n")
        if not valid_sha(self.source_sha256) or self.delta_m != CALIBRATION_DELTA_M:
            raise ValueError("calibration source or indentation mismatch")


def predict(algorithm, delta_m, calibration, outputs=("force",)):
    """Evaluate requested scalar outputs in SI; no qualification decisions here."""
    if algorithm not in ALGORITHMS or not isinstance(calibration, Calibration):
        raise ValueError("unknown algorithm or calibration")
    d = number(delta_m, "delta_m")
    outputs = tuple(outputs)
    if (
        not outputs
        or len(set(outputs)) != len(outputs)
        or any(v not in SUPPORTS[algorithm] for v in outputs)
    ):
        raise ValueError("unsupported outputs")
    if algorithm == "hertz_analytic":
        exponent = 1.5
        coefficient = (4.0 / 3.0) * MODULUS_PA * math.sqrt(RADIUS_M)
    else:
        exponent = POWERS[algorithm]
        coefficient = calibration.force_n / calibration.delta_m**exponent
    force = coefficient * d**exponent
    values = {}
    for output in outputs:
        if output == "force":
            values[output] = force
        elif output == "tangent":
            values[output] = exponent * force / d
        elif output == "energy":
            values[output] = force * d / (exponent + 1.0)
        elif output == "area":
            values[output] = math.pi * RADIUS_M * d
        elif output == "peak":
            values[output] = 2.0 * MODULUS_PA * math.sqrt(RADIUS_M * d) / (math.pi * RADIUS_M)
    if not all(math.isfinite(v) and v > 0 for v in values.values()):
        raise ValueError("unrepresentable prediction")
    return values


def domain():
    return {
        "physics": PHYSICS,
        "radius_m": RADIUS_M,
        "modulus_pa": MODULUS_PA,
        "domain_m": DOMAIN_M,
        "validated_deltas_m": list(VALIDATED_DELTAS_M),
        "scope": "empirical qualification at five points only",
    }


def seal_registry(calibration, validation, costs, references):
    """Package offline full validation rows and timing; validate before publishing."""
    value = {
        "schema": SCHEMA,
        "formula_version": FORMULA_VERSION,
        "domain": domain(),
        "calibration": asdict(calibration),
        "references": copy.deepcopy(references),
        "cards": {},
    }
    for algorithm in ALGORITHMS:
        rows = copy.deepcopy(validation[algorithm])
        maxima = {}
        for metric in SUPPORTS[algorithm]:
            worst = max(rows, key=lambda r: r["relative_errors"][metric])
            maxima[metric] = {
                "relative_error": worst["relative_errors"][metric],
                "delta_m": worst["delta_m"],
            }
        value["cards"][algorithm] = {
            "outputs": list(SUPPORTS[algorithm]),
            "rows": rows,
            "worst": maxima,
            "costs": copy.deepcopy(costs[algorithm]),
        }
    value["binding_sha256"] = digest(value)
    Selector(value)
    return value


class Selector:
    """Load a frozen registry; selection never computes reference errors."""

    def __init__(self, registry):
        self._registry = copy.deepcopy(registry)
        r = self._registry
        try:
            binding = r.pop("binding_sha256")
            if not valid_sha(binding) or digest(r) != binding:
                raise ValueError("registry checksum mismatch")
            r["binding_sha256"] = binding
            if r["schema"] != SCHEMA or r["formula_version"] != FORMULA_VERSION:
                raise ValueError("unsupported registry version")
            if r["domain"] != domain() or set(r["cards"]) != set(ALGORITHMS):
                raise ValueError("registry domain or candidate mismatch")
            self._calibration = Calibration(**r["calibration"])
            if set(r["references"]) != {"calibration", "historical", "prospective"}:
                raise ValueError("missing reference binding")
            if not all(valid_sha(v) for v in r["references"].values()):
                raise ValueError("invalid reference binding")
            if r["references"]["calibration"] != self.calibration.source_sha256:
                raise ValueError("calibration reference mismatch")
            for algorithm in ALGORITHMS:
                card = r["cards"][algorithm]
                if card["outputs"] != list(SUPPORTS[algorithm]):
                    raise ValueError("capability mismatch")
                rows = card["rows"]
                if tuple(row["delta_m"] for row in rows) != VALIDATED_DELTAS_M:
                    raise ValueError("incomplete validation points")
                for row in rows:
                    if set(row["relative_errors"]) != set(SUPPORTS[algorithm]):
                        raise ValueError("incomplete metric validation")
                    for error in row["relative_errors"].values():
                        number(error, "validation error", positive=False)
                if set(card["worst"]) != set(SUPPORTS[algorithm]):
                    raise ValueError("incomplete worst error")
                for metric in SUPPORTS[algorithm]:
                    worst = max(rows, key=lambda row: row["relative_errors"][metric])
                    if card["worst"][metric] != {
                        "relative_error": worst["relative_errors"][metric],
                        "delta_m": worst["delta_m"],
                    }:
                        raise ValueError("worst error mismatch")
                expected_profiles = {
                    p for p, outputs in PROFILES.items() if set(outputs) <= set(SUPPORTS[algorithm])
                }
                if set(card["costs"]) != expected_profiles:
                    raise ValueError("incomplete comparable timing")
                for timing in card["costs"].values():
                    lo = number(timing["min_s"], "min_s")
                    med = number(timing["median_s"], "median_s")
                    hi = number(timing["max_s"], "max_s")
                    if (
                        not lo <= med <= hi
                        or type(timing["batches"]) is not int
                        or type(timing["calls_per_batch"]) is not int
                        or timing["batches"] != 7
                        or timing["calls_per_batch"] != 1000
                    ):
                        raise ValueError("invalid timing")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"UNQUALIFIED_EVIDENCE: {exc}") from exc

    @property
    def calibration(self):
        return self._calibration

    def select(
        self,
        delta_m,
        *,
        profile="force",
        tolerances=None,
        allowed=None,
        budget_s=None,
        physics=PHYSICS,
        radius_m=RADIUS_M,
        modulus_pa=MODULUS_PA,
        domain_m=DOMAIN_M,
    ):
        result = {
            "status": "REFUSED",
            "algorithm": None,
            "prediction": None,
            "candidates": [],
            "binding_sha256": self._registry["binding_sha256"],
            "scope": self._registry["domain"]["scope"],
            "latency_is_empirical_not_realtime_guarantee": True,
        }
        try:
            d = number(delta_m, "delta_m")
            radius = number(radius_m, "radius_m")
            modulus = number(modulus_pa, "modulus_pa")
            span = number(domain_m, "domain_m")
            if physics != PHYSICS or (radius, modulus, span) != (RADIUS_M, MODULUS_PA, DOMAIN_M):
                return {**result, "reason": "UNSUPPORTED_PHYSICS"}
            if d not in VALIDATED_DELTAS_M:
                return {**result, "reason": "OUTSIDE_VALIDATED_POINTS"}
            if profile not in PROFILES:
                return {**result, "reason": "MISSING_OUTPUT"}
            outputs = PROFILES[profile]
            limits = {m: 0.05 for m in outputs} if tolerances is None else dict(tolerances)
            if set(limits) != set(outputs):
                raise ValueError("each required output needs its own tolerance")
            limits = {m: number(v, "tolerance", positive=False) for m, v in limits.items()}
            if any(v >= 1 for v in limits.values()):
                raise ValueError("relative tolerance must be less than 1")
            if budget_s is not None:
                budget_s = number(budget_s, "budget_s", positive=False)
            permitted = tuple(ALGORITHMS) if allowed is None else tuple(allowed)
            if len(permitted) != len(set(permitted)) or any(a not in ALGORITHMS for a in permitted):
                raise ValueError("unknown or duplicate allowed algorithm")
        except (TypeError, ValueError, OverflowError) as exc:
            return {**result, "reason": "INVALID_REQUEST", "detail": str(exc)}
        eligible = []
        for algorithm in ALGORITHMS:
            card = self._registry["cards"][algorithm]
            row = {"algorithm": algorithm, "eligible": False}
            if algorithm not in permitted:
                row["reason"] = "NOT_ALLOWED"
            elif any(m not in card["outputs"] for m in outputs):
                row["reason"] = "MISSING_OUTPUT"
            else:
                failures = {
                    m: {**card["worst"][m], "requested_limit": limits[m]}
                    for m in outputs
                    if card["worst"][m]["relative_error"] > limits[m]
                }
                timing = card["costs"][profile]
                row["cost"] = timing
                row["errors"] = {m: card["worst"][m] for m in outputs}
                if failures:
                    row.update(reason="ACCURACY_GATE_FAILED", failures=failures)
                elif budget_s is not None and timing["median_s"] > budget_s:
                    row["reason"] = "COST_BUDGET_EXCEEDED"
                else:
                    row.update(reason="QUALIFIED", eligible=True)
                    eligible.append((timing["median_s"], algorithm))
            result["candidates"].append(copy.deepcopy(row))
        if not eligible:
            return {**result, "reason": "NO_ALLOWED_CANDIDATE"}
        eligible.sort()
        _, algorithm = eligible[0]
        chosen = self._registry["cards"][algorithm]["costs"][profile]
        others = [self._registry["cards"][a]["costs"][profile] for _, a in eligible[1:]]
        prediction = predict(algorithm, d, self.calibration, outputs)
        return {
            **result,
            "status": "SELECTED",
            "reason": "LOWEST_OBSERVED_MEDIAN",
            "algorithm": algorithm,
            "prediction": prediction,
            "required_outputs": list(outputs),
            "tolerances": limits,
            "cost": copy.deepcopy(chosen),
            "speed_advantage_resolved": bool(others)
            and all(chosen["max_s"] < other["min_s"] for other in others),
            "speed_note": "Only one qualified candidate"
            if not others
            else "Overlapping batch ranges do not establish a stable speed advantage",
        }
