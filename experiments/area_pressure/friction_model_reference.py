"""Fixed friction-law comparison: independent, decoupled normal/tangent DOFs.

NOT TYX and not a general contact solver. There is one lumped elastic normal
coordinate and one translational tangent coordinate, each with the same scalar
mass. Contact geometry is not generated here. The normal law is backward Euler;
all three tangential laws share its final normal force, but Lagged uses the
INPUT elastic force to set its friction scale. That candidate can exert friction
on a separating step whose final normal force is zero. Negative friction power
alone therefore does not certify the final-normal Coulomb bound.

This experiment has four fixed inputs, two step sizes and three laws. It does
not run a trajectory, optimize parameters, change type7, or extend asset laws.
"""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCHEMA = "phymeo-friction-model-reference-v1"
QUALIFICATION = "DECOUPLED_2DOF_FRICTION_MODEL_REFERENCE"
MAX_BISECTION_STEPS = 80
RESIDUAL_TOLERANCE_M_S = 1e-13
POWER_TOLERANCE_W = 1e-14
STEPS = (0.001, 0.0005)
LAWS = ("coulomb-final-normal", "smooth-final-normal", "smooth-lagged-input-normal")
# name, gap, normal velocity, tangent velocity, external normal/tangent forces.
CASES = (
    ("support-slide", -0.001, 0.0, 0.02, -4.0, 0.0),
    ("support-start", -0.001, 0.0, 0.0, -4.0, 1.0),
    ("impact", -0.0001, -0.5, 0.02, 0.0, 0.0),
    ("release", -0.0001, 0.3, 0.02, 0.0, 0.0),
)
RESEARCH_SOURCES = (
    "https://arxiv.org/html/2312.03908v3",
    "https://drake.mit.edu/doxygen_cxx/group__friction__model.html",
    "https://drake.mit.edu/pydrake/pydrake.multibody.plant.html#pydrake.multibody.plant.DiscreteContactApproximation",
    "https://mujoco.readthedocs.io/en/stable/computation/index.html#friction-cones",
)


def _require(condition, message):
    if not condition:
        raise ValueError("friction model reference: " + message)


@dataclass(frozen=True)
class FrictionInput:
    """One preregistered input; physical parameters have no tuning arguments."""

    case: str
    h_s: float
    mass_kg: float = field(default=0.5, init=False)
    normal_stiffness_n_m: float = field(default=4000.0, init=False)
    mu: float = field(default=0.5, init=False)
    slip_speed_m_s: float = field(default=0.001, init=False)
    gap_m: float = field(init=False)
    normal_velocity_m_s: float = field(init=False)
    tangent_velocity_m_s: float = field(init=False)
    external_normal_force_n: float = field(init=False)
    external_tangent_force_n: float = field(init=False)

    def __post_init__(self):
        _require(type(self.case) is str and self.case in tuple(row[0] for row in CASES), "case")
        _require(
            type(self.h_s) in (int, float) and math.isfinite(self.h_s) and self.h_s in STEPS,
            "step size must be one of the two fixed finite values",
        )
        object.__setattr__(self, "h_s", float(self.h_s))
        values = next(row[1:] for row in CASES if row[0] == self.case)
        names = (
            "gap_m",
            "normal_velocity_m_s",
            "tangent_velocity_m_s",
            "external_normal_force_n",
            "external_tangent_force_n",
        )
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FrictionModelResult:
    input: FrictionInput
    law: str
    normal_free_velocity_m_s: float
    tangent_free_velocity_m_s: float
    normal_input_force_n: float
    normal_force_n: float
    friction_scale_force_n: float
    tangent_force_n: float
    normal_velocity_m_s: float
    tangent_velocity_m_s: float
    gap_m: float
    tangent_displacement_m: float
    normal_impulse_ns: float
    tangent_impulse_ns: float
    external_normal_impulse_ns: float
    external_tangent_impulse_ns: float
    iterations: int
    final_bracket_m_s: tuple[float, float]
    solver_residual_m_s: float
    normal_momentum_residual_ns: float
    tangent_momentum_residual_ns: float
    normal_law_residual_n: float
    normal_power_w: float
    tangent_power_w: float
    external_power_w: float
    kinetic_normal_entry_j: float
    kinetic_tangent_entry_j: float
    kinetic_normal_exit_j: float
    kinetic_tangent_exit_j: float
    kinetic_entry_j: float
    kinetic_exit_j: float
    spring_entry_j: float
    spring_exit_j: float
    normal_work_j: float
    tangent_work_j: float
    external_work_j: float
    mechanical_energy_change_j: float
    be_kinetic_numerical_loss_j: float
    be_kinetic_balance_residual_j: float
    normal_spring_balance_j: float
    mechanical_balance_j: float
    instantaneous_coulomb_excess_n: float
    qualification: str = QUALIFICATION


def _owned_input(sample):
    _require(type(sample) is FrictionInput, "expected FrictionInput")
    owned = FrictionInput(sample.case, sample.h_s)
    # Frozen public objects can still be deliberately mutated; do not let that
    # silently change this fixed experimental design or alias a caller object.
    for item in fields(owned):
        value, expected = getattr(sample, item.name), getattr(owned, item.name)
        _require(type(value) is type(expected) and value == expected, "modified input " + item.name)
    return owned


def _smooth_velocity(free, scale, sample):
    coefficient = sample.h_s * sample.mu * scale / sample.mass_kg
    if coefficient == 0.0 or free == 0.0:
        return free, 0, (free, free)
    lo, hi = min(0.0, free), max(0.0, free)
    residual = math.inf
    for iteration in range(1, MAX_BISECTION_STEPS + 1):
        speed = lo + 0.5 * (hi - lo)
        residual = speed - free + coefficient * speed / math.hypot(speed, sample.slip_speed_m_s)
        if not math.isfinite(residual):
            raise RuntimeError("nonfinite smooth bisection residual")
        if abs(residual) <= RESIDUAL_TOLERANCE_M_S:
            return speed, iteration, (lo, hi)
        if speed == lo or speed == hi:
            break
        if residual < 0.0:
            lo = speed
        else:
            hi = speed
    raise RuntimeError(f"smooth bisection residual did not converge: {residual!r}")


def _finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _finite_tree(child)
    elif type(value) is float:
        _require(math.isfinite(value), "nonfinite result")


def solve_friction_case(sample, law):
    """One simultaneous step of the fixed diagonal two-coordinate model.

    Exact Coulomb refers ONLY to its one-dimensional tangential maximum-
    dissipation problem with this analytically known final normal force.
    Smooth laws have finite slip; no exact static-lock claim is made for them.
    """
    a = _owned_input(sample)
    _require(type(law) is str and law in LAWS, "unknown law")
    h, m, k = a.h_s, a.mass_kg, a.normal_stiffness_n_m
    free_n = a.normal_velocity_m_s + h * a.external_normal_force_n / m
    free_t = a.tangent_velocity_m_s + h * a.external_tangent_force_n / m
    input_force = max(0.0, -k * a.gap_m)
    normal = max(0.0, -k * (a.gap_m + h * free_n) / (1.0 + h * h * k / m))
    vn = free_n + h * normal / m
    scale = input_force if law == LAWS[2] else normal
    if law == LAWS[0]:
        reduction = h * a.mu * normal / m
        if abs(free_t) <= reduction:
            vt = 0.0
            tangent = -m * free_t / h
        else:
            vt = free_t - math.copysign(reduction, free_t)
            tangent = -math.copysign(a.mu * normal, free_t)
        iterations, bracket = 0, (vt, vt)
    else:
        vt, iterations, bracket = _smooth_velocity(free_t, scale, a)
        tangent = -a.mu * scale * vt / math.hypot(vt, a.slip_speed_m_s)
    gap = a.gap_m + h * vn
    dn, dt = vn - a.normal_velocity_m_s, vt - a.tangent_velocity_m_s
    kn0, kt0 = 0.5 * m * a.normal_velocity_m_s**2, 0.5 * m * a.tangent_velocity_m_s**2
    kn1, kt1 = 0.5 * m * vn**2, 0.5 * m * vt**2
    u0, u1 = 0.5 * k * min(a.gap_m, 0.0) ** 2, 0.5 * k * min(gap, 0.0) ** 2
    pn, pt = normal * vn, tangent * vt
    pe = a.external_normal_force_n * vn + a.external_tangent_force_n * vt
    wn, wt, we = h * pn, h * pt, h * pe
    kinetic_change = (kn1 + kt1) - (kn0 + kt0)
    energy_change = kinetic_change + (u1 - u0)
    be_loss = 0.5 * m * (dn * dn + dt * dt)
    residual = abs(vt - free_t - h * tangent / m)
    result = FrictionModelResult(
        input=a,
        law=law,
        normal_free_velocity_m_s=free_n,
        tangent_free_velocity_m_s=free_t,
        normal_input_force_n=input_force,
        normal_force_n=normal,
        friction_scale_force_n=scale,
        tangent_force_n=tangent,
        normal_velocity_m_s=vn,
        tangent_velocity_m_s=vt,
        gap_m=gap,
        tangent_displacement_m=h * vt,
        normal_impulse_ns=h * normal,
        tangent_impulse_ns=h * tangent,
        external_normal_impulse_ns=h * a.external_normal_force_n,
        external_tangent_impulse_ns=h * a.external_tangent_force_n,
        iterations=iterations,
        final_bracket_m_s=bracket,
        solver_residual_m_s=residual,
        normal_momentum_residual_ns=m * dn - h * (a.external_normal_force_n + normal),
        tangent_momentum_residual_ns=m * dt - h * (a.external_tangent_force_n + tangent),
        normal_law_residual_n=abs(normal - max(0.0, -k * gap)),
        normal_power_w=pn,
        tangent_power_w=pt,
        external_power_w=pe,
        kinetic_normal_entry_j=kn0,
        kinetic_tangent_entry_j=kt0,
        kinetic_normal_exit_j=kn1,
        kinetic_tangent_exit_j=kt1,
        kinetic_entry_j=kn0 + kt0,
        kinetic_exit_j=kn1 + kt1,
        spring_entry_j=u0,
        spring_exit_j=u1,
        normal_work_j=wn,
        tangent_work_j=wt,
        external_work_j=we,
        mechanical_energy_change_j=energy_change,
        be_kinetic_numerical_loss_j=be_loss,
        be_kinetic_balance_residual_j=kinetic_change - we - wn - wt + be_loss,
        normal_spring_balance_j=u1 - u0 + wn,
        mechanical_balance_j=energy_change - we - wt,
        instantaneous_coulomb_excess_n=max(0.0, abs(tangent) - a.mu * normal),
    )
    _finite_tree(asdict(result))
    _require(residual <= RESIDUAL_TOLERANCE_M_S, "accepted tangent residual")
    _require(pt <= POWER_TOLERANCE_W, "friction direction/power")
    _require(abs(tangent) <= a.mu * scale + 1e-14, "declared friction scale bound")
    _require(normal >= 0.0 and result.normal_law_residual_n <= 1e-12, "normal law")
    return result


def _fixed_requests():
    return tuple((FrictionInput(row[0], h), law) for row in CASES for h in STEPS for law in LAWS)


def _check_distinctions(rows):
    _require(len(rows) == 24, "fixed comparison count")
    lookup = {(r.input.case, r.input.h_s, r.law): r for r in rows}
    _require(len(lookup) == 24, "fixed comparison identities")
    for h in STEPS:
        for law in LAWS:
            expected_impact = 50.0 / 21.0 if h == 0.001 else 700.0 / 501.0
            for case, expected in (
                ("support-slide", 4.0),
                ("support-start", 4.0),
                ("impact", expected_impact),
                ("release", 0.0),
            ):
                _require(
                    abs(lookup[case, h, law].normal_force_n - expected) <= 1e-12,
                    "hand normal oracle",
                )
        _require(
            lookup["support-start", h, LAWS[0]].tangent_velocity_m_s == 0.0, "Coulomb static branch"
        )
        for law in LAWS[1:]:
            _require(
                lookup["support-start", h, law].tangent_velocity_m_s > 0.0, "smooth finite slip"
            )
        for law in LAWS[:2]:
            _require(lookup["release", h, law].tangent_force_n == 0.0, "final-normal release")
        release = lookup["release", h, LAWS[2]]
        _require(
            abs(release.tangent_force_n) > 0.19 and release.tangent_power_w < 0.0,
            "Lagged release distinction",
        )


def compare_models():
    rows = tuple(solve_friction_case(sample, law) for sample, law in _fixed_requests())
    _check_distinctions(rows)
    return rows


def _markdown(rows):
    lines = [
        "# 固定摩擦模型对照：24 个独立单步结果",
        "",
        "**NOT TYX；仅解耦 2DOF 独立参考，不是通用接触求解器。**",
        "没有连续轨迹、几何生成或参数调优；type7 与资产 law 均未改变。",
        "",
        "所有力与出步速度来自同一次求解。功按 BE 的 h·F·v_next 计算。",
        "normal_force 是最终法向力；normal_input_force 是入步弹性力。",
        "Lagged 冻结后者作切向幅值，不能宣称满足最终法向力的 Coulomb 上界。",
        "",
        "| 输入 | h(ms) | 模型 | Fn(N) | Ft(N) | vt_next(mm/s) | 最终 Coulomb 超额(N) | 摩擦功率(W) |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.input.case} | {r.input.h_s * 1000:g} | {r.law} | {r.normal_force_n:.9g} | "
            f"{r.tangent_force_n:.9g} | {r.tangent_velocity_m_s * 1000:.9g} | "
            f"{r.instantaneous_coulomb_excess_n:.9g} | {r.tangent_power_w:.9g} |"
        )
    lines.extend(
        [
            "",
            "## 如何解释",
            "",
            "- 稳定支撑时法向力都是 4N；平滑与精确 Coulomb 的差来自正则化。",
            "- 低载启动时精确一维 Coulomb 静止，平滑模型有非零滑动，不能称精确静摩擦锁止。",
            "- 冲击中最终法向力高于入步力，Lagged 切向幅值偏小；同最终 Fn 的平滑对照分离了正则化与滞后效应。",
            "- 分离步最终 Fn=0，但 Lagged 仍有大于 0.19N 的切向力。这是已声明模型区别，不是 TYX 产品失败。",
            "- Lagged 此时依然耗散切向动能。被动性不等于满足最终 Fn 的 Coulomb 上界。",
            "- BE 数值耗散与物理摩擦功分开保存；normal_spring_balance 和 mechanical_balance 不是严格守恒误差。",
            "",
            "## 停止条件与下一步",
            "",
            "非有限、超过 80 次二分、真实速度残差超过 1e-13m/s、摩擦功率超过 1e-14W，或固定区别检查不符即停止并保存已有前缀。",
            "若产品要求最终 Fn 的 Coulomb 上界，Lagged 候选不合格。若考虑其凸近似优势，须先显式冻结 law、观察 ABI 与强瞬态限制；不直接替换 type7。",
            "",
            "## 原始研究来源",
            "",
        ]
    )
    lines.extend(f"- {url}" for url in RESEARCH_SOURCES)
    return "\n".join(lines) + "\n"


def write_comparison(output):
    """Write a fresh absolute directory; failures retain only completed rows."""
    path = Path(output)
    _require(path.is_absolute() and not path.exists(), "output must be a fresh absolute directory")
    path.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": SCHEMA,
        "status": "started",
        "scope": "decoupled-2dof-independent-reference",
        "qualification": QUALIFICATION,
        "is_tyx_runtime": False,
        "is_general_contact_solver": False,
        "force_phase": "same-step implicit exit velocity; Lagged scale from input elastic force",
        "work_convention": "backward Euler: h * force * exit_velocity",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "research_sources": RESEARCH_SOURCES,
        "requested_records": 24,
        "bisection_limit": MAX_BISECTION_STEPS,
        "residual_tolerance_m_s": RESIDUAL_TOLERANCE_M_S,
        "friction_power_tolerance_w": POWER_TOLERANCE_W,
        "records": [],
    }
    rows = []
    current = None
    try:
        for sample, law in _fixed_requests():
            current = {"case": sample.case, "h_s": sample.h_s, "law": law}
            result = solve_friction_case(sample, law)
            rows.append(result)
            payload["records"].append(asdict(result))
        _check_distinctions(tuple(rows))
        payload["status"] = "complete"
    except Exception as error:
        payload["status"] = "failed"
        payload["failure"] = {
            **(current or {}),
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        _finite_tree(payload)
        (path / "comparison.json").write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (path / "comparison.md").write_text(_markdown(tuple(rows)), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="fresh absolute directory")
    args = parser.parse_args()
    write_comparison(args.output)


if __name__ == "__main__":
    main()
