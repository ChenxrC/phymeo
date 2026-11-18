"""Owned historical-entry adapter to the independent area-friction solver.

The input is a trusted accepted receipt, not a live pre-integration capability.
Preparing or solving this counterfactual must never apply another force/step to
its already advanced runtime. No FK, contact solve, geometry generation, or
configuration integration occurs here. Original FP64 quadrature points and
coefficients are combined with FP64-decoded actual device dynamics; the solver
subsequently packs that complete problem to FP32. This is not a bitwise bridge
to the old type7 packed-point arithmetic, nor a new area-asset v1 contact law.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from ..builders.mujoco_compiler import MuJoCoCompilation, MuJoCoSource, _mujoco
from ..geometry.area_pressure import (
    AreaPressureQuadrature,
    _quadrature,
)
from ..solvers.area_friction import LAW, AreaFrictionProblem, AreaFrictionSample, _number
from .area_friction_law import AreaFrictionPairLaw as AreaFrictionPairLaw
from .mujoco_heterogeneous import MuJoCoTyxModelSpec, MuJoCoTyxWorldLayout

SCOPE = "historical-same-entry-counterfactual-no-integration"
PRECISION = "source-quadrature-fp64-and-decoded-device-dynamics; solver-packs-fp32"


def _require(condition, message):
    if not condition:
        raise ValueError("MuJoCo area-friction adapter: " + message)


def _array(value, shape, name):
    _require(
        isinstance(value, np.ndarray)
        and value.dtype == np.float64
        and value.shape == shape
        and np.isfinite(value).all(),
        name + " layout/finite",
    )
    return np.frombuffer(value.tobytes(order="C"), dtype=np.float64).reshape(shape)


def _same_array(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def _same_quad(left, right):
    for field in fields(AreaPressureQuadrature):
        a, b = getattr(left, field.name), getattr(right, field.name)
        _require(
            _same_array(a, b) if isinstance(a, np.ndarray) else a == b,
            "source quadrature differs: " + field.name,
        )


def _rn32(value, name):
    try:
        with np.errstate(over="raise", invalid="raise"):
            packed = np.asarray(value, dtype=np.float32)
    except (ValueError, OverflowError, FloatingPointError) as error:
        raise ValueError(name + " is not finite FP32") from error
    _require(np.isfinite(packed).all(), name + " is not finite FP32")
    return packed.astype(np.float64)


@dataclass(frozen=True, eq=False)
class PreparedAreaFrictionEntry:
    """Factory-produced owned historical problem and its complete sample catalog.

    This record is not a runtime command. Its scope forbids applying its solution
    after the source normal-only step; a future live adapter needs an earlier
    device phase and a separate integration/publication contract.
    """

    problem: AreaFrictionProblem
    quadratures: tuple[AreaPressureQuadrature, ...]
    pair_laws: tuple[AreaFrictionPairLaw, ...]
    sample_catalog: tuple[Mapping, ...]
    binding: Mapping

    @property
    def scope(self):
        return SCOPE

    @property
    def law(self):
        return LAW


class MuJoCoAreaFrictionAdapter:
    """Capture one source model's static identity; prepare historical receipts.

    Construction reconstructs a native model only to read static declarations.
    It never evaluates native FK/forces. No native model or caller-owned array is
    retained, and preparation neither reads nor changes a runtime/scene owner.
    """

    def __init__(self, model_spec, pair_laws):
        _require(type(model_spec) is MuJoCoTyxModelSpec, "expected MuJoCoTyxModelSpec")
        _require(type(model_spec.compilation) is MuJoCoCompilation, "expected source compilation")
        _require(type(model_spec.key) is str and bool(model_spec.key), "model key")
        _require(type(pair_laws) is tuple and 1 <= len(pair_laws) <= 16, "1..16 ordered pair laws")
        _require(
            type(model_spec.area_pressure_pairs) is tuple
            and len(model_spec.area_pressure_pairs) == len(pair_laws),
            "complete declared pairs required",
        )
        laws = []
        for item in pair_laws:
            _require(type(item) is AreaFrictionPairLaw, "expected AreaFrictionPairLaw")
            laws.append(AreaFrictionPairLaw(item.identity, item.mu, item.slip_speed_m_s))
        laws = tuple(laws)
        names = tuple((law.identity.compliant_body, law.identity.rigid_body) for law in laws)
        _require(names == model_spec.area_pressure_pairs, "ordered pair/body declaration differs")
        _require(type(model_spec.compilation.source) is MuJoCoSource, "expected MuJoCoSource")
        source = replace(model_spec.compilation.source)
        hashes = {}
        for data_name, digest_name in (
            ("primary_xml", "primary_sha256"),
            ("archive_bytes", "archive_sha256"),
            ("model_binary", "model_sha256"),
        ):
            raw = getattr(source, data_name)
            _require(type(raw) is bytes and bool(raw), data_name + " must own bytes")
            digest = sha256(raw).hexdigest()
            _require(
                type(getattr(source, digest_name)) is str
                and digest == getattr(source, digest_name),
                digest_name + " does not bind actual source bytes",
            )
            hashes[digest_name] = digest
        _require(
            type(source.mujoco_version) is str and bool(source.mujoco_version), "MuJoCo version"
        )
        model = replace(model_spec.compilation, source=source).fresh_native_model()
        _require(
            model.neq == 0 and model.ntendon == 0,
            "equality/tendon models are outside the existing TYX domain",
        )
        sizes = {name: int(getattr(model, name)) for name in ("nq", "nv", "nu", "na", "nbody")}
        _require(
            1 <= sizes["nv"] <= 6
            and 1 <= sizes["nq"] <= 45
            and 1 <= sizes["nbody"] <= 128
            and model.ngeom <= 128,
            "model capacity",
        )
        mj = _mujoco()
        _require(
            model.na == 0 and not np.any(model.jnt_type == mj.mjtJoint.mjJNT_BALL),
            "activation and BALL models are outside the area entry domain",
        )
        _require(
            not np.any(model.jnt_limited) and not np.any(model.dof_frictionloss),
            "joint limits/dry friction are not part of this counterfactual",
        )
        body_names = tuple(model.body(i).name for i in range(model.nbody))
        ids, seen = [], set()
        for a, b in names:
            _require(body_names.count(a) == body_names.count(b) == 1, "ambiguous/absent body name")
            ba, bb = body_names.index(a), body_names.index(b)
            pair = tuple(sorted((ba, bb)))
            _require(
                ba != bb and pair not in seen and model.body_weldid[ba] != model.body_weldid[bb],
                "duplicate/self/welded pair",
            )
            seen.add(pair)
            ids.append((ba, bb))
        from .mujoco_tyx._constraint_model import constraint_pairs

        _require(
            not any(
                tuple(sorted(map(int, model.geom_bodyid[list(pair)]))) not in seen
                for pair in constraint_pairs(model)
            ),
            "remaining native contact pair",
        )
        source_h = _number(model.opt.timestep, "source timestep")
        _require(
            source_h > 0 and float(_rn32(source_h, "source timestep")) > 0,
            "positive source timestep",
        )
        damping0 = np.asarray(model.dof_damping, dtype=np.float64)
        _require(np.isfinite(damping0).all() and np.all(damping0 >= 0), "joint damping")
        damping = _rn32(damping0, "joint damping")
        _require(not np.any((damping0 > 0) & (damping <= 0)), "positive damping underflows FP32")
        damping = _array(damping, (sizes["nv"],), "joint damping")
        self._laws = laws
        self._body_ids = tuple(ids)
        self._binding = MappingProxyType(
            dict(
                model_key=model_spec.key,
                **hashes,
                **sizes,
                mujoco_version=source.mujoco_version,
                body_names=body_names,
                body_ids=tuple(ids),
                damping=damping,
                source_h_s=source_h,
                area_pressure_pairs=names,
                law=LAW,
                scope=SCOPE,
            )
        )

    @property
    def binding(self):
        return MappingProxyType(
            dict(
                self._binding,
                damping=_array(self._binding["damping"], (self._binding["nv"],), "joint damping"),
            )
        )

    @property
    def pair_laws(self):
        return tuple(
            AreaFrictionPairLaw(item.identity, item.mu, item.slip_speed_m_s) for item in self._laws
        )

    @property
    def law(self):
        return LAW

    @property
    def scope(self):
        return SCOPE

    def prepare_from_receipt(self, receipt, *, layout, quadratures):
        """Build an owned FP64 problem from one trusted historical entry only.

        The opaque source configuration_id is retained, not recomputed against a
        restored/new owner. Old solved forces are never input to the new problem.
        """
        from ._mujoco_tyx_runtime import MuJoCoTyxDynamics, MuJoCoTyxStepReceipt

        _require(type(layout) is MuJoCoTyxWorldLayout, "expected MuJoCoTyxWorldLayout")
        model = self._binding
        for name in ("nq", "nv", "nu", "na", "nbody", "bucket", "row"):
            _require(
                type(getattr(layout, name)) is int and getattr(layout, name) >= 0,
                "layout integer: " + name,
            )
        for name in ("model_key", "primary_sha256", "archive_sha256", "model_sha256"):
            _require(type(getattr(layout, name)) is str, "layout text: " + name)
        for name in (
            "model_key",
            "primary_sha256",
            "archive_sha256",
            "model_sha256",
            "nq",
            "nv",
            "nu",
            "na",
            "nbody",
        ):
            _require(getattr(layout, name) == model[name], "layout/source mismatch: " + name)
        _require(
            type(layout.world) is int and 0 <= layout.world <= 2147483647,
            "world key must be nonnegative i32",
        )
        _require(isinstance(receipt, MuJoCoTyxStepReceipt), "expected TYX step receipt")
        _require(receipt.world == layout.world and type(receipt.world) is int, "receipt world")
        _require(type(receipt.tick) is int and 1 <= receipt.tick <= 2147483647, "receipt tick")
        _require(
            receipt.backend in ("mujoco-tyx-cuda", "mujoco-tyx-interpret")
            and receipt.state_precision == "fp32-device-integration",
            "device receipt required",
        )
        _require(receipt.contact_phase == "tyx-euler-input-constraint-evaluation", "contact phase")
        time = _number(receipt.input_time, "input_time")
        output_time = _number(receipt.output_time, "output_time")
        _require(
            time >= 0
            and output_time > time
            and _number(receipt.contact_time, "contact_time") == time,
            "entry clock/phase",
        )
        requested_h = _number(receipt.h, "h")
        _require(
            requested_h == model["source_h_s"], "receipt h differs from compiled source timestep"
        )
        h = float(_rn32(requested_h, "h"))
        _require(requested_h > 0 and h > 0, "positive integration h")
        _require(
            type(receipt.contacts) is tuple and not receipt.contacts,
            "non-area contacts are unsupported",
        )
        _require(
            type(receipt.constraints) is tuple
            and len(receipt.constraints) <= 128
            and all(
                type(getattr(row, "constraint_type", None)) is int and row.constraint_type == 7
                for row in receipt.constraints
            ),
            "non-area constraints are unsupported",
        )
        _require(
            type(quadratures) is tuple and len(quadratures) == len(self._laws),
            "every declared pair requires a quadrature, including empty",
        )
        total = 0
        for quad in quadratures:
            _require(
                type(quad) is AreaPressureQuadrature and type(quad.sample_ids) is tuple,
                "expected complete AreaPressureQuadrature",
            )
            total += len(quad.sample_ids)
            _require(total <= 128, "full quadrature exceeds 128 samples; no truncation")
        quads = tuple(_quadrature(quad) for quad in quadratures)
        raw_fields = receipt.raw_contact_fields
        _require(isinstance(raw_fields, Mapping), "receipt raw fields")
        _require(raw_fields.get("response_operator") == "M+h*joint_damping", "response operator")
        raw_pairs = raw_fields.get("area_pressure")
        _require(type(raw_pairs) is tuple and len(raw_pairs) == len(quads), "receipt pair count")
        counts = raw_fields.get("counts")
        _require(
            isinstance(counts, np.ndarray)
            and counts.shape == (3,)
            and counts.dtype.kind in "iu"
            and counts.tolist() == [0, 0, total],
            "native counts or incomplete area rows",
        )
        _require(
            len(receipt.constraints) == total
            and all(
                getattr(row, "source_id", None) == i for i, row in enumerate(receipt.constraints)
            ),
            "area row/sample order",
        )
        dyn = receipt.dynamics
        _require(
            type(dyn) is MuJoCoTyxDynamics
            and dyn.phase == "tyx-euler-input-dynamics"
            and _number(dyn.time, "dynamics time") == time,
            "same-entry dynamics required",
        )
        n, b, nq = model["nv"], model["nbody"], model["nq"]
        shapes = dict(
            body_position=(b, 3),
            body_rotation=(b, 3, 3),
            body_com=(b, 3),
            jacobian_com=(b, 6, n),
            mass_matrix=(n, n),
            bias=(n,),
            passive=(n,),
            applied=(n,),
            smooth_acceleration=(n,),
            input_q=(nq,),
            input_v=(n,),
        )
        data = {name: _array(getattr(dyn, name), shape, name) for name, shape in shapes.items()}
        _require(
            _same_array(data["input_q"], _array(receipt.input_q, (nq,), "receipt q"))
            and _same_array(data["input_v"], _array(receipt.input_v, (n,), "receipt v")),
            "dynamics q/v differ from receipt entry",
        )
        samples, catalog = [], []
        stamp = quads[0].stamp
        _require(
            stamp.world_id == str(layout.world)
            and stamp.step_index == receipt.tick - 1
            and stamp.sample_time_s == time,
            "historical quadrature stamp",
        )
        for pair_index, (quad, raw, law, ids) in enumerate(
            zip(quads, raw_pairs, self._laws, self._body_ids, strict=True)
        ):
            _require(
                quad.identity == law.identity and quad.stamp == stamp, "mixed pair identity/stamp"
            )
            _require(
                isinstance(raw, Mapping)
                and type(raw.get("source_quadrature")) is AreaPressureQuadrature,
                "receipt original source quadrature missing",
            )
            original_quad = raw["source_quadrature"]
            _require(
                type(original_quad.sample_ids) is tuple
                and len(original_quad.sample_ids) == len(quad.sample_ids),
                "source quadrature sample extent",
            )
            source_quad = _quadrature(original_quad)
            _same_quad(quad, source_quad)
            _require(
                raw.get("identity") == quad.identity
                and raw.get("stamp") == quad.stamp
                and raw.get("body_ids") == ids
                and raw.get("sample_ids") == quad.sample_ids
                and raw.get("source_ids") == quad.source_ids,
                "receipt pair/sample binding",
            )
            _require(
                _same_array(quad.dA, _array(raw.get("dA"), quad.dA.shape, "receipt dA")),
                "receipt/source dA differs",
            )
            ba, bb = ids
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    k = quad.dA * quad.gradient
                    phi = -quad.pressure / quad.gradient
                    for index, point in enumerate(quad.points):
                        ja = (
                            data["jacobian_com"][ba, :3]
                            + np.cross(
                                data["jacobian_com"][ba, 3:].T, point - data["body_com"][ba]
                            ).T
                        )
                        jb = (
                            data["jacobian_com"][bb, :3]
                            + np.cross(
                                data["jacobian_com"][bb, 3:].T, point - data["body_com"][bb]
                            ).T
                        )
                        samples.append(
                            AreaFrictionSample(
                                point,
                                quad.normals[index],
                                ja,
                                jb,
                                k[index],
                                phi[index],
                                law.mu,
                                law.slip_speed_m_s,
                            )
                        )
                        catalog.append(
                            MappingProxyType(
                                dict(
                                    pair_index=pair_index,
                                    identity=quad.identity,
                                    stamp=quad.stamp,
                                    sample_id=quad.sample_ids[index],
                                    source_id=quad.source_ids[index],
                                    body_ids=ids,
                                    point_world=tuple(map(float, point)),
                                    normal_world=tuple(map(float, quad.normals[index])),
                                    dA=float(quad.dA[index]),
                                    pressure_pa=float(quad.pressure[index]),
                                    gradient_pa_m=float(quad.gradient[index]),
                                    mu=law.mu,
                                    slip_speed_m_s=law.slip_speed_m_s,
                                )
                            )
                        )
            except (FloatingPointError, OverflowError) as error:
                raise ValueError("nonfinite quadrature/point-Jacobian algebra") from error
        # Passive already includes -D*v0. Cancel it because the solver's A
        # contains the same implicit +h*D once; never add old constraint forces.
        try:
            with np.errstate(over="raise", invalid="raise"):
                force = (
                    data["applied"]
                    + data["passive"]
                    - data["bias"]
                    + model["damping"] * data["input_v"]
                )
        except (FloatingPointError, OverflowError) as error:
            raise ValueError("nonfinite external-force assembly") from error
        problem = AreaFrictionProblem(
            data["mass_matrix"], model["damping"], data["input_v"], force, h, tuple(samples)
        )
        binding = MappingProxyType(
            dict(
                self.binding,
                world=layout.world,
                input_tick=receipt.tick - 1,
                output_tick=receipt.tick,
                input_time=time,
                output_time=output_time,
                requested_h_s=requested_h,
                h_s=h,
                configuration_id=stamp.configuration_id,
                input_q=data["input_q"],
                input_v=data["input_v"],
                dynamics=MappingProxyType(data),
                precision=PRECISION,
            )
        )
        return PreparedAreaFrictionEntry(problem, quads, self.pair_laws, tuple(catalog), binding)
