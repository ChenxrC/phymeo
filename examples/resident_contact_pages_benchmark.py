"""Bounded real loaded-contact page benchmark using only public Phymeo APIs.

Examples (run from an environment containing this feature):
  python examples/resident_contact_pages_benchmark.py --mode cuda --worlds 256 \
      --steps 16 --output artifacts/pages-w256.json
  python examples/resident_contact_pages_benchmark.py --mode interpret \
      --worlds 1 32 128 --steps 3 --compare-single --output artifacts/pages-compare.json

The default is ONE bundle path at W32, not a repeated correctness suite. W256
requires pages. The optional single-page comparison uses a separate original
initial trajectory only for W<=128. There are no resets or warm-up steps.
Construction is cold per batch in this process; imports and any pre-existing
compiler cache are outside that timing, so this is not a cold-driver benchmark.
No Newton comparison or performance superiority is implied.
"""

import argparse
import hashlib
import json
import platform
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

import phymeo as ph

H = 1 / 1024
MASS = 2.0
STIFFNESS = 64.0
INITIAL_VELOCITY = 1 / 64
MOTOR = 1 / 8
EXTERNAL = 1 / 4
STATE_ATOL = 5e-6
STATE_RTOL = 5e-5
FORCE_ATOL = 1e-8
SUPPORTED_WORLDS = (1, 32, 128, 256)


def _json(value):
    """Preserve every dataclass field; identify nonfinite failure values explicitly."""
    if is_dataclass(value):
        return {item.name: _json(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported benchmark record type: {type(value).__name__}")


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _close(actual, expected, *, atol=STATE_ATOL, rtol=STATE_RTOL):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    _require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite check operand")
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _box(low, high):
    return np.asarray(tuple(product(*zip(low, high, strict=True))), dtype=float)


def _world():
    model = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.array([[0.0, 0.0, 3 / 32]]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([MASS]),
        np.zeros((1, 3)),
        np.full((1, 3), 1 / 8),
        np.zeros(3),
    )
    program = ph.DrivenRobotProgram(
        model,
        kp=np.zeros(1),
        kd=np.zeros(1),
        effort_limit=np.array([8.0]),
        delay_steps=0,
    )
    initial = program.initial_state(ph.RobotState(np.zeros(1), np.array([INITIAL_VELOCITY])))
    tree = ph.FixedRobotTree(program, initial, control_max_iterations=512)
    command = program.inputs(
        np.zeros(1),
        np.zeros(1),
        feedforward=np.array([MOTOR]),
        loads=ph.RobotInputs(np.array([EXTERNAL]), np.zeros((1, 3)), np.zeros((1, 3))),
    )
    return model, tree, command


def _page(worlds, initial):
    """Two finite original shapes per world; coincident worlds reuse local IDs."""
    shapes, bindings, materials = [], [], {}
    local = _box((-1 / 8,) * 3, (1 / 8,) * 3)
    support = _box((-1 / 2, -1 / 2, -1 / 4), (1 / 2, 1 / 2, 0))
    for world in worlds:
        moving, fixed = ph.CollisionKey(world, 100, 11), ph.CollisionKey(world, 900, 1)
        shapes.extend(
            (
                ph.CollisionShape(moving, "hull", tuple(range(101, 109))),
                ph.CollisionShape(fixed, "hull", tuple(range(201, 209))),
            )
        )
        bindings.extend(
            (
                ph.ResidentTreeBinding(moving, world, 0, local, topology=initial[world].topology),
                ph.ResidentStaticBinding(fixed, ph.CollisionMotion(support)),
            )
        )
        materials[(moving, fixed)] = ph.RobotContactMaterial(STIFFNESS, 0, 0)
    return ph.ResidentContactPageSpec(
        ph.CollisionModel(tuple(shapes)),
        tuple(bindings),
        materials,
        candidate_capacity=len(worlds),
    )


def _check_step(worlds, before, states, receipts, acceptance, tick, q, v):
    _require(dict(acceptance) == dict.fromkeys(worlds, True), "not all worlds accepted")
    _require(set(states) == set(receipts) == set(worlds), "incomplete state/receipt publication")
    maxima = dict(position_error=0.0, velocity_error=0.0, force_error=0.0, same_input_error=0.0)
    forces = []
    for world in worlds:
        state, receipt, entry = states[world], receipts[world], before[world]
        _require(state.tick == tick, "physical tick must advance exactly once")
        _require(
            receipt.world == world and receipt.entry_tick == tick - 1 and receipt.exit_tick == tick,
            "receipt world/tick mismatch",
        )
        _require(
            receipt.entry_state == entry and receipt.accepted_state == state,
            "state/force publication mismatch",
        )
        _require(receipt.dt_s == H, "receipt time step changed")
        _require(state.control == entry.control, "held control changed")
        maxima["position_error"] = max(maxima["position_error"], _close(state.position, [q]))
        maxima["velocity_error"] = max(maxima["velocity_error"], _close(state.velocity, [v]))
        # Independent finite horizontal-face law at the actual published entry pose.
        # The continuous ideal q/v check above is independent of device geometry.
        height = float(receipt.entry_geometry.link_position_m[0, 2])
        force = STIFFNESS * max(0.0, 1 / 8 - height)
        _require(force > 0 and bool(receipt.sites), "the declared loaded window lost contact")
        wrench = np.array([[0.0, 0.0, force, 0.0, 0.0, 0.0]])
        error = _close(receipt.law_wrench_fp64, wrench, atol=FORCE_ATOL, rtol=0)
        maxima["force_error"] = max(maxima["force_error"], error)
        _close(receipt.uploaded_wrench_fp32, wrench.astype(np.float32))
        _close(receipt.contact_generalized, [force])
        _close(receipt.input_external, [EXTERNAL])
        _close(receipt.selected_drive, [[0.0, 0.0, MOTOR]])
        _close(receipt.actual_drive, [MOTOR])
        _close(receipt.diagnostics["mass"], [[MASS]])
        _close(receipt.diagnostics["total_external"], [EXTERNAL + force])
        _require(len(receipt.static_reactions) == 1, "missing complete static reaction")
        reaction = next(iter(receipt.static_reactions.values()))
        _close(reaction.force_n, [0.0, 0.0, -force], atol=FORCE_ATOL, rtol=0)
        _close(reaction.torque_nm, np.zeros(3), atol=FORCE_ATOL, rtol=0)
        _require(
            all(site.pair[0].world == site.pair[1].world == world for site in receipt.sites),
            "contact crossed independent worlds",
        )
        acceleration = (force + EXTERNAL + MOTOR) / MASS
        same_v = float(entry.velocity[0]) + H * acceleration
        same_q = float(entry.position[0]) + H * same_v
        maxima["same_input_error"] = max(
            maxima["same_input_error"],
            _close(state.velocity, [same_v]),
            _close(state.position, [same_q]),
        )
        forces.append(force)
    return dict(
        checked_worlds=len(worlds),
        force_min_n=min(forces),
        force_max_n=max(forces),
        **maxima,
    )


def run_case(count, mode, steps, path, cache_root):
    """One fresh, uninterrupted physical trajectory; no warmup or reset."""
    _require(type(count) is int and count in SUPPORTED_WORLDS, "unsupported bounded world count")
    _require(type(steps) is int and 3 <= steps <= 64, "continuous steps must be in [3,64]")
    _require(mode in ("interpret", "cuda") and path in ("bundle", "single"), "invalid route")
    _require(path != "single" or count <= 128, "single-page path requires W<=128")
    worlds = tuple(1001 + 3 * index for index in range(count))
    result = dict(
        status="RUNNING",
        mode=mode,
        path=path,
        world_count=count,
        world_keys=worlds,
        requested_steps=steps,
        dt_s=H,
        resets=0,
        warmup_steps=0,
        timing_scope="in-process construction; public synchronized step calls; checks timed separately",
        cache_root=str(cache_root.resolve()),
        steps=[],
    )
    batch = plan = None
    first_step_s = steady_step_total_s = checks_total_s = 0.0
    started = perf_counter()
    try:
        model, tree, command = _world()
        batch = ph.TyxWorldBatch(
            tuple((world, tree) for world in worlds), mode, cache_root=cache_root
        )
        batch.set_controls(dict.fromkeys(worlds, command))
        initial = batch.snapshot()
        specs = tuple(_page(worlds[start : start + 128], initial) for start in range(0, count, 128))
        if path == "single":
            _require(len(specs) == 1, "single-page path cannot represent W256")
            spec = specs[0]
            plan = batch.prepare_contacts(
                spec.model,
                spec.bindings,
                spec.materials_by_pair,
                candidate_capacity=spec.candidate_capacity,
            )
        else:
            plan = batch.prepare_contact_bundle(specs)
        construction_s = perf_counter() - started
        result.update(
            model=model,
            command=command,
            initial_state=initial,
            page_worlds=tuple(
                tuple(sorted({s.key.world for s in spec.model.shapes})) for spec in specs
            ),
            page_shape_counts=tuple(len(spec.model.shapes) for spec in specs),
            page_candidate_capacities=tuple(spec.candidate_capacity for spec in specs),
            candidate_capacity=sum(spec.candidate_capacity for spec in specs),
            counters_after_construction=dict(batch=batch.stats(), contact=plan.stats()),
            timing=dict(cold_construction_in_process_s=construction_s),
        )
        q, v = 0.0, INITIAL_VELOCITY
        previous = initial
        loop_started = perf_counter()
        for tick in range(1, steps + 1):
            # Literal independent semi-implicit trajectory of the original 2 kg slider.
            force = STIFFNESS * max(0.0, 1 / 32 - q)
            v += H * (force + MOTOR + EXTERNAL) / MASS
            q += H * v
            result["last_attempt"] = dict(tick=tick, phase="physical_step")
            step_started = perf_counter()
            acceptance = batch.step_contacts(plan, H)
            step_s = perf_counter() - step_started
            result["last_attempt"].update(phase="checks", acceptance=acceptance, step_s=step_s)
            if tick == 1:
                first_step_s = step_s
            else:
                steady_step_total_s += step_s
            # Snapshot/check work is explicit and outside the step-call timer.
            check_started = perf_counter()
            counters_after_step = dict(batch=batch.stats(), contact=plan.stats())
            states = batch.snapshot()
            receipts = plan.last_accepted
            checks = _check_step(worlds, previous, states, receipts, acceptance, tick, q, v)
            check_s = perf_counter() - check_started
            checks_total_s += check_s
            result["steps"].append(
                dict(
                    tick=tick,
                    time_s=tick * H,
                    acceptance=acceptance,
                    step_s=step_s,
                    check_s=check_s,
                    reference_position=q,
                    reference_velocity=v,
                    checks=checks,
                    counters_after_step=counters_after_step,
                )
            )
            previous = states
            result["last_attempt"]["phase"] = "completed"
        result["timing"].update(
            first_step_s=first_step_s,
            steady_step_count=steps - 1,
            steady_step_total_s=steady_step_total_s,
            checks_total_s=checks_total_s,
            continuous_loop_total_s=perf_counter() - loop_started,
        )
        result.update(
            final_state=previous,
            final_receipts=plan.last_accepted,
            final_status=plan.last_status,
            counters_after_observation=dict(batch=batch.stats(), contact=plan.stats()),
            checked_steps=len(result["steps"]),
            accepted_steps=batch.stats()["completed_steps"],
            status="PASS",
        )
        _require(
            batch.stats()["completed_steps"] == steps
            and batch.stats()["accepted_world_steps"] == count * steps
            and batch.stats()["failed_steps"] == batch.stats()["rejected_world_steps"] == 0,
            "runtime counters disagree with the checked continuous trajectory",
        )
    except Exception as error:
        result.update(
            status="FAIL",
            failure=dict(
                type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
            ),
            checked_steps=len(result["steps"]),
            accepted_steps=0 if batch is None else None,
        )
        if batch is not None:
            try:
                result["counters_at_failure"] = batch.stats()
                result["accepted_steps"] = result["counters_at_failure"]["completed_steps"]
                result["retained_state"] = batch.snapshot()
            except Exception as read_error:
                result["failure_readback"] = dict(
                    type=type(read_error).__name__, message=str(read_error)
                )
        if plan is not None:
            result["retained_receipts"] = plan.last_accepted
            result["retained_status"] = plan.last_status
            result["contact_counters_at_failure"] = plan.stats()
    finally:
        for label, owner in (("contact", plan), ("batch", batch)):
            if owner is None:
                continue
            try:
                owner.close()
                counters = owner.stats()
                result[label + "_counters_after_close"] = counters
                _require(
                    counters["device_owned_bytes"] == 0,
                    label + " retained live device bytes after close",
                )
            except Exception as error:
                result["status"] = "FAIL"
                result.setdefault("cleanup_failures", []).append(
                    dict(owner=label, type=type(error).__name__, message=str(error))
                )
    return result


def _steps(value):
    number = int(value)
    if not 3 <= number <= 64:
        raise argparse.ArgumentTypeError("steps must be an integer from 3 to 64")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("interpret", "cuda"), default="cuda")
    parser.add_argument("--worlds", type=int, nargs="+", choices=SUPPORTED_WORLDS, default=[32])
    parser.add_argument("--steps", type=_steps, default=16)
    parser.add_argument("--path", choices=("bundle", "single"), default="bundle")
    parser.add_argument("--compare-single", action="store_true")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(set(args.worlds)) != len(args.worlds):
        parser.error("world counts must be unique; repeated runs require separate output files")
    if args.path == "single" and (256 in args.worlds or args.compare_single):
        parser.error("single supports W<=128; --compare-single requires --path=bundle")
    if args.output.exists():
        parser.error("output already exists; preserve prior evidence and choose a new path")
    cache_root = args.cache_root or args.output.parent / "resident-contact-pages-cache"
    report = dict(
        schema="resident-contact-pages-benchmark-v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="RUNNING",
        performance_claim="none; no Newton comparison",
        source=dict(
            example=str(Path(__file__).resolve()),
            example_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            phymeo_module=str(Path(ph.__file__).resolve()),
            python=platform.python_version(),
            numpy=np.__version__,
        ),
        options=dict(
            mode=args.mode,
            worlds=args.worlds,
            steps=args.steps,
            path=args.path,
            compare_single=args.compare_single,
        ),
        checks=dict(state_atol=STATE_ATOL, state_rtol=STATE_RTOL, force_atol=FORCE_ATOL),
        results=[],
        comparisons=[],
    )
    for count in args.worlds:
        selected = run_case(
            count, args.mode, args.steps, args.path, cache_root / f"w{count}-{args.path}"
        )
        report["results"].append(selected)
        if selected["status"] != "PASS":
            break
        if args.compare_single and count <= 128:
            single = run_case(
                count, args.mode, args.steps, "single", cache_root / f"w{count}-single"
            )
            report["results"].append(single)
            if single["status"] != "PASS":
                break
            state_same = _json(selected["final_state"]) == _json(single["final_state"])
            receipts_same = _json(selected["final_receipts"]) == _json(single["final_receipts"])
            report["comparisons"].append(
                dict(
                    world_count=count,
                    final_state_exact=state_same,
                    final_receipts_exact=receipts_same,
                    status="PASS" if state_same and receipts_same else "FAIL",
                    timing_note="serial independent initial trajectories; process/compiler caching may differ",
                )
            )
            if not (state_same and receipts_same):
                break
        elif args.compare_single:
            report["comparisons"].append(
                dict(
                    world_count=count,
                    status="NOT_APPLICABLE",
                    reason="W256 exceeds one original model page",
                )
            )
    completed = sum(result["path"] == args.path for result in report["results"])
    passed = (
        completed == len(args.worlds)
        and all(result["status"] == "PASS" for result in report["results"])
        and all(item["status"] != "FAIL" for item in report["comparisons"])
    )
    report["status"] = "PASS" if passed else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(_json(report), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    print(
        json.dumps(
            dict(
                status=report["status"],
                output=str(args.output.resolve()),
                runs=len(report["results"]),
            )
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
