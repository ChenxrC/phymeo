"""G2a scale consumer; copy beside the exact pinned G1 consumer outside the repo.

No product geometry/dynamics oracle is imported. The sibling carries the
already qualified native/analytic reference and original family gates.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

REFERENCE_NAME = "check_mujoco_tyx_heterogeneous.py"
REFERENCE_SHA = "379a2139257398a2dec25f74a474b59c417618bbc7cfa5f15b0795cccabaf8cb"
H = 0.001
CELL_NAMES = ("w1-scalar", "w1-ball", "w1-free", "w8-b3", "w16-b3", "w16-b8")
KEYS = (41, 7, 305, 11, 203, 59, 997, 2, 88, 509, 17, 701, 109, 31, 809, 53)
FAMILIES = ("scalar", "ball", "free")
B8_FAMILIES = ("scalar", "ball", "free", "scalar", "ball", "free", "scalar", "ball")
TIMING_FIELDS = (
    "source_compile",
    "construction",
    "action",
    "physics",
    "observation",
    "reference",
    "validation",
    "save",
    "boundary",
    "cleanup",
)
MEMORY = dict(
    live_device_bytes="UNKNOWN",
    peak_device_bytes="UNKNOWN",
    reason="No independent live/peak device-memory observer is installed; "
    "allocation/upload counters retain cumulative semantics.",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_reference():
    path = Path(__file__).with_name(REFERENCE_NAME)
    require(path.is_file(), "copy the original reference consumer beside this file")
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA,
        "original reference consumer bytes differ",
    )
    spec = importlib.util.spec_from_file_location("g2a_pinned_native_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cell_layout(cell):
    require(cell in CELL_NAMES, "unknown predeclared scale cell")
    count = 1 if cell.startswith("w1-") else (8 if cell == "w8-b3" else 16)
    result = []
    occurrences = dict.fromkeys(FAMILIES, 0)
    for i, key in enumerate(KEYS[:count]):
        if cell == "w16-b8":
            source_index = i % 8
            family = B8_FAMILIES[source_index]
            model_key, slot = f"source-{source_index}-{family}", i // 8
        else:
            family = cell[3:] if count == 1 else FAMILIES[i % 3]
            model_key, slot = family, occurrences[family] % 2
            occurrences[family] += 1
        result.append(dict(world=key, model_key=model_key, family=family, slot=slot))
    return tuple(result)


def named_source(original, identity):
    # Change exactly one documentary name, not XML formatting or any physics.
    pattern = r'(<mujoco\b[^>]*\bmodel=")([^"]*)(")'
    match = re.search(pattern, original)
    if match is None:
        require("<mujoco>" in original, "expected original anonymous root tag")
        result = original.replace("<mujoco>", '<mujoco model="' + identity + '">', 1)
        restored = result.replace('<mujoco model="' + identity + '">', "<mujoco>", 1)
        count = 1
    else:
        result, count = re.subn(
            pattern, lambda m: m.group(1) + identity + m.group(3), original, count=1
        )
        restored = re.sub(
            pattern, lambda m: m.group(1) + match.group(2) + m.group(3), result, count=1
        )
    require(count == 1 and restored == original, "only original root model name may change")
    return result


def declarations(cell, helper):
    import mujoco as mj

    from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxModelSpec, MuJoCoTyxWorldSpec
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    require(mj.__version__ == "3.3.7", "fixed native MuJoCo 3.3.7")
    layout = cell_layout(cell)
    originals = dict(
        scalar=helper.scalar_source("coupled", early=True),
        ball=helper.source("r4-pyramid"),
        free=helper.free_source("mixed"),
    )
    source_families = {}
    for row in layout:
        source_families.setdefault(row["model_key"], row["family"])
    xml, native, compiled = {}, {}, {}
    for key, family in source_families.items():
        xml[key] = (
            named_source(originals[family], "g2a-" + key) if cell == "w16-b8" else originals[family]
        )
        native[key] = mj.MjModel.from_xml_string(xml[key])
        compiled[key] = compile_mujoco_xml(xml[key], require_asset=False)
        require(float(native[key].opt.timestep) == H, "unchanged source timestep")
    specs = tuple(MuJoCoTyxModelSpec(key, compiled[key]) for key in source_families)
    starts, world_models = {}, {}
    for row in layout:
        key, family, slot = row["world"], row["family"], row["slot"]
        model = native[row["model_key"]]
        world_models[key] = model
        if family == "scalar":
            starts[key] = MuJoCoInitialState(
                model.qpos0 + 0.06,
                np.full(model.nv, -0.035),
                act=np.full(model.na, 0.34 + slot * 0.01),
                time=0.125,
            )
        elif family == "ball":
            starts[key] = helper.initial(mj, model, "r4-pyramid", slot)
        else:
            starts[key] = helper.free_initial(mj, model, slot)
    worlds = tuple(
        MuJoCoTyxWorldSpec(row["world"], row["model_key"], starts[row["world"]]) for row in layout
    )
    sources = {}
    for key, compilation in compiled.items():
        source = compilation.source
        sources[key] = dict(
            family=source_families[key],
            xml=xml[key],
            original_xml=originals[source_families[key]],
            primary_sha256=hashlib.sha256(source.primary_xml).hexdigest(),
            archive_sha256=hashlib.sha256(source.archive_bytes).hexdigest(),
            model_sha256=hashlib.sha256(source.model_binary).hexdigest(),
            dimensions=tuple(
                int(getattr(native[key], field)) for field in ("nq", "nv", "nu", "na", "nbody")
            ),
        )
    if cell == "w16-b8":
        for field in ("primary_sha256", "archive_sha256", "model_sha256"):
            require(
                len({record[field] for record in sources.values()}) == 8,
                "eight independent compiled source identities: " + field,
            )
    return dict(
        mj=mj, layout=layout, models=world_models, specs=specs, worlds=worlds, sources=sources
    )


class Session:
    """One bounded cell, with prefix-safe streaming and measured host phases."""

    def __init__(self, mode, cell, output, installed_root=None):
        self.mode, self.cell = mode, cell
        self.output = Path(output)
        require(self.output.is_absolute() and not self.output.exists(), "fresh absolute output")
        self.installed_root = None if installed_root is None else Path(installed_root)
        self.helper = load_reference()
        self.batch, self.journal, self.result = None, None, None
        self.started = time.perf_counter()
        self.timings = dict.fromkeys(TIMING_FIELDS, 0.0)
        self.progress = dict(
            attempted_calls=0, physics_returned_calls=0, checked_calls=0, accepted_world_steps=0
        )
        self.seen_contact, self.seen_motor, self.seen_control = {}, {}, {}

    @contextmanager
    def timed(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] += time.perf_counter() - started

    def emit(self, event, **payload):
        with self.timed("save"):
            self.journal.emit(event, **payload)

    def __enter__(self):
        self.output.mkdir(parents=True)
        self.journal = self.helper.Journal(self.output / "trajectory.jsonl.gz")
        try:
            if self.installed_root is not None:
                require(self.installed_root.is_absolute(), "absolute installed root")
                provenance = self.helper.provenance(self.installed_root, preflight=True)
                self.emit("preflight", provenance=provenance)
            self.emit(
                "begin",
                cell=self.cell,
                mode=self.mode,
                reference_file=REFERENCE_NAME,
                reference_sha256=REFERENCE_SHA,
                consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                memory=MEMORY,
                h=H,
                continuous_steps=24,
            )
            with self.timed("source_compile"):
                declaration = declarations(self.cell, self.helper)
            self.mj = declaration["mj"]
            self.models = declaration["models"]
            self.layout = declaration["layout"]
            self.by_world = {row["world"]: row for row in self.layout}
            self.keys = tuple(self.by_world)
            self.bucket_count = len(declaration["specs"])
            self.emit(
                "declaration",
                layout=self.layout,
                sources=declaration["sources"],
                worlds=declaration["worlds"],
                gates=dict(
                    ball=self.helper.TOL,
                    free=self.helper.free_TOL,
                    scalar=self.helper.SCALAR_TOL,
                    ACT=self.helper.ACT,
                    FORCE=self.helper.FORCE,
                    CONTACT_FORCE=self.helper.CONTACT_FORCE,
                    free_Q_ATOL=self.helper.free_Q_ATOL,
                    free_V_ATOL=self.helper.free_V_ATOL,
                ),
            )
            from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch

            with self.timed("construction"):
                self.batch = MuJoCoTyxHeterogeneousBatch(
                    declaration["specs"], declaration["worlds"], mode=self.mode
                )
            with self.timed("observation"):
                self.original = self.batch.integration_snapshot()
            require(self.batch.world_keys == self.keys, "complete sparse declaration order")
            require(
                self.batch.capabilities["bucket_count"] == self.bucket_count,
                "actual exact-source bucket count",
            )
            memberships = {}
            for key in self.keys:
                row = self.batch.layout[key]
                requested = self.by_world[key]
                require(row.model_key == requested["model_key"], "source model identity")
                for field in ("primary_sha256", "archive_sha256", "model_sha256"):
                    require(
                        getattr(row, field) == declaration["sources"][row.model_key][field],
                        "published exact source binding",
                    )
                memberships.setdefault(row.bucket, []).append(key)
            require(len(memberships) == self.bucket_count, "all buckets have real rows")
            if self.cell == "w16-b8":
                require(
                    all(len(rows) == 2 for rows in memberships.values()),
                    "exactly two real worlds in every one of eight buckets",
                )
            self.refs = {}
            with self.timed("reference"):
                for key in self.keys:
                    self.refs[key] = self.helper.references(
                        self.mj, self.models[key], {key: self.original[key]}
                    )[key]
                self.initial_refs = {key: copy.copy(data) for key, data in self.refs.items()}
            self.seen_contact = dict.fromkeys(self.keys, False)
            self.seen_motor = dict.fromkeys(self.keys, False)
            self.seen_control = dict.fromkeys(self.keys, False)
            self.emit(
                "binding",
                layout=self.batch.layout,
                capabilities=self.batch.capabilities,
                initial=self.original,
                memberships=memberships,
                counters=self.batch.counters,
            )
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def commands(self, tick):
        funcs = dict(
            scalar=self.helper.scalar_control,
            ball=self.helper.control,
            free=self.helper.free_control,
        )
        return {
            key: funcs[self.by_world[key]["family"]](
                self.models[key], tick, self.by_world[key]["slot"]
            )
            for key in self.keys
        }

    @contextmanager
    def forbid_native_step(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("CPU MuJoCo physics used during actual TYX propagation")

        with ExitStack() as stack:
            for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward"):
                stack.enter_context(patch.object(self.mj, name, forbidden))
            yield

    def advance(self, tick, label):
        from phymeo.backends.mujoco_runtime import MuJoCoControl

        with self.timed("observation"):
            entry, full_entry = self.batch.integration_snapshot(), self.batch.snapshot()
        packets = self.commands(tick)
        self.emit(
            "attempt",
            label=label,
            entry=entry,
            full_entry=full_entry,
            requested=packets,
            revision=self.batch.checkpoint_revision,
            h=H,
        )
        expected = {}
        with self.timed("reference"):
            for key in self.keys:
                family, model = self.by_world[key]["family"], self.models[key]
                packet = packets[key]
                actual = MuJoCoControl(
                    np.asarray(packet.ctrl, np.float32),
                    qfrc_applied=np.asarray(packet.qfrc_applied, np.float32),
                    xfrc_applied=np.asarray(packet.xfrc_applied, np.float32),
                )
                oracle = (
                    self.helper.free_native_fields
                    if family == "free"
                    else self.helper.native_fields
                )
                expected[key] = oracle(self.mj, model, entry[key], actual)
                if family == "scalar":
                    expected[key]["scalar_analytic"] = self.helper.scalar_analytic(
                        model, entry[key], actual
                    )
                self.helper.set_control(self.refs[key], actual)
                self.mj.mj_step(model, self.refs[key])
                expected[key]["continuous"] = self.helper.native_state(self.refs[key])
        self.emit("reference", label=label, native=expected)
        with self.timed("action"):
            self.batch.set_controls(packets)
        self.progress["attempted_calls"] += 1
        with self.forbid_native_step():
            with self.timed("physics"):
                accepted = self.batch.step(H)
            self.progress["physics_returned_calls"] += 1
            self.progress["accepted_world_steps"] += sum(bool(value) for value in accepted.values())
            self.emit(
                "physics-return", label=label, accepted=accepted, progress=dict(self.progress)
            )
            with self.timed("observation"):
                output = self.batch.integration_snapshot()
            self.emit(
                "integration-return", label=label, output=output, receipt=self.batch.last_step
            )
        with self.timed("observation"):
            full, counters = self.batch.snapshot(), self.batch.counters
        self.emit("observation", label=label, full_output=full, counters=counters)
        with self.timed("validation"):
            require(
                tuple(output) == self.keys and dict(accepted) == dict.fromkeys(self.keys, True),
                "all worlds accepted",
            )
            require(
                tuple(self.batch.last_step.receipts) == self.keys, "complete receipt membership"
            )
            for key in self.keys:
                row, before, after = output[key].last_step, entry[key], output[key]
                require(
                    row.world == key and row.tick == after.tick == before.tick + 1,
                    "identity and one real physics tick",
                )
                require(
                    row.input_time == before.time and row.output_time == after.time and row.h == H,
                    "complete clock binding",
                )
                self.helper.exact(row.input_q, before.q, "same-entry q")
                self.helper.exact(row.input_v, before.v, "same-entry v")
                self.helper.exact(row.control, packets[key], "requested control")
                for field in ("ctrl", "qfrc_applied", "xfrc_applied"):
                    np.testing.assert_array_equal(
                        getattr(row.stored_control, field),
                        np.asarray(getattr(packets[key], field), np.float32),
                    )
                self.helper.exact(self.batch.last_step.receipts[key], row, "complete receipt")
                self.helper.exact(
                    self.batch.last_step.states[key], after, "same force/motion state"
                )
                for field in (
                    "world",
                    "q",
                    "v",
                    "act",
                    "ctrl",
                    "time",
                    "tick",
                    "qfrc_applied",
                    "xfrc_applied",
                    "held_control",
                    "last_step",
                ):
                    self.helper.exact(
                        getattr(full[key], field),
                        getattr(after, field),
                        "full/integration observation " + field,
                    )
                family = self.by_world[key]["family"]
                if family == "ball":
                    self.helper.compare(
                        row, before, after, packets[key], expected[key], self.models[key]
                    )
                elif family == "free":
                    self.helper.free_compare(row, before, after, expected[key], self.models[key])
                else:
                    self.helper.scalar_compare(row, before, after, expected[key])
                self.seen_contact[key] |= bool(np.any(row.qfrc_constraint != 0))
                self.seen_motor[key] |= bool(np.any(row.qfrc_actuator != 0))
                self.seen_control[key] |= bool(np.any(row.stored_control.ctrl != 0))
        self.progress["checked_calls"] += 1
        self.emit("checked", label=label, progress=dict(self.progress))
        return output

    def boundaries(self):
        from phymeo.backends.mujoco_heterogeneous_checkpoint import MuJoCoTyxHeterogeneousCheckpoint

        with self.timed("boundary"):
            saved = self.batch.checkpoint()
            raw = saved.to_bytes()
        with self.timed("save"):
            (self.output / "checkpoint.bin").write_bytes(raw)
        refs = {key: copy.copy(value) for key, value in self.refs.items()}
        first = self.advance(24, "checkpoint-successor-A")
        self.emit(
            "boundary-attempt",
            operation="restore",
            expected_revision=self.batch.checkpoint_revision,
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
        )
        with self.timed("boundary"):
            restored = self.batch.restore(
                MuJoCoTyxHeterogeneousCheckpoint.from_bytes(raw),
                expected_revision=self.batch.checkpoint_revision,
            )
        self.emit("boundary-return", operation="restore", receipt=restored)
        self.refs = {key: copy.copy(value) for key, value in refs.items()}
        repeated = self.advance(24, "checkpoint-successor-B")
        self.helper.exact(repeated, first, "complete restored successor")
        selected, families = [], set()
        for row in self.layout:
            if row["family"] not in families:
                families.add(row["family"])
                selected.append(row["world"])
        with self.timed("observation"):
            before = self.batch.integration_snapshot()
        self.emit("boundary-attempt", operation="reset", selected=tuple(selected), before=before)
        with self.timed("boundary"):
            reset = self.batch.reset(tuple(selected))
        self.emit("boundary-return", operation="reset", receipt=reset)
        with self.timed("observation"):
            after = self.batch.integration_snapshot()
        self.emit("boundary-observation", operation="reset", selected=tuple(selected), after=after)
        for key in self.keys:
            if key in selected:
                require(
                    after[key].tick == 0 and after[key].last_step is None, "selected original reset"
                )
                for field in ("q", "v", "act", "time"):
                    self.helper.exact(
                        getattr(after[key], field),
                        getattr(self.original[key], field),
                        "initial reset " + field,
                    )
                self.refs[key] = copy.copy(self.initial_refs[key])
            else:
                self.helper.exact(after[key], before[key], "all unselected fields/history retained")
        self.advance(25, "partial-reset-real-successor")

    def mark_pass(self, **extra):
        self.result = dict(
            status="PASS",
            cell=self.cell,
            mode=self.mode,
            bucket_count=self.bucket_count,
            world_count=len(self.keys),
            progress=dict(self.progress),
            memory=dict(MEMORY),
            **extra,
        )
        self.emit(
            "complete",
            result=self.result,
            counters=self.batch.counters,
            layout=self.batch.layout,
            capabilities=self.batch.capabilities,
        )

    def __exit__(self, exc_type, exc, tb):
        primary = exc
        errors = []
        if primary is not None:
            try:
                self.emit(
                    "failure",
                    error=repr(primary),
                    traceback="".join(traceback.format_exception(exc_type, exc, tb)),
                    progress=dict(self.progress),
                )
            except BaseException as error:
                errors.append(error)
        if self.batch is not None:
            try:
                with self.timed("cleanup"):
                    self.batch.close()
                self.emit("close", status="CLOSED")
            except BaseException as error:
                errors.append(error)
        summary = dict(
            self.result
            or dict(
                status="FAIL",
                cell=self.cell,
                mode=self.mode,
                progress=dict(self.progress),
                memory=dict(MEMORY),
            )
        )
        if primary is not None or errors:
            summary.update(
                status="FAIL", error=repr(primary), cleanup_errors=[repr(e) for e in errors]
            )
        try:
            if self.installed_root is not None:
                summary["provenance"] = self.helper.provenance(self.installed_root)
            else:
                summary["source_only_module_origins"] = self.helper.modules()
        except BaseException as error:
            errors.append(error)
            summary.update(status="FAIL", provenance_error=repr(error))
        if self.journal is not None:
            try:
                with self.timed("save"):
                    summary["trace"] = self.journal.close()
            except BaseException as error:
                errors.append(error)
                summary.update(status="FAIL", trace_error=repr(error))
        summary["host_stage_seconds"] = dict(self.timings)
        summary["wall_seconds"] = time.perf_counter() - self.started
        self.summary = summary
        try:
            (self.output / "summary.json").write_text(
                json.dumps(summary, allow_nan=False, indent=2), encoding="utf-8"
            )
        except BaseException as error:
            errors.append(error)
        if errors:
            raise BaseExceptionGroup(
                "scale execution/evidence/cleanup failed",
                ([primary] if primary is not None else []) + errors,
            ) from None
        return False


def run_cell(mode, cell, output, installed_root=None):
    with Session(mode, cell, output, installed_root) as run:
        for tick in range(24):
            run.advance(tick, f"continuous-{tick:02d}")
        for key in run.keys:
            require(
                run.seen_control[key] and run.seen_motor[key],
                "every world has real nonzero control and actuator force",
            )
            if run.by_world[key]["family"] == "ball":
                require(run.seen_contact[key], "every BALL world has actual contact load")
        require(run.batch.counters["boundary_clones"] == 0, "ordinary steps never clone")
        run.boundaries()
        require(
            run.progress
            == dict(
                attempted_calls=27,
                physics_returned_calls=27,
                checked_calls=27,
                accepted_world_steps=27 * len(run.keys),
            ),
            "complete cell counts, no truncation",
        )
        run.mark_pass()
    return run.summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--cell", choices=CELL_NAMES, required=True)
    parser.add_argument("--installed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.installed_root.is_absolute() and args.output.is_absolute(), "absolute paths required"
    )
    result = run_cell(args.mode, args.cell, args.output, args.installed_root)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
