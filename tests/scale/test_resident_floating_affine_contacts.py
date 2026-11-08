"""C11 real finite mu=0 contact and full floating-affine feedback, unexecuted."""

from dataclasses import replace
from itertools import combinations, product

import numpy as np
import pytest
from test_resident_affine_contacts import _audit_sites
from test_resident_floating_affine_step import (
    ATOL,
    WINDOW_ATOL,
    WORLD,
    _audit,
    _close,
    _compiled,
    _item,
    _json,
    _record,
    _reference,
    _save,
    _selected,
)

import phymeo as ph


def _box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), float)


def _geometry(batch, item, *, world=WORLD, reverse=False):
    data, asset = item["data"], item["asset"]
    checkpoint = batch.snapshot([world])[world]
    shapes, bindings, sources, laws = [], [], {}, {}

    def add(group, kind, points, radius=0.0, *, source=None):
        ordinal = len(shapes)
        key = ph.CollisionKey(
            world, 100 + group if group is not None else 900 + ordinal, 11 + ordinal
        )
        if source is None:
            ids = tuple(1000 * ordinal + 31 + 7 * i for i in range(len(points)))
            shape = ph.CollisionShape(key, kind, ids, radius, revision=7)
            binding = (
                ph.ResidentStaticBinding(key, ph.CollisionMotion(points))
                if group is None
                else ph.ResidentFloatingAffineTreeBinding(
                    key,
                    world,
                    group,
                    points,
                    topology=checkpoint.topology,
                    graph_token=item["command"].graph_token,
                )
            )
        else:
            shape = replace(source.shape, key=key)
            binding = ph.ResidentFloatingAffineTreeBinding.from_asset(
                asset, source.shape.key, key=key, world=world
            )
            assert binding.group_index == group
            points = np.asarray(binding.local_points_m)
        shapes.append(shape)
        bindings.append(binding)
        sources[key] = (shape, group, points)
        return key

    for rule in data["contacts"]:
        group = rule["first_group"]
        if rule["kind"] == "box_sphere":
            collider = next(c for c in asset.tree.collisions if c.link_name == "right_jaw")
            first = add(group, "hull", np.asarray(collider.local_vertices), source=collider)
            second = add(None, "sphere", rule["sphere"][None, :], rule["radius"])
        elif rule["kind"] == "sphere_pair":
            first = add(group, "sphere", np.zeros((1, 3)), rule["radius"])
            second = add(rule["second_group"], "sphere", np.zeros((1, 3)), rule["radius"])
        else:
            first = add(group, "sphere", np.zeros((1, 3)), rule["radius"])
            wall = _box([-1.0, -1.0, -0.25], [1.0, 1.0, 0.0])
            if data["placement"] == "cycle":
                wall = wall @ _reference().base().quat_matrix(np.full(4, 0.5)).T + [
                    0.5,
                    -0.25,
                    0.125,
                ]
            second = add(None, "hull", wall)
        laws[tuple(sorted((first, second)))] = ph.RobotContactMaterial(rule["stiffness"], 0.0, 0.0)
    exclusions = tuple(
        (a.key, b.key)
        for a, b in combinations(shapes, 2)
        if a.key.body != b.key.body and tuple(sorted((a.key, b.key))) not in laws
    )
    if reverse:
        shapes.reverse()
        bindings.reverse()
    return ph.CollisionModel(tuple(shapes), exclusions), tuple(bindings), sources, laws


def _audit_contact(batch, item, receipt, sources, *, world=WORLD):
    data, ref = item["data"], _reference()
    entry, after = receipt.entry_state, receipt.accepted_state
    n, k = data["P"].shape
    assert receipt.world == world and receipt.entry_tick == entry.tick
    assert receipt.exit_tick == after.tick == entry.tick + 1
    assert receipt.dt_s == data["h"] and after == batch.snapshot([world])[world]
    assert receipt.selected_drive.shape == (k, 3) and receipt.actual_drive.shape == (k,)
    _close(receipt.selected_drive, _selected(item, entry.tick), atol=0, rtol=0)
    assert receipt.law_wrench_fp64.shape == receipt.uploaded_wrench_fp32.shape == (n + 1, 6)
    assert receipt.contact_generalized.shape == receipt.input_external.shape == (n + 6,)
    full_q, full_v = ref.expand(data, np.asarray(entry.position), np.asarray(entry.velocity))
    independent_wrench, sites = ref.contact_wrenches(data, full_q, full_v)
    _close(receipt.law_wrench_fp64, independent_wrench)
    d = receipt.diagnostics
    assert d["coordinate_kind"] == "floating_affine"
    assert tuple(
        d[key] for key in ("joint_count", "body_count", "velocity_count", "position_count")
    ) == (n, n + 1, n + 6, n + 7)
    assert tuple(
        d[key] for key in ("full_dof_count", "reduced_dof_count", "independent_dof_count")
    ) == (n + 6, k + 6, k)
    assert d["graph_token"] == item["command"].graph_token
    _close(d["projection_matrix"], data["S"])
    _close(d["offset_vector"], data["offset"])
    _close(d["constraint_rows"], data["C"][:, 6:])
    nested = d["floating_affine_diagnostic"]
    audit = _audit(
        item,
        entry,
        after,
        receipt.accepted_geometry,
        nested,
        batch.summary()[world],
        wrench=np.asarray(receipt.uploaded_wrench_fp32, float),
    )
    truth = audit["same_entry"]
    _close(receipt.contact_generalized, truth["contact_generalized"])
    _close(receipt.input_external, truth["input_external"])
    _close(receipt.actual_drive, truth["drive"])
    _close(nested.external, receipt.contact_generalized + receipt.input_external)
    assert d["equation_residual"] == nested.equation_residual
    assert d["control_residual"] == nested.control_residual
    # The same material reconstruction checks both dynamic sides. These checks
    # use stored poses solely to audit records, not as a force/mass oracle.
    works = _audit_sites(receipt, nested, sources)
    for field, value in works.items():
        _close(d[field], value, atol=1e-12, rtol=1e-10)
    _close(d["entry_kinetic_energy_j"], truth["energy"])
    _close(d["accepted_kinetic_energy_j"], audit["actual_end"]["energy"])
    _close(
        d["entry_power_times_h_j"],
        data["h"]
        * sum(
            np.asarray(s.first_force_n) @ s.first_velocity_m_s
            + np.asarray(s.second_force_n) @ s.second_velocity_m_s
            for s in receipt.sites
        ),
        atol=1e-12,
        rtol=1e-10,
    )
    assert _json(nested) == _json(batch.tree_diagnostics()[world])
    for value in (
        receipt.input_external,
        receipt.contact_generalized,
        receipt.law_wrench_fp64,
        receipt.uploaded_wrench_fp32,
        receipt.actual_drive,
    ):
        assert not value.flags.writeable
    return dict(**audit, independent_contact_sites=sites, original_work=works)


def _contact_window(item, mode, directory, name, *, steps, reverse=False):
    ref, data = _reference(), item["data"]
    q, v = data["q"].copy(), data["v"].copy()
    output = dict(
        input=data,
        source=item["source"],
        asset=item["asset"],
        command=item["command"],
        mode=mode,
        rows=[],
        reverse=reverse,
    )
    with ph.TyxWorldBatch([(WORLD, item["tree"])], mode, cache_root=directory / "cache") as batch:
        batch.set_controls({WORLD: item["command"]})
        model, bindings, sources, laws = _geometry(batch, item, reverse=reverse)
        output.update(
            model=model,
            bindings=bindings,
            sources=sources,
            laws=laws,
            initial=_record(batch),
            compiled=_compiled(batch),
        )
        with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as plan:
            try:
                for _ in range(steps):
                    entry = batch.snapshot()[WORLD]
                    q, v, truth = ref.advance(data, q, v, _selected(item, entry.tick))
                    assert batch.step_contacts(plan, data["h"]) == {WORLD: True}
                    row = _record(batch, plan)
                    row["audit"] = _audit_contact(batch, item, plan.last_accepted[WORLD], sources)
                    row["trajectory_reference"] = dict(q=q.copy(), v=v.copy(), terms=truth)
                    output["rows"].append(row)
                    _close(row["state"][WORLD].position, q, atol=WINDOW_ATOL)
                    _close(row["state"][WORLD].velocity, v, atol=WINDOW_ATOL)
            except BaseException as exc:
                output["failure"] = dict(type=type(exc).__name__, message=str(exc))
                raise
            finally:
                output.update(
                    retained=_record(batch, plan), stats=batch.stats(), plan_stats=plan.stats()
                )
                _save(directory, name, output)
    return output


@pytest.mark.tyx
@pytest.mark.parametrize("variant", ("forward", "reverse"), ids=("forward", "reverse"))
def test_a02_follower_surface_load_projects_once(variant, tyx_mode, tmp_path):
    data = _reference().case("A02")
    item = _item(tmp_path / "asset", data)
    # Independent old geometry/law premise: actual finite box + sphere, no proxy.
    from phymeo.geometry.contact_patch_adapter import contact_patch_sample
    from phymeo.solvers.contact_patch_loads import sample_patch_loads

    col = next(c for c in item["asset"].tree.collisions if c.link_name == "right_jaw")
    first = col.shape
    second = ph.CollisionShape(ph.CollisionKey(0, 900, 11), "sphere", (71,), radius=1 / 16)
    points = np.asarray(col.local_vertices) + [1 / 4 + 0.004, 0, 0]
    sphere = data["contacts"][0]["sphere"][None, :]
    patch = contact_patch_sample(first, second, points, sphere)
    assert len(patch.sites) == 1
    assert patch.sites[0].weight == pytest.approx(1.0)
    assert patch.sites[0].witness.gap_m == pytest.approx(-1 / 1024, abs=1e-12)
    sampled = sample_patch_loads(
        patch, ph.RobotContactMaterial(2048, 0, 0), np.zeros((1, 3)), np.zeros((1, 3))
    )
    _close(sampled["first_force"], [[-2.0, 0, 0]], atol=1e-12, rtol=0)
    output = _contact_window(
        item, tyx_mode, tmp_path, "a02-" + variant, steps=2, reverse=variant == "reverse"
    )
    row = output["rows"][0]
    d = row["diagnostics"][WORLD]
    _close(d.acceleration[[0, 6]], [-2 / 7, 4 / 7])
    _close(d.row_multipliers, [2 / 7])
    reaction = next(iter(row["receipts"][WORLD].static_reactions.values()))
    _close(reaction.force_n, [2.0, 0, 0])


@pytest.mark.tyx
@pytest.mark.parametrize(
    "variant", ("forward", "reverse", "cycle"), ids=("forward", "reverse", "cycle")
)
def test_a05_same_tree_bilateral_response(variant, tyx_mode, tmp_path):
    data = _reference().case("A05", placement="cycle" if variant == "cycle" else "base")
    output = _contact_window(
        _item(tmp_path / "asset", data),
        tyx_mode,
        tmp_path,
        "a05-" + variant,
        steps=2,
        reverse=variant == "reverse",
    )
    first = output["rows"][0]
    d, receipt = first["diagnostics"][WORLD], first["receipts"][WORLD]
    axis = np.array([0.0, 1.0, 0.0]) if variant == "cycle" else np.array([1.0, 0.0, 0.0])
    _close(d.acceleration[:3], (2 / 7) * axis)
    _close(d.acceleration[6:], [-10 / 7])
    _close(d.full_acceleration[6:], [-10 / 7, 10 / 7])
    _close(d.row_multipliers, [-2 / 7])
    _close(receipt.law_wrench_fp64[:, :3], [np.zeros(3), -2 * axis, 2 * axis])
    _close(receipt.law_wrench_fp64[:, 3:], np.zeros((3, 3)))
    assert not receipt.static_reactions
    for row in output["rows"]:
        state = row["state"][WORLD]
        velocity = np.asarray(state.full_velocity)
        total = 5 * velocity[:3] + axis * (2 * velocity[6] + velocity[7])
        _close(total / 5, np.zeros(3), atol=ATOL, rtol=0)
