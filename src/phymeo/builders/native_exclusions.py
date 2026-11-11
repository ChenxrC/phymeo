"""Exact, cold lowering of source shape exclusions to existing native excludes.

This profile admits complete pairs of retained, non-world bodies only. It does
not approximate a shape rule with a broader body rule, rewrite masks, or add a
second collision policy to the solver Host.
"""

from dataclasses import replace
from itertools import product

import numpy as np

from ..backends.physolve_initial import NativeInitialState
from .collision_sources import CollisionExclusionPlan
from .mujoco_compiler import compile_mujoco_spec


def exclude_template_pairs(template, plan):
    import mujoco

    record = template.compilation
    mapping = record.collision_source
    if not isinstance(plan, CollisionExclusionPlan) or mapping is None:
        raise TypeError("native exclusions require a source-bound CollisionExclusionPlan")
    checked = mapping.plan_exclusions(plan.excluded, source_identity=plan.source_identity)
    if checked != plan:
        raise ValueError("collision exclusion plan content does not match its source")
    if not plan.native_geom_pairs:
        return template
    model = record.fresh_native_model()
    # Validate the original identity before binding fields to the new MJB hash.
    # Equal dimensions alone cannot authorize another model's initial state.
    template.initial_state.apply(record, mujoco.MjData(model))
    requested = set(plan.native_geom_pairs)
    body_geoms = {}
    for geom, body in enumerate(model.geom_bodyid):
        body_geoms.setdefault(int(body), []).append(geom)
    body_pairs = set()
    for a, b in requested:
        bodies = tuple(sorted((int(model.geom_bodyid[a]), int(model.geom_bodyid[b]))))
        if bodies[0] == 0 or bodies[0] == bodies[1]:
            raise NotImplementedError(
                "exact native exclusion requires distinct retained non-world bodies"
            )
        body_pairs.add(bodies)
    for a, b in body_pairs:
        affected = {
            tuple(sorted((int(x), int(y)))) for x, y in product(body_geoms[a], body_geoms[b])
        }
        if not affected <= requested:
            raise NotImplementedError("native body exclude would remove undeclared shape pairs")
    # XML serialization is not a precision-preserving spec snapshot. The private
    # compiler owner has complete VFS resources and is never returned to callers.
    if template._source_spec is None:
        raise ValueError("native template lacks its owned rule compilation source")
    spec = template._source_spec.copy()
    for a, b in sorted(body_pairs):
        names = model.body(a).name, model.body(b).name
        if not all(names):
            raise NotImplementedError("native exclusion requires named retained bodies")
        signature = (a << 16) + b
        if signature not in model.exclude_signature:
            spec.add_exclude(bodyname1=names[0], bodyname2=names[1])
    compiled = compile_mujoco_spec(spec, instance=mapping.instance, world=mapping.world)
    new = compiled.mj_model
    expected_excludes = set(map(int, model.exclude_signature)) | {
        (a << 16) + b for a, b in body_pairs
    }
    if set(map(int, new.exclude_signature)) != expected_excludes:
        raise ValueError("collision recompile changed undeclared exclusion signatures")
    # This cold transform changes filtering only. Native XML rounding, compiler
    # drift or an accidental source mutation must fail rather than alter physics.
    for name in dir(model):
        if name.startswith(("_", "name_")) or name in {"exclude_signature", "names", "names_map"}:
            continue
        value = getattr(model, name)
        if isinstance(value, np.ndarray) and not np.array_equal(value, getattr(new, name)):
            raise ValueError(f"collision recompile changed physical field {name}")
    for name in dir(model.opt):
        if not name.startswith("_"):
            value = getattr(model.opt, name)
            if isinstance(value, (int, float, np.ndarray)) and not np.array_equal(
                value, getattr(new.opt, name)
            ):
                raise ValueError(f"collision recompile changed physical option {name}")
    # A rule edit cannot reorder the indices used by initial/control bindings.
    for kind, count in (
        ("body", model.nbody),
        ("geom", model.ngeom),
        ("joint", model.njnt),
        ("actuator", model.nu),
    ):
        count_field = {"body": "nbody", "geom": "ngeom", "joint": "njnt", "actuator": "nu"}[kind]
        if getattr(new, count_field) != count:
            raise ValueError("collision recompile changed native entity count")
        if tuple(getattr(new, kind)(i).name for i in range(count)) != tuple(
            getattr(model, kind)(i).name for i in range(count)
        ):
            raise ValueError("collision recompile changed native identity layout")
    for name in ("jnt_type", "jnt_qposadr", "jnt_dofadr", "geom_bodyid"):
        if not np.array_equal(getattr(new, name), getattr(model, name)):
            raise ValueError("collision recompile changed native index binding")
    initial = NativeInitialState(compiled.source.model_sha256, template.initial_state.fields)
    initial.apply(compiled, mujoco.MjData(new))
    return replace(
        template,
        compilation=compiled,
        initial_state=initial,
        collision_plans=(*template.collision_plans, plan),
        _source_spec=compiled.mj_spec.copy(),
    )
