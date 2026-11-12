"""Describe actual PhySolve storage, without inventing Newton or ABI layouts."""

from ..core.runtime_storage import RuntimeField, RuntimeStorage


def bind_storage(owner, banks, stream, device, generation, controls=None, bodies=None, contacts=None, validation=None):
    import tyx as tx

    # q, DOF and body index spaces are deliberately distinct. The compiled
    # PHYMEO model supplies joint-dependent coordinate/actuator interpretation.
    semantics = {
        "qpos": ("m/rad/unit quaternion by joint", "joint; free translation world, quaternion WXYZ", "native_q", "integration endpoint", True),
        "qvel": ("m/s or rad/s by DOF", "joint; free linear world, angular body-local", "native_dof", "integration endpoint", True),
        "qacc": ("m/s^2 or rad/s^2 by DOF", "qvel tangent frame", "native_dof", "solver evaluation", False),
        "time": ("s", "simulation", "world", "integration endpoint", False),
        "ctrl": ("compiled actuator input", "compiled actuator", "native_actuator", "persistent command", True),
        "qfrc_applied": ("N or N*m by DOF", "generalized qvel dual", "native_dof", "persistent external command", True),
        "qfrc_actuator": ("N or N*m by DOF", "generalized qvel dual", "native_dof", "solver evaluation", False),
        "qfrc_constraint": ("N or N*m by DOF", "generalized qvel dual", "native_dof", "solver evaluation", False),
        "xpos": ("m", "world; body frame origin, not COM", "native_body", "solver evaluation", False),
        "xquat": ("unit quaternion", "body-to-world WXYZ", "native_body", "solver evaluation", False),
    }
    arrays = [{name: getattr(bank, name).view(tx.f32) for name in semantics} for bank in banks]
    if controls is not None:
        semantics["ctrl"] = (*semantics["ctrl"][:-1], False)
        for name, units in (("target_position", "m or rad by scalar joint"),
                            ("target_velocity", "m/s or rad/s by scalar joint")):
            semantics[name] = (units, "joint", "scalar_control", "persistent command", True)
        for array, bank in zip(arrays, controls.banks):
            array.update(bank)
    if bodies is not None:
        semantics.update({
            "body_position": ("m", "world; body frame origin, not COM", "native_body", "integration endpoint", False),
            "body_rotation_wxyz": ("unit quaternion", "body-to-world WXYZ", "native_body", "integration endpoint", False),
            "body_com_twist": ("m/s,rad/s", "world; linear at body COM, then angular", "native_body", "integration endpoint", False),
        })
        for array, bank in zip(arrays, bodies.banks):
            array.update(bank)
    fields = tuple(RuntimeField(name, tuple(arrays[0][name].shape), "float32", *spec)
                   for name, spec in semantics.items())
    if contacts is not None:
        for array, bank in zip(arrays, contacts.banks):
            array.update(bank)
        for name in contacts.banks[0]:
            floating = name in contacts.float_fields or name in ('contact_sample_time', 'contact_body_com_position')
            unit, frame = (contacts.float_fields[name][2:] if name in contacts.float_fields else
                           ("index/count", "native geom / 0-based world; rows are not persistent IDs"))
            space = 'native_contact'
            if name in ('contact_count_by_world', 'contact_sample_time'):
                space = 'world'
            elif name == 'contact_count':
                space = 'global_count'
            if name == 'contact_sample_time':
                unit, frame = 's', 'simulation clock of the retained solver evaluation'
            elif name == 'contact_body_com_position':
                unit, frame, space = 'm', 'world; body COM at retained contact solver evaluation', 'native_body'
            fields += (RuntimeField(name, tuple(arrays[0][name].shape),
                                    "float32" if floating else "int32", unit, frame,
                                    space, "solver evaluation; unselected reset worlds retain last sample"),)
    if validation is not None:
        for array in arrays:
            array['execution_error'] = validation
        fields += (RuntimeField('execution_error', (1,), 'int32', 'bit mask',
                                'runtime execution', 'global_error',
                                'current candidate; nonzero suppresses dependent observations'),)
    return RuntimeStorage(owner=owner, generation=generation, fields=fields,
                          banks=arrays,
                          stream=stream, device=str(device))
