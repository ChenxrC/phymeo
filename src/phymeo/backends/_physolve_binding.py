"""Describe actual PhySolve storage, without inventing Newton or ABI layouts."""

from ..core.runtime_storage import RuntimeField, RuntimeStorage


def bind_storage(owner, banks, stream, device, generation):
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
    fields = tuple(RuntimeField(name, tuple(arrays[0][name].shape), "float32", *spec)
                   for name, spec in semantics.items())
    return RuntimeStorage(owner=owner, generation=generation, fields=fields,
                          banks=arrays,
                          stream=stream, device=str(device))
