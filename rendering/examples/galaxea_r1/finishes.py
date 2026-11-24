"""Authored product finishes for the original R1 visual links. Colors are linear RGB."""
from dataclasses import replace
from hashlib import sha256
import json

from phymeo_rendering.assets import Material


# Coating determines the optical surface, even when the substrate is metal.
# Clean satin highlights distinguish the white arm covers from the blue paint.
_PALETTE = {
    "ocean enamel": ((.048, .155, .219), 0., .48),
    "ivory enamel": ((.78, .80, .77), 0., .41),
    "titanium": ((.44, .49, .52), .86, .56),
    "graphite alloy": ((.17, .192, .21), .74, .53),
    "champagne alloy": ((.64, .395, .175), .86, .43),
    "sensor housing": ((.52, .57, .58), 0., .40),
    "instrument polymer": ((.048, .061, .071), 0., .38),
    "gripping surface": ((.065, .073, .078), 0., .60),
    "wheel": ((.17, .18, .19), 0., .76),
}

_CHASSIS = {
    "base_link": "ocean enamel", "torso_link4": "ocean enamel",
    "torso_link1": "graphite alloy", "torso_link2": "titanium",
    "torso_link3": "graphite alloy", "zed_link": "sensor housing",
    **{f"steer_motor_link{i}": "graphite alloy" for i in (1, 2, 3)},
    **{f"wheel_motor_link{i}": "wheel" for i in (1, 2, 3)},
}
_ARM = {
    "arm_base_link": "graphite alloy", "arm_link1": "ivory enamel",
    "arm_link2": "ivory enamel", "arm_link3": "ivory enamel",
    "arm_link4": "graphite alloy", "arm_link5": "titanium",
    "arm_link6": "champagne alloy", "realsense_link": "instrument polymer",
    "gripper_link": "ivory enamel", "gripper_finger_link1": "gripping surface",
    "gripper_finger_link2": "gripping surface",
}
_LINK_FINISH = {**_CHASSIS, **{f"{side}_{link}": finish
                for side in ("left", "right") for link, finish in _ARM.items()}}


def apply_finish(scene, preset):
    """Keep URDF identity and transforms while changing the visual presentation."""
    if preset == "urdf":
        return scene
    if preset != "studio":
        raise ValueError(f"unknown R1 material preset: {preset}")
    materials = []
    for material in scene.materials:
        if material.id.startswith("r1-material-"):
            link = material.id.removeprefix("r1-material-")
            if link not in _LINK_FINISH:
                raise ValueError(f"no studio material declared for R1 link {link}")
            color, metallic, roughness = _PALETTE[_LINK_FINISH[link]]
            material = Material(material.id, color, metallic, roughness)
        materials.append(material)
    bindings = [(m.id, m.base_color, m.metallic, m.roughness)
                for m in materials if m.id.startswith("r1-material-")]
    signature = sha256(json.dumps(bindings, separators=(",", ":")).encode()).hexdigest()
    return replace(scene, key=f"{scene.key}:studio-finish-v1:{signature}", materials=tuple(materials))
