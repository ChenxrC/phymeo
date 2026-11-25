"""Restore the pinned Phymeo G1 standing inputs from a local Unitree rl_mjlab checkout."""

import argparse
import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

XML_NAME = "g1_primitive_actuated.xml"
XML_SHA = "f7e856e057a095a9f8708d302531d40ff2519440012c24488407ae8d0f0586aa"
DEPLOY_LF_SHA = "61e81a3ec7d3f3c7286ee39d043e8ca4f8e05257a5f6341825955a47e31bfbc1"


def restore(upstream, output):
    directory = upstream / "src/assets/robots/unitree_g1/xmls"
    scene = ET.parse(directory / "scene_g1.xml")
    robot = ET.parse(directory / "g1.xml")
    for parent in robot.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("mesh") and child.get("class") != "visual":
                raise ValueError("source contains a physical mesh")
            if child.tag == "mesh" or (child.tag == "geom" and child.get("mesh")):
                parent.remove(child)
    actuators = scene.find("actuator")
    robot.getroot().append(actuators)
    ET.SubElement(
        robot.find("worldbody"),
        "geom",
        name="floor",
        type="plane",
        size="10 10 .1",
        friction="1 .005 .0001",
    )
    # The archived process serialized and reparsed this intermediate MJCF.
    root = ET.fromstring(ET.tostring(robot.getroot(), encoding="unicode"))
    scalar = {j.get("name"): j for j in root.iter("joint")}
    source_defaults = scene.find("default/joint")
    motors = actuators.findall("motor")
    if len(motors) != 29 or len({motor.get("joint") for motor in motors}) != 29:
        raise ValueError("source does not have the original 29 named motors")
    for motor in motors:
        target = scalar[motor.attrib["joint"]]
        target.set("armature", repr(float(source_defaults.attrib["armature"])))
        target.set("damping", repr(float(source_defaults.attrib["damping"])))
        target.set("frictionloss", "0")
    # The original Windows research asset was written by ElementTree in text mode.
    xml = ET.tostring(root, encoding="unicode").replace("\n", "\r\n").encode("utf-8")
    deployment = (
        upstream / "deploy/robots/g1/config/policy/mimic/dance1_subject2/params/deploy.yaml"
    )
    yaml = deployment.read_bytes().replace(b"\r\n", b"\n")
    files = ((XML_NAME, xml, XML_SHA), ("deploy.yaml", yaml, DEPLOY_LF_SHA))
    for name, data, expected in files:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"{name}: expected {expected}, actual {actual}")
    output.mkdir(parents=True, exist_ok=True)
    for name, data, expected in files:
        path = output / name
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"existing {path} has different bytes")
        else:
            with path.open("xb") as stream:
                stream.write(data)
        print(f"{expected}  {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        type=Path,
        required=True,
        help="unitreerobotics/unitree_rl_mjlab at 1425b15f73bd4095f0df53709d7c389c3eb9e790",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    restore(args.upstream, args.output)
