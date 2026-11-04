"""Reproduce the distributable classic Panda URDF from the pinned Franka descriptions.

Only regeneration needs xacro==2.1.1 and PyYAML==6.0.3. Give --source the original
franka_description directory. No ROS installation, CAD files, simulator or network is needed.
"""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import xacro

SHA256 = {
    "package.xml": "88832657e816997861731e2363477a2e02d0e3c95ffd1999c4d87cce16c0b2cc",
    "robots/common/franka_arm.xacro": "5250eceb90296803ae32cd86189edeba4ab977cdfadfdddeb5a44860f4b9ec4e",
    "robots/common/franka_hand.xacro": "61bfc98d331fdcc9f340f028efff8b6ab6fbaf1db123384afb6c52d1c8ca9f17",
    "robots/common/franka_robot.xacro": "f65d0b824a2c3697e243b3fd87cf8248fbde9a1a6cda57fdf41b9904fac10d8b",
    "robots/common/utils.xacro": "f0124cd8ccb470b94be7c8da5e2dacf69671ec22c385013d34c9bed5409eca9b",
    "robots/common/inertial.yaml": "435dfba86cbab389c5e37f36e142daca65e79b357064c3fdf245b402d6e072cf",
    "robots/panda/joint_limits.yaml": "c531201d06715a3c90a9d793748b14d83f5b512c007dd56e3dc712eee6106ec7",
    "robots/panda/panda.urdf.xacro": "cb14a9dbde4b93592840dc865f256ae8343fd1e061d458ca620f5e549910ed8f",
}


def expanded(source, work, gazebo):
    for relative, digest in SHA256.items():
        data = (source / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"upstream bytes do not match the pinned SHA-256: {relative}")
        text = data.decode().replace("$(find franka_description)", work.as_posix())
        if relative.endswith("franka_robot.xacro"):
            # Physical proxies have no manufacturer self-collision anticipation margin.
            text = text.replace('safety_distance="0.03"', 'safety_distance="0"')
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    path = work / "robots/panda/panda.urdf.xacro"
    document = xacro.process_file(str(path), mappings={"hand": "true", "gazebo": gazebo})
    return ET.fromstring(document.toxml())


def export(source, output):
    with TemporaryDirectory() as temp:
        full = expanded(source, Path(temp), "true")
        proxy = expanded(source, Path(temp), "false")
    robot = ET.Element("robot", {"name": "panda"})
    robot.append(
        ET.Comment(
            " Modified by Phymeo from franka_ros 0.10.1, Apache-2.0. "
            "Fixed-base dynamics and proxy collision geometry. See accompanying NOTICE and SOURCES.md. "
        )
    )
    for link in full.findall("link"):
        name = link.attrib["name"]
        if name == "world":
            continue
        for item in list(link):
            if item.tag == "visual" or (
                item.tag == "collision" and item.find("geometry/mesh") is not None
            ):
                link.remove(item)
        if name.endswith("_sc"):
            collisions = proxy.findall(f"./link[@name='{name}']/collision")
            # The distal self-collision guard spans the working aperture; keep only the palm.
            if name == "panda_hand_sc":
                collisions = collisions[:3]
            for i, collision in enumerate(collisions):
                part = ("barrel", "cap1", "cap2")[i % 3]
                collision.set("name", f"{name}_{i // 3}_{part}")
                link.append(collision)
        if name in ("panda_leftfinger", "panda_rightfinger"):
            labels = ("screw", "sledge", "diagonal", "pad")
            for item, label in zip(link.findall("collision"), labels, strict=True):
                item.set("name", f"{name}_{label}")
        robot.append(link)
    robot.extend(j for j in full.findall("joint") if j.get("name") != "world_joint")
    ET.indent(robot)
    output.write_bytes(ET.tostring(robot, encoding="utf-8", xml_declaration=True) + b"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    export(options.source.resolve(), options.output)
