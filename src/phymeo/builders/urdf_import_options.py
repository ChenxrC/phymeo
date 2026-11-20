"""Standalone URDF compiler options and link-local display assets."""

from dataclasses import dataclass
from pathlib import Path

from ..geometry.visual import VisualInstance, VisualScene
from ._physolve_stl import stl_visual
from ._robot_xml import _attributes, _child, _children, _name, _numbers
from .urdf import _origin


@dataclass(frozen=True)
class URDFImportOptions:
    meshdir: str = ""
    discard_visual: bool = True
    strip_paths: bool = False


def take_compiler_options(root):
    """Consume the admitted extension in a compiler-owned XML tree."""
    extension = _child(root, "mujoco", "robot", required=False)
    if extension is None:
        return URDFImportOptions()
    _attributes(extension, set(), "URDF mujoco")
    _children(extension, {"compiler"}, "URDF mujoco")
    compiler = _child(extension, "compiler", "URDF mujoco")
    _attributes(compiler, {"meshdir", "assetdir", "strippath", "discardvisual"}, "URDF compiler")
    _children(compiler, set(), "URDF compiler")
    discard = compiler.get("discardvisual", "true")
    strip = compiler.get("strippath", "false")
    if discard not in ("true", "false") or strip not in ("true", "false"):
        raise ValueError("URDF discardvisual and strippath require true or false")
    root.remove(extension)
    return URDFImportOptions(compiler.get("meshdir", compiler.get("assetdir", "")),
                             discard == "true", strip == "true")


def resolve_mesh(asset, options, filename):
    """Resolve a filename against the declared directory and resource roots."""
    meshdir = options.meshdir
    if "://" in filename or "://" in meshdir:
        raise ValueError("autonomous URDF mesh references require filesystem paths")
    relative, directory = Path(filename), Path(meshdir)
    if options.strip_paths:
        relative = Path(relative.name)
    if not relative.is_absolute() and meshdir and not relative.is_relative_to(directory):
        relative = directory / relative
    bases = (Path(asset.path).parent, *map(Path, asset.resource_roots))
    for base in bases:
        candidate = base / relative
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"URDF mesh not found: {filename} (meshdir={meshdir!r})")


def authored_visuals(root, asset, options):
    """Decode source visuals in link coordinates, separate from collision input."""
    colors = {}
    for material in root.findall("material"):
        color = _child(material, "color", "URDF material", required=False)
        if color is not None:
            colors[_name(material, "name", "URDF material")] = tuple(_numbers(color, "rgba", 4, "URDF color"))
    meshes, instances = {}, []
    for link in root.findall("link"):
        body = f"{asset.name}::{link.attrib['name']}"
        for index, visual in enumerate(link.findall("visual")):
            name = f"{body}::visual_{index}"
            position, rotation = _origin(visual, name)
            geometry = _child(visual, "geometry", name)
            if len(geometry) != 1:
                raise ValueError(f"{name}: expected one visual geometry")
            shape = geometry[0]
            material = _child(visual, "material", name, required=False)
            rgba = (.5, .5, .5, 1.)
            if material is not None:
                _children(material, {"color"}, name)
                color = _child(material, "color", name, required=False)
                rgba = (tuple(_numbers(color, "rgba", 4, name)) if color is not None
                        else colors.get(material.get("name"), rgba))
            mesh = None
            if shape.tag == "mesh":
                path = resolve_mesh(asset, options, _name(shape, "filename", name))
                if path.suffix.lower() != ".stl":
                    raise NotImplementedError(f"{name}: authored mesh visuals require STL")
                scale = tuple(_numbers(shape, "scale", 3, name, (1., 1., 1.)))
                key = (path, scale)
                if key not in meshes:
                    meshes[key] = stl_visual(name, path.read_bytes(), scale)
                mesh = meshes[key].name
                size = ()
            elif shape.tag == "box":
                size = tuple(_numbers(shape, "size", 3, name) / 2)
            elif shape.tag == "sphere":
                size = tuple(_numbers(shape, "radius", 1, name))
            elif shape.tag in ("capsule", "cylinder"):
                radius, = _numbers(shape, "radius", 1, name)
                length, = _numbers(shape, "length", 1, name)
                size = (radius, length / 2)
            else:
                raise NotImplementedError(f"{name}: unsupported URDF visual {shape.tag}")
            if size and min(size) <= 0:
                raise ValueError(f"{name}: visual geometry dimensions must be positive")
            instances.append(VisualInstance(name, body, shape.tag, size, tuple(position),
                                            tuple(rotation), rgba, mesh=mesh, source_role="urdf_visual"))
    return VisualScene(tuple(instances), tuple(meshes.values()))
