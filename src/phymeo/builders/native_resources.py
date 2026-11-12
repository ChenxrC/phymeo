"""Own disk resources before native attachment without filename aliasing."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree as ET


def read_urdf_joint_catalog(asset):
    """Source-only joint identity/limits; no mesh loading, solver or device work."""
    raw = asset.urdf_xml if asset.urdf_xml is not None else Path(asset.path).read_bytes()
    root = ET.fromstring(raw)
    if root.tag != 'robot':
        raise NotImplementedError('native source joint catalog currently requires URDF')
    rows = []
    names = set()
    for joint in root.findall('joint'):
        name, kind = joint.get('name', ''), joint.get('type', '')
        if not name or name in names:
            raise ValueError('source joint names must be nonempty and unique')
        names.add(name)
        lower, upper = -float('inf'), float('inf')
        limit = joint.find('limit')
        if kind in ('revolute', 'prismatic') and limit is not None:
            lower, upper = float(limit.get('lower', '0')), float(limit.get('upper', '0'))
        rows.append((name, 'revolute' if kind == 'continuous' else kind, lower, upper))
    return tuple(rows)


def load_asset_source(asset):
    """Read one authoritative source; share its XML across physics and visuals.

    An in-memory URDF overrides the locator file. Roots resolve resources only,
    never replace its model text. Existing file-only MJCF/include loading stays
    native. Resource bytes are owned by the compiled template before publication.
    """
    import mujoco as mj

    path = Path(asset.path)
    raw = asset.urdf_xml.encode("utf-8") if asset.urdf_xml is not None else path.read_bytes()
    root = ET.fromstring(raw)
    digest = sha256(raw).hexdigest()
    if asset.urdf_xml is None and not asset.resource_roots:
        return mj.MjSpec.from_file(str(path)), root, digest
    if root.tag != "robot":
        raise ValueError("in-memory XML and resource_roots require URDF")
    physical = deepcopy(root)
    compiler = physical.find("mujoco/compiler")
    discard = "true" if compiler is None else compiler.get("discardvisual", "true")
    if discard == "true":
        for link in physical.findall("link"):
            for visual in link.findall("visual"):
                link.remove(visual)
    resolve_urdf_resources(physical, asset)
    spec = mj.MjSpec.from_string(ET.tostring(physical, encoding="unicode"))
    spec.modelfiledir = str(path.parent) + "/"
    return spec, root, digest


def resolve_urdf_resources(root, asset):
    """Resolve only the admitted geometry in this cold, caller-owned XML copy."""
    path = Path(asset.path)
    extension = root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(root, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    strip = compiler.get("strippath", "false")
    if strip not in {"true", "false"}:
        raise ValueError("URDF strippath requires true/false")
    directory = compiler.get("meshdir", compiler.get("assetdir", ""))
    bases = (path.parent, *(Path(p) for p in asset.resource_roots))
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if not filename:
            raise ValueError("URDF mesh requires filename")
        if "://" in filename:
            raise NotImplementedError("native URDF resource URI requires a filesystem path")
        relative = Path(filename).name if strip == "true" else filename
        candidates = [base / directory / relative for base in bases]
        resolved = next((p for p in candidates if p.is_file()), None)
        if resolved is None:
            raise FileNotFoundError(f"URDF mesh not found in declared resource roots: {filename}")
        mesh.set("filename", str(resolved.resolve()))
    # Apply native strip/directory policy once, then pass explicit full paths to
    # both compilers. This avoids a different root or basename being selected.
    compiler.set("strippath", "false")
    compiler.set("meshdir", "")


def own_spec_resources(spec):
    # Keep the input VFS separate from bytes added below: identical relative
    # filenames in meshdir/texturedir must not resolve to the first added file.
    existing = dict(spec.assets)

    def freeze(filename, directory, kind):
        if not filename:
            return filename
        if spec.strippath:
            filename = Path(filename).name
        if filename in existing:
            content = bytes(existing[filename])
        else:
            content = (Path(spec.modelfiledir) / directory / filename).read_bytes()
        if kind == "meshes" and Path(filename).suffix.lower() == ".stl":
            from ._native_stl import binary_stl

            content = binary_stl(content)
        key = "__phymeo_resource_" + sha256(content).hexdigest() + Path(filename).suffix
        spec.assets[key] = content
        return key

    for kind, directory in (
        ("meshes", spec.compiler.meshdir),
        ("hfields", spec.compiler.meshdir),
        ("textures", spec.compiler.texturedir),
    ):
        for resource in getattr(spec, kind):
            if kind == "textures" and any(resource.cubefiles):
                raise NotImplementedError(
                    "cube-face texture attachment is not qualified; use a single-file texture"
                )
            resource.file = freeze(resource.file, directory, kind)
    # Every admitted file now has a flat owned VFS key. Disk search directories
    # must not be prepended again during attachment/recompilation.
    spec.compiler.meshdir = ""
    spec.compiler.texturedir = ""
    return tuple(
        (singular, resource.name, resource.file)
        for plural, singular in (("meshes", "mesh"), ("hfields", "hfield"), ("textures", "texture"))
        for resource in getattr(spec, plural)
        if resource.file
    )


def bind_attached_resources(parent, resources, prefix):
    # MuJoCo 3.8 attach namespaces VFS keys but does not rewrite every resource
    # file reference (notably hfields). Bind only the newly attached definitions.
    for singular, name, filename in resources:
        target = getattr(parent, singular)(prefix + name)
        key = prefix + filename
        if target is None or key not in parent.assets:
            raise ValueError("attached native resource has no unambiguous owned VFS entry")
        target.file = key
