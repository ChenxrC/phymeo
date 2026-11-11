"""Own disk resources before native attachment without filename aliasing."""

from hashlib import sha256
from pathlib import Path


def own_spec_resources(spec):
    # Keep the input VFS separate from bytes added below: identical relative
    # filenames in meshdir/texturedir must not resolve to the first added file.
    existing = dict(spec.assets)

    def freeze(filename, directory):
        if not filename:
            return filename
        if spec.strippath:
            filename = Path(filename).name
        if filename in existing:
            content = bytes(existing[filename])
        else:
            content = (Path(spec.modelfiledir) / directory / filename).read_bytes()
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
            resource.file = freeze(resource.file, directory)
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
