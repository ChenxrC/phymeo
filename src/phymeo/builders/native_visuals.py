"""Native cold display export; URDF visuals never enter the physical model."""

from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ..geometry.visual import VisualInstance, VisualMesh, VisualScene, VisualTexture


def capture_urdf_visuals(asset, source_root):
    """Return a display-only snapshot, or None for native/collision fallback."""
    import mujoco as mj

    path = Path(asset.path)
    root = deepcopy(source_root)
    if root.tag != "robot" or not root.findall("link/visual"):
        return None
    if root.findall(".//material/texture"):
        raise NotImplementedError("URDF visual texture export is not qualified yet")
    extension = root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(root, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    # MuJoCo discards URDF visuals while parsing, before MjSpec is returned.
    compiler.set("discardvisual", "false")
    compiler.set("fusestatic", "false")
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            link.remove(collision)
        for index, visual in enumerate(link.findall("visual")):
            visual.set("name", f"{link.attrib['name']}::visual_{index}")
    if asset.urdf_xml is not None or asset.resource_roots:
        from .native_resources import resolve_urdf_resources

        resolve_urdf_resources(root, asset)
    spec = mj.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.modelfiledir = str(path.parent) + "/"
    from .native_resources import own_spec_resources

    own_spec_resources(spec)
    model = spec.compile()
    return export_model(model, prefix=asset.name + "::", role="visual")


def export_model(
    model, *, prefix="", role="native_geom", excluded=(), fallback=(), identities=None
):
    import mujoco as mj

    kinds = {
        int(getattr(mj.mjtGeom, "mjGEOM_" + kind.upper())): (kind, width)
        for kind, width in (
            ("plane", 3),
            ("sphere", 1),
            ("capsule", 2),
            ("ellipsoid", 3),
            ("cylinder", 2),
            ("box", 3),
            ("mesh", 0),
        )
    }
    instances, meshes = [], {}
    texture_kinds = {
        int(mj.mjtTexture.mjTEXTURE_2D): "2d",
        int(mj.mjtTexture.mjTEXTURE_CUBE): "cube",
        int(mj.mjtTexture.mjTEXTURE_SKYBOX): "skybox",
    }
    textures = []
    for index in range(model.ntex):
        height, width, channels = (
            int(getattr(model, "tex_" + key)[index]) for key in ("height", "width", "nchannel")
        )
        start = int(model.tex_adr[index])
        pixels = model.tex_data[start : start + height * width * channels].reshape(
            height, width, channels
        )
        textures.append(
            VisualTexture(
                prefix + f"texture_{index}",
                texture_kinds[int(model.tex_type[index])],
                pixels,
                model.texture(index).name,
                int(model.tex_colorspace[index]),
            )
        )
    visibility = mj.MjvOption().geomgroup
    for i in range(model.ngeom):
        name = model.geom(i).name
        if any(name.startswith(p) for p in excluded):
            continue
        if int(model.geom_type[i]) not in kinds:
            raise NotImplementedError(f"visual geom type {model.geom_type[i]} is not qualified")
        kind, width = kinds[int(model.geom_type[i])]
        body = int(model.geom_bodyid[i])
        mesh_key = None
        if kind == "plane":
            x, y = map(float, model.geom_size[i, :2])
            if x > 0 and y > 0:
                # A native plane's finite display rectangle is not an infinite
                # collision plane or a box with arbitrary thickness.
                mesh_key = prefix + f"plane_{i}"
                meshes[mesh_key] = VisualMesh(
                    mesh_key,
                    [[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0]],
                    [[0, 1, 2], [0, 2, 3]],
                )
                kind, width = "mesh", 0
            elif (x > 0) != (y > 0):
                raise NotImplementedError("partially infinite visual plane is not qualified")
        if kind == "mesh" and mesh_key is None:
            mid = int(model.geom_dataid[i])
            mesh_key = prefix + f"mesh_{mid}"
            if mesh_key not in meshes:
                va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
                vertices = model.mesh_vert[va : va + vn]
                triangles = model.mesh_face[fa : fa + fn]
                uvs = None
                ta, tn = int(model.mesh_texcoordadr[mid]), int(model.mesh_texcoordnum[mid])
                if tn:
                    uv_indices = model.mesh_facetexcoord[fa : fa + fn]
                    if np.any(uv_indices < 0) or np.any(uv_indices >= tn):
                        raise ValueError("visual mesh has incomplete UV face indices")
                    # Native UVs have their own index space. Expand corners once
                    # in the display asset, leaving physical mesh indices intact.
                    vertices = vertices[triangles.reshape(-1)]
                    uvs = model.mesh_texcoord[ta : ta + tn][uv_indices.reshape(-1)]
                    triangles = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
                meshes[mesh_key] = VisualMesh(mesh_key, vertices, triangles, uvs)
        rgba = model.geom_rgba[i]
        material = []
        mat = int(model.geom_matid[i])
        if mat >= 0:
            if np.array_equal(rgba, np.array([0.5, 0.5, 0.5, 1], np.float32)):
                rgba = model.mat_rgba[mat]
            material = [
                ("roughness", float(model.mat_roughness[mat])),
                ("metallic", float(model.mat_metallic[mat])),
                ("specular", float(model.mat_specular[mat])),
                ("shininess", float(model.mat_shininess[mat])),
                ("reflectance", float(model.mat_reflectance[mat])),
                ("texture_repeat", tuple(map(float, model.mat_texrepeat[mat]))),
                ("texture_uniform", bool(model.mat_texuniform[mat])),
                (
                    "texture_bindings",
                    tuple(
                        (role, textures[int(index)].name)
                        for role, index in enumerate(model.mat_texid[mat])
                        if index >= 0
                    ),
                ),
            ]
        q = model.geom_quat[i]
        group = int(model.geom_group[i])
        item_role = "collision_geometry" if any(name.startswith(p) for p in fallback) else role
        instances.append(
            VisualInstance(
                identities["geom"][i] if identities is not None else prefix + name,
                (
                    identities["body"][body]
                    if identities is not None
                    else prefix + model.body(body).name
                )
                if body
                else None,
                kind,
                tuple(map(float, model.geom_size[i, :width])),
                tuple(map(float, model.geom_pos[i])),
                tuple(float(q[j]) for j in (1, 2, 3, 0)),
                tuple(map(float, rgba)),
                mesh_key,
                item_role,
                bool(rgba[3] > 0 and visibility[min(max(group, 0), 5)]),
                tuple(material),
            )
        )
    return VisualScene(
        tuple(instances), tuple(meshes.values()), 2 if textures else 1, tuple(textures)
    )


def assemble_visual_scene(model, captured, assets):
    """Bind display snapshots to named physical bodies, not native geom IDs."""
    excluded = tuple(name + "::" for name in captured)
    fallback = tuple(
        a.name + "::"
        for a in assets
        if a.name not in captured and Path(a.path).suffix.lower() == ".urdf"
    )
    base = export_model(model, excluded=excluded, fallback=fallback)
    instances = base.instances + tuple(i for scene in captured.values() for i in scene.instances)
    meshes = base.meshes + tuple(m for scene in captured.values() for m in scene.meshes)
    bodies = {model.body(i).name for i in range(1, model.nbody)}
    if any(i.body is not None and i.body not in bodies for i in instances):
        raise ValueError("visual instance has no retained public physical body")
    textures = base.textures + tuple(t for scene in captured.values() for t in scene.textures)
    return VisualScene(instances, meshes, 2 if textures else 1, textures)
