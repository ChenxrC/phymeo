"""Cold conversion of a compiled native template to public local ray surfaces."""

from dataclasses import dataclass

from ..geometry.ray_surface import RaySurface


@dataclass(frozen=True)
class NativeRayScene:
    compilation: object
    surfaces: tuple
    bodies: tuple
    contract_version: int = 1


def compile_ray_scene(template):
    import mujoco as mj

    model = template.compilation.fresh_native_model()
    bodies = {index: name for name, index in template.body_bindings}
    bodies.setdefault(0, "world")
    shapes = {index: name for name, index in template.shape_bindings}
    if set(shapes) != set(range(model.ngeom)):
        raise ValueError("ray scene requires complete public shape identity")
    kinds = {
        int(mj.mjtGeom.mjGEOM_PLANE): ("plane", 0),
        int(mj.mjtGeom.mjGEOM_SPHERE): ("sphere", 1),
        int(mj.mjtGeom.mjGEOM_CAPSULE): ("capsule", 2),
        int(mj.mjtGeom.mjGEOM_ELLIPSOID): ("ellipsoid", 3),
        int(mj.mjtGeom.mjGEOM_CYLINDER): ("cylinder", 2),
        int(mj.mjtGeom.mjGEOM_BOX): ("box", 3),
        int(mj.mjtGeom.mjGEOM_MESH): ("mesh", 0),
    }
    surfaces, bindings = [], []
    for i in range(model.ngeom):
        if int(model.geom_type[i]) not in kinds:
            raise NotImplementedError(f"native ray geom type {model.geom_type[i]} is unqualified")
        kind, width = kinds[int(model.geom_type[i])]
        body = int(model.geom_bodyid[i])
        if body not in bodies:
            raise ValueError("ray surface has no public body identity")
        vertices, triangles = (), ()
        if kind == "mesh":
            mesh = int(model.geom_dataid[i])
            va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            vertices = tuple(map(tuple, model.mesh_vert[va : va + vn]))
            triangles = tuple(map(tuple, model.mesh_face[fa : fa + fn]))
        q = model.geom_quat[i]
        surfaces.append(
            RaySurface(
                shapes[i],
                bodies[body],
                kind,
                tuple(model.geom_size[i, :width]),
                tuple(model.geom_pos[i]),
                (q[1], q[2], q[3], q[0]),
                vertices,
                triangles,
            )
        )
        bindings.append(body)
    return NativeRayScene(template.compilation, tuple(surfaces), tuple(bindings))
