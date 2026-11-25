"""A tabletop product-photography set for the eight live manipulation models (metres)."""
from dataclasses import replace
from math import cos, pi, sin
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from _panda_scene import write_mesh
from galaxea_r1.finish_details import _store, _texture, _with_uv
from phymeo_rendering.assets import Material, Mesh, TextureChannel, TexturedMaterial
from phymeo_rendering.mesh_data import _local_path, load_mesh
from phymeo_rendering.photography import STUDIO_DISPLAY, STUDIO_EXPOSURE, studio_lighting
from phymeo_rendering.scene import Instance
from preview_stable_lighting.shell_normals import _replace_normals


_SET_NODE = 'photo-studio-set'


def _bench_maps(directory):
    # One orthographic chart for the authored 2 m tabletop. The working area
    # is marked by satin ink on the physical receiver, including in the depth/ID views.
    # Values here are sRGB reflectances. No lighting or contact shadows are baked.
    side = 768
    coord = (np.arange(side) + .5) * (2. / side) - 1.
    x, y = coord[None, :], coord[:, None]
    # Low-frequency laid finish, broad enough for the single-mip resident sampler.
    lay = .75*np.sin(2*pi*(4*x + 7*y)) + .45*np.cos(2*pi*(3*x - 17*y))
    color = np.empty((side, side, 3))
    color[:] = np.array((170, 173, 170)) + lay[..., None]
    # Silkscreen corner registration marks and the edge scale sit outside the
    # contact footprint. Subdued ink lets the saturated workpiece read first.
    near = lambda a, center, half: np.abs(a-center) < half
    ink = np.zeros((side, side), bool)
    for edge_x, sign_x in ((-.218, 1), (.71, -1)):
        for edge_y, sign_y in ((-.235, 1), (.235, -1)):
            ink |= (near(y, edge_y, .002) & near(x, edge_x + sign_x*.018, .018))
            ink |= (near(x, edge_x, .002) & near(y, edge_y + sign_y*.018, .018))
    for i in range(19):
        length = .014 if i % 5 == 0 else .007
        ink |= near(x, -.20 + i*.05, .0018) & near(y, -.268-length/2, length/2)
    color[ink] = (127, 137, 133)
    rough = np.broadcast_to((.65 + .006*lay)[..., None], color.shape)
    return (_texture(directory, 'worktop-reflectance', np.rint(color)),
            TextureChannel(_texture(directory, 'worktop-roughness', np.rint(rough*255))))


def _arm_maps(directory):
    # An axial finish on the actual capsule: dark joint boot, satin alloy cuff,
    # broad petrol-blue enamel and a narrow warm identification band at one end.
    u = (np.arange(256)+.5)/256
    color = np.tile((47, 124, 134), (256, 1))
    rough = np.full(256, .32)
    for mask, swatch, value in (
        ((u < .135) | (u > .865), (78, 87, 93), .45),
        (((.135 <= u) & (u < .19)) | ((.81 < u) & (u <= .865)), (167, 174, 174), .34),
        ((.21 <= u) & (u < .245), (227, 167, 82), .39),
    ):
        color[mask], rough[mask] = swatch, value
    return (_texture(directory, 'enamel-sleeve', color[None, :, :]),
            TextureChannel(_texture(directory, 'enamel-finish', np.tile(np.rint(rough*255)[None, :, None], (1, 1, 3)))))


def _finish_mesh(mesh, model, directory, arm, bench):
    """Keep the P4 positions/triangles; derive exact primitive normals and local UVs."""
    import mujoco as mj

    geom = model.geom(mesh.id)
    kind = int(model.geom_type[geom.id])
    is_link = mesh.id.startswith('link') and mesh.id.endswith('-shape')
    curved = kind in (int(mj.mjtGeom.mjGEOM_SPHERE), int(mj.mjtGeom.mjGEOM_CAPSULE))
    if not curved and mesh.id != 'support':
        return mesh, None
    source = load_mesh(mesh)
    raw = _local_path(mesh.uri).read_bytes()
    points = np.asarray(source.positions)
    finish = None
    if curved:
        half = float(model.geom_size[geom.id, 1]) if kind == int(mj.mjtGeom.mjGEOM_CAPSULE) else 0.
        normals = points.copy()
        normals[:, 2] -= np.clip(points[:, 2], -half, half)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        raw = _replace_normals(raw, source.normals, normals)
        if is_link:
            total = 2*(half + float(model.geom_size[geom.id, 0]))
            uv = np.column_stack((np.clip(.5+points[:, 2]/total, .002, .998), np.full(len(points), .5)))
            raw = _with_uv(raw, uv)
            finish = TexturedMaterial(mesh.id, (1., 1., 1.), .08, 1., *arm)
    elif mesh.id == 'support':
        raw = _with_uv(raw, .5 + .5*points[:, :2])
        finish = TexturedMaterial(mesh.id, (1., 1., 1.), .04, 1., *bench)
    path, _ = _store(directory, mesh.id+'-photo', raw, 'gltf')
    return replace(mesh, uri=path.as_uri()), finish


class _Surface:
    def __init__(self):
        self.points, self.normals, self.faces = [], [], []

    def quad(self, points, normals):
        i = len(self.points)
        self.points.extend(points)
        self.normals.extend(normals)
        self.faces.extend(((i, i+1, i+2), (i, i+2, i+3)))

    def block(self, low, high):
        """A square-edged sheet-metal component, with separate normals per face."""
        center = (np.asarray(low) + high)*.5
        half = (np.asarray(high) - low)*.5
        axes = np.eye(3)
        for axis in range(3):
            for sign in (-1., 1.):
                n, u = sign*axes[axis], axes[(axis+1)%3]
                v = np.cross(n, u)
                self.quad([center + (n + s*u + t*v)*half
                           for s,t in ((-1,-1),(1,-1),(1,1),(-1,1))], (n,)*4)

    def turned(self, x, y, profile, segments=48):
        """Lathe an ascending radius/z profile, keeping each chamfer's hard break."""
        for (r0, z0), (r1, z1) in zip(profile, profile[1:]):
            dr, dz = r1-r0, z1-z0
            length = np.hypot(dr, dz)
            for i in range(segments):
                a, b = 2*pi*i/segments, 2*pi*(i+1)/segments
                corners = ((r0, z0, a), (r0, z0, b), (r1, z1, b), (r1, z1, a))
                points = [(x+r*cos(t), y+r*sin(t), z) for r,z,t in corners]
                normals = [(cos(t)*dz/length, sin(t)*dz/length, -dr/length) for _,_,t in corners]
                self.quad(points, normals)

    def mesh(self, directory, name):
        with TemporaryDirectory(dir=directory) as temp:
            asset = write_mesh(temp, name, self.points, self.normals, self.faces)
            path, _ = _store(directory, name, _local_path(asset.uri).read_bytes(), 'gltf')
        return Mesh(name, path.as_uri())


def _room(wall, floor, panels, upper, hardware):
    # Closed faces well behind/below the physical bench. The room corner and
    # service cabinet give depth cues in the fixed view, even at low exposure.
    # The walls extend below the floor, which fills the space past the table.
    wall.quad(((-1.22,1.20,-.62),(3.,1.20,-.62),(3.,1.20,2.4),(-1.22,1.20,2.4)),
              ((0.,-1.,0.),)*4)
    wall.quad(((-1.22,-3.,-.62),(-1.22,1.20,-.62),(-1.22,1.20,2.4),(-1.22,-3.,2.4)),
              ((1.,0.,0.),)*4)
    floor.quad(((-1.25,-3.,-.54),(3.,-3.,-.54),(3.,1.24,-.54),(-1.25,1.24,-.54)),
               ((0.,0.,1.),)*4)
    # Lower doors and a separate upper drawer fascia. Their 7 mm horizontal
    # reveal and offset lip are visible just above bench height from the fixed camera.
    for left in (-1.15, -.64, -.13):
        right, front, rear = left+.498, 1.174, 1.197
        panels.block((left,front,-.20), (right,rear,.219))
        upper.block((left,front-.003,.226), (right,rear,.56))
        hardware.block((left,front+.012,.218), (right,rear,.227))
        # A 132 mm cast pull on two short returns. Its underside clearance and
        # satin warm metal are visible against the cooler painted door.
        center = (left+right)*.5
        for dx in (-.057,.057):
            hardware.block((center+dx-.006,front-.030,.116), (center+dx+.006,front,.129))
        hardware.block((center-.066,front-.039,.114), (center+.066,front-.027,.131))


def _set(model, directory):
    wall, floor, panels, upper, hardware, iron, alloy, enamel = (_Surface() for _ in range(8))
    _room(wall, floor, panels, upper, hardware)
    # The physical arm has no rendered mounting stand beneath its first joint.
    # The fixed stand meets the underside of the root capsule at its nominal pose.
    x, y, pivot = map(float, model.body('link0').pos)
    top = pivot - float(model.geom_size[model.geom('link0-shape').id, 0])
    iron.turned(x, y, ((.080, .0003), (.082, .0025), (.082, .010), (.077, .014), (.045, .014)))
    alloy.turned(x, y, ((.064, .014), (.069, .018), (.069, .024), (.065, .028), (.050, .028)))
    enamel.turned(x, y, ((.052, .028), (.053, .031), (.045, top-.033), (.043, top-.030)))
    alloy.turned(x, y, ((.043, top-.030), (.046, top-.028), (.046, top-.018), (.042, top-.015)))
    iron.turned(x, y, ((.042, top-.015), (.042, top-.005), (.037, top), (.003, top)))
    # Real flush fastener heads around the anchor flange. The planar annular
    # tops end at their recessed dark hex socket rather than a specular decal.
    for i in range(6):
        a = i*pi/3
        hx, hy = x+.072*cos(a), y+.072*sin(a)
        alloy.turned(hx, hy, ((.0040,.0140), (.0042,.0147), (.0038,.0154), (.0018,.0154)), segments=12)
        iron.turned(hx, hy, ((.0018,.0153), (.0001,.0153)), segments=6)
    # Vertical fascia starts below the physics plane and meets its actual edges.
    sx, sy = map(float, model.geom_size[model.geom('support').id, :2])
    rim = ((-sx,-sy), (sx,-sy), (sx,sy), (-sx,sy))
    for (ax,ay), (bx,by) in zip(rim, (*rim[1:],rim[0])):
        n = np.array((by-ay, ax-bx, 0.))
        n /= np.linalg.norm(n)
        iron.quad(((ax,ay,-.043),(bx,by,-.043),(bx,by,-.004),(ax,ay,-.004)), (n,)*4)
        alloy.quad(((ax,ay,-.004),(bx,by,-.004),(bx,by,0.),(ax,ay,0.)), (n,)*4)
    finishes = (
        ('graphite', iron, (.058,.073,.077), .28, .43),
        ('satin-alloy', alloy, (.56,.60,.60), .82, .32),
        ('plinth', enamel, (.035,.128,.143), .08, .38),
        ('plaster', wall, (.68,.70,.68), 0., .90),
        ('room-floor', floor, (.31,.34,.34), 0., .76),
        ('cabinet', panels, (.53,.58,.57), .02, .62),
        ('cabinet-drawers', upper, (.61,.63,.60), .04, .56),
        ('cabinet-hardware', hardware, (.20,.15,.093), .69, .38),
    )
    assets = []
    for label, surface, rgb, metallic, roughness in finishes:
        name = 'photo-studio-'+label
        assets.append((surface.mesh(directory, name), Material(name, rgb, metallic, roughness)))
    return assets


def style_manipulation_studio(scene, model, directory):
    """Return a styled P4 scene and extra node/body bindings for DeviceVisualBinding."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    arm, bench = _arm_maps(directory), _bench_maps(directory)
    meshes, replacements = [], {}
    for source in scene.meshes:
        mesh, finish = _finish_mesh(source, model, directory, arm, bench)
        meshes.append(mesh)
        if finish is not None:
            replacements[finish.id] = finish
    materials = []
    for mat in scene.materials:
        if mat.id in replacements:
            mat = replacements[mat.id]
        elif mat.id in ('wrist', 'pusher'):
            mat = replace(mat, base_color=(.24,.29,.31), metallic=.62, roughness=.31)
        elif mat.id.endswith('-pad'):
            mat = replace(mat, base_color=(.085,.098,.101), metallic=.0, roughness=.55)
        elif mat.id.startswith('object-'):
            mat = replace(mat, roughness=.35)
        materials.append(mat)
    instances = list(scene.instances)
    last_id = max(i.id for i in instances)
    for i, (mesh, finish) in enumerate(_set(model, directory), 1):
        meshes.append(mesh)
        materials.append(finish)
        instances.append(Instance(last_id+i, mesh.id, finish.id, _SET_NODE))
    return (replace(scene, key=scene.key+':photo-studio-v3', meshes=tuple(meshes),
                    materials=tuple(materials), instances=tuple(instances), lights=()), {_SET_NODE: 0})
