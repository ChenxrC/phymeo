"""Serve a live MuJoCo experiment while its clock advances independently of viewers.

Examples:
    python rendering/examples/view_background_simulation.py --scene cradle
    python rendering/examples/view_background_simulation.py --scene dominoes --no-browser --steps 4000

The native Vulkan renderer uses the physics model's compiled geometry, body
poses and material bindings. Ctrl+C stops the simulation and local monitor.
"""
import argparse
from dataclasses import replace
from hashlib import sha256
from math import cos, radians, sin, tan
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
import webbrowser
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from _panda_scene import box, look_at, write_mesh
from material_orbit import sphere_mesh
from phymeo_rendering import StudioRenderer
from phymeo_rendering.assets import Material
from phymeo_rendering.cameras import Camera
from phymeo_rendering.live_viewer import LiveSceneViewer
from phymeo_rendering.photography import studio_lighting
from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene


# Linear RGB, metalness, roughness. Names bind the physical geoms to their finish.
FINISHES = {
    'limestone': ((.47, .40, .29), 0., .76),
    'slate': ((.035, .065, .071), .16, .45),
    'sand': ((.30, .27, .22), 0., .88),
    'brass': ((.80, .48, .17), .84, .32),
    'nickel': ((.48, .55, .60), .83, .24),
    'jade': ((.035, .32, .26), .06, .32),
    'ivory': ((.78, .68, .45), .04, .40),
    'terracotta': ((.54, .14, .061), .04, .42),
}
TITLES = {'cradle': '碰撞摆 · 牛顿摆', 'dominoes': '多米诺 · 桌面连锁'}


def _vec(values):
    return ' '.join(format(x, '.10g') for x in values)


def _geom(body, name, kind, size, pos, material, **kwargs):
    return ET.SubElement(body, 'geom', name=name, type=kind, size=_vec(size),
                         pos=_vec(pos), material=material, **kwargs)


def make_model(choice):
    root = ET.Element('mujoco', model=choice)
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep='.002', gravity='0 0 -9.81', integrator='implicitfast')
    assets = ET.SubElement(root, 'asset')
    for name, (color, _, _) in FINISHES.items():
        ET.SubElement(assets, 'material', name=name, rgba=_vec((*color, 1.)))
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', friction='.58 .006 .0005', solref='.008 .8',
                  solimp='.95 .99 .001', density='590')
    world = ET.SubElement(root, 'worldbody')
    _geom(world, 'floor', 'box', (3.5, 3., .025), (0., 0., -.155), 'sand')
    _geom(world, 'plinth', 'box', (.85, .34, .07), (0., 0., -.06), 'slate')
    _geom(world, 'deck', 'box', (.9, .39, .035), (0., 0., .045), 'limestone')
    if choice == 'cradle':
        for x, name in [(-.53, 'left'), (.53, 'right')]:
            _geom(world, 'post-'+name, 'box', (.024, .035, .30), (x, .12, .38), 'slate')
        _geom(world, 'bridge', 'box', (.56, .075, .025), (0., .057, .68), 'brass')
        for j in range(5):
            link = ET.SubElement(world, 'body', name=f'pendulum-{j}', pos=_vec(((j-2)*.145, 0., .677)))
            ET.SubElement(link, 'joint', name=f'pivot-{j}', type='hinge', axis='0 1 0', damping='.001')
            _geom(link, f'stem-{j}', 'box', (.005, .005, .198), (0., 0., -.198), 'nickel',
                  contype='0', conaffinity='0', mass='.012')
            _geom(link, f'bob-{j}', 'sphere', (.072,), (0., 0., -.407), 'brass' if j in (0, 4) else 'nickel',
                  mass='.28', solref='.006 .35', friction='.17 .002 .0002')
    else:
        for j in range(12):
            tile = ET.SubElement(world, 'body', name=f'domino-{j}', pos=_vec((-.61+j*.098, 0., .201)))
            ET.SubElement(tile, 'freejoint', name=f'free-{j}')
            _geom(tile, f'tile-{j}', 'box', (.015, .061, .12), (0., 0., 0.),
                  ('jade','jade','ivory','terracotta')[j%4], mass='.090')
        ball = ET.SubElement(world, 'body', name='target-marble', pos='.60 0 .133')
        ET.SubElement(ball, 'freejoint', name='marble-free')
        _geom(ball, 'marble', 'sphere', (.052,), (0., 0., 0.), 'brass', mass='.085')
        _geom(world, 'end-stop', 'box', (.018, .145, .040), (.82, 0., .12), 'slate')
    xml = ET.tostring(root, encoding='unicode')
    return mj.MjModel.from_xml_string(xml), sha256(xml.encode()).hexdigest()


def start_episode(model, data, choice):
    mj.mj_resetData(model, data)
    first = model.joint('pivot-0' if choice == 'cradle' else 'free-0').id
    if choice == 'cradle':
        data.qpos[model.jnt_qposadr[first]] = .85
    else:
        # Start the first domino past its tipping point; leave a 1 mm gap
        # under its lowest corner. All later contacts are solver-generated.
        angle = .22
        i = model.jnt_qposadr[first]
        data.qpos[i + 2] = .081 + .12*cos(angle) + .015*sin(angle)
        data.qpos[i + 3:i + 7] = (cos(angle/2), 0., sin(angle/2), 0.)
        data.qvel[model.jnt_dofadr[first] + 4] = 1.5
    mj.mj_forward(model, data)


def _pose(position, wxyz):
    w, x, y, z = map(float, wxyz)
    return Pose(tuple(map(float, position)), (x, y, z, w))


def visual_scene(model, model_digest, directory):
    meshes, materials, instances = {}, {}, []
    for i in range(model.ngeom):
        kind, size = model.geom_type[i], model.geom_size[i]
        shape = (int(kind), *map(float, size))
        if shape not in meshes:
            key = 'shape-'+str(len(meshes))
            if kind == mj.mjtGeom.mjGEOM_BOX:
                meshes[shape] = box(directory, key, size * 2)
            elif kind == mj.mjtGeom.mjGEOM_SPHERE:
                meshes[shape] = write_mesh(directory, key, *sphere_mesh(float(size[0]), 32, 64))
            else:
                raise ValueError(f'unsupported monitor shape {kind}')
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_MATERIAL, int(model.geom_matid[i]))
        if name not in materials:
            materials[name] = Material(name, *FINISHES[name])
        node = model.body(int(model.geom_bodyid[i])).name
        instances.append(Instance(i+1, meshes[shape].id, name, node, _pose(model.geom_pos[i], model.geom_quat[i])))
    return VisualScene('mujoco-monitor-'+model_digest, tuple(meshes.values()), tuple(materials.values()), tuple(instances))


def snapshot(scene, model, data, scratch, world, epoch, tick):
    # mj_step can leave cached Cartesian poses at the pre-integration state.
    # Evaluate on scratch so display sampling cannot perturb the physics state.
    scratch.qpos[:] = data.qpos
    mj.mj_kinematics(model, scratch)
    nodes = tuple(NodePose(model.body(i).name, _pose(scratch.xpos[i], scratch.xquat[i])) for i in range(model.nbody))
    key = FrameKey(scene.key, world, epoch, tick, tick, 'accepted_endpoint' if tick else 'authored')
    return SceneFrame(key, float(data.time), nodes)


def cameras(width=960, height=540):
    # These cameras depend only on the published frame key and studio work volume.
    views = {
        'overview': ((.98, -1.85, 1.06), (0., 0., .29), (0., 0., 1.), 39.),
        'front': ((.04, -1.96, .51), (0., 0., .29), (0., 0., 1.), 35.),
        'overhead': ((0., -.13, 2.17), (0., 0., .12), (0., 1., 0.), 37.),
    }
    def make(name, eye, target, up, fov):
        pose = look_at(eye, target, up)
        focal = .5*height/tan(radians(fov)/2)
        return lambda frame: Camera(name, frame.key, pose, width, height, focal, focal, width/2, height/2, .03, 12.)
    return {name: make(name, *setting) for name, setting in views.items()}


def run(args):
    if args.steps is not None and args.steps < 0:
        raise ValueError('--steps must be nonnegative')
    model, digest = make_model(args.scene)
    data, scratch = mj.MjData(model), mj.MjData(model)
    step_s = model.opt.timestep
    # Both examples demonstrate free response following a single initial
    # excitation. Repeat after it settles; epoch and simulated time expose resets.
    episode_steps = round((6. if args.scene == 'dominoes' else 12.)/step_s)
    start_episode(model, data, args.scene)
    lights = replace(studio_lighting(), center_m=(0., 0., .28), scale_m=.95, panel_strength=.34)
    def backend():
        return StudioRenderer(library_path=args.library, shader_directory=args.library.parent,
                              device_index=args.device_index, studio=lights, panel_samples=16)
    with TemporaryDirectory(prefix='phymeo-live-monitor-') as temp:
        scene = visual_scene(model, digest, Path(temp))
        with LiveSceneViewer(scene, cameras=cameras(), renderer_factory=backend, title=TITLES[args.scene],
                             fps=15, port=args.port,
                             camera_labels={'overview':'全景', 'front':'正面', 'overhead':'俯视'},
                             depth_ranges={'overview':(.6,3.2),'front':(.6,3.0),'overhead':(.65,2.6)}) as viewer:
            tick = epoch = total = 0
            viewer.publish(snapshot(scene, model, data, scratch, args.scene, epoch, tick))
            print(f'{viewer.url}  scene={args.scene}  physics={step_s:g}s/step  Ctrl+C to stop', flush=True)
            if not args.no_browser:
                webbrowser.open(viewer.url)
            start = perf_counter()
            try:
                while args.steps is None or total < args.steps:
                    if tick == episode_steps:
                        epoch += 1
                        tick = 0
                        start_episode(model, data, args.scene)
                        viewer.publish(snapshot(scene, model, data, scratch, args.scene, epoch, tick))
                    # Publish completed physics at 50 Hz. A slow subscriber may
                    # observe fewer exposures, but never sets the timestep.
                    steps = min(10, episode_steps-tick)
                    if args.steps is not None:
                        steps = min(steps, args.steps-total)
                    delay = start + (total+steps)*step_s - perf_counter()
                    if delay > 0:
                        sleep(delay)
                    mj.mj_step(model, data, nstep=steps)
                    tick += steps
                    total += steps
                    if not np.isfinite(data.qpos).all() or abs(data.time-tick*step_s) > 1e-8:
                        raise RuntimeError('MuJoCo did not reach the requested finite physics endpoint')
                    viewer.publish(snapshot(scene, model, data, scratch, args.scene, epoch, tick))
            except KeyboardInterrupt:
                pass
            print(f'Complete: {total} physics steps, epoch {epoch}, simulation time {data.time:.3f} s', flush=True)


if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=TITLES, default='cradle')
    parser.add_argument('--steps', type=int, help='stop after this many actual physics steps across all episodes')
    parser.add_argument('--port', type=int, default=0, help='loopback HTTP port; 0 chooses an available port')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--device-index', type=int, default=1)
    parser.add_argument('--library', type=Path,
                        default=repo/'artifacts/physics-vulkan-interop/p1-raster-20261124/build/phymeo_native.dll')
    run(parser.parse_args())
