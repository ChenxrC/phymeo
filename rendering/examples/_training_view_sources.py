"""On-demand, single-camera training image sources for the local viewer."""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from phymeo_rendering._viewer_planes import channel_metadata, display_png as _image
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOTION = ROOT/'artifacts/physics-vulkan-interop/studio-contact-film-20261125/approved'
DEFAULT_ASSETS = ROOT/'artifacts/training-view/assets'
PLANES = {'rgb': Channel.RGB, 'depth': Channel.FORWARD_DEPTH, 'id': Channel.INSTANCE_ID}


def _channels():
    return channel_metadata(tuple(PLANES.values()))


class _TrainingSource:
    def describe(self):
        return dict(title=self.title, mode=self.mode, frame_count=self.frame_count, fps=self.fps,
                    cameras=list(self.cameras.values()), channels=_channels(), seekable=self.seekable,
                    max_step_frames=None if self.seekable else 1,
                    capture_ms_scope='pose update and requested native plane readback, excluding PNG encoding',
                    **self.provenance)

    def capture(self, camera_id, frame_index, channel_id):
        if self.closed:
            raise RuntimeError('training image source is closed')
        if camera_id not in self.cameras or channel_id not in PLANES:
            raise ValueError('unknown camera or image channel')
        if type(frame_index) is not int or not 0 <= frame_index < self.frame_count:
            raise ValueError(f'frame index must be in 0..{self.frame_count-1}')
        channel = PLANES[channel_id]
        started = perf_counter()
        shot = self._capture(camera_id, frame_index, channel)
        ms = 1000*(perf_counter()-started)
        data = _image(shot, channel, self.cameras[camera_id]['depth_range_m'])
        return data, frame_index, float(shot.time_s), ms

    def close(self):
        if not self.closed:
            self.closed = True
            self._session.close()


class _EightStudios(_TrainingSource):
    def __init__(self, args, live):
        from _eight_studio_view import StudioView
        self._session = StudioView(args)
        self.closed = False
        self.seekable = not live
        self.mode = 'live' if live else 'replay'
        self.title = '八工位实时训练' if live else '八工位训练回放'
        self.frame_count = len(self._session.frame_ticks)
        self.fps = self._session.fps
        self.provenance = dict(trajectory='MuJoCo live physics' if live else str(args.motion))
        self.cameras = {'overview': dict(id='overview', label='八工位总览', depth_range_m=[5.,12.])}
        self._indices = {'overview':0}
        for i, key in enumerate(self._session.world_keys, 1):
            name = f'station-{i}'
            self._indices[name] = i
            self.cameras[name] = dict(id=name, label=f'工位 {i} · {self._session.names[key]}',
                                      depth_range_m=[.1,2.5])

    def _capture(self, camera_id, frame_index, channel):
        return self._session.capture_at(self._indices[camera_id], frame_index, channel)


class _TwoPandas(_TrainingSource):
    def __init__(self, library, assets, device_index, motion):
        from _panda_scene import box, look_at
        from phymeo_rendering.adapters.panda_replay import PandaReplaySource
        from phymeo_rendering.assets import Material
        from phymeo_rendering.backends.native import NativeBackend
        from phymeo_rendering.panda_visuals import panda_visual_assets
        from phymeo_rendering.scene import DirectionalLight, Instance, Pose, VisualScene

        fixtures = ROOT/'rendering/tests/fixtures'
        film_trace = fixtures/'panda_film/trace.json'
        trace = Path(motion) if motion is not None else film_trace
        if trace.is_dir():
            trace = trace/'trace.json'
        replay = PandaReplaySource(VisualScene('recorded-panda', (), (), ()), fixtures/'panda_replay/panda.urdf', trace)
        self._records = tuple(replay.frames())
        source = replay.provenance(self._records[0])
        meshes, materials, parts = panda_visual_assets()
        assets.mkdir(parents=True, exist_ok=True)
        payload = box(assets, 'test-payload', source['geometry']['cube_full_size_m'])
        self._offsets = (Pose((0.,-.46,0.)), Pose((0.,.46,0.)))
        colors = (Material('test-red', (.78,.045,.025), 0.,.65), Material('test-blue', (.028,.042,.78), 0.,.65))
        instances = []
        for arm, material in enumerate(colors):
            instances.extend(replace(p, id=p.id+1000*arm, node_id=f'{arm}/{p.node_id}') for p in parts)
            instances.append(Instance(1+1000*arm, payload.id, material.id, f'{arm}/cube'))
        self._scene = VisualScene('paired-pandas:'+source['archive_sha256'], meshes+(payload,), materials+colors,
                                  tuple(instances), (DirectionalLight((.36,.48,-.8),(1.3,1.3,1.3)),
                                                     DirectionalLight((-.8,0.,-.6),(.9,.9,.9)),
                                                     DirectionalLight((0.,-.8,-.6),(.7,.7,.7))))
        self._overview_pose = look_at((1.77,-.28,1.21),(.15,0.,.43),(0.,0.,1.))
        self.cameras = {'overview':dict(id='overview', label='双 Panda 总览', depth_range_m=[.4,3.5])}
        for arm, label in enumerate(('A 腕部 · 固定记录', 'B 腕部 · 轨迹')):
            name = f'wrist-{arm}'
            self.cameras[name] = dict(id=name, label=label, depth_range_m=[.01,1.5])
        self.title, self.mode, self.seekable = '双 Panda 训练回放', 'replay', True
        self.frame_count = len(self._records)
        span = self._records[-1].time_s-self._records[0].time_s
        self.fps = 30 if trace.resolve() == film_trace else ((self.frame_count-1)/span if span else 1.)
        self.provenance = dict(trajectory=str(trace), archive_sha256=source['archive_sha256'],
                               description='A 固定首记录，B 回放原始采样。返回时间为 B 的物理出步时间。')
        self._session = NativeBackend(library_path=library, shader_directory=library.parent,
                                      device_index=device_index, validation=True, resident=True).open_scene(self._scene)
        self.closed = False

    def _capture(self, camera_id, frame_index, channel):
        from _panda_scene import cameras as panda_cameras
        from phymeo_rendering.cameras import Camera
        from phymeo_rendering.scene import FrameKey, NodePose, SceneFrame, compose_pose

        states = (self._records[0], self._records[frame_index])
        right = states[1]
        key = FrameKey(self._scene.key, 'shared-arm-scene', 0, frame_index, right.key.tick, 'authored')
        nodes = tuple(NodePose(f'{i}/{node.node_id}', compose_pose(self._offsets[i], node.world_pose))
                      for i, state in enumerate(states) for node in state.nodes)
        frame = SceneFrame(key, right.time_s, nodes)
        if camera_id == 'overview':
            camera = Camera(camera_id,key,self._overview_pose,960,720,850.,850.,480.,360.,.005,5.)
        else:
            arm = int(camera_id[-1])
            wrist = panda_cameras(states[arm])[1]
            camera = replace(wrist, id=camera_id, frame_key=key,
                             world_pose=compose_pose(self._offsets[arm],wrist.world_pose))
        shot = self._session.capture(frame, CaptureRequest(camera, RenderPath.RASTER, (channel,)))
        if (shot.camera, shot.frame_key, shot.time_s) != (camera, key, frame.time_s):
            raise RuntimeError('native Panda session returned a different exposure')
        return shot


def open_training_source(name, *, library, device_index=0, motion=None, assets=None, area_grid=4,
                         wide_size=(1440,810), station_size=(640,480), live_fps=30):
    """Open eight (archive), eight-live (forward-only), or panda (checked trajectory)."""
    library = Path(library).resolve(strict=True)
    if name == 'panda':
        return _TwoPandas(library, Path(assets) if assets else DEFAULT_ASSETS/'panda', device_index, motion)
    if name not in ('eight', 'eight-live'):
        raise ValueError("source must be 'eight', 'eight-live' or 'panda'")
    live = name == 'eight-live'
    if live and motion is not None:
        raise ValueError('eight-live advances physics; use eight to read a motion archive')
    archive = None if live else Path(motion) if motion is not None else DEFAULT_MOTION
    if archive is not None and archive.name == 'motion.npz':
        archive = archive.parent
    if archive is not None and not (archive/'motion.npz').is_file():
        raise FileNotFoundError(f'No studio motion archive in {archive}; pass --motion with its directory')
    assets = Path(assets) if assets else DEFAULT_ASSETS/'eight'
    args = SimpleNamespace(library=library, device_index=device_index, motion=archive, assets=assets,
                           area_grid=area_grid, wide_size=wide_size, station_size=station_size, live_fps=live_fps)
    return _EightStudios(args, live)
