"""On-demand Vulkan cameras on eight live studio mechanisms or one accepted pose archive."""
from bisect import bisect_right
from contextlib import ExitStack, closing
from dataclasses import replace
from pathlib import Path

import mujoco as mj
import numpy as np

from _eight_studio_motion import _model, _schedule, load_motion
from _eight_studio_set import build_eight_studio, overview_lighting
from _heterogeneous_camera_worlds import WORLD_KEYS
from _manipulation_photo_studio import studio_lighting
from _studio_grasp_control import grasp_setup
from _studio_pusht_control import PushController
from phymeo.training.manipulation_tasks import H, manipulation_scenes
from record_eight_studio import CAMERA_SIZE, build_scenes, local_frame, renderer, station_camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath


class _LiveWorlds:
    """All controllers share an integer clock, independently of the selected view."""

    def __init__(self):
        self.tick = 0
        samples, self.last_tick = _schedule(30, 30)
        self.models, self.data, self.controllers, self.kinematic_scratch = {}, {}, {}, {}
        self.report = {'worlds': []}
        self.exposures = set(samples.tolist())
        for key, scene in zip(WORLD_KEYS, manipulation_scenes(), strict=True):
            model = _model(scene)
            if model.opt.timestep != H:
                raise ValueError('the studio clock expects 1 ms model steps')
            state = mj.MjData(model)
            mj.mj_forward(model, state)
            controller = grasp_setup(scene, model, state) if scene.finger_geom_names else PushController(scene, model, state)
            self.models[key], self.data[key], self.controllers[key] = model, state, controller
            self.kinematic_scratch[key] = mj.MjData(model)
            self.report['worlds'].append(dict(world=key, model_key=scene.key, task=controller.setup))

    def advance(self, target):
        while self.tick < target:
            # The recorder refreshes kinematics at its canonical 30 Hz sample
            # boundaries before reading the four-tick held motor control.
            steps = min(4 - self.tick % 4, target - self.tick)
            for offset in range(1, steps):
                if self.tick + offset in self.exposures:
                    steps = offset
                    break
            for key, model in self.models.items():
                state = self.data[key]
                if self.tick in self.exposures:
                    mj.mj_kinematics(model, state)
                if self.tick % 4 == 0:
                    self.controllers[key].apply(state, self.tick * H)
                mj.mj_step(model, state, nstep=steps)
            self.tick += steps

    def observation(self, key):
        # Query a scratch state. A UI exposure at an arbitrary millisecond
        # must not alter the state caches read by the next feedback update.
        state, model = self.data[key], self.models[key]
        if not np.isfinite(state.time) or abs(state.time - self.tick * H) > 1.e-8:
            raise RuntimeError(f'world {key} failed to reach the requested simulation time')
        scratch = self.kinematic_scratch[key]
        scratch.qpos[:] = state.qpos
        mj.mj_kinematics(model, scratch)
        rows = np.concatenate((scratch.xmat.reshape(-1, 3, 3), scratch.xpos[:, :, None]), axis=2)
        return rows, float(state.time)


class StudioView:
    """Single-worker capture source; index zero is the overview, 1–8 are local cameras."""

    world_keys = WORLD_KEYS

    def __init__(self, args):
        self._stack = ExitStack()
        self._sessions = {}
        self._closed = False
        self.capture_counts = {i: 0 for i in range(1 + len(WORLD_KEYS))}
        self.physics_tick = 0
        self._requested_tick = -1
        self._wide_size = tuple(args.wide_size)
        self._station_size = tuple(getattr(args, 'station_size', None) or CAMERA_SIZE)
        for size in (self._wide_size, self._station_size):
            if len(size) != 2 or any(type(n) is not int or n < 1 for n in size):
                raise ValueError('positive integer width and height required')
        self._assets = Path(args.assets)
        self._live = args.motion is None
        self._source = _LiveWorlds() if self._live else load_motion(Path(args.motion))
        if set(self._source.models) != set(WORLD_KEYS):
            raise ValueError('the viewer needs all eight registered studio worlds')
        self.last_tick = self._source.last_tick if self._live else int(self._source.ticks[-1])
        self.fps = getattr(args, 'live_fps', 30) if self._live else self._source.report['fps']
        if self._live:
            if type(self.fps) is not int or not 1 <= self.fps <= 1000:
                raise ValueError('live camera fps must be an integer in 1..1000')
            self.frame_ticks = (*_schedule(self.fps, 30)[0].tolist(), self.last_tick)
        else:
            self.frame_ticks = tuple(int(tick) for tick in self._source.ticks)
        self._end_on = {r['world'] for r in self._source.report['worlds']
                        if (r.get('task') or {}).get('worktop_contact')}
        self._scenes, self._nodes, self.names, _ = build_scenes(self._source, self._assets)
        self._overview = None
        self._wide_camera = None
        self._renderers = renderer(args, overview_lighting()), renderer(args, studio_lighting())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            self._stack.close()
            self._sessions.clear()

    def _frame(self, key, sample):
        if self._live:
            rows, time = self._source.observation(key)
        else:
            rows, time = self._source.rows[key][sample], float(self._source.times[sample])
        return local_frame(self._scenes[key], self._nodes[key], rows, key, self.physics_tick, time)

    def capture_view(self, index, tick):
        """Render one RGB exposure. Archive playback holds the last available sample."""
        shot = self._capture_tick(index, tick, Channel.RGB, seek=False)
        return np.frombuffer(shot.planes[0].data, np.uint8).reshape(shot.camera.height, shot.camera.width, 3)

    def capture_at(self, index, frame_index, channel):
        """Render one native plane at an exact archive index or current/next live exposure."""
        if type(frame_index) is not int or not 0 <= frame_index < len(self.frame_ticks):
            raise ValueError('frame index is outside the studio timeline')
        if self._live:
            current = max(0, bisect_right(self.frame_ticks, self._requested_tick)-1)
            if frame_index not in (current, current+1):
                raise ValueError(f'live physics is at frame {current}; request the current or next frame')
        return self._capture_tick(index, self.frame_ticks[frame_index], channel, seek=not self._live)

    def _capture_tick(self, index, tick, channel, *, seek):
        if self._closed:
            raise RuntimeError('studio viewer is closed')
        if type(index) is not int or index not in self.capture_counts:
            raise ValueError('choose overview 0 or station 1 through 8')
        if channel not in (Channel.RGB, Channel.FORWARD_DEPTH, Channel.INSTANCE_ID):
            raise ValueError('unsupported studio image channel')
        last = self.last_tick if seek else 30000
        if type(tick) is not int or not 0 <= tick <= last or (not seek and tick < self._requested_tick):
            raise ValueError('invalid studio tick or attempted backwards advancement')
        self._requested_tick = tick
        sample = None
        if self._live:
            self._source.advance(tick)
            self.physics_tick = self._source.tick
        else:
            sample = int(np.searchsorted(self._source.ticks, tick, side='right')) - 1
            self.physics_tick = int(self._source.ticks[sample])

        if index == 0:
            if self._overview is None:
                self._overview = build_eight_studio(self._scenes, self._nodes, self._source.models,
                                                    self._assets / 'overview', end_on=self._end_on)
            frames = {key: self._frame(key, sample) for key in WORLD_KEYS}
            scene = self._overview.scene
            frame = self._overview.frame(frames)
            # The benches, lens and camera are fixed. Only the exposure key changes.
            if self._wide_camera is None:
                self._wide_camera = self._overview.camera(frame.key, *self._wide_size)
            camera = replace(self._wide_camera, frame_key=frame.key)
        else:
            key = WORLD_KEYS[index - 1]
            scene = self._scenes[key]
            frame = self._frame(key, sample)
            camera = station_camera(frame, key, self._end_on, self._station_size)
        if index not in self._sessions:
            backend = self._renderers[0 if index == 0 else 1]
            session = self._stack.enter_context(closing(backend.open_scene(scene)))
            self._sessions[index] = session
        shot = self._sessions[index].capture(frame, CaptureRequest(camera, RenderPath.RASTER, (channel,)))
        if (shot.camera, shot.frame_key, shot.time_s) != (camera, frame.key, frame.time_s):
            raise RuntimeError('native studio returned a different exposure')
        self.capture_counts[index] += 1
        return shot
