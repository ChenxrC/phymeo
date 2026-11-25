"""Encode one overview and eight synchronized cameras from each native RGB exposure."""
import json
import os
from fractions import Fraction
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from _studio_camera_gallery import OBJECTS, _font


def _paths(keys, names):
    if len(keys) != 8 or len(set(keys)) != 8 or set(keys) != set(names):
        raise ValueError('expected eight distinct worlds and one model name per world')
    paths = [Path('overview.mp4'), Path('multiview.mp4')]
    for key in keys:
        if type(key) is not int:
            raise ValueError('world keys must be integers')
        slug = re.sub(r'[^a-z0-9]+', '-', names[key].lower()).strip('-')
        if not slug:
            raise ValueError(f'model {key} is missing a usable name')
        paths.append(Path('cameras') / f'{key}-{slug}.mp4')
    return paths


def _binary(name):
    exe = shutil.which(name)
    if not exe:
        raise FileNotFoundError(f'the studio movie example requires {name} on PATH')
    return exe


def _size(size):
    if len(size) != 2 or any(type(v) is not int or v <= 0 or v % 2 for v in size):
        raise ValueError('H.264 yuv420p requires positive even image dimensions')
    return tuple(size)


def _rgb_image(frame, size):
    if isinstance(frame, Image.Image):
        if frame.mode != 'RGB' or frame.size != size:
            raise ValueError(f'expected a native {size} RGB image')
        return frame
    data = np.asarray(frame)
    if data.dtype != np.uint8 or data.shape != (size[1], size[0], 3):
        raise ValueError(f'expected a native {size} HWC uint8 RGB frame')
    return Image.fromarray(data)


class StudioMovieWriter:
    """All ten MP4s share one input packet, timestamp, and lossless set of crop locations."""

    def __init__(self, directory, keys, names, *, overview_size, camera_size, fps=30, contact_tasks=False):
        if type(fps) is not int or fps <= 0:
            raise ValueError('fps must be a positive integer')
        self.root = Path(directory)
        self.keys = tuple(keys)
        self.paths = _paths(self.keys, names)
        self.fps = fps
        self.overview_size = _size(overview_size)
        self.camera_size = _size(camera_size)
        ow, oh = self.overview_size
        w, h = self.camera_size
        gutter, title_height = 8, max(36, 2 * (w // 24))
        strip = title_height + (2 * (title_height // 4) if contact_tasks else 0)
        board_w, board_h = 4*w + 5*gutter, 2*(h+strip) + 3*gutter
        self.atlas = Image.new('RGB', (max(ow, board_w), oh + board_h), '#19282b')
        draw = ImageDraw.Draw(self.atlas)
        title_font = _font(max(14, title_height//2), bold=True)
        minor_font = _font(max(12, title_height*2//5))
        self.origins = []
        for i, key in enumerate(self.keys):
            col, row = i%4, i//4
            x, top = gutter + col*(w+gutter), oh + gutter + row*(h + strip + gutter)
            label = OBJECTS.get(names[key], names[key].replace('-', ' ').capitalize())
            title_y = top + (strip//3 if contact_tasks else strip//2)
            draw.text((x+6, title_y), label, fill='#f4f3ef', font=title_font, anchor='lm')
            draw.text((x+w-6, title_y), f'World {key}', fill='#c6d9d7', font=minor_font, anchor='rm')
            if contact_tasks:
                pushing = names[key].startswith('pusht-')
                action = 'Push into the amber T outline' if pushing else 'Pick up and place'
                ink = '#ffd37e' if pushing else '#c4e5dd'
                draw.text((x+6, top+strip*3//4), action, fill=ink, font=minor_font, anchor='lm')
            self.origins.append((x, top+strip))

        # Convert the RGB atlas once. The overview and eight single-camera movies
        # are native-resolution crops; only the labeled board is downsampled.
        self.filters = ['[0:v]scale=iw:ih:in_range=full:out_range=tv:out_color_matrix=bt709,'
                        'format=yuv420p,split=10' + ''.join(f'[s{i}]' for i in range(10))]
        self.filters.append(f'[s0]crop={ow}:{oh}:0:0[v0]')
        display_w = min(board_w, 1920)
        display_h = 2 * round(board_h * display_w / board_w / 2)
        self.filters.append(f'[s1]crop={board_w}:{board_h}:0:{oh},'
                            f'scale={display_w}:{display_h}:flags=lanczos,setsar=1[v1]')
        for i, (x, y) in enumerate(self.origins, 2):
            self.filters.append(f'[s{i}]crop={w}:{h}:{x}:{y}[v{i}]')
        self.frames = 0
        self.proc = None

    def __enter__(self):
        exe = _binary('ffmpeg')
        self.root.mkdir(exist_ok=True, parents=True)
        (self.root/'cameras').mkdir(exist_ok=True)
        for p in self.paths:
            if (self.root/p).exists():
                raise FileExistsError(self.root/p)
        self.temp = TemporaryDirectory(dir=self.root, prefix='encoding-')
        self.stage = Path(self.temp.name)
        try:
            (self.stage/'cameras').mkdir()
            self.log = (self.stage/'ffmpeg.log').open('w+b')
            cmd = [exe, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                   '-threads', '1', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                   '-video_size', f'{self.atlas.width}x{self.atlas.height}', '-framerate', str(self.fps),
                   '-i', 'pipe:0', '-filter_complex_threads', '1', '-filter_complex', ';'.join(self.filters)]
            for i, path in enumerate(self.paths):
                cmd += ['-map', f'[v{i}]', '-an', '-c:v', 'libx264', '-threads', '1',
                        '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
                        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'iec61966-2-1',
                        '-color_range', 'tv', '-movflags', '+faststart', str(self.stage/path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
            return self
        except BaseException:
            if hasattr(self, 'log'):
                self.log.close()
            self.temp.cleanup()
            raise

    def _error(self):
        self.log.seek(0)
        return RuntimeError('studio video encoder failed: ' + self.log.read().decode('utf-8', errors='replace'))

    def write(self, overview_rgb, cameras):
        if set(cameras) != set(self.keys):
            raise ValueError('one synchronized camera exposure is required for each world')
        self.atlas.paste(_rgb_image(overview_rgb, self.overview_size), (0, 0))
        for key, xy in zip(self.keys, self.origins, strict=True):
            self.atlas.paste(_rgb_image(cameras[key], self.camera_size), xy)
        try:
            self.proc.stdin.write(self.atlas.tobytes())
        except (BrokenPipeError, OSError) as exc:
            if self.proc.poll() is not None:
                raise self._error() from exc
            raise
        self.frames += 1

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.proc.stdin.close()
                if self.proc.wait() != 0:
                    raise self._error()
                if self.frames == 0:
                    raise ValueError('cannot publish an empty studio movie')
                # All outputs have finished encoding. Link within this volume to
                # publish without overwriting an existing or concurrent capture.
                published = []
                try:
                    for p in self.paths:
                        os.link(self.stage/p, self.root/p)
                        published.append(self.root/p)
                except BaseException:
                    for p in published:
                        p.unlink()
                    raise
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            self.log.close()
            self.temp.cleanup()


def probe_movies(directory, keys, names, *, fps=30, frames=900):
    """Read container timing; decode and count every frame and measure visible motion."""
    ffprobe, ffmpeg = _binary('ffprobe'), _binary('ffmpeg')
    result = {}
    for p in _paths(tuple(keys), names):
        path = str(Path(directory)/p)
        info = subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,pix_fmt,width,height,avg_frame_rate,duration_ts,time_base',
            '-of', 'json', path])
        stream, = json.loads(info)['streams']
        duration = int(stream['duration_ts']) * Fraction(stream['time_base'])
        if (stream['codec_name'] != 'h264' or stream['pix_fmt'] != 'yuv420p' or
                Fraction(stream['avg_frame_rate']) != fps or duration != Fraction(frames, fps)):
            raise ValueError(f'incorrect stream format or timing: {p} {stream}')
        decoded = subprocess.check_output([ffmpeg, '-v', 'error', '-xerror', '-nostdin', '-threads', '1',
            '-i', path, '-map', '0:v:0', '-an', '-filter_threads', '1', '-vf', 'scale=128:72:flags=area',
            '-fps_mode', 'passthrough', '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1'])
        gray = np.frombuffer(decoded, dtype=np.uint8).reshape(-1, 72, 128)
        if len(gray) != frames:
            raise ValueError(f'{p} decoded to {len(gray)} frames; expected {frames}')
        lag = max(1, fps//2)
        deltas = np.abs(gray[lag:].astype(np.int16) - gray[:-lag].astype(np.int16))
        # Fractions use raw decoded images. Robot-specific displacement is
        # checked independently against the common physics trajectory.
        result[p.as_posix()] = dict(codec=stream['codec_name'], pixel_format=stream['pix_fmt'],
            size=[stream['width'], stream['height']], decoded_frames=len(gray), fps=fps,
            duration_s=float(duration), motion_lag_frames=lag,
            max_changed_fraction_at_5_levels=float((deltas >= 5).mean(axis=(1, 2)).max()) if deltas.size else 0.)
    return result
