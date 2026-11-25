"""Watch eight manipulation worlds in a desktop window. Keys 0–8, Space and Escape."""
import argparse
from collections import deque
from math import ceil
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event, Thread
from time import perf_counter
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk


VIEWS = ('总览', '红球', '蓝盒', '绿块', '黄胶囊', '阶梯块', '标准 T', '宽 T', '窄 T')


def _render_loop(args, requests, replies, stopping, factory):
    backend = None
    try:
        if factory is None:
            from _eight_studio_view import StudioView
            factory = StudioView
        backend = factory(args)
        replies.put(('ready', int(backend.last_tick)))
        while not stopping.is_set():
            request = requests.get()
            if request is None or stopping.is_set():
                break
            index, tick, step_index, deadline, cancelled = request
            if cancelled.wait(max(0., deadline-perf_counter())):
                replies.put(('cancelled',))
                continue
            if stopping.is_set():
                break
            start = perf_counter()
            rgb = backend.capture_view(index, tick)
            duration = (perf_counter()-start)*1000.
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError('expected H×W×3 uint8 RGB from the studio renderer')
            # Take ownership before the backend reuses a capture buffer.
            source_tick = int(getattr(backend,'physics_tick',tick))
            image = Image.frombytes('RGB', (rgb.shape[1],rgb.shape[0]), np.ascontiguousarray(rgb))
            replies.put(('frame', index, tick, step_index, source_tick, image, duration, perf_counter()-start))
    except Exception as error:
        replies.put(('error', f'{type(error).__name__}: {error}'))
    finally:
        try:
            if backend is not None:
                backend.close()
        except Exception as error:
            replies.put(('error', f'Closing studio: {error}'))
        replies.put(('stopped',))


class StudioWindow:
    """Tk owns the screen. One worker owns physics, Vulkan, and at most one capture."""

    def __init__(self, root, args, *, view_factory=None):
        self.root, self.args = root, args
        self.requests, self.replies = Queue(), Queue()
        self.stopping = Event()
        self.ready = self.busy = self.paused = self.finished = False
        self.error = None
        self.index = self.tick = self.step_index = self.presented_frames = 0
        self.last_tick = None
        self.last_image = self.last_presented = self._photo = None
        self._back_photo = self._cancel_request = self._poll_after = None
        self._painted = None
        self._deadline = 0.
        self._expected = 0.
        self._worker_times = {}
        self._history = deque(maxlen=24)
        root.title('Phymeo · 八工位实时查看')
        width, height = args.wide_size
        root.geometry(f'{min(width,root.winfo_screenwidth()-80)}x{min(height+96,root.winfo_screenheight()-110)}')
        toolbar = ttk.Frame(root, padding=4)
        toolbar.pack(side='top', fill='x')
        self.choice = tk.StringVar(root, '0  总览')
        selector = ttk.Combobox(toolbar, state='readonly', width=15, textvariable=self.choice,
                                values=[f'{i}  {name}' for i,name in enumerate(VIEWS)], takefocus=False)
        selector.pack(side='left', padx=3)
        selector.bind('<<ComboboxSelected>>', lambda _: self.select_view(selector.current()))
        self.pause_button = ttk.Button(toolbar, text='暂停', command=self.toggle_pause, takefocus=False)
        self.pause_button.pack(side='left', padx=5)
        ttk.Label(toolbar, text='0 总览　1–8 工位　空格 暂停/继续　Esc 退出').pack(side='left', padx=8)
        self.status = tk.StringVar(root, '正在初始化仿真与渲染设备…')
        ttk.Label(root, textvariable=self.status, padding=5).pack(side='bottom', fill='x')
        self.canvas = tk.Canvas(root, background='#162123', borderwidth=0, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self._item = self.canvas.create_image(0,0,anchor='center')
        self._resize = None
        self.canvas.bind('<Configure>', self._on_resize)
        root.bind('<KeyPress>', self._key)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.worker = Thread(target=_render_loop, name='studio-render', args=(
            args,self.requests,self.replies,self.stopping,view_factory))
        self.worker.start()
        root.after_idle(root.focus_set)
        self._schedule_poll(5)

    def _key(self, event):
        k = event.keysym.removeprefix('KP_')
        if k in tuple('012345678'):
            self.select_view(int(k))
        elif k == 'space':
            self.toggle_pause()
        elif k == 'Escape':
            self.close()
        else:
            return
        return 'break'

    def select_view(self, index):
        if type(index) is not int or not 0 <= index < len(VIEWS):
            raise ValueError('view index must be 0 through 8')
        if index != self.index and not self.stopping.is_set():
            self.index = index
            self.choice.set(f'{index}  {VIEWS[index]}')
            self.canvas.itemconfigure(self._item, state='normal' if self._painted==(index,self.tick) else 'hidden')
            self.status.set(f'正在切换到{VIEWS[index]}…')
            self.root.focus_set()
            self._change_request()

    def toggle_pause(self):
        if self.stopping.is_set() or (self.ready and self.tick==self.last_tick):
            return
        self.paused = not self.paused
        self._history.clear()
        self.pause_button.configure(text='继续' if self.paused else '暂停')
        self._status()
        self._change_request()

    def _change_request(self):
        if self.finished:
            return
        if self._cancel_request is not None:
            self._cancel_request.set()
        self._deadline = perf_counter()
        self._schedule_poll(0)

    def _schedule_poll(self, delay):
        if self._poll_after is not None:
            self.root.after_cancel(self._poll_after)
        self._poll_after = self.root.after(delay,self._poll)

    def _on_resize(self, _):
        if self._resize is not None:
            self.root.after_cancel(self._resize)
        self._resize = self.root.after(75,self._redraw)

    def _redraw(self):
        self._resize = None
        if self.last_image is None or self.stopping.is_set() or self._painted != (self.index,self.tick):
            return
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        image = self.last_image
        scale = min(w/image.width, h/image.height)
        if 1. <= scale < 1.03:
            scale = 1.
        size = (max(1,round(image.width*scale)), max(1,round(image.height*scale)))
        if size != image.size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        # Upload into the hidden buffer. The Tk image currently on screen
        # stays intact until the canvas switches to the completed upload.
        back = self._back_photo
        if back is None or (back.width(),back.height()) != image.size:
            back = ImageTk.PhotoImage(image, master=self.canvas)
        else:
            back.paste(image)
        self.canvas.coords(self._item,w/2,h/2)
        self.canvas.itemconfigure(self._item,image=back,state='normal')
        self._back_photo,self._photo = self._photo,back

    def _status(self):
        if not self.ready or self.error or self.stopping.is_set():
            return
        mode = '现场仿真' if self.args.motion is None else '轨迹回放'
        if self.tick==self.last_tick:
            mode += ' · 已到片尾'
        elif self.paused:
            mode += ' · 已暂停'
        stats = self.last_presented
        detail = '' if not stats else (f"　{stats['display_fps']:.1f} fps　时间倍率 {stats['sim_rate']:.2f}×"
                                       f"　采集 {stats['render_ms']:.1f} ms　绘窗 {stats['paint_ms']:.1f} ms")
        shown_tick = stats['tick'] if stats else self.tick
        self.status.set(f'{mode}　{VIEWS[self.index]}　{shown_tick/1000:.3f} s{detail}')

    def _request_next(self):
        if not self.ready or self.busy or self.error or self.stopping.is_set():
            return
        now = perf_counter()
        if self._painted != (self.index,self.tick):
            tick,step,deadline = self.tick,self.step_index,now
        elif not self.paused and self.tick < self.last_tick:
            step = self.step_index+1
            tick = min(self.last_tick,round(step*1000/self.args.fps))
            deadline = max(now,self._deadline)
        else:
            return
        self._cancel_request = Event()
        self.busy = True
        self._deadline = deadline+1/self.args.fps
        self._expected = deadline+self._worker_times.get(self.index,.025)
        self.requests.put((self.index,tick,step,deadline,self._cancel_request))

    def _poll(self):
        self._poll_after = None
        while True:
            try:
                kind,*values = self.replies.get_nowait()
            except Empty:
                break
            if kind=='ready':
                self.ready, self.last_tick = True, values[0]
            elif kind=='error':
                self.error = values[0]
                self.status.set(self.error)
                print(self.error,file=sys.stderr,flush=True)
            elif kind=='stopped':
                self.finished = True
                if self.stopping.is_set() or self.args.smoke_frames:
                    self.root.destroy()
                return
            elif kind=='cancelled':
                self.busy = False
                self._cancel_request = None
            elif kind=='frame':
                index,self.tick,self.step_index,source_tick,image,duration,worker_time = values
                self.busy = False
                self._cancel_request = None
                self._worker_times[index] = min(worker_time, .050)
                if index!=self.index or self.stopping.is_set():
                    continue
                self._painted = (index,self.tick)
                self.last_image = image
                # Reserve at most one next capture *before* uploading this
                # frame to Tk. The worker sleeps until its cadence deadline.
                if not self.args.smoke_frames or self.presented_frames+1 < self.args.smoke_frames:
                    self._request_next()
                if self._resize is not None:
                    self.root.after_cancel(self._resize)
                start = perf_counter()
                self._redraw()
                self.canvas.update_idletasks()
                now = perf_counter()
                self._history.append((now,source_tick))
                t0,k0 = self._history[0]
                span = now-t0
                self.presented_frames += 1
                self.last_presented = dict(index=index,tick=source_tick,requested_tick=self.tick,
                    render_ms=duration,worker_ms=worker_time*1000,paint_ms=(now-start)*1000,
                    display_fps=(len(self._history)-1)/span if span else 0.,
                    sim_rate=(source_tick-k0)/1000/span if span else 0.,frames=self.presented_frames)
                if self.tick==self.last_tick:
                    self.pause_button.configure(text='片尾',state='disabled')
                self._status()
                self.root.event_generate('<<StudioFramePresented>>',when='tail')
                if self.args.smoke_frames and self.presented_frames>=self.args.smoke_frames:
                    self.root.after_idle(self.close)
                elif self.args.smoke_frames and self.tick==self.last_tick:
                    self.error = f'End of timeline after {self.presented_frames} frames, before the requested smoke count'
                    print(self.error,file=sys.stderr,flush=True)
                    self.root.after_idle(self.close)
                # Yield to Tk input even when the next result is already here.
                break
        smoke_done = self.args.smoke_frames and self.presented_frames>=self.args.smoke_frames
        if not smoke_done:
            self._request_next()
        # Check close to the expected completion. A slow device or scene load
        # backs off; paused and idle windows do not run a high-frequency timer.
        delay = 40
        if self.busy:
            remaining = self._expected-perf_counter()
            delay = min(40,ceil(remaining*1000)) if remaining>0 else min(20,2+int(-remaining*30))
        if not self.replies.empty():
            delay = 0
        elif self.ready and not self.busy and not self.error and not self.stopping.is_set():
            # Paused, at the end, or waiting for smoke close. Input wakes us.
            return
        self._schedule_poll(delay)

    def close(self):
        if not self.stopping.is_set():
            self.stopping.set()
            if self._cancel_request is not None:
                self._cancel_request.set()
            self.requests.put(None)
            self.root.withdraw()
        if self.finished:
            self.root.destroy()
        else:
            self._schedule_poll(1)


def _size(value):
    try:
        w,h = map(int,value.lower().split('x'))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('use WIDTHxHEIGHT, for example 1280x720') from exc
    if not 1<=w<=4096 or not 1<=h<=4096:
        raise argparse.ArgumentTypeError('width and height must each be in 1..4096')
    return w,h


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True, type=Path, help='native Vulkan library alongside the resident SPIR-V files')
    p.add_argument('--device-index',type=int,default=0)
    p.add_argument('--area-grid',type=int,choices=(2,4,8),default=4)
    p.add_argument('--wide-size',type=_size,default=(1440,810))
    p.add_argument('--assets',type=Path,default=Path('artifacts/eight-studio-view/assets'))
    p.add_argument('--motion',type=Path,help='replay an existing motion archive instead of stepping live physics')
    p.add_argument('--fps',type=int,default=30,help='target refresh rate')
    p.add_argument('--smoke-frames',type=int,metavar='N',help='close after N actual window frames')
    p.add_argument('--web',action='store_true',help='open the multi-camera RGB/depth/ID browser viewer')
    p.add_argument('--port',type=int,default=0,help='local web viewer port (0 chooses an available port)')
    args = p.parse_args(argv)
    if not 1<=args.fps<=1000 or args.device_index<0 or (args.smoke_frames is not None and args.smoke_frames<1):
        p.error('fps must be 1..1000, device index nonnegative and smoke-frame count positive')
    if args.web:
        if args.smoke_frames or not 0 <= args.port <= 65535:
            p.error('--web requires a valid port and does not accept --smoke-frames')
        from view_training import show
        return show('eight' if args.motion else 'eight-live', library=args.library, device_index=args.device_index,
                    motion=args.motion, assets=args.assets, area_grid=args.area_grid, wide_size=args.wide_size,
                    port=args.port, live_fps=args.fps)
    root = tk.Tk()
    window = StudioWindow(root,args)
    try:
        root.mainloop()
    finally:
        window.stopping.set()
        if window._cancel_request is not None:
            window._cancel_request.set()
        window.requests.put(None)
        window.worker.join()
    if args.smoke_frames and window.last_presented:
        print('Presented window frames:', window.last_presented, flush=True)
    return int(window.error is not None)


if __name__=='__main__':
    raise SystemExit(main())
