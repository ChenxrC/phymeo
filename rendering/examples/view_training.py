"""Start the local multi-camera training viewer on a real studio or Panda source."""
import argparse
from pathlib import Path
import webbrowser

from phymeo_rendering.viewer import ViewerServer


def show(name, *, library, device_index, motion=None, assets=None, area_grid=4,
         wide_size=(1440, 810), port=0, open_browser=True, live_fps=30):
    from _training_view_sources import open_training_source

    def source():
        options = dict(library=Path(library).resolve(strict=True), device_index=device_index,
                       motion=motion, assets=assets, area_grid=area_grid, wide_size=wide_size)
        if name == 'eight-live':
            options['live_fps'] = live_fps
        return open_training_source(name, **options)

    with ViewerServer(source, port=port) as server:
        print(server.url, flush=True)
        if open_browser:
            webbrowser.open(server.url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=('eight', 'eight-live', 'panda'), default='eight')
    parser.add_argument('--library', required=True, type=Path, help='Vulkan library next to matching SPIR-V')
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--motion', type=Path, help='eight-station archive directory or Panda trace JSON')
    parser.add_argument('--assets', type=Path, help='generated studio geometry cache')
    parser.add_argument('--area-grid', type=int, choices=(2,4,8), default=4)
    parser.add_argument('--port', type=int, default=0, help='loopback port, 0 chooses an available port')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args(argv)
    if args.motion and args.scene == 'eight-live':
        parser.error('--motion requires a replay scene')
    if not 0 <= args.port <= 65535 or args.device_index < 0:
        parser.error('port must be 0..65535 and device index nonnegative')
    return show(args.scene, library=args.library, device_index=args.device_index, motion=args.motion,
                assets=args.assets, area_grid=args.area_grid, port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    raise SystemExit(main())
