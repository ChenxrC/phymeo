"""Check a studio Vulkan device without preparing assets or running physics."""
import argparse
import json
from pathlib import Path

from .backends import BackendUnavailableError
from .photography import StudioRenderer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True, help='native renderer with its matching SPIR-V')
    parser.add_argument('--device-index', type=int, help='pin a Vulkan physical device (default: automatic)')
    parser.add_argument('--profile', choices=('auto', 'strict', 'compatible'), default='auto')
    parser.add_argument('--validation', action='store_true', help='also require the Vulkan validation layer')
    args = parser.parse_args(argv)
    if args.device_index is not None and not 0 <= args.device_index <= 2**32 - 1:
        parser.error('--device-index must be a nonnegative uint32')

    library = args.library.resolve()
    renderer = StudioRenderer(library_path=library, shader_directory=library.parent,
                              device_index=args.device_index, profile=args.profile, validation=args.validation)
    try:
        report = renderer.probe_device()
    except (RuntimeError, OSError) as error:
        category = 'native_installation' if isinstance(error, BackendUnavailableError) else 'native_runtime'
        report = getattr(error, 'report', {'category': category, 'error': str(error)})
        print(json.dumps({'ok': False, 'diagnostic': report}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({'ok': True, 'device': report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
