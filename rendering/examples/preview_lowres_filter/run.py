"""Run the unchanged 300-frame paired rigid driver with the explicit lowres owner."""
import argparse
import importlib
from pathlib import Path
import sys


def load_parent_preview(examples):
    root=Path(examples).resolve()
    expected=root/'preview_compact_rigid_history/preview.py'
    if not expected.is_file():raise FileNotFoundError(expected)
    families=('preview_compact_rigid_history','preview_compact_history','preview_compact','preview_sparse',
              'panda_render_performance','panda_realtime_film')
    for name,module in tuple(sys.modules.items()):
        if any(name==prefix or name.startswith(prefix+'.') for prefix in families):
            path=getattr(module,'__file__',None)
            if path is not None and not Path(path).resolve().is_relative_to(root):
                raise RuntimeError('foreign prepared driver already imported: '+name)
            paths=getattr(module,'__path__',())
            if any(not Path(p).resolve().is_relative_to(root) for p in paths):
                raise RuntimeError('foreign driver package already imported: '+name)
    # The requested checkout wins over caller PYTHONPATH or this author tree.
    sys.path.insert(0,str(root));importlib.invalidate_caches()
    module=importlib.import_module('preview_compact_rigid_history.preview')
    if Path(module.__file__).resolve()!=expected.resolve():
        raise RuntimeError('requested driver file was not imported')
    return module

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples',type=Path,required=True,help='frozen parent rendering/examples tree')
    parser.add_argument('--debug-output',type=Path,required=True,help='new low-guide audit directory, outside capture output')
    own,rest=parser.parse_known_args()
    if rest and rest[0]=='--':rest=rest[1:]
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

    from preview_lowres_filter.build_source import digest,save
    from preview_lowres_filter.runtime import backend_class,check_seal
    preview=load_parent_preview(own.examples)
    args=preview.parse_args(rest);check_seal(args.prepared)
    own.debug_output.mkdir(parents=True,exist_ok=False)
    save(own.debug_output/'producer.json',dict(schema='phymeo-preview-lowres-driver-v1',
        prepared=str(args.prepared.resolve()),driver_path=str(Path(preview.__file__).resolve()),driver_sha256=digest(preview.__file__),
        wrappers={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        captures='original current/rigid-history 300-frame driver; no new samples; debug outside capture timers'))
    preview.RigidHistoryBackend=backend_class(own.debug_output)
    preview.capture(args)

if __name__=='__main__':main()
