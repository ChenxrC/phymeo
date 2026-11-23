"""Run the unchanged 300-frame paired rigid driver with the explicit lowres owner."""
import argparse
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples',type=Path,required=True,help='frozen parent rendering/examples tree')
    parser.add_argument('--debug-output',type=Path,required=True,help='new low-guide audit directory, outside capture output')
    own,rest=parser.parse_known_args()
    if rest and rest[0]=='--':rest=rest[1:]
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.append(str(own.examples.resolve()))
    from preview_lowres_filter.build_source import digest,save
    from preview_lowres_filter.runtime import backend_class,check_seal
    from preview_compact_rigid_history import preview
    args=preview.parse_args(rest);check_seal(args.prepared)
    own.debug_output.mkdir(parents=True,exist_ok=False)
    save(own.debug_output/'producer.json',dict(schema='phymeo-preview-lowres-driver-v1',
        prepared=str(args.prepared.resolve()),driver_path=str(Path(preview.__file__).resolve()),driver_sha256=digest(preview.__file__),
        wrappers={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        captures='original current/rigid-history 300-frame driver; no new samples; debug outside capture timers'))
    preview.RigidHistoryBackend=backend_class(own.debug_output)
    preview.capture(args)

if __name__=='__main__':main()
