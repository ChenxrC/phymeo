"""Apply primary panel quadrature to a copy of the native or compact source."""
import argparse
from hashlib import sha256
from pathlib import Path


STUDIO = (Path(__file__).resolve().parents[2] / 'src/phymeo_rendering/backends'
          / '_shaders/resident_studio.glsl')
# The frozen lf100 parent has the same studio integrator as the original
# product source. Do not replace an unknown local variant of that integrator.
PARENT_SHA256 = 'a77d07b8e03b5b3847e9ae81801af2818f951a14ae424f91cd9189b37ca86a8e'
MARKER_ID = 107
MARKER_NAME = 'PR_PRIMARY_PANEL_SEQUENCE_SAMPLES'


def apply(source_dir, samples=16):
    """Opt in to 8 or 16 primary samples; return the actual build contract."""
    if type(samples) is not int or samples not in (8, 16):
        raise ValueError('primary panel quadrature requires 8 or 16 samples')
    target = Path(source_dir).resolve() / '_shaders/resident_studio.glsl'
    original = target.read_text(encoding='utf-8')
    shared = STUDIO.read_text(encoding='utf-8')
    if original != shared and sha256(original.encode()).hexdigest() != PARENT_SHA256:
        raise ValueError('unknown or already configured studio integrator')
    if target == STUDIO:
        raise ValueError('apply to a separate source build directory')
    # Compact uses a dedicated primary draw and a separate low-rate ray draw.
    # The full progressive study uses the original 2x2 base configuration.
    # All other compilation modes preprocess to the unchanged shader path.
    selection = (
        '#if defined(PR_PREVIEW_SPARSE_MAIN) || '
        '(defined(PR_RESIDENT_PROGRESSIVE) && !defined(PR_PREVIEW_TILE) && '
        '(!defined(PR_STUDIO_AREA_GRID) || PR_STUDIO_AREA_GRID == 2))\n'
        f'#define PR_PRIMARY_PANEL_SEQUENCE {samples}\n'
        '#endif\n'
    )
    target.write_text(selection + shared, encoding='utf-8', newline='\n')
    return dict(primary_samples_per_panel=samples, secondary_samples_per_panel=4,
                brdf_directions=8, marker_id=MARKER_ID, marker_name=MARKER_NAME,
                motion='fixed in emitter space, current geometry, no direct-light history',
                progressive='within-cell shift per unchanged-state batch, reset on state change')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_dir', type=Path)
    parser.add_argument('--samples', type=int, choices=(8, 16), default=16)
    args = parser.parse_args()
    print(apply(args.source_dir, args.samples))
