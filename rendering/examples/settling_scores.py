"""Independent 4d saved-image settling scores, with frozen registered targets.

Inputs are displayed RGB8, not raw GPU means. Inverse IEC sRGB gives
DISPLAY-linear values, not scene radiance. References remain finite/noisy.
No renderer, denoiser, image registration, or threshold fitting is used here.
"""
from __future__ import annotations

import math
import numpy as np
import guided_motion_metrics as motion

POSES = (0, 15, 24)
STAGES = (1, 2, 4, 8, 16)
REVEAL_POSES = (15,)
SUBSETS = (*motion.SUBSETS, 'reveal_right_158', 'foreground')
EXPECTED_REFERENCE_SPP = 1024


def _keys(mapping, expected, name):
    if not isinstance(mapping, dict) or any(type(k) is not int for k in mapping) or set(mapping) != set(expected):
        raise ValueError(f'{name} requires exactly the registered integer keys {expected}')


def _rgb(value, shape=None):
    image = np.asarray(value)
    if image.dtype != np.uint8:
        raise TypeError('RGB inputs must be uint8 displayed sRGB codes')
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('RGB inputs must be nonempty HxWx3')
    if shape is not None and image.shape != shape:
        raise ValueError('matched pose RGB extents differ')
    return image


def _statistics(delta):
    count = len(delta)
    summed = delta.sum(axis=0)
    squared = float(np.square(delta).sum())
    pixel_error = delta.mean(axis=1) if count else np.empty(0)
    return {'squared_error_sum': squared, 'error_sum_rgb': summed.tolist(),
            'pixel_error_squared_sum': float(np.square(pixel_error).sum()),
            'rmse': math.sqrt(squared / (3 * count)) if count else None,
            'mean_bias': float(delta.mean()) if count else None,
            'mean_bias_rgb': (summed / count).tolist() if count else None,
            'mean_bias_luminance': float(np.dot(summed / count, (.2126, .7152, .0722))) if count else None,
            'pixel_error_sd': float(np.std(pixel_error, ddof=1)) if count > 1 else None}


def _delta(current, baseline, mask):
    # Convert only selected pixels. Neither caller arrays nor masks are modified.
    return _statistics(motion.srgb_to_display_linear(current[mask]) -
                       motion.srgb_to_display_linear(baseline[mask]))


def _pool_statistics(rows, counts):
    count = sum(counts)
    squared = sum(row['squared_error_sum'] for row in rows)
    summed = np.sum([row['error_sum_rgb'] for row in rows], axis=0) if rows else np.zeros(3)
    pixel_ss = sum(row['pixel_error_squared_sum'] for row in rows)
    mean = float(summed.sum() / (3 * count)) if count else None
    return {'squared_error_sum': squared, 'error_sum_rgb': summed.tolist(),
            'pixel_error_squared_sum': pixel_ss,
            'rmse': math.sqrt(squared / (3 * count)) if count else None,
            'mean_bias': mean, 'mean_bias_rgb': (summed / count).tolist() if count else None,
            'mean_bias_luminance': float(np.dot(summed / count, (.2126, .7152, .0722))) if count else None,
            'pixel_error_sd': math.sqrt(max(0., pixel_ss - count * mean * mean) / (count - 1)) if count > 1 else None}


def _selected_poses(subset):
    return REVEAL_POSES if subset in ('reveal', 'reveal_right_158') else POSES


def _progression(stage_rows, mode):
    """Observed reference-error rises, not claims of mathematical convergence."""
    adjacent = []
    for before, after in zip(STAGES, STAGES[1:]):
        a = stage_rows[str(before)]['fingers'][mode]['rmse']
        b = stage_rows[str(after)]['fingers'][mode]['rmse']
        observed = a is not None and b is not None
        unbounded = bool(observed and a == 0 and b > 0)
        relative = ((b - a) / a if a > 0 else 0.) if observed and not unbounded else None
        adjacent.append({'from_stage': before, 'to_stage': after,
                         'rmse_before': a, 'rmse_after': b,
                         'absolute_change': b - a if observed else None,
                         'relative_rise': relative, 'unbounded_relative_rise': unbounded,
                         'over_10_percent': bool(observed and b > 1.10 * a)})
    observed = [r for r in adjacent if r['absolute_change'] is not None]
    worst = max(observed, key=lambda r: r['absolute_change']) if observed else None
    max_rise = max(0., worst['absolute_change']) if worst else None
    return {'adjacent': adjacent,
            'any_fingers_rise_over_10_percent': any(r['over_10_percent'] for r in adjacent),
            'max_adjacent_rmse_rise': max_rise,
            'worst_transition': [worst['from_stage'], worst['to_stage']] if worst and max_rise > 0 else None,
            'interpretation': 'A distinct observed adverse-change outcome; later stages are not subjected to a fabricated strict-monotonicity gate.'}


def _gate(value, baseline, limit, *, count, require_reduction=False):
    sufficient = count >= motion.MIN_PIXELS and value is not None and baseline is not None
    interpretation = 'relative_reduction' if require_reduction else 'noninferiority'
    passed = bool(sufficient and value <= limit * baseline)
    ratio = value / baseline if sufficient and baseline > 0 else (1. if sufficient and value == 0 else None)
    zero_preservation = bool(sufficient and baseline == 0 and value == 0)
    if require_reduction and sufficient and baseline == 0:
        passed = False
        ratio = None
        interpretation = 'exact_preservation_only' if value == 0 else 'introduced_error'
    return {'passed': passed, 'value_rmse': value, 'baseline_rmse': baseline,
            'limit': limit, 'ratio': ratio, 'count': count,
            'interpretation': interpretation, 'zero_error_preservation': zero_preservation,
            'sufficient_target': sufficient}


def score_settling(*, old_rgb8, new_rgb8, reference_rgb8, geometry):
    """Score the fixed 3 poses x 5 sample budgets without changing registration.

    old/new: {pose_int: {batch_int: uint8 HxWx3}}.
    reference: {pose_int: uint8 HxWx3}, independently captured 1024 spp;
      sample count/seed and same-pose provenance must be verified by the caller.
    geometry: {pose_int: kwargs for guided_motion_metrics.build_masks};
      previous_* denotes the actual preceding MOTION frame, never a batch.

    Each stage is a still-pose comparison on unchanged supplied geometry. Signed
    brightness jumps are changes of displayed pixels, not evidence of a physical
    light/depth change. Exact n1/n16 identities and geometry preservation are
    separate device obligations, not inferred from these error scores.
    """
    for name, mapping in (('old', old_rgb8), ('new', new_rgb8), ('reference', reference_rgb8), ('geometry', geometry)):
        _keys(mapping, POSES, name)
    per_pose = {}
    masks_by_pose = {}
    for pose in POSES:
        _keys(old_rgb8[pose], STAGES, 'old stages')
        _keys(new_rgb8[pose], STAGES, 'new stages')
        reference = _rgb(reference_rgb8[pose])
        old = {n: _rgb(old_rgb8[pose][n], reference.shape) for n in STAGES}
        new = {n: _rgb(new_rgb8[pose][n], reference.shape) for n in STAGES}
        g = geometry[pose]
        masks = motion.build_masks(**g)
        if masks['fingers'].shape != reference.shape[:2]:
            raise ValueError('geometry and displayed image extents differ')
        ids, depth, valid = (np.asarray(g[k]) for k in ('instance_id', 'depth', 'valid'))
        masks['reveal_right_158'] = masks['reveal'] & (ids == 158)
        masks['foreground'] = (ids > 0) & (valid == 1) & np.isfinite(depth) & (depth > 0)
        masks_by_pose[pose] = masks
        stages = {}
        for n in STAGES:
            stages[str(n)] = {name: {'count': int(mask.sum()),
                                     'old': _delta(old[n], reference, mask),
                                     'new': _delta(new[n], reference, mask)}
                              for name, mask in masks.items()}
        transitions, from_n1 = {}, {}
        for mode, images in (('old', old), ('new', new)):
            transitions[mode] = {f'{a}->{b}': {name: {'count': int(mask.sum()), **_delta(images[b], images[a], mask)}
                                              for name, mask in masks.items()}
                                 for a, b in zip(STAGES, STAGES[1:])}
            from_n1[mode] = {str(n): {name: {'count': int(mask.sum()), **_delta(images[n], images[1], mask)}
                                     for name, mask in masks.items()} for n in STAGES}
        per_pose[str(pose)] = {'stages': stages, 'transitions': transitions, 'from_n1': from_n1,
                               'progression': {mode: _progression(stages, mode) for mode in ('old', 'new')},
                               'saturation_pixels': {
                                   'reference': int(np.any(reference == 255, axis=2).sum()),
                                   **{mode: {str(n): int(np.any(images[n] == 255, axis=2).sum()) for n in STAGES}
                                      for mode, images in (('old', old), ('new', new))}}}
    per_stage = {}
    for n in STAGES:
        per_stage[str(n)] = {}
        for subset in SUBSETS:
            rows = [per_pose[str(p)]['stages'][str(n)][subset] for p in _selected_poses(subset)]
            counts = [row['count'] for row in rows]
            per_stage[str(n)][subset] = {'count': sum(counts),
                                        **{mode: _pool_statistics([row[mode] for row in rows], counts)
                                           for mode in ('old', 'new')}}
    pooled_drift = {'adjacent': {}, 'from_n1': {}}
    for mode in ('old', 'new'):
        for section, destination in (('transitions', 'adjacent'), ('from_n1', 'from_n1')):
            pooled_drift[destination][mode] = {}
            keys = tuple(per_pose['0'][section][mode])
            for key in keys:
                pooled_drift[destination][mode][key] = {}
                for subset in SUBSETS:
                    rows = [per_pose[str(p)][section][mode][key][subset] for p in _selected_poses(subset)]
                    counts = [row['count'] for row in rows]
                    pooled_drift[destination][mode][key][subset] = {'count': sum(counts), **_pool_statistics(rows, counts)}
    missing = []
    for pose in POSES:
        for name in ('fingers', 'edge'):
            if int(masks_by_pose[pose][name].sum()) < motion.MIN_PIXELS:
                missing.append(f'pose_target:{pose}:{name}')
    for name in ('finger_156', 'finger_158', 'fingers', 'edge', 'reveal', 'reveal_right_158'):
        if per_stage['1'][name]['count'] < motion.MIN_PIXELS:
            missing.append(f'pooled_target:{name}')
    gates = {}
    n2 = per_stage['2']['fingers']
    gates['n2_vs_new_n1'] = _gate(n2['new']['rmse'], per_stage['1']['fingers']['new']['rmse'], 1.05, count=n2['count'])
    gates['n2_vs_old_n2'] = _gate(n2['new']['rmse'], n2['old']['rmse'], .80, count=n2['count'], require_reduction=True)
    if gates['n2_vs_old_n2']['interpretation'] == 'exact_preservation_only':
        missing.append('no_baseline_noise:old_n2_fingers')
    for n in STAGES[1:]:
        for name in ('edge', 'reveal'):
            row = per_stage[str(n)][name]
            gates[f'{name}_n{n}_vs_old'] = _gate(row['new']['rmse'], row['old']['rmse'], 1.05, count=row['count'])
    progression = {mode: _progression(per_stage, mode) for mode in ('old', 'new')}
    for mode in ('old', 'new'):
        brightness = [r['foreground']['mean_bias_luminance'] for r in pooled_drift['adjacent'][mode].values()]
        brightness = [x for x in brightness if x is not None]
        progression[mode]['max_adjacent_absolute_luminance_jump'] = max(map(abs, brightness)) if brightness else None
        progression[mode]['max_adjacent_luminance_increase'] = max(0., max(brightness)) if brightness else None
    passed = not missing and all(g['passed'] for g in gates.values())
    return {'schema': 'phymeo-guided-settling-scores-v1',
            'status': 'insufficient_evidence' if missing else ('passed' if passed else 'failed'),
            'quality_claim_supported': passed, 'insufficient_reasons': missing,
            'space': 'display-linear inverse IEC sRGB; not scene radiance or raw GPU mean',
            'registration': {'poses': list(POSES), 'stages': list(STAGES), 'required_reveal_poses': list(REVEAL_POSES),
                             'expected_reference_spp': EXPECTED_REFERENCE_SPP, 'minimum_pixels': motion.MIN_PIXELS,
                             'n2_vs_n1_limit': 1.05, 'n2_vs_old_limit': .80, 'edge_reveal_limit': 1.05,
                             'adverse_adjacent_rise_fraction': .10,
                             'mask_source': 'guided_motion_metrics.build_masks, unchanged fixed IDs/erosion/edge/reveal definitions'},
            'per_pose': per_pose, 'per_stage': per_stage, 'pooled_drift': pooled_drift,
            'gates': gates, 'progression': progression,
            'geometry_scope': 'Same supplied pose geometry for all batch stages; caller verifies actual unchanged depth/ID/VALID, n1/n16 identity and true prior-motion blocker binding.',
            'uncertainty': {'reference_standard_error': None,
                            'reference_scope': 'One independently sampled finite 1024-spp image per pose; caller binds actual seeds/counts. It is not a noiseless truth.',
                            'pixel_error_sd': 'Spatial SD of signed pixel-mean RGB errors, not an IID SEM; no statistical significance is inferred from correlated pixels.',
                            'adverse_rises': 'Reported relative to the finite reference; brightness jumps are signed displayed changes at fixed geometry, not physical illumination changes.'},
            'pooling': 'Sum pixel squared errors divided by 3*pixel count, then square root. Reveal/right158 pools only pose15; extra observations remain per_pose. n4/n8/n16 finger progression is measured, not forced monotone.'}
