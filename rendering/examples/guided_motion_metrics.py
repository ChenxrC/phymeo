"""Independent saved-image metrics, without renderer/filter implementations.

RGB8 is decoded using inverse IEC sRGB into *display-linear* values. This
cannot recover scene radiance, clipped highlights, or reference sampling error.
Masks use unshifted, same-pixel geometry. In particular, reveal is a screen-space
occlusion transition; the caller must separately prove actual blocker withdrawal
and same-pose image/geometry binding. No optical flow or wrapped indexing is used.
"""
from __future__ import annotations

import math
import numpy as np

FINGER_IDS = (156, 158)
MIN_PIXELS = 16
DEPTH_EDGE_RELATIVE = .02
THRESHOLDS = {'fingers': .80, 'edge': 1.05, 'reveal': 1.05}
SUBSETS = ('finger_156', 'finger_158', 'fingers', 'edge', 'reveal')
SCHEMA = 'phymeo-guided-motion-pose-metrics-v1'


def srgb_to_display_linear(rgb8):
    """Decode actual uint8 codes; no exposure inversion or fitted scaling."""
    a = np.asarray(rgb8)
    if a.dtype != np.uint8:
        raise TypeError('RGB must contain uint8 sRGB codes')
    x = a.astype(np.float64) / 255.
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def _geometry(instance_id, depth, valid, shape=None):
    ids, z, v = map(np.asarray, (instance_id, depth, valid))
    if ids.ndim != 2 or min(ids.shape) < 1:
        raise ValueError('geometry must be a nonempty HxW image')
    if shape is not None and ids.shape != shape:
        raise ValueError('geometry extent differs from images')
    if z.shape != ids.shape or v.shape != ids.shape:
        raise ValueError('geometry extents differ')
    if ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids > 0xffffffff):
        raise TypeError('instance IDs must be nonnegative uint32-range integers')
    if z.dtype.kind not in 'fiu':
        raise TypeError('depth must be numeric')
    if v.dtype.kind not in 'biu' or not np.all((v == 0) | (v == 1)):
        raise ValueError('VALID must contain only 0 or 1')
    z = z.astype(np.float64)
    evidence = (v == 1) & (ids > 0) & np.isfinite(z) & (z > 0)
    return ids, z, evidence


def _erode_one(mask):
    """Nine-neighbor erosion with false exterior, including image corners."""
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    out = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy:dy+h, dx:dx+w]
    return out


def build_masks(*, instance_id, depth, valid, previous_instance_id=None,
                previous_depth=None, previous_valid=None, blocker_ids=()):
    """Return independent boolean masks. Invalid geometry supplies no evidence.

    Edge includes the valid side(s) of 4-neighbor identity/validity transitions
    and same-object depth jumps exceeding 2% of the nearer positive depth. The
    image border itself supplies no neighbor and is not declared an edge.
    Reveal requires an actual prior valid blocker pixel and a currently farther,
    valid, positive-ID nonblocker surface; background is never a reveal target.
    """
    ids, z, good = _geometry(instance_id, depth, valid)
    masks = {f'finger_{i}': _erode_one(good & (ids == i)) for i in FINGER_IDS}
    masks['fingers'] = masks['finger_156'] | masks['finger_158']
    edge = np.zeros(ids.shape, dtype=bool)
    for a, b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        both = good[a] & good[b]
        jump = np.zeros(both.shape, dtype=bool)
        # Index before subtracting: invalid infinities/NaNs are not arithmetic inputs.
        za, zb = z[a][both], z[b][both]
        jump[both] = np.abs(za - zb) > DEPTH_EDGE_RELATIVE * np.minimum(za, zb)
        boundary = (good[a] != good[b]) | (both & (ids[a] != ids[b])) | jump
        edge[a] |= boundary & good[a]
        edge[b] |= boundary & good[b]
    masks['edge'] = edge
    previous = (previous_instance_id, previous_depth, previous_valid)
    supplied = tuple(x is not None for x in previous)
    if any(supplied) and not all(supplied):
        raise ValueError('previous ID, depth and VALID must be supplied together')
    blockers = tuple(blocker_ids)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           or not 0 < int(i) <= 0xffffffff for i in blockers):
        raise ValueError('blocker IDs must be positive integer identities')
    if len(set(blockers)) != len(blockers):
        raise ValueError('duplicate blocker identity')
    reveal = np.zeros(ids.shape, dtype=bool)
    if all(supplied):
        old_ids, old_z, old_good = _geometry(*previous, shape=ids.shape)
        candidates = good & old_good & np.isin(old_ids, blockers) & ~np.isin(ids, blockers)
        reveal[candidates] = z[candidates] > old_z[candidates]
    masks['reveal'] = reveal
    return masks


def _stats(error):
    count = len(error)
    sums = error.sum(axis=0)
    sse = float(np.square(error).sum())
    pixel_error = error.mean(axis=1) if count else np.empty(0)
    return {'squared_error_sum': sse, 'error_sum_rgb': sums.tolist(),
            'pixel_error_squared_sum': float(np.square(pixel_error).sum()),
            'rmse': math.sqrt(sse / (3 * count)) if count else None,
            'mean_bias': float(error.mean()) if count else None,
            'mean_bias_rgb': (sums / count).tolist() if count else None,
            'mean_bias_luminance': float(np.dot(sums / count, (.2126, .7152, .0722))) if count else None,
            'pixel_error_sd': float(np.std(pixel_error, ddof=1)) if count > 1 else None}


def _sign(raw_sse, filtered_sse):
    return 'improved' if filtered_sse < raw_sse else ('degraded' if filtered_sse > raw_sse else 'unchanged')


def _pose_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == '':
        raise ValueError('pose_id must be a nonempty string or a non-bool integer')
    return value


def score_pose(*, pose_id, raw_rgb8, filtered_rgb8, reference_rgb8,
               instance_id, depth, valid, previous_instance_id=None,
               previous_depth=None, previous_valid=None, blocker_ids=()):
    """Compare one registered pose; no image registration or mask selection by RGB."""
    _pose_id(pose_id)
    images = [np.asarray(x) for x in (raw_rgb8, filtered_rgb8, reference_rgb8)]
    shape = images[0].shape
    if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 1:
        raise ValueError('RGB must have shape HxWx3')
    if any(x.shape != shape for x in images):
        raise ValueError('raw, filtered and reference RGB extents must match')
    linear = [srgb_to_display_linear(x) for x in images]
    _geometry(instance_id, depth, valid, shape=shape[:2])
    masks = build_masks(instance_id=instance_id, depth=depth, valid=valid,
                        previous_instance_id=previous_instance_id, previous_depth=previous_depth,
                        previous_valid=previous_valid, blocker_ids=blocker_ids)
    subsets = {}
    for name, mask in masks.items():
        raw, filtered = [_stats((x - linear[2])[mask]) for x in linear[:2]]
        count = int(mask.sum())
        subsets[name] = {'count': count, 'raw': raw, 'filtered': filtered,
                         'filtered_equals_raw': bool(np.array_equal(images[0][mask], images[1][mask])) if count else None,
                         'rmse_change_sign': _sign(raw['squared_error_sum'], filtered['squared_error_sum']) if count else 'unobserved'}
    return {'schema': SCHEMA, 'pose_id': pose_id, 'width': shape[1], 'height': shape[0],
            'space': 'display-linear inverse IEC sRGB; not scene radiance',
            'subsets': subsets,
            'saturated_pixels': {name: int(np.any(x == 255, axis=2).sum()) for name, x in zip(('raw', 'filtered', 'reference'), images)},
            'uncertainty': {'reference_standard_error': None,
                            'reference_reason': 'One supplied reference image cannot establish independent-seed uncertainty.',
                            'pixel_error_sd': 'Sample SD of signed per-pixel mean RGB error; spatial description, not an IID standard error or confidence interval.',
                            'signs': 'Observed errors relative to the supplied noisy reference; not statistical significance.'}}


def _pool(rows):
    count = sum(row['count'] for row in rows)
    result = {'count': count}
    for mode in ('raw', 'filtered'):
        sse = sum(row[mode]['squared_error_sum'] for row in rows)
        sums = np.sum([row[mode]['error_sum_rgb'] for row in rows], axis=0) if rows else np.zeros(3)
        pixel_ss = sum(row[mode]['pixel_error_squared_sum'] for row in rows)
        mean = float(sums.sum() / (3 * count)) if count else None
        result[mode] = {'squared_error_sum': sse, 'error_sum_rgb': sums.tolist(),
                        'pixel_error_squared_sum': pixel_ss,
                        'rmse': math.sqrt(sse / (3 * count)) if count else None,
                        'mean_bias': mean, 'mean_bias_rgb': (sums / count).tolist() if count else None,
                        'mean_bias_luminance': float(np.dot(sums / count, (.2126, .7152, .0722))) if count else None,
                        'pixel_error_sd': math.sqrt(max(0., pixel_ss - count * mean * mean) / (count - 1)) if count > 1 else None}
    result['filtered_equals_raw'] = all(row['filtered_equals_raw'] for row in rows if row['count']) if count else None
    result['rmse_change_sign'] = _sign(result['raw']['squared_error_sum'], result['filtered']['squared_error_sum']) if count else 'unobserved'
    return result


def summarize_scores(scores, *, registered_pose_ids, required_reveal_pose_ids):
    """Frozen pooled gates; missing registered targets explicitly drop the claim.

    Call independently for each sample budget. Later unfiltered batches are not
    exempted from the fixed improvement gate: their observed equality is reported
    separately so callers can describe them as preservation controls.
    """
    scores = list(scores)
    registered = [_pose_id(x) for x in registered_pose_ids]
    reveals = [_pose_id(x) for x in required_reveal_pose_ids]
    if not registered or len(set(registered)) != len(registered) or len(set(reveals)) != len(reveals):
        raise ValueError('registered identities must be nonempty and unique; reveal identities unique')
    if not set(reveals) <= set(registered):
        raise ValueError('reveal poses must be registered poses')
    by_id = {}
    for score in scores:
        if score.get('schema') != SCHEMA:
            raise ValueError('unsupported pose score schema')
        identity = _pose_id(score['pose_id'])
        if identity not in registered or identity in by_id:
            raise ValueError('unexpected or duplicate scored pose')
        by_id[identity] = score
    missing = [f'missing_pose:{p}' for p in registered if p not in by_id]
    if not reveals:
        missing.append('no_registered_reveal_pose')
    for p in registered:
        if p in by_id:
            for subset in ('fingers', 'edge'):
                if by_id[p]['subsets'][subset]['count'] < MIN_PIXELS:
                    missing.append(f'pose_target:{p}:{subset}')
    for p in reveals:
        if p in by_id and by_id[p]['subsets']['reveal']['count'] < MIN_PIXELS:
            missing.append(f'reveal_target:{p}')
    pooled = {name: _pool([s['subsets'][name] for s in scores]) for name in SUBSETS}
    # Only predeclared reveal poses decide that gate. Other observed reveals
    # remain inspectable, but cannot supply or dilute the required evidence.
    pooled['reveal'] = _pool([s['subsets']['reveal'] for s in scores if s['pose_id'] in reveals])
    extras = [s for s in scores if s['pose_id'] not in reveals]
    extra_reveal = {'pose_ids': [s['pose_id'] for s in extras],
                    'pooled': _pool([s['subsets']['reveal'] for s in extras]),
                    'scope': 'Additional observed reveals; excluded from registered reveal acceptance'}
    for name in SUBSETS:
        if pooled[name]['count'] < MIN_PIXELS:
            missing.append(f'pooled_target:{name}')
    gates = {}
    for name, limit in THRESHOLDS.items():
        row = pooled[name]
        raw, filtered = row['raw']['rmse'], row['filtered']['rmse']
        sufficient = row['count'] >= MIN_PIXELS
        passed = bool(sufficient and filtered <= limit * raw)
        ratio = filtered / raw if sufficient and raw > 0 else (1. if sufficient and filtered == 0 else None)
        interpretation = 'relative_improvement' if limit < 1 else 'noninferiority'
        if name == 'fingers' and sufficient and raw == 0:
            passed = False
            ratio = None
            interpretation = 'exact_preservation_only' if filtered == 0 else 'introduced_error'
            if filtered == 0:
                missing.append('no_baseline_noise:fingers')
        gates[name] = {'passed': passed, 'limit': limit, 'rmse_ratio': ratio,
                       'interpretation': interpretation,
                       'zero_raw_error': raw == 0 if sufficient else None,
                       'rule': ('positive raw RMSE and filtered_rmse <= limit * raw_rmse' if name == 'fingers'
                                else 'filtered_rmse <= limit * raw_rmse; zero raw requires zero filtered')}
    success = not missing and all(g['passed'] for g in gates.values())
    return {'schema': 'phymeo-guided-motion-summary-v1',
            'status': 'insufficient_evidence' if missing else ('passed' if success else 'failed'),
            'quality_claim_supported': success, 'registered_pose_ids': registered,
            'required_reveal_pose_ids': reveals, 'insufficient_reasons': missing,
            'registration': {'finger_ids': list(FINGER_IDS), 'erosion_radius': 1,
                             'edge_connectivity': 4, 'depth_edge_relative': DEPTH_EDGE_RELATIVE,
                             'minimum_pixels': MIN_PIXELS, 'rmse_limits': dict(THRESHOLDS)},
            'pooled': pooled, 'extra_reveal': extra_reveal, 'gates': gates, 'poses': scores,
            'pooling': 'Pixel-weighted squared error; not mean of pose RMSE. Correlated pixels/poses are not treated as IID trials.',
            'evidence_boundary': 'Caller verifies matching poses, camera, geometry channels and actual blocker withdrawal. Metrics do not establish those identities or reference convergence.'}

