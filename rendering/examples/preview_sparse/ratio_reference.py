"""CPU algebra oracle for the explicit finite-sample visibility-ratio option.

No ray tracer or shader import. Each sample is (sky_rgb, MIS_weight, outcome,
terminal_rgb), where outcome is miss/hit/clipped. Geometry clipping leaves the
unoccluded denominator in place and contributes no visible sky or terminal hit.
"""
import math


def estimate_ratio(sh, samples):
    def rgb(value):
        if len(value)!=3 or any(not math.isfinite(x) or x<0 for x in value):
            raise ValueError("expected finite nonnegative RGB")
        return tuple(float(x) for x in value)
    sh=rgb(sh)
    samples=tuple(samples)
    if not samples:
        raise ValueError("sample count must be positive")
    sky=[0.,0.,0.];miss=[0.,0.,0.];hit=[0.,0.,0.]
    total_weight=0.;visible_weight=0.
    for sky_value,weight,outcome,terminal in samples:
        sky_value=rgb(sky_value);terminal=rgb(terminal)
        if not math.isfinite(weight) or weight<0 or outcome not in ("miss","hit","clipped"):
            raise ValueError("invalid weighted outcome")
        total_weight+=weight
        for k in range(3):
            sky[k]+=sky_value[k]*weight
        if outcome=="miss":
            visible_weight+=weight
            for k in range(3):miss[k]+=sky_value[k]*weight
        elif outcome=="hit":
            for k in range(3):hit[k]+=terminal[k]*weight
    fallback=visible_weight/total_weight if total_weight>0 else 0.
    ratio=tuple(miss[k]/sky[k] if sky[k]>0 else fallback for k in range(3))
    hit_mean=tuple(value/len(samples) for value in hit)
    incoming=tuple(sh[k]*ratio[k]+hit_mean[k] for k in range(3))
    if any(not math.isfinite(x) for x in (*sky,*miss,*hit,*ratio,*incoming)):
        raise ValueError("nonfinite ratio estimator")
    return dict(incoming=incoming,ratio=ratio,sky_denominator=tuple(sky),
                hit_mean=hit_mean,fallback_visibility=fallback)
