"""Support-feature and exact time brackets for curved solids translating past a plane."""

from dataclasses import replace
from fractions import Fraction as F
from math import inf, nextafter, ulp

from ..constraints.collision_shapes import CollisionMotion
from ._curved_solids import CurvedEvent, _rotation, _static, _support_bounds
from ._round_sweep import _translation, _vector
from .collision_witness import CollisionSweep


def _outward_time(values):
    lo, hi = sorted(values)
    lo, hi = max(F(0), lo), min(F(1), hi)
    if lo > hi:
        raise OverflowError("curved plane time outside the step")
    a, b = float(lo), float(hi)
    return (nextafter(a,-inf) if F(a)>lo else a,
            nextafter(b,inf) if F(b)<hi else b)


def translating_curved_plane(shape, plane, motion, plane_motion, curved_first):
    n = _vector(plane.normal)
    center = _vector(motion.start[0])
    anchor = _vector(plane_motion.start[0])
    dc, dp = _translation(motion), _translation(plane_motion)
    frame = _rotation(shape.curved, motion)
    reach = _support_bounds(shape.kind, _vector(shape.curved.dimensions_m), -(frame.T @ n))
    height0, slope = (center-anchor) @ n, (dc-dp) @ n

    def sample(time):
        t=F(time)
        cp=CollisionMotion([[float(x) for x in center+t*dc]])
        pp=CollisionMotion([[float(x) for x in anchor+t*dp]])
        _,_,w,trace=_static(shape,plane,cp,pp,curved_first,frame=frame)
        if max(ulp(x) for x in (*w.first_point,*w.second_point,w.gap_m))>1e-7:
            raise OverflowError("curved plane world witness resolution")
        return w,trace

    start,_=sample(0.)
    end,trace=sample(1.)
    g0,g1=start.gap_m,end.gap_m
    if g0>0 and g1>0:
        return "separated","affine_curved_halfspace",end,trace,CollisionSweep(g0,g1,None,None,None)

    start_event,end_event=CurvedEvent((0.,0.),0.,start),CurvedEvent((1.,1.),1.,end)
    if g0<=0 and g1<=0:
        first,last=start_event,end_event
    else:
        interval=_outward_time([(h-height0)/slope for h in reach])
        time=(interval[0]+interval[1])/2
        w,_=sample(time)
        boundary=CurvedEvent(interval,time,w)
        first,last=(start_event,boundary) if g0<=0 else (boundary,end_event)
    trace=replace(trace,first_contact=first,last_contact=last)
    sweep=CollisionSweep(g0,g1,first.interval,first.witness_time,first.witness)
    return "hit","affine_curved_halfspace",end,trace,sweep
