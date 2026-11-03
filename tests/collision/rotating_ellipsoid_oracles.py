"""Independent closed-form oracle. Decimal angles are computed from the input floats."""

from decimal import Decimal as D, localcontext
from math import cos, sin

import numpy as np


def _atan_small(x):
    term, result, j = x, x, 1
    while abs(term) > D("1e-78"):
        term *= -x*x
        result += term/(2*j+1)
        j += 1
    return result


def analytic_extrema(dims, height, axis_xy, angle):
    """Identity initial frame, normal +Z, arbitrary stored in-plane axis."""
    with localcontext() as ctx:
        ctx.prec = 84
        a,b,c = [D.from_float(float(q)) for q in dims]
        x,y = [D.from_float(float(q)) for q in axis_xy[:2]]
        h,A = D.from_float(float(height)), D.from_float(abs(float(angle)))
        peak2 = (a*a*y*y + b*b*x*x)/(x*x+y*y)
        if h <= c:
            return D(0), D(1)
        if h*h > peak2:
            return None
        pi = 16*_atan_small(D(1)/5) - 4*_atan_small(D(1)/239)
        if h*h == peak2:
            alpha = pi/2
        else:
            tan = ((h*h-c*c)/(peak2-h*h)).sqrt()
            reciprocal = tan > 1
            z = 1/tan if reciprocal else tan
            arc = pi/4 + _atan_small((z-1)/(z+1)) if z>D("0.5") else _atan_small(z)
            alpha = pi/2-arc if reciprocal else arc
        if A < alpha:
            return None
        return alpha/A, min(D(1),(pi-alpha)/A)


def physical_points(center, anchor, plane_normal, dims, initial_frame, axis, signed_phase):
    """Independent FP64 Rodrigues rotation and analytic plane/ellipsoid support map."""
    k=np.asarray(axis,dtype=float)/np.linalg.norm(axis)
    x,y,z=k
    cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    turn=np.eye(3)+sin(signed_phase)*cross+(1-cos(signed_phase))*(cross@cross)
    frame=turn@initial_frame
    n=np.asarray(plane_normal,dtype=float)/np.linalg.norm(plane_normal)
    matrix=(frame*np.square(dims))@frame.T
    ellipsoid=center-matrix@n/np.sqrt(n@matrix@n)
    gap=float(n@(ellipsoid-anchor))
    return gap,ellipsoid,ellipsoid-gap*n
