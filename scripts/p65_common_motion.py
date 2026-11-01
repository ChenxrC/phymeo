"""Bounded common-path collocation for constant-mass translational research models."""

from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
from math import lcm

import numpy as np
from numpy.polynomial import chebyshev as ch

DEGREE = 20
GRID = 1 << 60
PACKET_BYTES = 1024


def need(ok, message):
    if not ok:
        raise ValueError(message)


def int_bytes(n):
    n = int(n)
    need(abs(n).bit_length() <= 512, "checkpoint integer limit")
    data = abs(n).to_bytes((abs(n).bit_length() + 7) // 8, "big")
    return bytes([len(data) + (128 if n < 0 else 0)]) + data


def checkpoint(q, v, time, signature):
    values = list(map(Q, [time, *q, *v]))
    den = lcm(*(x.denominator for x in values))
    payload = b"CP01" + bytes.fromhex(signature) + bytes([len(q)]) + int_bytes(den)
    payload += b"".join(int_bytes(int(x * den)) for x in values)
    data = len(payload).to_bytes(2, "big") + payload + sha256(payload).digest()
    need(len(q) == len(v) and len(data) <= PACKET_BYTES, "checkpoint shape or byte limit")
    return np.frombuffer(data.ljust(PACKET_BYTES, b"\0"), dtype="<i4").copy()


def unpack(packet, signature):
    need(packet.dtype == np.dtype("<i4") and packet.shape == (256,), "invalid history field")
    data = packet.tobytes()
    length = int.from_bytes(data[:2], "big")
    p = data[2 : 2 + length]
    need(
        length >= 38
        and len(data) == PACKET_BYTES
        and data[2 + length : 34 + length] == sha256(p).digest()
        and not any(data[34 + length :]),
        "checkpoint checksum, size or pad",
    )
    need(p[:36] == b"CP01" + bytes.fromhex(signature), "checkpoint model binding")
    n = p[36]
    idx = 37

    def integer():
        nonlocal idx
        need(idx < len(p), "short checkpoint integer")
        tag = p[idx]
        size = tag & 127
        idx += 1
        need(size <= 64 and idx + size <= len(p), "checkpoint integer limit")
        raw = p[idx : idx + size]
        idx += size
        return int.from_bytes(raw, "big") * (-1 if tag & 128 else 1)

    den = integer()
    need(den > 0 and n > 0, "invalid dimension or denominator")
    values = [Q(integer(), den) for _ in range(2 * n + 1)]
    t = values[0]
    q = values[1 : n + 1]
    v = values[n + 1 :]
    need(
        idx == len(p) and t >= 0 and np.array_equal(checkpoint(q, v, t, signature), packet),
        "noncanonical endpoint",
    )
    return q, v, t


@dataclass
class Forces:
    material: np.ndarray
    rows: np.ndarray
    pressures: np.ndarray


def total(frame, model):
    need(
        frame.material.shape == model.mass.shape
        and frame.rows.shape == (len(model.owners), len(model.mass))
        and frame.pressures.shape == (len(model.owners),)
        and all(np.isfinite(a).all() for a in (frame.material, frame.rows, frame.pressures))
        and np.min(frame.pressures) >= 0,
        "invalid force provider output",
    )
    return (model.external + frame.material + frame.rows.sum(axis=0)) / model.mass


def tables(degree=DEGREE):
    need(1 <= degree <= DEGREE, "unsupported collocation degree")
    nodes = np.rint((1 - np.cos(np.pi * np.arange(degree + 1) / degree)) / 2 * (1 << 32)) / (
        1 << 32
    )
    table = ch.chebvander(2 * nodes - 1, degree)
    cardinal = np.linalg.solve(table, np.eye(len(nodes)))

    def integral(m):
        coeff = ch.chebint(cardinal, m=m, lbnd=-1, scl=0.5, axis=0)
        result = ch.chebvander(2 * nodes - 1, len(coeff) - 1) @ coeff
        result[0] = 0.0
        return result

    return nodes, cardinal, integral(1), integral(2)


def quantized(coeff):
    words = np.rint(coeff * GRID)
    need(np.isfinite(words).all() and np.max(abs(words)) < 2**63, "force coefficient grid overflow")
    return words.astype(np.int64)


def paired_words(frames, cardinal, pressure, axes):
    """Interpolate the varying map components and encode exact unit reactions from one flux."""
    count, owners = pressure.shape
    size = frames[0].rows.shape[-1]
    independent = [i for i in range(size) if i not in axes]
    rows = np.empty((count, owners, size), dtype=np.int64)
    if independent:
        samples = np.stack([f.rows[:, independent] for f in frames])
        coeff = cardinal @ samples.reshape(len(frames), -1)
        rows[:, :, independent] = quantized(coeff.reshape(count, owners, len(independent)))
    for i, sign in axes.items():
        need(0 <= i < size and sign in (-1, 1), "invalid constant map component")
        rows[:, :, i] = sign * pressure
    return rows


def fit(model, qin, vin, h, bind, charge_limit=512, *, degree=DEGREE, sweeps=4, paired_axes=()):
    """The provider returns forces on ONE state at each node; its inlet tangent is only a preconditioner."""
    q0, v0 = np.asarray(qin, float), np.asarray(vin, float)
    n = len(q0)
    h = float(h)
    need(
        n == len(model.mass)
        and np.all(model.mass > 0)
        and len(set(model.owners)) == len(model.owners),
        "invalid owners or mass",
    )
    need(1 <= sweeps <= 4, "unsupported collocation sweep count")
    nodes, card, velocity_integral, position_integral = tables(degree)
    calls = 0
    history = []
    axes = dict(paired_axes)

    def force(q, v):
        nonlocal calls
        need((calls + 1) * len(model.owners) <= charge_limit, "owner evaluation budget")
        calls += 1
        f = model.forces(q, v)
        for i, sign in axes.items():
            need(
                0 <= i < len(q)
                and sign in (-1, 1)
                and np.array_equal(f.rows[:, i], sign * f.pressures),
                "unproved sampled constant map component",
            )
        return f

    first = force(q0, v0)
    a0 = total(first, model)
    Fq, Fv = model.linearization(q0, v0, first)
    Aq = Fq / model.mass[:, None]
    Av = Fv / model.mass[:, None]
    matrix = (
        np.eye(len(nodes) * n)
        - np.kron(h * velocity_integral, Av)
        - np.kron(h * h * position_integral, Aq)
    )
    rhs = np.tile(a0, (len(nodes), 1)) + h * nodes[:, None] * (Aq @ v0)
    acc = np.linalg.solve(matrix, rhs.ravel()).reshape(-1, n)
    for iteration in range(sweeps):
        q = q0 + h * nodes[:, None] * v0 + h * h * (position_integral @ acc)
        v = v0 + h * (velocity_integral @ acc)
        frames = [first] + [force(pos, vel) for pos, vel in zip(q[1:], v[1:], strict=True)]
        evaluated = np.stack([total(frame, model) for frame in frames])
        residual = evaluated - acc
        history.append(
            dict(
                q=q.copy(),
                v=v.copy(),
                acc=acc.copy(),
                evaluated=evaluated,
                pressures=np.stack([f.pressures for f in frames]),
                residual=float(np.max(abs(residual))),
            )
        )
        if iteration + 1 < sweeps:
            acc += np.linalg.solve(matrix, residual.ravel()).reshape(acc.shape)

    def coefficients(field):
        samples = np.stack([getattr(f, field) for f in frames])
        return quantized((card @ samples.reshape(len(nodes), -1)).reshape(samples.shape))

    pressure = coefficients("pressures")
    rows = paired_words(frames, card, pressure, axes) if axes else coefficients("rows")
    return dict(
        schema="p65-common-curve-v1",
        binding=bind,
        owners=list(model.owners),
        h=h,
        grid=GRID,
        material=coefficients("material"),
        rows=rows,
        pressure=pressure,
        external=model.external.copy(),
        mass=model.mass.copy(),
        force_calls=calls,
        owner_charges=calls * len(model.owners),
        nodes=nodes,
        iterations=history,
    )


def endpoint(claim, q, v):
    """Closed Chebyshev moments; the checker instead integrates exact power polynomials."""
    h = Q(claim["h"])
    n = len(q)
    gr = claim["grid"]

    def mean(j):
        return Q(1, 1 - j * j) if j % 2 == 0 else Q(0)

    xend = []
    vend = []
    for i in range(n):
        J = B = Q(0)
        mass = Q(float(claim["mass"][i]))
        for k, word in enumerate(claim["material"][:, i]):
            f = Q(int(word) + sum(int(z) for z in claim["rows"][k, :, i]), gr)
            if k == 0:
                f += Q(float(claim["external"][i]))
            J += h * mean(k) * f
            B += h * h * (mean(k) / 2 - (mean(k + 1) + mean(abs(k - 1))) / 4) * f
        vend.append(v[i] + J / mass)
        xend.append(q[i] + h * v[i] + B / mass)
    return xend, vend
