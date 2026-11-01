"""Current force tangents for the active, translational four-node research model."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev as ch
from p65_common_motion import GRID, Forces, need, paired_words, quantized, tables, total

from phymeo.constraints.bending import bending_angle_gradient


@dataclass
class Jet:
    value: Forces
    mq: np.ndarray
    pq: np.ndarray
    pv: np.ndarray
    rq: np.ndarray
    rv: np.ndarray
    gap_hessian: np.ndarray
    gap: np.ndarray

    def at_linear_increment(self, dx, du):
        return Forces(
            self.value.material + self.mq @ dx,
            self.value.rows + self.rq @ dx + self.rv @ du,
            self.value.pressures + self.pq @ dx + self.pv @ du,
        )


def face_normal_jet(e, u, de, du):
    c = np.cross(e, u)
    length = np.linalg.norm(c)
    need(length > 1e-10 * max(np.dot(e, e), np.dot(u, u)) and c[2] > 0, "unsupported oriented face")
    n = c / length
    dc = np.cross(de, u) + np.cross(e, du)
    mixed = np.cross(de[:, None, :], du[None, :, :])
    ddc = mixed + mixed.transpose(1, 0, 2)
    dl = dc @ n
    dn = (dc - dl[:, None] * n) / length
    ddl = dc @ dn.T + ddc @ n
    ddnz = (ddc[:, :, 2] - n[2] * ddl - np.outer(dn[:, 2], dl) - np.outer(dl, dn[:, 2])) / length
    return n, dn[:, 2], ddnz


def material_jet(model, x, basis):
    size = basis.shape[1]
    force = np.zeros(size)
    jac = np.zeros((size, size))
    dist = model.material.distance
    for (a, b), rest, alpha in zip(dist.indices, dist.rest_length, dist.compliance, strict=True):
        edge = x[b] - x[a]
        length = np.linalg.norm(edge)
        need(length > 0, "zero distance edge")
        unit = edge / length
        delta = (basis[b] - basis[a]).T
        force -= delta.T @ ((length - rest) * unit / alpha)
        hess = ((1 - rest / length) * np.eye(3) + (rest / length) * np.outer(unit, unit)) / alpha
        jac -= delta.T @ hess @ delta
    hinge = model.material.bending
    for ids, rest, alpha in zip(hinge.indices, hinge.rest_angle, hinge.compliance, strict=True):
        a, b, c, d = ids
        e, u, w = x[b] - x[a], x[c] - x[a], x[d] - x[a]
        de, du, dw = basis[b] - basis[a], basis[c] - basis[a], basis[d] - basis[a]
        theta, raw = bending_angle_gradient(x[ids])
        C = theta - rest
        need(abs(C) < np.pi, "unsupported hinge branch")
        le = np.linalg.norm(e)
        dle = (de @ e) / le

        def outer_gradient(n, dn):
            n2 = np.dot(n, n)
            return (
                -le * n / n2,
                -(dle[:, None] * n + le * dn) / n2 + 2 * le * (dn @ n)[:, None] * n / n2**2,
            )

        gc, dgc = outer_gradient(np.cross(e, u), np.cross(de, u) + np.cross(e, du))
        gd, dgd = outer_gradient(np.cross(w, e), np.cross(dw, e) + np.cross(w, de))

        def projection(z, dz):
            t = np.dot(e, z) / le**2
            return t, (de @ z + dz @ e - 2 * t * (de @ e)) / le**2

        tc, dtc = projection(u, du)
        td, dtd = projection(w, dw)
        dg_b = -dtc[:, None] * gc - tc * dgc - dtd[:, None] * gd - td * dgd
        dg_a = -dg_b - dgc - dgd
        grad = np.zeros(size)
        H = np.zeros((size, size))
        for node, g, dg in zip(ids, raw, (dg_a, dg_b, dgc, dgd), strict=True):
            sl = slice(3 * node, 3 * node + 3)
            grad[sl] = g
            H[sl] = dg.T
        force -= C * grad / alpha
        jac -= (np.outer(grad, grad) + C * H) / alpha
    return force, jac


def force_jet(model, q, v):
    """One fused value-and-derivative evaluation for each original paired address."""
    size = len(q)
    x = q[:12].reshape(4, 3)
    basis = np.eye(size)[:12].reshape(4, 3, size).transpose(0, 2, 1)
    material, mq = material_jet(model, x, basis)
    ps = []
    rows = []
    pq = []
    pv = []
    rq = []
    rv = []
    hess = []
    gaps = []
    K = 500 / 6
    d = 1000
    for ids in model.faces:
        a, b, c = ids
        n, dnz, ddnz = face_normal_jet(
            x[b] - x[a], x[c] - x[a], basis[b] - basis[a], basis[c] - basis[a]
        )
        H = -model.rho * ddnz
        for node in ids:
            D = -model.rho * dnz.copy()
            D[3 * node + 2] += 1
            D[-1] -= 1
            gap = q[3 * node + 2] - q[-1] + model.rho * (1 - n[2])
            rate = D @ v
            comp = 0.002 - gap
            fac = 1 - d * rate
            need(
                gap >= 0
                and comp > 0
                and fac > 0
                and np.max(abs((x[node] - model.rho * n)[:2])) < 2,
                "trial outside the finite active chart",
            )
            p = K * comp * fac
            px = -K * (fac * D + d * comp * (H @ v))
            pu = -K * d * comp * D
            ps.append(p)
            rows.append(p * D)
            pq.append(px)
            pv.append(pu)
            rq.append(p * H + np.outer(D, px))
            rv.append(np.outer(D, pu))
            hess.append(H)
            gaps.append(gap)
    return Jet(
        Forces(material, np.array(rows), np.array(ps)),
        mq,
        np.array(pq),
        np.array(pv),
        np.array(rq),
        np.array(rv),
        np.array(hess),
        np.array(gaps),
    )


def residual_matrix(h, V, P, mass, Fq, Fv):
    nodes, dofs = Fq.shape[:2]
    mat = np.eye(nodes * dofs).reshape(nodes, dofs, nodes, dofs)
    for k in range(nodes):
        mat[k] -= (
            h**2 * P[k][None, :, None] * Fq[k][:, None, :]
            + h * V[k][None, :, None] * Fv[k][:, None, :]
        ) / mass[:, None, None]
    return mat.reshape(nodes * dofs, nodes * dofs)


def newton_grid(model, q0, v0, h, first, degree, sweeps, guess, charged, lifting=None):
    nodes, card, V, P = tables(degree)
    n = len(q0)
    a = guess(nodes)
    actual_inlet = total(first.value, model)
    a[0] = actual_inlet
    history = []
    for _ in range(sweeps):
        x = q0 + h * nodes[:, None] * v0 + h * h * (P @ a)
        v = v0 + h * (V @ a)
        if lifting is not None and lifting.coeff is not None:
            ex_v, ex_x = lifting.motion(nodes, model, h)
            x += ex_x
            v += ex_v
        jets = [first] + [charged(pos, vel) for pos, vel in zip(x[1:], v[1:], strict=True)]
        target = np.stack([total(j.value, model) for j in jets])
        Kq = np.stack([j.mq + j.rq.sum(axis=0) for j in jets])
        Kv = np.stack([j.rv.sum(axis=0) for j in jets])
        matrix = residual_matrix(h, V, P, model.mass, Kq, Kv)
        rhs = (target - a).ravel()
        need(not np.any(rhs[:n]), "inlet changed in the nonlinear solve")
        da = np.zeros_like(a)
        da[1:] = np.linalg.solve(matrix[n:, n:], rhs[n:]).reshape(degree, n)
        linear_error = float(np.max(abs(matrix @ da.ravel() - rhs)))
        dx = h * h * (P @ da)
        du = h * (V @ da)
        frames = [j.at_linear_increment(p, u) for j, p, u in zip(jets, dx, du, strict=True)]
        # Newton's equation identifies each corrected acceleration with the linearized force.
        linear_acc = np.stack(
            [(model.external + f.material + f.rows.sum(axis=0)) / model.mass for f in frames]
        )
        projected_error = float(np.max(abs(linear_acc - (a + da))))
        need(linear_error < 1e-10 and projected_error < 1e-10, "linearized common motion mismatch")
        history.append(
            dict(
                q=x,
                v=v,
                acc=a,
                evaluated=target,
                pressures=np.stack([j.value.pressures for j in jets]),
                delta_acc=da,
                residual=float(np.max(abs(rhs))),
                linear_error=linear_error,
                projected_error=projected_error,
            )
        )
        if lifting is not None:
            frames, da, update = lifting.update(
                model, h, nodes, card, V, P, jets, frames, matrix, a, da, v, dx, du
            )
            history[-1].update(delta_acc=da, slope=update)
        a = a + da
    return nodes, card, a, frames, history


def peak_fit(
    model,
    q,
    v,
    h,
    bind,
    available,
    ledger,
    *,
    lifting=None,
    multilevel=False,
    provider=force_jet,
    paired_axes=(),
):
    need(not (multilevel and lifting is not None), "incompatible time bases")
    paired_axes = dict(paired_axes)
    need(
        all(0 <= j < len(q) and sign in (-1, 1) for j, sign in paired_axes.items()),
        "invalid constant map component",
    )
    q0, v0 = np.asarray(q, float), np.asarray(v, float)
    h = float(h)
    calls = 0

    def evaluate_joint(x, u):
        nonlocal calls
        need((calls + 1) * len(model.owners) <= available, "original step force budget")
        calls += 1
        row = dict(call=calls, addresses=len(model.owners), q=x.copy(), v=u.copy(), returned=False)
        ledger.append(row)
        jet = provider(model, x, u)
        for j, sign in paired_axes.items():
            need(
                np.array_equal(jet.value.rows[:, j], sign * jet.value.pressures)
                and np.array_equal(jet.rq[:, j], sign * jet.pq)
                and np.array_equal(jet.rv[:, j], sign * jet.pv),
                "unproved constant map component",
            )
        row["pressures"] = jet.value.pressures.copy()
        row["returned"] = True
        return jet

    first = evaluate_joint(q0, v0)
    inlet = total(first.value, model)

    def initial(nodes):
        return np.tile(inlet, (len(nodes), 1))

    _, coarse_card, coarse_a, _, coarse_history = newton_grid(
        model, q0, v0, h, first, 3, 2 if multilevel else 3, initial, evaluate_joint
    )
    previous_coeff = coarse_card @ coarse_a

    def transferred(nodes):
        return ch.chebvander(2 * nodes - 1, len(previous_coeff) - 1) @ previous_coeff

    bridge = []
    if multilevel:
        _, mid_card, mid_a, _, bridge = newton_grid(
            model, q0, v0, h, first, 9, 1, transferred, evaluate_joint
        )
        previous_coeff = mid_card @ mid_a

    nodes, card, _, frames, history = newton_grid(
        model,
        q0,
        v0,
        h,
        first,
        20 if multilevel else 13,
        1 if multilevel else 2,
        transferred,
        evaluate_joint,
        lifting,
    )

    def words(field):
        values = np.stack([getattr(f, field) for f in frames])
        low = (card @ values.reshape(len(nodes), -1)).reshape(values.shape)
        if lifting is None:
            return quantized(low)
        complete = getattr(lifting.fields(len(q0), len(model.owners)), field).copy()
        complete[: len(nodes)] += low
        return quantized(complete)

    pressure = words("pressures")
    schema = "p65-active-newton-curve-v1" if lifting is None else "p65-slope-newton-curve-v1"
    if multilevel:
        schema = "p65-levels-newton-curve-v1"
    rows = paired_words(frames, card, pressure, paired_axes) if paired_axes else words("rows")
    claim = dict(
        schema=schema,
        binding=bind,
        h=h,
        grid=GRID,
        owners=list(model.owners),
        mass=model.mass.copy(),
        external=model.external.copy(),
        material=words("material"),
        pressure=pressure,
        rows=rows,
        nodes=nodes,
        coarse=coarse_history,
        iterations=history,
        force_calls=calls,
        owner_charges=calls * len(model.owners),
    )
    if multilevel:
        claim["bridge"] = bridge
    return claim, evaluate_joint
