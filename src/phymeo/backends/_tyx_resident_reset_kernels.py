"""Select frozen initial or accepted banks; no physical integration or FK."""

import tyx as tx


def reset_scalar(
    count: tx.i32,
    initial: tx.ndarray(tx.f32, 2),
    selected_ticks: tx.ndarray(tx.i32, 1),
    state: tx.ndarray(tx.f32, 2),
    ticks: tx.ndarray(tx.i32, 1),
    candidate: tx.ndarray(tx.f32, 2),
    candidate_ticks: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(count):
        for j in tx.static_range(12):
            candidate[i, j] = state[i, j]
            if selected_ticks[i] >= 0:
                candidate[i, j] = initial[i, j]
        candidate_ticks[i] = ticks[i]
        if selected_ticks[i] >= 0:
            candidate_ticks[i] = selected_ticks[i]


def reset_tree(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    selected_ticks: tx.ndarray(tx.i32, 1),
    initial: tx.ndarray(tx.f32, 3),
    initial_fk: tx.ndarray(tx.f32, 3),
    initial_energy: tx.ndarray(tx.f32, 1),
    state: tx.ndarray(tx.f32, 3),
    fk: tx.ndarray(tx.f32, 3),
    energy: tx.ndarray(tx.f32, 1),
    candidate: tx.ndarray(tx.f32, 3),
    candidate_fk: tx.ndarray(tx.f32, 3),
    candidate_energy: tx.ndarray(tx.f32, 1),
):
    for w in tx.range(count):
        selected = selected_ticks[meta[w, 0]] >= 0
        for i in range(meta[w, 1]):
            for j in tx.static_range(7):
                candidate[w, i, j] = state[w, i, j]
                if selected:
                    candidate[w, i, j] = initial[w, i, j]
            for j in tx.static_range(18):
                candidate_fk[w, i, j] = fk[w, i, j]
                if selected:
                    candidate_fk[w, i, j] = initial_fk[w, i, j]
        candidate_energy[w] = energy[w]
        if selected:
            candidate_energy[w] = initial_energy[w]


def make_reset_kernels():
    return (tx.kernel(reset_scalar), tx.kernel(reset_scalar)), (
        tx.kernel(reset_tree),
        tx.kernel(reset_tree),
    )
