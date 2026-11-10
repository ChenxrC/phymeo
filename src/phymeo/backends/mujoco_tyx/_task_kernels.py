"""Native accepted-state feature extraction on independent TYX task storage.

This kernel never mutates physical q/v and never runs collision or dynamics.
"""

import tyx as tx


def evaluate(
    count: tx.i32,
    features: tx.i32,
    ids: tx.ndarray(tx.i32, 2),
    parameters: tx.ndarray(tx.f32, 2),
    q: tx.ndarray(tx.f32, 2),
    v: tx.ndarray(tx.f32, 2),
    observation: tx.ndarray(tx.f32, 2),
    reward: tx.ndarray(tx.f32, 1),
    terminated: tx.ndarray(tx.i32, 1),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        status[w] = 0
        total = 0.0
        done = 0
        for feature in range(features):
            value = 0.0
            if ids[feature, 0] == 0:
                value = q[w, ids[feature, 1]]
            else:
                value = v[w, ids[feature, 1]]
            observed = parameters[feature, 0] * value + parameters[feature, 1]
            observation[w, feature] = observed
            if value < parameters[feature, 4] or value > parameters[feature, 5]:
                done = 1
            if parameters[feature, 3] > 0.0:
                difference = value - parameters[feature, 2]
                total = total - parameters[feature, 3] * difference * difference
            if not tx.isfinite(value) or not tx.isfinite(observed):
                status[w] = 1
        reward[w] = total
        terminated[w] = done
        if not tx.isfinite(total):
            status[w] = 2


evaluate = tx.kernel(evaluate)
