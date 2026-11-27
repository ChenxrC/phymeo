"""State-feedback pickup imitation using a small normalized radial basis network.

Training inputs are decision-time measurements paired with the teacher's next
16 normalized motor commands. Deployment needs the same measurement vector.
The model has no clock, phase machine, teacher dependency, or action sequence.
"""
import json
from pathlib import Path

import numpy as np


SCHEMA = "dual-panda-pickup-rbf-student-v1"


def _distances(x, centers):
    if len(x) == 1:
        delta = centers - x[0]
        return np.einsum("ij,ij->i", delta, delta)[None, :]
    # Squared distances without allocating [examples, centers, features].
    result = (x * x).sum(axis=1)[:, None] + (centers * centers).sum(axis=1)
    result -= 2. * (x @ centers.T)
    return np.maximum(result, 0.)


def _basis(x, centers):
    distance = _distances(x, centers)
    # Set the response width from the local center spacing, so an approach
    # and a nearly stationary closed grasp can have different resolutions.
    neighbor = min(5, len(centers) - 1)
    width_squared = np.partition(distance, neighbor, axis=1)[:, neighbor:neighbor + 1]
    logits = -(distance - distance.min(axis=1, keepdims=True)) / np.maximum(.25 * width_squared, 1e-10)
    activation = np.exp(logits)
    return activation / activation.sum(axis=1, keepdims=True)


def _select_centers(x, count, rng):
    # Farthest-point coverage prevents long stationary holds from using all
    # available basis functions. Time and episode ordering are not features.
    selected = [int(rng.integers(len(x)))]
    nearest = np.full(len(x), np.inf)
    while len(selected) < count:
        diff = x - x[selected[-1]]
        nearest = np.minimum(nearest, np.einsum("ij,ij->i", diff, diff))
        index = int(nearest.argmax())
        if nearest[index] <= 1e-12:
            break
        selected.append(index)
    return np.asarray(selected)


class PickupStudent:
    def __init__(self, mean, inverse_scale, centers, weights, fit_summary, observation_contract):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.inverse_scale = np.asarray(inverse_scale, dtype=np.float64)
        self.centers = np.asarray(centers, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        if (self.mean.ndim != 1 or not self.mean.size or self.inverse_scale.shape != self.mean.shape
                or np.any(self.inverse_scale < 0.) or self.centers.ndim != 2 or not len(self.centers)
                or self.centers.shape[1] != len(self.mean) or self.weights.shape != (len(self.centers), 16)
                or not all(np.isfinite(a).all() for a in (self.mean, self.inverse_scale, self.centers, self.weights))):
            raise ValueError("Invalid pickup student parameter dimensions or nonfinite values")
        self.fit_summary = fit_summary
        self.observation_contract = observation_contract

    @classmethod
    def fit(cls, observations, actions, *, seed=271133, max_centers=192, observation_contract="caller measurement vector"):
        """Fit all 16 output channels on training examples. The time axis is flattened by the caller."""
        x = np.asarray(observations, dtype=np.float64)
        y = np.asarray(actions, dtype=np.float64)
        if (x.ndim != 2 or not x.shape[0] or not x.shape[1] or y.shape != (len(x), 16)
                or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.abs(y) > 1.)):
            raise ValueError("Expected finite [N,D] observations and [N,16] actions bounded by [-1, 1]")
        if max_centers < 1 or seed < 0:
            raise ValueError("A positive center count and nonnegative seed are required")
        if not isinstance(observation_contract, str) or not observation_contract.strip():
            raise ValueError("Expected a nonempty observation contract")
        rng = np.random.default_rng(seed)
        samples = len(x)
        if samples > 16384:
            indices = rng.choice(samples, 16384, replace=False)
            x, y = x[indices], y[indices]
        mean, deviation = x.mean(axis=0), x.std(axis=0)
        # Constant or numerically stationary channels have no data-supported
        # scale. Their coefficients stay zero for future unseen measurements.
        active = deviation > 1e-6 * np.maximum(1., np.abs(mean))
        inverse_scale = np.divide(1., deviation, out=np.zeros_like(deviation), where=active)
        standardized = np.clip((x - mean) * inverse_scale, -8., 8.)
        center_ids = _select_centers(standardized, min(int(max_centers), len(x)), rng)
        centers = standardized[center_ids]
        phi = _basis(standardized, centers)
        # Learn a regularized least-squares motor readout. The small diagonal
        # prior keeps sparsely occupied centers close to their training labels.
        ridge = .002 * len(x) / len(centers)
        gram = phi.T @ phi + ridge * np.eye(len(centers))
        weights = np.linalg.solve(gram, phi.T @ y + ridge * y[center_ids])
        residual = np.clip(phi @ weights, -1., 1.) - y
        report = dict(seed=int(seed), samples_available=samples, samples_fitted=len(x), centers=len(centers),
                      observation_size=x.shape[1], active_features=int(active.sum()), action_size=16,
                      training_arm_rmse=float(np.sqrt(np.mean(residual[:, :14] ** 2))),
                      training_finger_rmse=float(np.sqrt(np.mean(residual[:, 14:] ** 2))),
                      training_rmse_by_motor=np.sqrt(np.mean(residual ** 2, axis=0)).tolist())
        return cls(mean, inverse_scale, centers, weights, report, observation_contract)

    def act(self, obs):
        """Return a float32 [16] or [world,16] command from current measurements."""
        x = np.asarray(obs, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Expected finite observations matching the fitted pickup input width")
        batch = np.atleast_2d(x)
        standardized = np.clip((batch - self.mean) * self.inverse_scale, -8., 8.)
        basis = _basis(standardized, self.centers)
        action = np.clip(np.einsum("nk,ka->na", basis, self.weights), -1., 1.).astype(np.float32)
        return action[0] if x.ndim == 1 else action

    def save(self, path):
        with Path(path).open("xb") as f:
            np.savez_compressed(f, schema=SCHEMA, mean=self.mean, inverse_scale=self.inverse_scale,
                                centers=self.centers, weights=self.weights, observation_contract=self.observation_contract,
                                fit_summary=json.dumps(self.fit_summary, allow_nan=False))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if data["schema"].item() != SCHEMA:
                raise ValueError("Unknown pickup student checkpoint schema")
            return cls(data["mean"], data["inverse_scale"], data["centers"], data["weights"],
                       json.loads(data["fit_summary"].item()), str(data["observation_contract"].item()))
