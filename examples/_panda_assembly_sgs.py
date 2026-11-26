"""Reset-bank sampling and terminal-success CEM for the Panda assembly experiment.

AlphaNIST, section IV-B, equation 4: https://arxiv.org/html/2610.13354v1
SGS, section 3.2 and appendix B.2: https://arxiv.org/html/2610.12465v1
The diagonal CEM optimizer below is specific to this small experiment.
"""

from copy import deepcopy

import numpy as np


def _positive_int(x, name):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(x)


def _binary(x):
    x = np.asarray(x)
    if x.dtype.kind not in "biuf" or not np.all((x == 0) | (x == 1)):
        raise ValueError("expected completed-episode binary success indicators")
    return x.astype(np.uint8, copy=False)


def uniform_validation(n_states, count, seed):
    """Fixed balanced-uniform indices into a separately constructed validation bank."""
    n_states = _positive_int(n_states, "n_states")
    count = _positive_int(count, "count")
    rng = np.random.default_rng(seed)
    passes, rest = divmod(count, n_states)
    batches = [rng.permutation(n_states) for _ in range(passes)]
    if rest:
        batches.append(rng.choice(n_states, size=rest, replace=False))
    return np.concatenate(batches)


class StateSampler:
    """Categorical sampling over train-bank entries with per-entry rolling outcomes."""

    def __init__(self, n_states, seed, *, history=100, mode="alphanist", epsilon=None):
        self.n_states = _positive_int(n_states, "n_states")
        self.history = _positive_int(history, "history")
        if mode not in ("alphanist", "sgs", "uniform"):
            raise ValueError("mode must be alphanist, sgs or uniform")
        self.mode = mode
        # AlphaNIST specifies a positive floor, with no numeric value in eq. 4.
        # The small-bank experiment makes its chosen floor explicit. The SGS
        # option uses the manipulation paper's own epsilon and temperature.
        self.epsilon = float(epsilon if epsilon is not None else (1.e-4 if mode == "sgs" else .05))
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.rng = np.random.default_rng(seed)
        self.visits = np.zeros(self.n_states, dtype=np.int64)
        self._buffer = np.zeros((self.n_states, self.history), dtype=np.uint8)
        self._successes = np.zeros(self.n_states, dtype=np.int64)

    def rates(self):
        return self._successes/np.maximum(1, np.minimum(self.history, self.visits))

    def probabilities(self):
        p = self.rates()
        if self.mode == "uniform":
            return np.full(self.n_states, 1./self.n_states)
        if self.mode == "alphanist":
            weights = self.epsilon+np.sqrt(p*(1.-p))
        else:
            # Mode .5, concentration 1, temperature 2. The package's floor
            # before logits follows the source paper's implementation appendix.
            kernel = np.sqrt((p+self.epsilon)*(1.-p+self.epsilon))
            weights = np.sqrt(np.maximum(kernel, self.epsilon)+self.epsilon)
        return weights/weights.sum()

    def sample(self, count):
        count = _positive_int(count, "count")
        return self.rng.choice(self.n_states, size=count, replace=True, p=self.probabilities())

    def update(self, state_ids, terminal_success):
        """Accept each training result in completion order, including repeated IDs."""
        ids = np.asarray(state_ids)
        outcomes = _binary(terminal_success)
        if ids.dtype.kind not in "iu" or ids.shape != outcomes.shape:
            raise ValueError("a training state ID is required for each terminal result")
        if np.any(ids < 0) or np.any(ids >= self.n_states):
            raise ValueError("state ID outside training bank")
        for idx, won in zip(ids.flat, outcomes.flat, strict=True):
            slot = self.visits[idx] % self.history
            self._successes[idx] += int(won)-int(self._buffer[idx, slot])
            self._buffer[idx, slot] = won
            self.visits[idx] += 1

    def snapshot(self):
        return dict(history=self.history, n_states=self.n_states, mode=self.mode, epsilon=self.epsilon,
                    visits=self.visits.tolist(), buffer=self._buffer.tolist(),
                    rng=deepcopy(self.rng.bit_generator.state))

    @classmethod
    def restore(cls, data):
        out = cls(data["n_states"], 0, history=data["history"], mode=data["mode"], epsilon=data["epsilon"])
        buffer = _binary(data["buffer"])
        visits = np.asarray(data["visits"])
        if buffer.shape != out._buffer.shape or visits.shape != out.visits.shape or visits.dtype.kind not in "iu" or np.any(visits < 0):
            raise ValueError("invalid sampler history")
        if np.any((np.arange(out.history)[None, :] >= visits[:, None]) & (buffer != 0)):
            raise ValueError("outcomes present beyond visit count")
        out._buffer[:] = buffer
        out._successes[:] = buffer.sum(axis=1)
        out.visits[:] = visits
        out.rng.bit_generator.state = deepcopy(data["rng"])
        return out


class TerminalCEM:
    """Diagonal Gaussian CEM. Evaluate each candidate on identical sampled starts."""

    def __init__(self, mean, std, seed, population=8, elite_fraction=.25, smoothing=.7, min_std=.01):
        self.mean = np.asarray(mean, dtype=float).copy()
        if self.mean.ndim != 1 or not self.mean.size or not np.isfinite(self.mean).all():
            raise ValueError("mean must be a finite parameter vector")
        self.std = np.broadcast_to(np.asarray(std, dtype=float), self.mean.shape).copy()
        self.min_std = np.broadcast_to(np.asarray(min_std, dtype=float), self.mean.shape).copy()
        if (not np.isfinite(self.std).all() or not np.isfinite(self.min_std).all()
                or np.any(self.min_std <= 0) or np.any(self.std < self.min_std)):
            raise ValueError("need positive finite exploration floor and initial std >= floor")
        self.population = _positive_int(population, "population")
        if self.population < 2 or not 0 < elite_fraction < 1 or not 0 < smoothing <= 1:
            raise ValueError("invalid CEM population, elite fraction or smoothing")
        self.elite_fraction, self.smoothing = float(elite_fraction), float(smoothing)
        self.rng = np.random.default_rng(seed)
        self.generation = 0
        self._pending = None

    def ask(self):
        """Candidate zero is the current mean. A repeated ask returns the same batch."""
        if self._pending is None:
            n = self.population-1
            delta = self.rng.standard_normal(((n+1)//2, len(self.mean)))*self.std
            perturbations = np.stack((delta, -delta), axis=1).reshape(-1, len(self.mean))[:n]
            self._pending = np.vstack((self.mean, self.mean+perturbations))
        return self._pending.copy()

    def tell(self, candidates, terminal_success):
        if self._pending is None or not np.array_equal(candidates, self._pending):
            raise ValueError("candidates must match the outstanding ask batch")
        wins = _binary(terminal_success)
        if wins.ndim == 1:
            wins = wins[:, None]
        if wins.ndim != 2 or wins.shape[0] != self.population or wins.shape[1] < 1:
            raise ValueError("need a binary result for every candidate and common start")
        scores = wins.mean(axis=1)
        if np.ptp(scores) > 0:
            mass = max(1, int(np.ceil(self.elite_fraction*self.population)))
            cutoff = np.sort(scores)[-mass]
            better, tied = scores > cutoff, scores == cutoff
            weights = (better.astype(float)+tied*((mass-better.sum())/tied.sum()))/mass
            elite_mean = weights@self._pending
            elite_std = np.sqrt(weights@(self._pending-elite_mean)**2)
            self.mean += self.smoothing*(elite_mean-self.mean)
            self.std = np.maximum(self.min_std, self.std+self.smoothing*(elite_std-self.std))
        self._pending = None
        self.generation += 1
        return scores

    def snapshot(self):
        return dict(mean=self.mean.tolist(), std=self.std.tolist(), min_std=self.min_std.tolist(),
                    population=self.population, elite_fraction=self.elite_fraction, smoothing=self.smoothing,
                    generation=self.generation, rng=deepcopy(self.rng.bit_generator.state),
                    pending=None if self._pending is None else self._pending.tolist())

    @classmethod
    def restore(cls, data):
        out = cls(data["mean"], data["std"], 0, population=data["population"],
                  elite_fraction=data["elite_fraction"], smoothing=data["smoothing"], min_std=data["min_std"])
        out.generation = int(data["generation"])
        if out.generation < 0:
            raise ValueError("negative generation")
        if data["pending"] is not None:
            batch = np.asarray(data["pending"], dtype=float)
            if batch.shape != (out.population, out.mean.size) or not np.isfinite(batch).all() or not np.array_equal(batch[0], out.mean):
                raise ValueError("invalid pending population")
            out._pending = batch
        out.rng.bit_generator.state = deepcopy(data["rng"])
        return out
