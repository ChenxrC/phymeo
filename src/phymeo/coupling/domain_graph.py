"""Atomic instantaneous point loads across uniquely owned physical state domains."""

from collections.abc import Mapping
from numbers import Integral
from types import MappingProxyType

import numpy as np

from .point_port import _group_sum
from .ports import _numerical
from .surface_port import _array, _result


class MultiDomainPointOperator:
    """Sum signed domain responses while retaining all shared-point cross terms.

    This routes world impulses; it does not certify that endpoint geometry is
    touching, choose forces, or advance a scene. Distinct domain labels must
    represent distinct physical states, each through one immutable PointPort.
    """

    def __init__(self, ports, left, right):
        if not isinstance(ports, Mapping):
            raise TypeError("ports must be a domain-to-port mapping")
        self._ports, self._shapes = {}, {}
        for name, port in ports.items():
            if not isinstance(name, str) or not name or name != port.domain:
                raise ValueError("each key must match its port domain")
            if port.dtype != np.dtype(np.float64):
                raise TypeError("domain graph requires float64 state storage")
            positions = _array(port.positions(), "domain positions")
            if positions.shape[1] != 3:
                raise ValueError("domain world points require three coordinates")
            self._ports[name], self._shapes[name] = port, positions.shape
        self._domains = tuple(sorted(self._ports))
        self._left, self._right = self._endpoints(left), self._endpoints(right)
        if len(self._left) != len(self._right):
            raise ValueError("each edge requires two endpoints")
        self._shape = (len(self._left), 3)
        routes = {name: [] for name in self._domains}
        for edge, (a, b) in enumerate(zip(self._left, self._right, strict=True)):
            if a == b:
                continue
            routes[a[0]].append((a[1], edge, 1))
            routes[b[0]].append((b[1], edge, -1))
        self._routes = {}
        for name in self._domains:
            values = np.array(routes[name], dtype=np.int64).reshape(-1, 3)
            values.flags.writeable = False
            self._routes[name] = values

    def _endpoints(self, endpoints):
        result = []
        for endpoint in endpoints:
            if isinstance(endpoint, (str, bytes)) or len(endpoint) != 2:
                raise ValueError("endpoint must be (domain, index)")
            name, index = endpoint
            if not isinstance(name, str) or name not in self._ports:
                raise ValueError("endpoint references an unknown domain")
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
                raise TypeError("endpoint index must be an integer")
            if index < 0 or index >= self._shapes[name][0]:
                raise ValueError("endpoint index outside domain port")
            result.append((name, int(index)))
        return tuple(result)

    @property
    def domains(self):
        return self._domains

    @property
    def dtype(self):
        return np.dtype(np.float64)

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    def snapshot(self):
        return MappingProxyType({name: self._ports[name].snapshot() for name in self._domains})

    def validate_candidates(self, states):
        if not isinstance(states, Mapping) or set(states) != set(self._domains):
            raise ValueError("candidate domains must exactly match the graph")
        return MappingProxyType(
            {name: self._ports[name].validate_candidate(states[name]) for name in self._domains}
        )

    def _relative(self, values):
        owned = {name: _result(values[name], self._shapes[name]) for name in self._domains}
        result = np.empty(self._shape, dtype=np.float64)
        for edge, (a, b) in enumerate(zip(self._left, self._right, strict=True)):
            result[edge] = owned[a[0]][a[1]] - owned[b[0]][b[1]]
        return _result(result, self._shape)

    @_numerical
    def relative_positions(self):
        return self._relative({name: self._ports[name].positions() for name in self._domains})

    @_numerical
    def relative_velocities(self):
        return self._relative({name: self._ports[name].velocities() for name in self._domains})

    @_numerical
    def domain_impulses(self, impulses):
        impulses = _array(impulses, "impulses", self._shape)
        result = {}
        for name in self._domains:
            route = self._routes[name]
            loads = _group_sum(
                route[:, 0], impulses[route[:, 1]] * route[:, 2, None], self._shapes[name][0]
            )
            loads = _result(loads, self._shapes[name])
            loads.flags.writeable = False
            result[name] = loads
        return MappingProxyType(result)

    @_numerical
    def response(self, impulses):
        loads = self.domain_impulses(impulses)
        return self._relative(
            {name: self._ports[name].response(loads[name].copy()) for name in self._domains}
        )

    @_numerical
    def apply_impulses(self, impulses):
        loads = self.domain_impulses(impulses)
        candidates = {}
        for name in self._domains:
            port = self._ports[name]
            candidates[name] = port.validate_candidate(port.apply_impulses(loads[name].copy()))
        # Earlier candidates stay local until the last domain validates.
        return MappingProxyType(candidates)
