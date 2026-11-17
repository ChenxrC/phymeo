"""Prepared public IO for the existing compiled scene, borrowing its sole batch."""

import numpy as np


class SceneIOPlan:
    """Preserve scene admission, limits and metadata around native prepared IO.

    No physical state, current control mirror or new device workspace is owned
    here. Native plans remain owned and closed by the scene's existing batch.
    """

    contract_version = 1

    def __init__(self, scene, native, kind, bounds):
        self._scene, self._native, self._kind = scene, native, kind
        self._bounds = bounds
        self._closed = False

    @property
    def layout(self):
        return self._native.layout

    def _check(self):
        if self._closed:
            raise RuntimeError("compiled scene IO plan is closed")
        self._scene._check_open(ready=True)

    def read_into(self, values, **kwargs):
        with self._scene._lock:
            self._check()
            if self._kind == "reset":
                raise TypeError("reset plan has no readable fields")
            return self._native.read_into(values, **kwargs)

    def write(self, values):
        with self._scene._lock:
            self._check()
            if self._kind != "control":
                raise TypeError("only control plans accept writes")
            if not isinstance(values, np.ndarray) or values.dtype != np.float32:
                raise TypeError("scene controls require a float32 NumPy array")
            if values.shape != self.layout["shape"]:
                raise ValueError("scene control shape differs from the prepared layout")
            for start, end, low, high, scratch in self._bounds:
                part = values[:, start:end]
                np.less(part, low, out=scratch)
                if scratch.any():
                    raise ValueError("control target exceeds declared asset metadata limits")
                np.greater(part, high, out=scratch)
                if scratch.any():
                    raise ValueError("control target exceeds declared asset metadata limits")
            return self._native.write(values)

    def reset(self, mask=None):
        scene = self._scene
        with scene._lock:
            if self._closed:
                raise RuntimeError("compiled scene IO plan is closed")
            scene._check_open()
            if self._kind != "reset":
                raise TypeError("only reset plans accept reset")
            if scene._poisoned and mask is not None and not np.all(mask):
                raise RuntimeError("reset all worlds after partial or uncertain publication")
            self._native.reset(mask)
            for index, key in enumerate(scene.world_keys):
                if mask is None or mask[index]:
                    scene._times[key] = 0.0
                    scene._receipts.pop(key, None)
            scene._poisoned = False

    def stats(self):
        with self._scene._lock:
            result = self._native.stats()
            result["scene_limit_workspace_bytes"] = sum(
                low.nbytes + high.nbytes + scratch.nbytes
                for _, _, low, high, scratch in self._bounds
            )
            return result

    def close(self):
        with self._scene._lock:
            if not self._closed:
                self._native.close()
                self._bounds = ()
                self._closed = True
                if self in self._scene._owners:
                    self._scene._owners.remove(self)

    def __enter__(self):
        with self._scene._lock:
            self._check()
        return self

    def __exit__(self, *_):
        self.close()


def prepare_scene_io(scene, kind, keys, fields):
    with scene._lock:
        scene._check_open(ready=True)
        if scene.backend == "host":
            raise NotImplementedError("prepared scene IO requires resident interpret/CUDA")
        selected = scene._selected(keys)
        fields = tuple(fields)
        # Limits are cold structural metadata, not a second runtime control state.
        bounds = []
        if kind == "control":
            for field in fields:
                if field in ("target_position", "target_velocity"):
                    worlds = [scene._worlds[k] for k in selected]
                    if field == "target_position":
                        low = np.array([w.asset.lower for w in worlds])
                        high = np.array([w.asset.upper for w in worlds])
                    else:
                        high = np.array(
                            [
                                [np.inf if v is None else v for v in w.asset.velocity_limit]
                                for w in worlds
                            ]
                        )
                        low = -high
                    bounds.append((field, low, high))
        if kind == "reset":
            native = scene._raw.prepare_reset()
        elif kind == "body":
            native = scene._raw.prepare_body_read(selected, fields=fields)
        elif kind == "field":
            native = scene._raw.prepare_field_read(selected, fields=fields)
        else:
            native = scene._raw.prepare_control_write(selected, fields=fields)
        try:
            layout = native.layout
            resolved = []
            for field, low, high in bounds:
                meta = layout["fields"][field]
                low.flags.writeable = high.flags.writeable = False
                resolved.append(
                    (
                        meta["offset"],
                        meta["offset"] + meta["width"],
                        low,
                        high,
                        np.empty(low.shape, bool),
                    )
                )
            plan = SceneIOPlan(scene, native, kind, tuple(resolved))
            scene._owners.append(plan)
            return plan
        except BaseException:
            # Native plan remains in the batch registry if release fails.
            native.close()
            raise
