"""Retryable release of resident dependencies, under their existing owner lock."""


class ReleaseStack:
    """Release in reverse acquisition order; retain the failed callback and ancestors."""

    def __init__(self):
        self._callbacks = []

    def callback(self, function, *args):
        self._callbacks.append((function, args))

    def close(self):
        while self._callbacks:
            function, args = self._callbacks[-1]
            function(*args)
            self._callbacks.pop()


class ResidentCleanupError(RuntimeError):
    """Construction failed and cleanup must be retried through cleanup_owner.close()."""

    def __init__(self, owner):
        super().__init__("resident construction cleanup failed; retry cleanup_owner.close()")
        self.cleanup_owner = owner


def close_dependencies(owner):
    pending = getattr(owner, "_cleanup_dependencies", ())
    while pending:
        pending[-1].close()
        pending.pop()


def cleanup_failed_init(owner, error):
    """Keep a failed child's dependencies alive when its constructor could not return."""
    dependency = getattr(error, "cleanup_owner", None)
    if dependency is not None and dependency is not owner:
        owner._cleanup_dependencies = [dependency]
    try:
        owner.close()
    except BaseException as cleanup_error:
        raise ResidentCleanupError(owner) from cleanup_error
