"""Retryable release of resident dependencies, under their existing owner lock."""


# Unknown outcomes retain the callback and its owners for process lifetime.
_UNKNOWN_RELEASE_STACKS = {}


class RetryableReleaseError(RuntimeError):
    """The releaser guarantees no release occurred; close may safely be retried."""


class UnknownReleaseError(RuntimeError):
    """A native release outcome is unknown; the callback must not run again."""


class ReleaseStack:
    """Release in reverse acquisition order; retain the failed callback and ancestors."""

    def __init__(self):
        self._callbacks = []
        self._unknown = None
        self._started = False
        self._active = False

    def callback(self, function, *args):
        if self._started:
            raise RuntimeError("cannot register resources after cleanup started")
        self._callbacks.append((function, args))

    def close(self):
        if self._active:
            raise RuntimeError("recursive resource cleanup")
        if self._unknown is not None:
            raise self._unknown
        self._started = True
        self._active = True
        try:
            self._release()
        finally:
            self._active = False

    def _release(self):
        while self._callbacks:
            function, args = self._callbacks[-1]
            try:
                function(*args)
            except RetryableReleaseError:
                raise
            except BaseException as error:
                self._unknown = UnknownReleaseError(
                    "UNKNOWN_RELEASE; retained owner cannot retry native release"
                )
                _UNKNOWN_RELEASE_STACKS[id(self)] = self
                raise self._unknown from error
            self._callbacks.pop()


class ResidentCleanupError(RuntimeError):
    """Construction failed; cleanup_owner retains pending or quarantined resources."""

    def __init__(self, owner):
        super().__init__("resident construction cleanup failed; inspect cleanup_owner.close()")
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
