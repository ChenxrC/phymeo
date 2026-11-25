"""Atomically publish derived geometry, textures and receipts in local caches."""
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep


def _matches(path, data):
    try:
        with path.open('rb') as src:
            return os.fstat(src.fileno()).st_size == len(data) and src.read(len(data)+1) == data
    except FileNotFoundError:
        return False


def _retry_sharing(operation):
    # A virus scanner or another native reader can briefly deny access on Windows.
    for attempt in range(10):
        try:
            return operation()
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 9:
                raise
            sleep(min(.01 * 2**attempt, .12))


def publish_bytes(path: Path, data: bytes) -> None:
    """Publish fully formed bytes; competing writers leave a verified winner in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _retry_sharing(lambda: _matches(path, data)):
        return
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as out:
            temporary = Path(out.name)
            out.write(data)

        def publish():
            if _matches(path, data):
                return
            try:
                if os.name == 'nt':
                    os.rename(temporary, path)  # Win32 rename without replace-existing.
                else:
                    os.link(temporary, path)    # Atomically create the name if absent.
            except FileExistsError:
                if not _matches(path, data):
                    os.replace(temporary, path)  # Only a stale or corrupt entry needs repair.

        _retry_sharing(publish)
    finally:
        if temporary is not None:
            _retry_sharing(lambda: temporary.unlink(missing_ok=True))
