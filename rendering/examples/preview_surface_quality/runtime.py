"""Public capture and sensor API for the optional surface-quality package."""
from preview_compact_rigid_history.runtime import RigidHistoryBackend
from .build import check_package


class SurfaceQualityBackend(RigidHistoryBackend):
    def __init__(self, directory, **settings):
        root, _ = check_package(directory)
        super().__init__(root, **settings)
