"""Deterministic full-rate diffuse IBL, local visibility and current-frame contour AA."""
from preview_compact_rigid_history.runtime import (
    RigidHistoryBackend, RigidHistorySession, activate as activate_transport,
)
from .build import check_package


def activate(package):
    root, manifest = check_package(package)
    activate_transport(manifest["parent"])
    return root, manifest


class StableSession(RigidHistorySession):
    def diagnostics(self):
        data = super().diagnostics()
        data.update(backend="native-stable-motion-preview",
                    samples_scope="Full-resolution receivers use their own prefiltered environment and local visibility.",
                    unobserved_secondary_visibility=False)
        data["history"].update(mode="off", scope="current frame", dynamic_scene="current rigid transforms and TLAS",
                               age_scope="no accumulated lighting")
        profile = data["device"].get("resident_profile")
        if profile is not None:
            profile["gpu_scope"] = "TLAS, primary panel lighting and local visibility, current-frame resolve, contour AA and copies"
        return data


class StableBackend(RigidHistoryBackend):
    def __init__(self, package, **settings):
        root, manifest = check_package(package)
        # Both AO controls share the sealed Python runtime, unchanged native DLL
        # and contour shader. Only the fragment/reconstruction SPIR-V differs.
        super().__init__(manifest["parent"], reuse_history=False, **settings)
        self.root = root
        self.manifest["contract"]["stable_lighting"] = manifest["stable_lighting"]

    def _new_session(self, underlying):
        return StableSession(self, underlying)


def open_session(package, scene, studio, device_index=1, validation=False):
    return StableBackend(package, studio=studio, device_index=device_index,
                         validation=validation, exposure=8.).open_scene(scene)
