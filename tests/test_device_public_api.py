"""Device factories stay importable without installing the optional runtime."""

import phymeo


def test_public_device_participants_and_runtime_are_exported():
    for name in (
        "TyxRobotParticipant",
        "TyxClothParticipant",
        "TyxTetrahedralParticipant",
        "TyxArticulatedCouplingProgram",
        "make_tyx_articulated_runtime",
    ):
        assert name in phymeo.__all__
        assert callable(getattr(phymeo, name))
