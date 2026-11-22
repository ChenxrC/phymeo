import numpy as np
import pytest

from experiments.area_pressure.pusht_lookahead import shortlisted


def test_screen_keeps_wrong_yaw_for_path_test_but_excludes_unreachable_or_missing_faces():
    def proposal(index, x, direction):
        return dict(
            index=index,
            distance_ordinal=0,
            input_binding=str(index),
            point_world=(x, 0, 0),
            precontact_center_world=(x, 0, 0),
            direction_world=direction,
            reference_usable=True,
            selection_reason="yaw_wrong_signed_moment",
        )

    proposals = [
        proposal(27, 0.6, (0, 1, 0)),
        proposal(2, 0.62, (0, 1, 0)),
        proposal(7, 0.5, (1, 0, 0)),
        proposal(5, 0.4, (0, 1, 0)),
    ]
    faces = [
        dict(
            index=p["index"],
            distance_ordinal=0,
            input_binding=p["input_binding"],
            eligible=p["index"] != 5,
        )
        for p in proposals
    ]
    args = (proposals, faces, np.zeros(2), [0, 1], np.zeros(2), 0.6, 0.0005)
    screen = shortlisted(*args)
    assert screen["indices"] == [27]
    assert screen["rows"][1]["precontact_excess_m"] == pytest.approx(0.02)
    faces[0]["input_binding"] = "wrong-source"
    with pytest.raises(ValueError, match="proposal identity"):
        shortlisted(*args)
