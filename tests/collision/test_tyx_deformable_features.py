"""Only face broad phase uses TYX; the full material primitive set is constructed on HOST."""
import json
import pytest
import phymeo as ph
from test_deformable_features import CAT,key,query,readbacks,register

pytestmark=pytest.mark.tyx


def test_real_broad_phase_supplies_faces_for_every_native_fold_feature(tyx_mode,tmp_path):
    shapes,paths=register(CAT["DD-REMOTE-FOLD"])
    model=ph.CollisionModel(shapes)
    with ph.CollisionWorkspace(model,tyx_mode,cache_root=tmp_path/"cache") as work:
        before=work.stats()
        result=query(model,paths,work)
        after=work.stats()
        assert work.last_result is None
    assert result.status=="ok" and result.backend==tyx_mode and result.narrow_backend=="host"
    assert after["completed_launches"]>before["completed_launches"]
    if tyx_mode=="cuda":
        assert after["downloads"]>before["downloads"] and after["downloaded_bytes"]>before["downloaded_bytes"]
    rows={key(r):r for r in result.contacts}
    readbacks(rows,"cpu")
    readbacks(rows,"cuda:0")
    assert len(rows)==32 and result.broad_face_pairs>0
    (tmp_path/"deformable-features-device.json").write_text(json.dumps(dict(backend=result.backend,narrow=result.narrow_backend,
        broad=result.broad_face_pairs,feature_tests=result.feature_tests,retained=len(rows),
        stats={k:v for k,v in after.items() if k in ("completed_launches","uploads","downloads","uploaded_bytes","downloaded_bytes","synchronizations")}),
        allow_nan=False),encoding="utf-8")
