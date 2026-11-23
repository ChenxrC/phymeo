"""Portable saved-data/page contracts; no ffmpeg or rendering."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

_PATH=list(sys.path)
try:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import deliver
finally:sys.path[:]=_PATH


class DeliveryContracts(unittest.TestCase):
    def test_display_cost_comes_from_completed_wall_capture_not_gpu_or_playback(self):
        frames=[dict(capture_ms=float(i)) for i in range(1,301)]
        saved=dict(count=300,mean_ms=150.5,p50_ms=150.,p95_ms=285.,max_ms=300.,completed_fps_from_mean=1000/150.5)
        value=deliver.measured_timing(frames,saved)
        self.assertEqual(value["p95_ms"],285.)
        self.assertAlmostEqual(value["fps"],1000/150.5)
        self.assertNotEqual(value["fps"],30.)
        for field in ("p95_ms","mean_ms"):
            bad=deepcopy(saved);bad[field]=30.
            with self.assertRaises(ValueError):deliver.measured_timing(frames,bad)

    def test_same_view_requires_all_actual_frame_and_camera_identities(self):
        row=dict(pose_index=0,frame_key={"tick":1},time_s=.033,state_sha256="state",camera_sha256="camera")
        deliver.same_inputs([row],[deepcopy(row)])
        for key in ("frame_key","camera_sha256","state_sha256","time_s"):
            bad=deepcopy(row);bad[key]="wrong"
            with self.assertRaises(ValueError):deliver.same_inputs([row],[bad])

    def test_chinese_cost_table_escapes_labels_and_preserves_failed_quality(self):
        lane=dict(label="<candidate>",profile="compact",camera="sequence",look="recorded",extent=[1280,720],
            quality_status="failed_observed_gate",timing=dict(fps=21.,mean_ms=47.,p95_ms=60.,max_ms=80.),
            measured_groups=[dict(round=1,channels="rgb",timing=dict(fps=21.,mean_ms=47.,p95_ms=60.,max_ms=80.))])
        text=deliver.cost_table([lane])
        self.assertIn("&lt;candidate&gt;",text)
        self.assertIn("failed_observed_gate",text)
        self.assertIn("P95",text)
        self.assertIn("播放",text)
        self.assertIn("sequence",text)
        self.assertNotIn("<candidate>",text)


    @staticmethod
    def saved_media_fixture():
        preset="compact-t4-p4-g2-k25-r2-visibility-ratio-local-120mm"
        row=dict(name="sparse",channels="rgb",round=0,status="complete",extent=[1280,720],
            profile={"capture_ms":47.},profile_label=preset,receipt="original/receipt.json",receipt_sha256="source-receipt",
            frames=[dict(pose_index=i,frame_key={"tick":100+i},time_s=i/30,
                state_sha256=f"state-{i}",camera_sha256=f"camera-{i}",png_sha256=f"png-{i}") for i in range(300)])
        view_row=dict(deepcopy(row),preset=preset,receipt=preset+"/receipt.json",receipt_sha256="view-receipt")
        view=dict(schema="phymeo-panda-motion-preview-v1",status="complete",quick=False,
            requested_presets=[preset],runs=[view_row])
        movie=dict(file="panda-"+preset+".mp4",source_png_sha256=[f["png_sha256"] for f in row["frames"]])
        return row,view,dict(movies=[movie])

    def test_saved_encoded_profile_keeps_bound_local_distance_suffix(self):
        row,view,media=self.saved_media_fixture()
        profile,movie,borrowed=deliver.bound_encoded_movie(view,media,row)
        self.assertEqual(profile,row["profile_label"])
        self.assertTrue(profile.endswith("-local-120mm"))
        self.assertEqual(movie["file"],"panda-"+profile+".mp4")
        self.assertEqual(borrowed,view["runs"][0])
        # Historical timing dictionaries stay timing dictionaries.
        self.assertEqual(row["profile"],{"capture_ms":47.})

    def test_encoded_view_cannot_relabel_or_change_the_capture(self):
        row,view,media=self.saved_media_fixture()
        for change in ("label","camera","timing","preset"):
            bad=deepcopy(view)
            if change=="label":bad["runs"][0]["profile_label"]="different-policy"
            if change=="camera":bad["runs"][0]["frames"][59]["camera_sha256"]="other-camera"
            if change=="timing":bad["runs"][0]["profile"]["capture_ms"]=1.
            if change=="preset":bad["requested_presets"]=["other-movie"]
            with self.subTest(change=change),self.assertRaises(ValueError):
                deliver.bound_encoded_movie(bad,media,row)

    def test_encoded_movie_requires_one_safe_name_and_exact_png_sequence(self):
        row,view,media=self.saved_media_fixture()
        for change in ("duplicate","missing","png","unsafe"):
            bad_view=deepcopy(view);bad_media=deepcopy(media);bad_row=deepcopy(row)
            if change=="duplicate":bad_media["movies"]*=2
            if change=="missing":bad_media["movies"][0]["file"]="unbound.mp4"
            if change=="png":bad_media["movies"][0]["source_png_sha256"][149]="different-pixel-source"
            if change=="unsafe":
                bad_view["runs"][0]["preset"]="../other"
                bad_view["requested_presets"]=["../other"]
                bad_view["runs"][0]["profile_label"]="../other";bad_row["profile_label"]="../other"
                bad_media["movies"][0]["file"]="panda-../other.mp4"
            with self.subTest(change=change),self.assertRaises(ValueError):
                deliver.bound_encoded_movie(bad_view,bad_media,bad_row)

if __name__=="__main__":unittest.main()