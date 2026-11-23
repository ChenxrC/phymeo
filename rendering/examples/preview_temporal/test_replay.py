"""Saved-data binding and failure controls, independent of device execution."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from PIL import Image

_PATH=list(sys.path)
try:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import collect
    import replay
finally:sys.path[:]=_PATH


def fixture(root):
    pose=dict(position_m=[0.,0.,0.],orientation_xyzw=[0.,0.,0.,1.])
    key=dict(scene_key="scene",world_id="world",epoch=0,revision=55,tick=56,phase="accepted_endpoint")
    frame=dict(key=key,time_s=56/30,nodes=[dict(node_id="body",world_pose=deepcopy(pose))])
    camera=dict(id="cam",frame_key=key,world_pose=deepcopy(pose),width=8,height=8,fx_px=50.,fy_px=50.,cx_px=4.,cy_px=4.)
    scene=dict(instances=[dict(id=156,node_id="body",local_pose=deepcopy(pose))])
    binding=dict(scene=scene,frames=[dict(pose_index=55,frame=frame,camera=camera)])
    raw=np.zeros((8,8,4),dtype="<f4");raw[...,:3]=2.;raw[...,3]=512+(512<<10)+(1<<20)
    depth=np.ones((8,8),dtype="<f4");ids=np.full((8,8),156,dtype="<u4");valid=np.ones((8,8),dtype="u1")
    np.savez_compressed(root/"frame.npz",guide=raw,depth=depth,instance_id=ids,valid=valid)
    pixels=np.full((8,8,3),100,dtype="u1");Image.fromarray(pixels).save(root/"public.png")
    tokens=dict(material_token="material",lighting_token="lights",signal_token="incoming")
    meta=collect.metadata_for(scene,frame,camera,**tokens)
    row=dict(pose_index=55,file="frame.npz",sha256=collect.digest(root/"frame.npz"),metadata=meta,
        frame_sha256=sha256(collect.canonical(frame)).hexdigest(),camera_sha256=sha256(collect.canonical(camera)).hexdigest(),
        guide_sha256=sha256(raw.tobytes()).hexdigest(),public_png="public.png",
        public_png_sha256=collect.digest(root/"public.png"),public_rgb_sha256=sha256(pixels.tobytes()).hexdigest(),
        sensor_sha256={"forward_depth_m_f32":sha256(depth.tobytes()).hexdigest(),
            "instance_id_u32":sha256(ids.tobytes()).hexdigest(),"valid_u8":sha256(valid.tobytes()).hexdigest()})
    return binding,row


class ReplayContracts(unittest.TestCase):
    def test_camera_control_is_a_fixed_point_one_millimetre_per_ten_steps_in_local_x(self):
        pose=dict(position_m=[1.,2.,3.],orientation_xyzw=[0.,0.,2**-.5,2**-.5])
        np.testing.assert_allclose(collect.control_camera_position(pose,0),[1.,2.,3.],atol=0,rtol=0)
        np.testing.assert_allclose(collect.control_camera_position(pose,1),[1.,2.0001,3.],atol=1e-15,rtol=0)
        np.testing.assert_allclose(collect.control_camera_position(pose,2),[1.,2.0002,3.],atol=1e-15,rtol=0)
    def test_real_packet_layout_is_bound_to_input_and_each_original_plane(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);binding,row=fixture(root)
            packet=replay.load_packet(root,binding,row)
            self.assertEqual(packet["key"]["tick"],56)
            self.assertEqual(packet["instance_id"].shape,(8,8))
            self.assertTrue(packet["traced"].all())
            for key in ("guide_sha256","public_rgb_sha256"):
                bad=deepcopy(row);bad[key]="0"*64
                with self.assertRaises(ValueError):replay.load_packet(root,binding,bad)

    def test_saved_camera_transform_or_identity_cannot_be_relabelled(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);binding,row=fixture(root)
            for mutate in (lambda r:r["metadata"]["camera_matrix"][0].__setitem__(3,.01),
                           lambda r:r["metadata"]["key"].__setitem__("tick",57),
                           lambda r:r["sensor_sha256"].__setitem__("valid_u8","bad")):
                bad=deepcopy(row);mutate(bad)
                with self.assertRaises(ValueError):replay.load_packet(root,binding,bad)

    def test_background_and_untraced_values_cannot_count_as_history_improvement(self):
        current=np.ones((2,2,3));result=dict(incoming=current.copy(),age=np.zeros((2,2),dtype=np.uint16),
            status=np.zeros((2,2),dtype=np.uint8))
        result["incoming"][0,0]=2.
        with self.assertRaises(ValueError):replay.audit_fallback_exact(current,result)
        result["incoming"]=current.copy()
        self.assertEqual(replay.audit_fallback_exact(current,result)["accepted_pixels"],0)

    def test_collection_rejects_low_resolution_compact_or_missing_frames(self):
        report=dict(schema=collect.SCHEMA,status="complete",start=55,count=2,
            source={"contract":{"bounces":1}},records=[{"pose_index":55},{"pose_index":56}])
        replay.audit_collection(report)
        bad=deepcopy(report);bad["source"]["contract"]["compact"]={"schema":"phymeo-preview-compact-v1"}
        with self.assertRaises(ValueError):replay.audit_collection(bad)
        bad=deepcopy(report);bad["records"][1]["pose_index"]=57
        with self.assertRaises(ValueError):replay.audit_collection(bad)


if __name__=="__main__":unittest.main()