"""R1 viewing presets preserve scene bindings and distinguish physical finishes."""
from dataclasses import replace
import unittest

from galaxea_r1.finishes import apply_finish
from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.scene import DirectionalLight, Instance, Pose, VisualScene


class R1Finishes(unittest.TestCase):
    def setUp(self):
        self.links = ("wheel_motor_link1", "zed_link", "base_link", "left_gripper_finger_link1", "torso_link2")
        materials = tuple(Material(f"r1-material-{link}", (.45,.45,.45), 0., .65) for link in self.links)
        self.floor = Material("r1-stage-mineral", (.21,.265,.28), 0., .85)
        mesh = Mesh("registered-piece", "fixture.gltf")
        instances = tuple(Instance(i, mesh.id, mat.id, link, Pose((.01*i,0.,0.)))
                          for i, (mat,link) in enumerate(zip(materials,self.links), 1))
        self.scene = VisualScene("r1-fixture", (mesh,), (*materials,self.floor), instances,
                                 (DirectionalLight((0.,0.,-1.), (1.,1.,1.)),))

    def test_urdf_preserves_original_scene_and_rejects_unknown_preset(self):
        self.assertIs(apply_finish(self.scene, "urdf"), self.scene)
        with self.assertRaises(ValueError):
            apply_finish(self.scene, "unknown-preset")

    def test_studio_changes_only_declared_finishes_and_scene_identity(self):
        result = apply_finish(self.scene, "studio")
        self.assertNotEqual(result.key, self.scene.key)
        self.assertEqual(replace(result, key=self.scene.key, materials=self.scene.materials), self.scene)
        by_id = {item.id: item for item in result.materials}
        self.assertEqual(set(by_id), {item.id for item in self.scene.materials})
        self.assertEqual(by_id[self.floor.id], self.floor)

        parts = [by_id[f"r1-material-{link}"] for link in self.links]
        self.assertEqual(len({(p.base_color,p.roughness,p.metallic) for p in parts}), len(parts))
        tire, _, shell, _, link = parts
        self.assertGreater(tire.roughness, shell.roughness)
        self.assertGreater(link.metallic, tire.metallic)
        self.assertLess(max(tire.base_color), max(shell.base_color))

    def test_polish_changes_only_the_head_and_white_arm_covers(self):
        covers = {"zed_link"} | {f"{side}_{part}" for side in ("left", "right")
                  for part in ("arm_link1", "arm_link2", "arm_link3", "gripper_link")}
        links = covers | {"torso_link4", "left_arm_link5", "left_gripper_finger_link1",
                          "wheel_motor_link2"}
        scene = replace(self.scene, instances=(), materials=tuple(
            Material("r1-material-"+link, (.45,.45,.45), 0., .65) for link in sorted(links)))
        previous = apply_finish(scene, "studio")
        polished = apply_finish(scene, "polished")
        self.assertNotEqual(polished.key, previous.key)
        for before, after in zip(previous.materials, polished.materials, strict=True):
            link = after.id.removeprefix("r1-material-")
            if link in covers:
                self.assertLess(after.roughness, before.roughness)
                self.assertEqual(replace(after, roughness=before.roughness), before)
            else:
                self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
