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


if __name__ == "__main__":
    unittest.main()
