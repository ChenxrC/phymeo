"""Device-free contracts for opaque visual assets; no rendering is exercised."""

import builtins
import io
import math
import os
import socket
import tempfile
import unittest
import urllib.request
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, asdict
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from phymeo_rendering.assets import Material, Mesh


class MeshTests(unittest.TestCase):
    def test_asset_identity_and_uri_are_not_normalized(self):
        mesh = Mesh("  Tool.Mesh.α  ", "  file:///Assets/Part%20A.glb  ")
        self.assertEqual(mesh.id, "  Tool.Mesh.α  ")
        self.assertEqual(mesh.uri, "  file:///Assets/Part%20A.glb  ")

    def test_mesh_rejects_nonstring_identity(self):
        for value in (None, True, 1, b"mesh", ["mesh"]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError) as caught:
                    Mesh(value, "part.glb")
                self.assertIn("id", str(caught.exception))

    def test_mesh_rejects_blank_identity(self):
        for value in ("", " ", "\t\n", "\u2003"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError) as caught:
                    Mesh(value, "part.glb")
                self.assertIn("id", str(caught.exception))

    def test_mesh_rejects_invalid_uri_with_asset_context(self):
        cases = (
            (None, TypeError),
            (False, TypeError),
            (1, TypeError),
            (b"part.glb", TypeError),
            (Path("part.glb"), TypeError),
            ("", ValueError),
            ("\t \n", ValueError),
        )
        for value, error in cases:
            with self.subTest(value=value):
                with self.assertRaises(error) as caught:
                    Mesh("tool-mesh", value)
                self.assertIn("uri", str(caught.exception))
                self.assertIn("tool-mesh", str(caught.exception))

    def test_construction_is_a_declaration_without_file_or_network_io(self):
        with tempfile.TemporaryDirectory() as directory:
            absent_mesh = Path(directory) / "not-created.glb"
            self.assertFalse(absent_mesh.exists())
            uri = absent_mesh.as_uri()
            # Guard the external boundary, while exercising real constructors.
            with ExitStack() as guards:
                for module, name in (
                    (builtins, "open"),
                    (io, "open"),
                    (os, "stat"),
                    (os, "listdir"),
                    (os, "scandir"),
                    (urllib.request, "urlopen"),
                    (socket, "create_connection"),
                ):
                    guards.enter_context(
                        patch.object(
                            module,
                            name,
                            side_effect=AssertionError("asset construction performed IO"),
                        )
                    )
                mesh = Mesh("unloaded-mesh", uri)
                material = Material("unloaded-material", (0.2, 0.3, 0.4), 0, 0.5)
            self.assertEqual(mesh.uri, uri)
            self.assertEqual(material.base_color, (0.2, 0.3, 0.4))
            self.assertFalse(absent_mesh.exists())

    def test_mesh_declaration_remains_frozen(self):
        mesh = Mesh("tool-mesh", "part.glb")
        with self.assertRaises(FrozenInstanceError):
            mesh.uri = "other.glb"
        self.assertEqual(mesh.uri, "part.glb")


class MaterialTests(unittest.TestCase):
    def test_three_opaque_materials_preserve_linear_values(self):
        # Numeric test fixtures, not measured physical material calibration.
        fixtures = (
            ("metal", (0.7, 0.7, 0.7), 1, 0.2),
            ("plastic", (0.2, 0.3, 0.4), 0, 0.5),
            ("rubber", (0.03, 0.03, 0.03), 0, 0.9),
        )
        for identity, color, metallic, roughness in fixtures:
            with self.subTest(material=identity):
                material = Material(identity, color, metallic, roughness)
                self.assertEqual(
                    asdict(material),
                    {
                        "id": identity,
                        "base_color": color,
                        "metallic": metallic,
                        "roughness": roughness,
                    },
                )

    def test_material_identity_is_not_normalized(self):
        material = Material("  Material.α  ", (0.2, 0.3, 0.4), 0, 0.5)
        self.assertEqual(material.id, "  Material.α  ")

    def test_material_rejects_nonstring_identity(self):
        for value in (None, False, 1, b"material", ["material"]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError) as caught:
                    Material(value, (0.2, 0.3, 0.4), 0, 0.5)
                self.assertIn("id", str(caught.exception))

    def test_material_rejects_blank_identity(self):
        for value in ("", " ", "\t\n", "\u2003"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError) as caught:
                    Material(value, (0.2, 0.3, 0.4), 0, 0.5)
                self.assertIn("id", str(caught.exception))

    def test_closed_unit_interval_accepts_endpoints_without_coercion(self):
        material = Material("endpoints", (0, 1, -0.0), 0, 1)
        self.assertEqual(material.base_color, (0, 1, -0.0))
        self.assertIs(type(material.base_color[0]), int)
        self.assertEqual(material.base_color[2].hex(), "-0x0.0p+0")
        self.assertIs(type(material.metallic), int)
        self.assertIs(type(material.roughness), int)

    def test_real_numbers_are_valid_without_float_conversion(self):
        half = Fraction(1, 2)
        material = Material("rational", (half, 0, 1), half, half)
        self.assertEqual(material.base_color[0], Fraction(1, 2))
        self.assertIs(type(material.base_color[0]), Fraction)
        self.assertIs(type(material.metallic), Fraction)
        self.assertIs(type(material.roughness), Fraction)

    def test_material_rejects_mutable_or_non_tuple_color(self):
        color = [0.2, 0.3, 0.4]
        for value in (color, None, "rgb", {"r": 0.2, "g": 0.3, "b": 0.4}):
            with self.subTest(value=value):
                with self.assertRaises(TypeError) as caught:
                    Material("surface", value, 0, 0.5)
                self.assertIn("base_color", str(caught.exception))
                self.assertIn("surface", str(caught.exception))
        self.assertEqual(color, [0.2, 0.3, 0.4])

    def test_material_rejects_wrong_color_length(self):
        for value in ((), (0.2,), (0.2, 0.3), (0.2, 0.3, 0.4, 0.5)):
            with self.subTest(color=value):
                with self.assertRaises(ValueError) as caught:
                    Material("surface", value, 0, 0.5)
                self.assertIn("base_color", str(caught.exception))
                self.assertIn("surface", str(caught.exception))

    def test_material_rejects_nonreal_or_boolean_components(self):
        for field in (
            "base_color[0]",
            "base_color[1]",
            "base_color[2]",
            "metallic",
            "roughness",
        ):
            for value in (None, True, False, "0.5", 0.5 + 0j, [0.5]):
                with self.subTest(field=field, value=value):
                    self._assert_bad_component(field, value, TypeError)

    def test_material_rejects_nonfinite_or_out_of_range_components(self):
        invalid_values = (
            math.nan,
            math.inf,
            -math.inf,
            -0.01,
            1.01,
            math.nextafter(0.0, -math.inf),
            math.nextafter(1.0, math.inf),
            10**1000,
            -(10**1000),
        )
        for field in (
            "base_color[0]",
            "base_color[1]",
            "base_color[2]",
            "metallic",
            "roughness",
        ):
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    self._assert_bad_component(field, value, ValueError)

    def test_material_declaration_remains_frozen(self):
        material = Material("surface", (0.2, 0.3, 0.4), 0, 0.5)
        with self.assertRaises(FrozenInstanceError):
            material.base_color = (0.7, 0.7, 0.7)
        self.assertEqual(material.base_color, (0.2, 0.3, 0.4))

    def _assert_bad_component(self, field, value, error):
        color = [0.2, 0.3, 0.4]
        metallic = 0
        roughness = 0.5
        if field.startswith("base_color["):
            color[int(field[-2])] = value
        elif field == "metallic":
            metallic = value
        else:
            roughness = value
        with self.assertRaises(error) as caught:
            Material("surface", tuple(color), metallic, roughness)
        self.assertIn(field, str(caught.exception))
        self.assertIn("surface", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
