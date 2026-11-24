# Galaxea R1 visual model

The film uses Galaxea’s original R1 with three independently steered wheels, a four-joint torso, twin A1 arms and G1 grippers. Its official ROS description is `r1_description`, formerly `r1_v2_1_0`. [Manufacturer’s simulation guide](https://docs.galaxea-ai.com/Guide/R1/R1_Simulation_Tutorial/) · [Pinned manufacturer files](https://github.com/userguide-galaxea/robot_descriptions/tree/72008ea950a4c9258387d30821a7d8d8a2b588ca/r1_description)

The package metadata declares BSD. The 34 binary STL visual files are ordinary Git blobs. The loader verifies each input against the object IDs in [origin.json](origin.json). Use this checkout to reproduce the model.

```powershell
git clone --filter=blob:none --no-checkout https://github.com/userguide-galaxea/robot_descriptions.git artifacts/rendering/galaxea-r1-assets
git -C artifacts/rendering/galaxea-r1-assets sparse-checkout init --cone
git -C artifacts/rendering/galaxea-r1-assets sparse-checkout set r1_description
git -C artifacts/rendering/galaxea-r1-assets checkout --detach 72008ea950a4c9258387d30821a7d8d8a2b588ca
```

`load_model(asset_dir, cache_dir)` reads the visual meshes in metres, their local origins, the URDF joint tree and each visual’s declared RGB. URDF supplies no metallic or roughness measurements. The preview assigns metallic 0 and roughness 0.65. It uses each rim and tire’s source coordinates to set the chassis height and rolling radius.

The source meshes exceed the Vulkan renderer’s index capacity. For a part above 35,000 nondegenerate facets, conversion uses border-preserving quadric error decimation with `fast-simplification==0.2.0`, targeting 58% of those triangles. Smaller parts keep their geometry except exact zero-area triangles. All visual links remain present. The glTF files store smooth corner normals across shallow connected facets and preserve sharp seams. Per-part counts, input and output bounds, and largest bounds change in metres are available in `model.diagnostics`. Conversion runs when preparing a fresh asset cache.
