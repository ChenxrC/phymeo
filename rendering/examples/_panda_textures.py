"""Photographic wood mapped onto the existing workbench triangles in metres."""
from dataclasses import replace
from pathlib import Path

from _panda_scene import write_mesh
from phymeo_rendering.assets import Material, TextureMap, TexturedMaterial
from phymeo_rendering.mesh_data import load_mesh

IMAGE_DIR = Path(__file__).resolve().parent / "materials"
WOOD_SHA256 = "460dd08d240f4a1f02982415048c3c5c200f385db212da18dc7e2df68bf4d0be"


def textured_workshop(scene, directory):
    mesh = next(m for m in scene.meshes if m.id == "floor")
    source = load_mesh(mesh)
    uv = []
    for (x,y,z), (nx,ny,nz) in zip(source.positions,source.normals,strict=True):
        if abs(nz) > .5:
            horizontal, vertical = x,y
        elif abs(nx) > .5:
            horizontal, vertical = y,z
        else:
            horizontal, vertical = x,z
        # The source scan covers 1.5 metres; glTF V grows down the image.
        uv.append((.5+horizontal/1.5,.5-vertical/1.5))
    mapped = write_mesh(directory,"worktop-with-uv",source.positions,source.normals,source.triangles,
                        texcoords=uv)
    image = TextureMap((IMAGE_DIR/"wood_table_001_diff_1k.jpg").as_uri(),WOOD_SHA256)
    wood = TexturedMaterial("photographic-wood",(1.,1.,1.),0.,.40,image)
    rubber = Material("rubber-mat",(.035,.041,.045),0.,.83)
    steel = Material("steel-post",(.48,.52,.57),1.,.28)
    finishes = {2:rubber.id,3:wood.id,4:steel.id}
    return replace(scene,key=scene.key+":photographic-workshop-v1",
                   meshes=tuple(replace(m,uri=mapped.uri) if m.id==mesh.id else m for m in scene.meshes),
                   materials=(*scene.materials,wood,rubber,steel),
                   instances=tuple(replace(i,material_id=finishes[i.id]) if i.id in finishes else i
                                   for i in scene.instances))
