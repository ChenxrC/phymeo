"""Direct free-point reads from the distance and derivative function shipped with Newton 1.7.0."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import warp as wp
import newton
from newton._src.geometry.sdf_texture import TextureSDFData,texture_sample_sdf_grad

import runpy

HERE=Path(__file__).resolve().parent


@wp.kernel
def read_texture(texture:TextureSDFData,points:wp.array(dtype=wp.vec3),values:wp.array(dtype=wp.float32),
                 gradients:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    value,gradient=texture_sample_sdf_grad(texture,points[i])
    values[i]=value
    gradients[i]=gradient


def asset(kind):
    if kind=="u":
        return runpy.run_path(str(HERE/"baseline.py"))["_u_geometry"]([1,1,1],False)
    vertices,indices=runpy.run_path(str(HERE/"mesh_round_poses.py"))["geometry"]("box")
    half=(1/64,0.5,0.5) if kind=="thin" else (0.5,0.375,0.25)
    offset=(0.125,0,0) if kind=="thin" else (0,0,0)
    return np.asarray(vertices*half+offset,dtype=np.float32),indices


def main():
    import sys
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    assert version("newton")=="1.7.0"
    wp.config.kernel_cache_dir=str(Path(sys.prefix)/"warp-kernels")
    wp.init()
    device=wp.get_device("cuda:0")
    assert device.is_cuda
    probes=json.loads((HERE/"sdf_grid_points.json").read_text("utf-8"))
    output={}
    with wp.ScopedDevice(device),TemporaryDirectory(dir=args.out.parent,prefix="free-sdf-") as cache:
        for kind,case in probes.items():
            vertices,indices=asset(kind)
            mesh=newton.Mesh(vertices,indices,compute_inertia=False)
            sdf=mesh.build_sdf(device=device,max_resolution=64,margin=0.125,cache_dir=cache)
            tex=sdf.to_texture_kernel_data()
            assert tex is not None and str(tex.subgrid_start_slots.device)=="cuda:0"
            host=np.asarray(case["points_m"],dtype=np.float32)
            points=wp.array(host,dtype=wp.vec3,device=device)
            values=wp.empty(len(host),dtype=wp.float32,device=device)
            gradients=wp.empty(len(host),dtype=wp.vec3,device=device)
            wp.launch(read_texture,dim=len(host),inputs=[tex,points],outputs=[values,gradients],device=device)
            wp.synchronize_device(device)
            vv,gg=values.numpy(),gradients.numpy()
            assert np.isfinite(vv).all() and np.isfinite(gg).all()
            lo,hi=np.asarray(list(tex.sdf_box_lower)),np.asarray(list(tex.sdf_box_upper))
            output[kind]=dict(voxel_size_m=list(tex.voxel_size),num_subgrids=int(tex.num_subgrids),
                native_box_m=[lo.tolist(),hi.tolist()],native_queries=[
                    dict(position_m=p.tolist(),distance_m=float(v),gradient=g.tolist(),
                         outside_texture_box=bool(np.any((p<lo)|(p>hi))))
                    for p,v,g in zip(host,vv,gg,strict=True)])
            print(kind, "free queries",len(host),"range",float(vv.min()),float(vv.max()))
    with args.out.open("x",encoding="utf-8") as fp:
        json.dump(dict(newton=version("newton"),warp=version("warp-lang"),device=str(device),
                       interpolation="texture_sample_sdf_grad",assets=output),fp,allow_nan=False,indent=2)


if __name__=="__main__":
    main()
