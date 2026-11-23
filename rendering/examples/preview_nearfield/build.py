"""Compile only the two compact GI guide variants for finite-range diffuse light.

The source package and every other stage keep their own immutable identity.
A far-field HDR environment handles distant transport. A nearby occluder
progressively replaces it with the radiance of the actual first intersection.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess


GUIDES = ("resident_rayquery.frag.spv", "resident_rayquery_normal.frag.spv")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def replace(text, old, new, *, count=1):
    if text.count(old) != count:
        raise ValueError("unsupported upstream source: " + old[:70])
    return text.replace(old, new)


def build(parent, compiler, output, radius_mm):
    if type(radius_mm) is not int or not 1 <= radius_mm <= 10000:
        raise ValueError("radius must be an integer within 1..10000 mm")
    parent = parent.resolve(strict=True)
    compiler = compiler.resolve(strict=True)
    source_manifest = parent / "source-manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    contract = manifest["contract"]
    if (contract.get("compact", {}).get("tile") != 4 or contract.get("estimator") != "visibility-ratio"
            or contract.get("area_light_grid") != 2 or contract.get("bounces") != 1):
        raise ValueError("requires the compact 2x2 area-light, one-bounce visibility-ratio parent")
    names = {}
    for key, expected in manifest["generated_sha256"].items():
        path = (parent / key).resolve(strict=True)
        if not path.is_relative_to(parent) or digest(path) != expected:
            raise ValueError("invalid original source " + key)
        name = path.relative_to(parent).as_posix()
        if name in names:
            raise ValueError("duplicate input file")
        names[name] = expected
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(parent / "source", output / "source")
    shutil.copytree(parent / "runtime", output / "runtime")
    built = output / "build"
    built.mkdir()
    for path in [parent / "build/phymeo_native.dll", *sorted((parent / "build").glob("*.spv"))]:
        if path.name not in GUIDES:
            shutil.copyfile(path, built / path.name)
    shaders = output / "source/_shaders"

    def edit(name, change):
        path = shaders / name
        path.write_text(change(path.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
        names[path.relative_to(output).as_posix()] = digest(path)

    # Capture the real Vulkan ray parameter before material decoding, including
    # a black/degenerate terminal hit. A miss never consumes this value.
    edit("resident.frag", lambda s: replace(s, "float metal,roughness; bool valid; };",
        "float metal,roughness; bool valid;\n#ifdef PR_PREVIEW_SPARSE_GUIDE\nfloat previewDistance;\n#endif\n};"))

    radius = radius_mm / 1000.
    def trace(s):
        a = s.index("bool residentTraceSurface(")
        b = s.index("uint slot=uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query,true));", a)
        prefix = s[a:b]
        if "3.402823466e38" not in prefix:
            raise ValueError("expected unbounded original queries")
        # Only the two guide modules are compiled. Keep opaque strict-endpoint
        # retry and candidate filtering; exactly at radius the model is sky.
        prefix = prefix.replace("3.402823466e38", "PR_DIFFUSE_GUIDE_LIMIT")
        limit=(f"#ifdef PR_PREVIEW_SPARSE_GUIDE\n#define PR_DIFFUSE_GUIDE_LIMIT {radius:.4f}\n"
               "#else\n#define PR_DIFFUSE_GUIDE_LIMIT 3.402823466e38\n#endif\n")
        return (limit+s[:a]+prefix+"\n#ifdef PR_PREVIEW_SPARSE_GUIDE\n"
                "    hit.previewDistance=rayQueryGetIntersectionTEXT(query,true);\n#endif\n    "+s[b:])
    edit("resident_reflection.glsl", trace)

    def guide(s):
        s = "layout(constant_id=96) const uint PR_NEARFIELD_DIFFUSE_ABI=1u;\n" + s
        s = replace(s, "if(PR_PREVIEW_SKY_VISIBILITY_RATIO_ABI!=1u", "if(PR_NEARFIELD_DIFFUSE_ABI!=1u||PR_PREVIEW_SKY_VISIBILITY_RATIO_ABI!=1u")
        # Both sealed guide schemas exist. Later guides additionally publish
        # event bits for diagnostics; preserve these on the hit branch.
        call = "if(residentTraceTerminal(P,faceN,direction,terminal))"
        if call+"hitSum+=terminal*weight;" in s:
            s = replace(s, call+"hitSum+=terminal*weight;", call+" {hitSum+=terminal*weight;}")
        s = replace(s, call, "BounceSurface hit;\n        if(residentTraceSurface(P,faceN,direction,true,terminal,hit))")
        return replace(s, "hitSum+=terminal*weight;",
            "            float nearWeight=1.0-smoothstep(0.0,PR_DIFFUSE_GUIDE_LIMIT,hit.previewDistance);\n"
            "            hitSum+=terminal*(nearWeight*weight);\n"
            "            skyVisible+=sky*((1.0-nearWeight)*weight);\n"
            "            visibleWeight+=(1.0-nearWeight)*weight;")
    edit("preview_sparse_ratio.glsl", guide)
    for name in GUIDES:
        flags = ["PR_RESIDENT_RAY_QUERY=1","PR_RESIDENT_INDIRECT=1","PR_RESIDENT_PROGRESSIVE=1",
                 "PR_PREVIEW_SPARSE_GUIDE=1","PR_PREVIEW_TILE=4","PR_STUDIO_AREA_GRID=2",
                 "PR_PREVIEW_NEIGHBORS=25","PR_PREVIEW_RECONSTRUCTION_REVISION=2",
                 f"PR_PREVIEW_PATHS={contract['paths_per_anchor']}"]
        if "_normal." in name:
            flags.append("PR_RESIDENT_NORMAL_MAPS=1")
        cmd = [str(compiler), "-V", "--target-env", "vulkan1.2", "--glsl-version", "460",
               *["-D"+f for f in flags], str(shaders/"resident.frag"), "-o", str(built/name)]
        with (output/(name+".txt")).open("wb") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        if b"PR_NEARFIELD_DIFFUSE_ABI" not in (built/name).read_bytes():
            raise ValueError("missing local transport shader")
    files = {p.name:digest(p) for p in built.iterdir() if p.is_file()}
    for name,h in files.items():
        if name not in GUIDES and h != digest(parent/"build"/name):
            raise ValueError("unrelated binary changed")
    contract["local_diffuse"] = dict(radius_m=radius, fade="1-smoothstep(0,R,t)", marker=96,
        model="finite-range diffuse replacement against the original far-field SH; weighted hit radiance and sky visibility")
    manifest["generated_sha256"] = names
    manifest["local_transport_parent_sha256"] = digest(source_manifest)
    manifest["local_transport_source"] = str(parent)
    manifest["compiled_sha256"] = files
    (output/"source-manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for field in ("parent","compiler","output"):
        ap.add_argument("--"+field,type=Path,required=True)
    ap.add_argument("--radius-mm",type=int,required=True)
    args = ap.parse_args()
    build(args.parent,args.compiler,args.output,args.radius_mm)
