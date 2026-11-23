"""Prepare a NEW private native/runtime copy, without invoking a compiler.

python -m preview_sparse.build_source --rendering-root RENDERING --output NEW --tile 2 --paths 4 --grid 2
Then configure NEW/source/_native with CMake into NEW/build. All exact patch
anchors fail closed on upstream drift; source and generated hashes are archived.
"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import struct


HERE=Path(__file__).resolve().parent


def contract(*,tile,paths,grid,neighbors=4,reconstruction_revision=1,estimator="radiance"):
    for name,value,allowed in (("tile",tile,(1,2,4)),("paths",paths,(2,4,16)),("grid",grid,(2,4,8)),("neighbors",neighbors,(4,25))):
        if type(value) is not int:raise TypeError(name+" must be an integer")
        if value not in allowed:raise ValueError("unsupported "+name)
    if type(reconstruction_revision) is not int or reconstruction_revision not in (1,2):
        raise ValueError("reconstruction_revision must be 1 or 2")
    if estimator not in ("radiance","visibility-ratio"):
        raise ValueError("unknown incoming estimator")
    return dict(schema="phymeo-preview-sparse-v1",tile=tile,paths_per_anchor=paths,area_light_grid=grid,neighbor_candidates=neighbors,
                estimator=estimator,
                guide_outcome_flags=estimator=="visibility-ratio",
                estimator_scope="ordinary weighted Monte Carlo incoming" if estimator=="radiance" else "per-channel sampled visible-sky/unoccluded-sky ratio times existing SH plus unchanged weighted hit/N; finite-sample ratio bias and SH approximation, Ng-clipped samples remain denominator-only; zero channel denominator uses same-weight miss fraction",
                reconstruction_revision=reconstruction_revision,
                reconstruction_implementation="clamped candidates + duplicate scan + per-neighbor atomic" if reconstruction_revision==1 else "bounded unique tile indices + one accepted-neighbor atomic per supported receiver",
                history="none; one independent exposure per capture",guide_format="RGBA32F",
                fallback="current-pixel unoccluded diffuse environment SH; not traced GI",
                bounces=1,normal_encoding="octahedral 10+10 bits plus traced flag, exactly represented float integer",
                rho_encoding="RGB UNORM8 as exact positive float integer; absolute channel error <= 0.5/255",
                reconstruction=f"at most {neighbors} unique tile anchors; same actual ID, valid coverage, relative depth <= .003, shading-normal dot >= .98; normalized supported weights",
                weights="inverse 1+pixel-distance-squared" if neighbors==4 else "Gaussian sigma=1.5 tile widths, nearest 5x5 anchor window",
                anchor_filter="none" if neighbors==4 or tile==1 else "same 25-anchor filter, including the always-supported own sample",
                primary="full-rate direct + environment specular + optional near mirror; diffuse IBL overridden by zero",
                guide="full-rate current shading normal and local SH fallback; only eligible tile anchors trace real incoming light",
                timing="includes two full raster draws, true anchor rays/terminal shadows, reconstruction, all copies; no extra session",
                unsupported=["MSAA4","temporal accumulation","software ray traversal","two diffuse bounces"])


def anchor(pixel,extent,tile):
    return tuple(min((int(p)//tile)*tile+tile//2,int(e)-1) for p,e in zip(pixel,extent))


def pack_rho(rgb):
    if len(rgb)!=3 or any(not math.isfinite(x) or not 0<=x<=1 for x in rgb):raise ValueError("invalid rho")
    values=[math.floor(x*255+.5) for x in rgb]
    return float(values[0]|(values[1]<<8)|(values[2]<<16))


def unpack_rho(value):
    encoded=struct.unpack("<f",struct.pack("<f",value))[0]
    code=int(encoded)
    return tuple(((code>>(8*k))&255)/255 for k in range(3))


def replace_once(text,old,new,label):
    if text.count(old)!=1:raise ValueError(f"upstream patch anchor count differs: {label}: {text.count(old)}")
    return text.replace(old,new,1)


def _write(path,text):path.write_text(text,encoding="utf-8",newline="\n")
def _hash(path):return sha256(path.read_bytes()).hexdigest()


def prepare(rendering_root,output,*,tile=2,paths=4,grid=2,neighbors=4,reconstruction_revision=1,estimator="radiance"):
    policy=contract(tile=tile,paths=paths,grid=grid,neighbors=neighbors,reconstruction_revision=reconstruction_revision,estimator=estimator)
    root=Path(rendering_root).resolve();out=Path(output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    package=root/"src/phymeo_rendering"
    native=out/"source/_native";shaders=out/"source/_shaders"
    native.mkdir(parents=True);shaders.mkdir()
    source_hashes={}
    for origin,target,suffixes in ((package/"backends/_native",native,{".cpp",".h",".inc"}),
                                    (package/"backends/_shaders",shaders,{".glsl",".vert",".frag",".comp"})):
        for path in sorted(origin.iterdir()):
            if path.is_file() and path.suffix in suffixes:
                if path.stat().st_size>8*1024*1024:raise ValueError("unexpected oversized source")
                shutil.copyfile(path,target/path.name)
                source_hashes[str(path.relative_to(root))]=_hash(path)
    runtime=out/"runtime/phymeo_rendering"
    shutil.copytree(package,runtime,ignore=shutil.ignore_patterns("__pycache__","*.pyc","*.dll","*.spv","build*","_panda_assets"))
    profile=f"PR_PREVIEW_SPARSE_PROFILE_T{tile}_P{paths}_G{grid}_K{neighbors}"
    if reconstruction_revision!=1:profile+=f"_R{reconstruction_revision}"
    estimator_code=1 if estimator=="visibility-ratio" else 0
    if estimator_code:profile+="_Eratio"

    def change(relative,edits):
        path=out/relative;text=path.read_text(encoding="utf-8")
        for label,old,new in edits:text=replace_once(text,old,new,label)
        _write(path,text)

    change("source/_native/renderer.cpp",[
        ("context", "    PrResidentSpatialInfo spatial_info{};", "    PrResidentSpatialInfo spatial_info{};\n    bool sparse_shader=false;\n    uint32_t sparse_captures=0,sparse_sensor_captures=0,sparse_last_rgb=0;\n    std::array<uint32_t,8> sparse_counts{};"),
        ("guide attachment", "if(i==5)return VK_FORMAT_R8G8B8A8_UNORM;", "if(i==5)return sparse_shader?VK_FORMAT_R32G32B32A32_SFLOAT:VK_FORMAT_R8G8B8A8_UNORM;"),
        ("main marker", "        spatial_info.shader_compatible=progressive_info.shader_compatible;",
         f'''        sparse_shader=named_shader_marker(frag,frag_bytes,78,"PR_PREVIEW_SPARSE_PRIMARY_ABI")&&!named_shader_marker(frag,frag_bytes,79,"PR_PREVIEW_SPARSE_GUIDE_ABI")&&named_shader_marker(frag,frag_bytes,83,"{profile}");
        require(sparse_shader&&progressive_info.shader_compatible&&ray_query,"private sparse DLL requires matching sparse primary shader");
        spatial_info.shader_compatible=progressive_info.shader_compatible;'''),
        ("strict reset", "require(c.progressive_info.enabled&&!c.poisoned&&reset<=1,\"invalid progressive begin\");",
         "require(c.progressive_info.enabled&&!c.poisoned&&reset==1,\"sparse experiment requires fresh independent exposure\");"),
        ("private setup export", "int pr_resident_set_spatial(void* handle,uint32_t bounces,const uint32_t* code,size_t bytes,const uint32_t* guide,size_t guide_bytes){",
         "PR_API int pr_preview_sparse_set(void* handle,uint32_t bounces,const uint32_t* code,size_t bytes,const uint32_t* guide,size_t guide_bytes){"),
        ("private setup", "try{require(handle,\"null spatial context\");spatial_setup(*static_cast<Context*>(handle),bounces,code,bytes,guide,guide_bytes);return 0;}",
         f'''try{{require(handle,"null spatial context");
        require(bounces==1&&named_shader_marker(code,bytes,80,"PR_PREVIEW_SPARSE_COMPUTE_ABI",5)&&named_shader_marker(code,bytes,83,"{profile}",5),"private sparse compute/profile ABI differs");
        require(named_shader_marker(guide,guide_bytes,83,"{profile}"),"private sparse guide profile differs");
        require(named_shader_marker(guide,guide_bytes,84,"PR_PREVIEW_SKY_VISIBILITY_RATIO_ABI")==bool({estimator_code}),"private sparse estimator ABI differs");
        spatial_setup(*static_cast<Context*>(handle),bounces,code,bytes,guide,guide_bytes);return 0;}}'''),
    ])
    with (native/"renderer.cpp").open("a",encoding="utf-8") as stream:
        stream.write(f'''
PR_API uint32_t pr_preview_sparse_abi_version(){{return 1;}}
PR_API int pr_preview_sparse_info(void* handle,uint32_t* output,size_t bytes){{
    if(!handle||!output||bytes!=64){{last_error="invalid private sparse info size";return -1;}}
    auto& c=*static_cast<Context*>(handle);
    const uint32_t header[8]={{1,static_cast<uint32_t>(c.sparse_shader),{tile},{paths},c.sparse_captures,c.sparse_sensor_captures,c.sparse_last_rgb,{grid}|({neighbors}u<<8u)|({0 if reconstruction_revision==1 else reconstruction_revision}u<<16u)|({estimator_code}u<<24u)}};
    std::memcpy(output,header,32);std::memcpy(output+8,c.sparse_counts.data(),32);return 0;
}}
PR_API int pr_preview_sparse_read_guide(void* handle,void* output,size_t bytes){{
    try{{require(handle,"null sparse guide context");spatial_read(*static_cast<Context*>(handle),output,bytes);return 0;}}
    catch(const std::exception& e){{last_error=e.what();return -1;}}
}}
''')
    change("source/_native/resident_spatial.inc",[
        ("linear guide has no display transform", "display_shader_marker(fragment,fragment_bytes)&&", "c.sparse_shader&&"),
        ("guide ABI", '''!named_shader_marker(fragment,fragment_bytes,64,"PR_REFLECTION_ABI")&&
            !named_shader_marker(fragment,fragment_bytes,65,"PR_INDIRECT_ABI")&&
            !named_shader_marker(fragment,fragment_bytes,66,"PR_PROGRESSIVE_ABI")''',
         '''named_shader_marker(fragment,fragment_bytes,79,"PR_PREVIEW_SPARSE_GUIDE_ABI")&&
            named_shader_marker(fragment,fragment_bytes,65,"PR_INDIRECT_ABI")&&
            named_shader_marker(fragment,fragment_bytes,66,"PR_PROGRESSIVE_ABI")'''),
        ("sampling format", "for(auto format:{VK_FORMAT_R8G8B8A8_UNORM,VK_FORMAT_R32_SFLOAT,VK_FORMAT_R32_UINT,VK_FORMAT_R8_UINT})",
         "for(auto format:{VK_FORMAT_R32G32B32A32_SFLOAT,VK_FORMAT_R32_SFLOAT,VK_FORMAT_R32_UINT,VK_FORMAT_R8_UINT})"),
        ("guide renderpass format", "auto& a=attachments[i];a.format=FORMATS[i];a.samples=VK_SAMPLE_COUNT_1_BIT;",
         "auto& a=attachments[i];a.format=i==0?VK_FORMAT_R32G32B32A32_SFLOAT:FORMATS[i];a.samples=VK_SAMPLE_COUNT_1_BIT;"),
    ])
    change("source/_native/resident_progressive.inc",[
        ("only one exposure", "    require(c.progressive_info.supported&&c.progressive_info.shader_compatible&&c.indirect_info.enabled,",
         "    require(guided&&!settling&&bounces==1&&c.sparse_shader,\"private sparse setup only\");\n    require(c.progressive_info.supported&&c.progressive_info.shader_compatible&&c.indirect_info.enabled,"),
        ("status allocation", "c.make_buffer(state.status,4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT", "c.make_buffer(state.status,32,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT"),
        ("status descriptor", "VkDescriptorBufferInfo status{state.status.handle,0,4};", "VkDescriptorBufferInfo status{state.status.handle,0,32};"),
        ("status clear", "vkCmdFillBuffer(commands,state.status.handle,0,4,0);", "vkCmdFillBuffer(commands,state.status.handle,0,32,0);"),
        ("status barrier", "status.buffer=state.status.handle;status.offset=0;status.size=4;", "status.buffer=state.status.handle;status.offset=0;status.size=32;"),
        ("status copy", "VkBufferCopy copy{0,status_offset,4};", "VkBufferCopy copy{0,status_offset,32};"),
        ("float guide debug extent", "bytes==static_cast<size_t>(state.width)*state.height*4,\"spatial guide byte extent mismatch\"", "bytes==static_cast<size_t>(state.width)*state.height*16,\"private RGBA32F incoming guide byte extent mismatch\""),
    ])
    change("source/_native/resident.inc",[
        ("readback extent", "height*13+(c.progressive_info.shader_compatible?8:0)", "height*13+(c.progressive_info.shader_compatible?36:0)"),
        ("single batch", "require(!progressive_rgb||progressive_batch<4096,\"progressive sample budget exceeded\");",
         "require(!progressive_rgb||progressive_batch==0,\"private sparse experiment has no history\");"),
        ("read actual counters", "        require(invalid==0,\"progressive shader produced nonfinite or negative radiance; no publication\");",
         "        require(invalid==0,\"sparse shader produced invalid radiance/attributes; no publication\");\n        std::memcpy(c.sparse_counts.data(),static_cast<const uint8_t*>(frame.readback_mapping)+progressive_status_offset,32);"),
        ("capture accounting", "    c.telemetry.readback_bytes += output_bytes+(progressive_rgb?4:0);",
         "    c.sparse_last_rgb=progressive_rgb?1u:0u;\n    if(progressive_rgb)++c.sparse_captures;else{++c.sparse_sensor_captures;c.sparse_counts.fill(0);}\n    c.telemetry.readback_bytes += output_bytes+(progressive_rgb?32:0);"),
        ("copy accounting", "c.readback_info.copied_bytes += output_bytes+(progressive_rgb?4:0);", "c.readback_info.copied_bytes += output_bytes+(progressive_rgb?32:0);"),
        ("status accounting", "progressive.status_readback_bytes+=4;", "progressive.status_readback_bytes+=32;"),
    ])
    change("runtime/phymeo_rendering/backends/_native_progressive.py",[
        ("private dispatch import", "import hashlib", "import hashlib\nfrom ._preview_sparse_dispatch import progressive_setter"),
        ("dedicated ABI", """        setter = (device._lib.pr_resident_set_spatial if getattr(device, '_spatial_requested', False)
                  else device._lib.pr_resident_set_progressive)
        if getattr(device, '_settling_requested', False):
            setter = device._lib.pr_resident_set_settling""", "        setter = progressive_setter(device)"),
    ])
    _write(runtime/"backends/_preview_sparse_dispatch.py",(HERE/"dispatch.py").read_text(encoding="utf-8"))
    change("runtime/phymeo_rendering/backends/native.py",[
        ("explicit sparse transport", "            display_info = configure_display(device, self.display_transform)",
         "            device._preview_sparse_requested = bool(getattr(self, '_preview_sparse_requested', False))\n            display_info = configure_display(device, self.display_transform)"),
    ])
    # Private shader code: same full-rate primary and sensor math, early-return
    # sparse guide, and exact selected budget. No change to original files.
    common=(HERE/"sparse_common.glsl").read_text(encoding="utf-8")
    common+=f'\nlayout(constant_id=83) const uint {profile}=1u;\n'
    if estimator_code:
        _write(shaders/"preview_sparse_ratio.glsl",(HERE/"sparse_ratio.glsl").read_text(encoding="utf-8"))
        common+='\n#ifdef PR_PREVIEW_SPARSE_GUIDE\n#include "preview_sparse_ratio.glsl"\n#endif\n'
    _write(shaders/"preview_sparse_common.glsl",common)
    guide='''#ifdef PR_PREVIEW_SPARSE_GUIDE
    float sparseMetal=scene.instances[ids.y].material.x;
    vec3 sparseRho=max(vec3(0.0),(1.0-sparseMetal)*albedo*(1.0-mix(vec3(.04),albedo,sparseMetal)));
    bool traced=u.counts.z!=0u&&shadingFacesView&&reflectionFaceValid&&sparsePackRho(sparseRho)!=0u&&sparseAnchor(ivec2(gl_FragCoord.xy));
    vec3 incoming=shadingFacesView?sparseFallback(N):vec3(0.0);
    uint sparseOutcomes=0u;
    if(traced)incoming=ESTIMATOR(worldPosition,N,reflectionFace);
    if(PR_PREVIEW_SPARSE_GUIDE_ABI!=1u||PR_PROGRESSIVE_ABI!=1u||PROFILE!=1u)incoming=vec3(importanceInvalid());
    rgb=vec4(incoming,float(sparsePackNormal(N,traced)|sparseOutcomes));return;
#endif
'''.replace("PROFILE",profile).replace("ESTIMATOR","sparseRatioIncoming" if estimator_code else "residentDiffuseIncoming")
    if estimator_code:
        guide=guide.replace("sparseRatioIncoming(worldPosition,N,reflectionFace)","sparseRatioIncoming(worldPosition,N,reflectionFace,sparseOutcomes)")
    change("source/_shaders/resident.frag",[
        ("private helper", '#include "resident_msaa.glsl"', '#include "resident_msaa.glsl"\n#include "preview_sparse_common.glsl"'),
        ("guide early return", '#if defined(PR_RESIDENT_REFLECTION) || defined(PR_RESIDENT_INDIRECT)\n    if(u.counts.z!=0u && shadingFacesView)',
         guide+'#if defined(PR_RESIDENT_REFLECTION) || defined(PR_RESIDENT_INDIRECT)\n    if(u.counts.z!=0u && shadingFacesView)'),
        ("primary zero diffuse incoming", '    if(indirectActive&&reflectionFaceValid&&any(greaterThan(rho,vec3(0.0))))\n        diffuseIncoming=residentDiffuseIncoming(worldPosition,N,reflectionFace);',
         '#ifndef PR_PREVIEW_SPARSE_MAIN\n    if(indirectActive&&reflectionFaceValid&&any(greaterThan(rho,vec3(0.0))))\n        diffuseIncoming=residentDiffuseIncoming(worldPosition,N,reflectionFace);\n#endif'),
        ("rho metadata", '    rgb=vec4(PR_PROGRESSIVE_ABI==1u?color:vec3(0.0),1.0);',
         f'''    vec3 sparseOutputRho=(indirectActive&&reflectionFaceValid)?rho:vec3(0.0);
    if(PR_PREVIEW_SPARSE_PRIMARY_ABI!=1u||PR_INDIRECT_ABI!=1u||PR_ENVIRONMENT_CDF_ABI!=1u||{profile}!=1u)color=vec3(importanceInvalid());
    rgb=vec4(PR_PROGRESSIVE_ABI==1u?color:vec3(0.0),float(sparsePackRho(sparseOutputRho)));'''),
    ])
    # Guide has an unconditional runtime return but must still compile the tail.
    # Declare the primary marker there too; only C++ selected main ABI enables a context.
    text=(shaders/"preview_sparse_common.glsl").read_text()
    text=text.replace('#if defined(PR_PREVIEW_SPARSE_MAIN)\nlayout(constant_id=78)', '#if defined(PR_PREVIEW_SPARSE_MAIN) || defined(PR_PREVIEW_SPARSE_GUIDE)\nlayout(constant_id=78)')
    _write(shaders/"preview_sparse_common.glsl",text)
    x=4 if paths==16 else 2;y=paths//x
    half=paths//2;mx=min(4,half);my=half//mx
    change("source/_shaders/resident_indirect.glsl",[
        ("primary paths", "for(uint i=0u;i<16u;++i)", f"for(uint i=0u;i<{paths}u;++i)"),
        ("cosine strata", "vec2 q=(vec2(i&3u,i>>2u)+jitter)*.25;", f"vec2 q=(vec2(i%{x}u,i/{x}u)+jitter)/vec2({float(x)},{float(y)});"),
        ("MIS cosine strata", "if(environmentMIS)q=(vec2(i&3u,(i&7u)>>2u)+jitter)*vec2(0.25,0.5);",
         f"if(environmentMIS)q=(vec2((i%{half}u)%{mx}u,(i%{half}u)/{mx}u)+jitter)/vec2({float(mx)},{float(my)});"),
        ("MIS split", "bool fromEnvironment=environmentMIS && i>=8u;", f"bool fromEnvironment=environmentMIS && i>={half}u;"),
        ("environment strata", "importanceSample(key,i-8u,8u,direction,sampledPdf)", f"importanceSample(key,i-{half}u,{half}u,direction,sampledPdf)"),
        ("normalization", "return sum*(1.0/16.0);", f"return sum*(1.0/{float(paths)});"),
    ])
    compute=(HERE/"sparse_reconstruct.comp").read_text(encoding="utf-8")
    if estimator_code:
        compute=compute.replace("2097151.0","16777215.0")
    compute=compute.replace('layout(constant_id=80)',f'layout(constant_id=83) const uint {profile}=1u;\nlayout(constant_id=80)')
    compute=compute.replace('if(PR_SPATIAL_MEAN_ABI!=1u||',f'if({profile}!=1u||PR_SPATIAL_MEAN_ABI!=1u||')
    _write(shaders/"resident_spatial.comp",compute)
    _write(native/"CMakeLists.txt",_cmake(tile,paths,grid,neighbors,reconstruction_revision))
    manifest=dict(contract=policy,source_root=str(root),source_sha256=source_hashes,
                  generated_sha256={str(p.relative_to(out)):_hash(p) for p in sorted(out.rglob("*")) if p.is_file()},
                  builder_sha256=_hash(Path(__file__)),profile_marker=profile,
                  state="source prepared only; no compiler/GPU validation claimed")
    _write(out/"source-manifest.json",json.dumps(manifest,indent=2)+"\n")
    return manifest


def _cmake(tile,paths,grid,neighbors,reconstruction_revision):
    return f'''cmake_minimum_required(VERSION 3.20)
project(phymeo_preview_sparse LANGUAGES CXX)
find_path(VULKAN_INCLUDE_DIR vulkan/vulkan.h HINTS "$ENV{{VULKAN_SDK}}/Include")
find_library(VULKAN_LIBRARY NAMES vulkan-1 vulkan HINTS "$ENV{{VULKAN_SDK}}/Lib")
find_program(GLSLANG_VALIDATOR NAMES glslang glslangValidator HINTS "$ENV{{VULKAN_SDK}}/Bin")
if(NOT VULKAN_INCLUDE_DIR OR NOT VULKAN_LIBRARY OR NOT GLSLANG_VALIDATOR)
  message(FATAL_ERROR "Explicit local Vulkan SDK/compiler required")
endif()
add_library(phymeo_native SHARED renderer.cpp)
target_compile_features(phymeo_native PRIVATE cxx_std_17)
if(MSVC)
target_compile_options(phymeo_native PRIVATE /W4 /EHsc)
endif()
target_include_directories(phymeo_native PRIVATE "${{VULKAN_INCLUDE_DIR}}")
target_link_libraries(phymeo_native PRIVATE "${{VULKAN_LIBRARY}}")
set_target_properties(phymeo_native PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}" LIBRARY_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}")
foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
set_target_properties(phymeo_native PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${{configuration}} "${{CMAKE_BINARY_DIR}}")
endforeach()
file(GLOB shader_deps "${{CMAKE_CURRENT_SOURCE_DIR}}/../_shaders/*")
function(shader output source)
add_custom_command(OUTPUT "${{CMAKE_BINARY_DIR}}/${{output}}.spv"
 COMMAND "${{GLSLANG_VALIDATOR}}" -V --target-env vulkan1.2 --glsl-version 460
 -DPR_PREVIEW_TILE={tile} -DPR_PREVIEW_PATHS={paths} -DPR_STUDIO_AREA_GRID={grid} -DPR_PREVIEW_NEIGHBORS={neighbors} -DPR_PREVIEW_RECONSTRUCTION_REVISION={reconstruction_revision}
 ${{ARGN}} "${{CMAKE_CURRENT_SOURCE_DIR}}/../_shaders/${{source}}" -o "${{CMAKE_BINARY_DIR}}/${{output}}.spv"
 DEPENDS ${{shader_deps}} VERBATIM)
set(shader_outputs ${{shader_outputs}} "${{CMAKE_BINARY_DIR}}/${{output}}.spv" PARENT_SCOPE)
endfunction()
set(suffix "")
if(NOT {grid} EQUAL 2)
set(suffix "_area{grid}")
endif()
foreach(normal plain normal)
set(prefix resident_rayquery)
set(normal_options "")
if(normal STREQUAL "normal")
set(prefix resident_rayquery_normal)
set(normal_options -DPR_RESIDENT_NORMAL_MAPS=1)
endif()
shader("${{prefix}}.vert" resident.vert -DPR_RESIDENT_RAY_QUERY=1 ${{normal_options}})
shader("${{prefix}}${{suffix}}.frag" resident.frag -DPR_RESIDENT_RAY_QUERY=1 -DPR_RESIDENT_INDIRECT=1 -DPR_RESIDENT_PROGRESSIVE=1 -DPR_PREVIEW_SPARSE_GUIDE=1 ${{normal_options}})
foreach(reflection plain reflection)
set(r "")
set(r_options "")
if(reflection STREQUAL "reflection")
set(r "_reflection")
set(r_options -DPR_RESIDENT_REFLECTION=1)
endif()
shader("${{prefix}}${{r}}_indirect_progressive${{suffix}}.frag" resident.frag -DPR_RESIDENT_RAY_QUERY=1 -DPR_RESIDENT_INDIRECT=1 -DPR_RESIDENT_PROGRESSIVE=1 -DPR_PREVIEW_SPARSE_MAIN=1 ${{normal_options}} ${{r_options}})
endforeach()
endforeach()
shader(resident_sky.vert resident_sky.vert)
shader(resident_sky_progressive.frag resident_sky.frag -DPR_RESIDENT_PROGRESSIVE=1)
shader(resident_spatial.comp resident_spatial.comp)
add_custom_target(native_shaders ALL DEPENDS ${{shader_outputs}})
add_dependencies(phymeo_native native_shaders)
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendering-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--tile",type=int,default=2)
    parser.add_argument("--paths",type=int,default=4)
    parser.add_argument("--grid",type=int,default=2)
    parser.add_argument("--neighbors",type=int,default=4,choices=[4,25])
    parser.add_argument("--reconstruction-revision",type=int,choices=(1,2),default=1)
    parser.add_argument("--estimator",choices=("radiance","visibility-ratio"),default="radiance")
    args=parser.parse_args()
    prepare(args.rendering_root,args.output,tile=args.tile,paths=args.paths,grid=args.grid,neighbors=args.neighbors,reconstruction_revision=args.reconstruction_revision,estimator=args.estimator)


if __name__=="__main__":main()
