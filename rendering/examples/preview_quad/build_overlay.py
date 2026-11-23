"""Build and verify a NEW shader overlay; never edit an existing DLL/SPV tree."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import struct
import subprocess

TARGET = "resident_rayquery_normal_reflection_indirect_progressive.frag.spv"
SOURCE_HASHES = {
    "resident.frag": "14e9b111bf9eade60219a4b828232d25f5a4e55d3fde1d55ce13cec3c5a95663",
    "resident_indirect.glsl": "247eb988be2cb2ac39b647e1dc11a62be04e15966f7fa42b23fd3ddfb730980f",
}
MARKERS = {"PR_PREVIEW_QUAD_ABI": (120, 1),
           "PR_PREVIEW_PRIMARY_PROPOSALS": (121, None),
           "PR_PREVIEW_SPATIAL_REUSE": (122, None)}
SCHEMA = "phymeo-preview-quad-overlay-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def patch_main(text):
    include = '#include "resident_indirect.glsl"'
    call = ("    if(indirectActive&&reflectionFaceValid&&any(greaterThan(rho,vec3(0.0))))\n"
            "        diffuseIncoming=residentDiffuseIncoming(worldPosition,N,reflectionFace);")
    require(text.count("#version 450\n") == 1 and text.count(include) == 1 and text.count(call) == 1,
            "registered main shader anchors differ")
    extensions = ("#extension GL_KHR_shader_subgroup_basic : require\n"
                  "#extension GL_KHR_shader_subgroup_ballot : require\n"
                  "#extension GL_KHR_shader_subgroup_quad : require\n")
    text = text.replace("#version 450\n", "#version 450\n" + extensions, 1)
    text = text.replace(include, include + '\n#include "preview_quad.glsl"', 1)
    return text.replace(call,
        "    // Unconditional entry; eligibility gates rays, never quad exchanges.\n"
        "    diffuseIncoming=previewQuadIncoming(worldPosition,N,reflectionFace,forwardDepth,ids,\n"
        "        indirectActive&&reflectionFaceValid&&any(greaterThan(rho,vec3(0.0))));", 1)


def inspect_spirv(path, samples, sharing):
    data = Path(path).read_bytes()
    require(len(data) >= 20 and len(data) % 4 == 0, "invalid SPIR-V extent")
    words = struct.unpack("<" + "I" * (len(data)//4), data)
    require(words[0] == 0x07230203, "invalid SPIR-V magic")
    names, spec_ids, constants, uint_types = {}, {}, {}, set()
    capabilities, instructions = set(), set()
    offset = 5
    while offset < len(words):
        word = words[offset]
        count, opcode = word >> 16, word & 65535
        require(count > 0 and offset + count <= len(words), "truncated SPIR-V instruction")
        w = words[offset+1:offset+count]
        instructions.add(opcode)
        if opcode == 17 and len(w) == 1:
            capabilities.add(w[0])
        if opcode == 5:  # OpName
            raw = struct.pack("<" + "I" * (len(w)-1), *w[1:])
            names[w[0]] = raw.split(b"\0", 1)[0].decode("utf-8")
        elif opcode == 71 and len(w) >= 3 and w[1] == 1:  # OpDecorate SpecId
            spec_ids[w[0]] = w[2]
        elif opcode == 21 and len(w) == 3 and w[1:] == (32, 0):
            uint_types.add(w[0])
        elif opcode == 50 and len(w) == 3:  # OpSpecConstant
            constants[w[1]] = (w[0], w[2])
        offset += count
    require({61, 64, 68} <= capabilities and {339, 341, 365} <= instructions,
            "SPIR-V lacks core subgroup ballot/bit-extract/quad-broadcast operations")
    result = {"binary_requirements": dict(capabilities=sorted(capabilities),
               ballot_opcode=339, ballot_bit_extract_opcode=341, quad_broadcast_opcode=365)}
    for name, (spec_id, default) in MARKERS.items():
        wanted = samples if name.endswith("PROPOSALS") else int(sharing) if name.endswith("REUSE") else default
        ids = [i for i, n in names.items() if n == name]
        require(len(ids) == 1, "missing/ambiguous experimental SPIR-V marker: " + name)
        i = ids[0]
        require(spec_ids.get(i) == spec_id and i in constants and
                constants[i][0] in uint_types and constants[i][1] == wanted,
                "experimental SPIR-V default/type differs: " + name)
        result[name] = dict(spec_id=spec_id, default=wanted)
    return result


def executable(value, sibling=None):
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if sibling:
        candidate = Path(sibling).with_name(value + (".exe" if Path(sibling).suffix == ".exe" else ""))
        if candidate.is_file():
            return candidate.resolve()
    found = shutil.which(value)
    require(found is not None, "required tool unavailable: " + value)
    return Path(found).resolve()


def verify_overlay(directory):
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory / "overlay.json").read_text(encoding="utf-8"))
    require(manifest["schema"] == SCHEMA and manifest["status"] == "complete", "uncompleted shader overlay")
    require(manifest["samples_per_eligible_pixel"] in (2, 4), "unknown experimental budget")
    require(set(manifest["shaders"]) == {p.name for p in directory.glob("*.spv")}, "SPIR-V file set differs")
    for name, expected in manifest["shaders"].items():
        require(Path(name).name == name and digest(directory / name) == expected, "overlay SPIR-V hash differs: " + name)
    changed = [k for k, v in manifest["shaders"].items() if v != manifest["base_shaders"][k]]
    require(changed == [TARGET], "overlay modified a nonregistered shader")
    inspect_spirv(directory / TARGET, manifest["samples_per_eligible_pixel"], manifest["sharing"])
    return manifest


def build(args):
    require(args.samples in (2, 4), "only 2/4 primary proposals supported")
    source = args.source_shaders.resolve(strict=True)
    base = args.base_shaders.resolve(strict=True)
    output = args.output.resolve()
    require(not output.exists() and not output.is_relative_to(base) and not base.is_relative_to(output),
            "overlay must be a new, separate shader directory")
    texts = {name: (source/name).read_text(encoding="utf-8-sig") for name in SOURCE_HASHES}
    for name, expected in SOURCE_HASHES.items():
        require(sha256(texts[name].encode()).hexdigest() == expected, "unregistered baseline shader source: " + name)
    require((base / TARGET).is_file(), "baseline lacks requested normal+reflection+progressive grid2 shader")
    compiler = executable(args.compiler)
    def optional_tool(explicit, name):
        if explicit:
            return executable(explicit, compiler)
        try:
            return executable(name, compiler)
        except ValueError:
            return None
    validator = optional_tool(args.spirv_val, "spirv-val")
    disassembler = optional_tool(args.spirv_dis, "spirv-dis")
    require(not args.strict_validation or validator is not None,
            "strict validation requested but independent spirv-val is unavailable")
    output.mkdir(parents=True)
    generated = output / "source"
    generated.mkdir()
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".glsl", ".frag", ".vert", ".comp"):
            shutil.copy2(path, generated/path.name)
    (generated/"resident.frag").write_text(patch_main(texts["resident.frag"]), encoding="utf-8")
    local = Path(__file__).with_name("preview_quad.glsl")
    (generated/local.name).write_text(local.read_text(encoding="utf-8-sig"), encoding="utf-8")
    base_hashes = {}
    for path in sorted(base.glob("*.spv")):
        base_hashes[path.name] = digest(path)
        shutil.copy2(path, output/path.name)
    record = dict(schema=SCHEMA, status="building", source_directory=str(source),
                  source_hashes={p.name: digest(p) for p in sorted(source.iterdir()) if p.is_file()},
                  generated_source_hashes={p.name: digest(p) for p in sorted(generated.iterdir())},
                  base_directory=str(base), base_shaders=base_hashes,
                  samples_per_eligible_pixel=args.samples, sharing=not args.no_share,
                  runtime_nominal_samples_per_pixel=16,
                  actual_budget_scope="2/4 primary proposals per covered eligible pixel; hemisphere-rejected proposals remain zero; not a measured ray/hit count",
                  compatible_profile="MSAA1, progressive, first diffuse bounce, normal+reflection shader, panel grid2",
                  limitations=["experimental spatial bias; different secondary visibility is not detected by primary geometry",
                               "active/helper, instance/slot, depth and normal gates only; no primitive-ID equality gate; no temporal reprojection",
                               "legacy public diagnostics retain16; experimental count is this hash-bound shader budget"],
                  source_policy="registered baseline main/indirect normalized-text SHA plus exact once-only anchors")
    save(output/"overlay.json", record)
    command = [str(compiler), "-V", "-H", "--target-env", "vulkan1.2", "--glsl-version", "460",
               "-DPR_RESIDENT_RAY_QUERY=1", "-DPR_RESIDENT_NORMAL_MAPS=1",
               "-DPR_RESIDENT_REFLECTION=1", "-DPR_RESIDENT_INDIRECT=1", "-DPR_RESIDENT_PROGRESSIVE=1",
               "-DPR_STUDIO_AREA_GRID=2", f"-DPR_PREVIEW_SAMPLES={args.samples}",
               f"-DPR_PREVIEW_SHARE={int(not args.no_share)}",
               str(generated/"resident.frag"), "-o", str(output/TARGET)]
    record["commands"] = [command]
    try:
        result = subprocess.run(command, cwd=generated, capture_output=True, text=True)
        (output/"compile.log").write_text(result.stdout+result.stderr, encoding="utf-8")
        result.check_returncode()
        # glslang -H emits readable SPIR-V itself; no separate SDK is required.
        (output/"prototype.spvasm").write_text(result.stdout, encoding="utf-8")
        record["disassembly_source"] = "glslang -H during the actual compilation"
        if validator is not None:
            val_command = [str(validator), "--target-env", "vulkan1.2", str(output/TARGET)]
            record["commands"].append(val_command)
            checked = subprocess.run(val_command, capture_output=True, text=True)
            (output/"validation.log").write_text(checked.stdout+checked.stderr, encoding="utf-8")
            checked.check_returncode()
            record["standalone_spirv_validation"] = dict(status="passed", tool=str(validator))
        else:
            record["standalone_spirv_validation"] = dict(status="not_available",
                required_next="real Vulkan pipeline/capture with validation layers; parser is not a full SPIR-V validator")
        if disassembler is not None:
            dis_command = [str(disassembler), str(output/TARGET), "-o", str(output/"independent.spvasm")]
            record["commands"].append(dis_command)
            subprocess.run(dis_command, check=True, capture_output=True, text=True)
        dis = (output/"prototype.spvasm").read_text(encoding="utf-8")
        for instruction in ("GroupNonUniformBallot", "GroupNonUniformQuadBroadcast"):
            require(instruction in dis, "glslang disassembly lacks required subgroup instructions")
        record["markers"] = inspect_spirv(output/TARGET, args.samples, not args.no_share)
        record["shaders"] = {p.name: digest(p) for p in sorted(output.glob("*.spv"))}
        record["tools"] = {k: dict(file=str(v), sha256=digest(v)) for k,v in
                           (("compiler",compiler),("validator",validator),("disassembler",disassembler)) if v is not None}
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        save(output/"overlay.json", record)
    verify_overlay(output)
    print(json.dumps(dict(status="complete", overlay=str(output), samples=args.samples), ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-shaders", type=Path, required=True)
    ap.add_argument("--base-shaders", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples", type=int, choices=(2,4), required=True)
    ap.add_argument("--no-share", action="store_true", help="same 2/4 proposal estimator without spatial reuse")
    ap.add_argument("--compiler", default="glslangValidator")
    ap.add_argument("--spirv-val", help="optional independent validator; an explicit missing path is an error")
    ap.add_argument("--spirv-dis", help="optional independent disassembler; glslang -H is always retained")
    ap.add_argument("--strict-validation", action="store_true", help="require independent spirv-val")
    build(ap.parse_args())


if __name__ == "__main__":
    main()

