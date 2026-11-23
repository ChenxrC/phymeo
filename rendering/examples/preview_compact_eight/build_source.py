"""Build one isolated, full-distance eight-proposal compact/rigid candidate.

This is a source composer, not a renderer or a new production default. It copies
only hash-bound files from a plain compact4 package, patches the proposal budget,
then uses the existing static/rigid composers without changing their algorithms.
No compilation or rendering happens here. Existing packages are never edited.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re


HERE = Path(__file__).resolve().parent
OLD_PROFILE = "PR_PREVIEW_SPARSE_PROFILE_T4_P4_G2_K25_R2_Eratio"
PROFILE = "PR_PREVIEW_SPARSE_PROFILE_T4_P8_G2_K25_R2_Eratio"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def proposal_layout():
    return dict(total=8, cosine_grid=(4, 2), mis_cosine_grid=(2, 2),
                mis_cosine_count=4, mis_environment_count=4, normalization=.125)


def proposal(index, jitter, mis):
    """CPU contract for strata, not a substitute for actual shader execution."""
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError("proposal index must be an integer in [0,8)")
    if type(mis) is not bool:
        raise TypeError("mis must be bool")
    if len(jitter) != 2 or any(not math.isfinite(v) or not 0 < v < 1 for v in jitter):
        raise ValueError("jitter must be two finite values strictly inside (0,1)")
    if mis and index >= 4:
        return dict(kind="environment", cdf_stratum=index - 4, cdf_strata=4)
    nx, ny = (2, 2) if mis else (4, 2)
    return dict(kind="cosine", q=((index % nx + jitter[0]) / nx,
                                 (index // nx + jitter[1]) / ny))


def balance_weight(cosine_pdf, environment_pdf):
    """Four samples per technique imply a 1/2 + 1/2 mixture, followed by /8."""
    if any(not math.isfinite(v) or v < 0 for v in (cosine_pdf, environment_pdf)):
        raise ValueError("PDFs must be finite and nonnegative")
    if cosine_pdf == 0:
        return 0.
    # Algebraically p_cos / (.5 * (p_cos + p_env)); this form also avoids
    # overflow in a CPU contract evaluated with very large finite densities.
    scale = max(cosine_pdf, environment_pdf)
    return 2 * (cosine_pdf / scale) / (cosine_pdf / scale + environment_pdf / scale)


def validate_parent(policy):
    expected = dict(schema="phymeo-preview-sparse-v1", tile=4, paths_per_anchor=4,
                    area_light_grid=2, neighbor_candidates=25,
                    reconstruction_revision=2, estimator="visibility-ratio", bounces=1)
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("exact full-distance compact4/K25/R2/ratio parent required")
    if any(key in policy for key in ("local_diffuse", "pooled_sky", "rigid_history",
                                     "proposal_budget", "irradiance_filter", "edge_rescue")):
        raise ValueError("parent already contains an unsupported estimator/history overlay")
    if not isinstance(policy.get("history"), str) or not policy["history"].startswith("none"):
        raise ValueError("parent must have no history")
    compact = policy.get("compact", {})
    if (compact.get("schema"), compact.get("tile"), compact.get("history")) != (
            "phymeo-preview-compact-v1", 4, "none"):
        raise ValueError("plain compact source required")


def source_edits():
    """Exact anchors keep a different source version from silently half-patching."""
    return {
        "source/_native/renderer.cpp": [
            (OLD_PROFILE, PROFILE, 3),
            ("const uint32_t header[8]={1,static_cast<uint32_t>(c.sparse_shader),4,4,",
             "const uint32_t header[8]={1,static_cast<uint32_t>(c.sparse_shader),4,8,", 1),
        ],
        "source/_native/CMakeLists.txt": [
            ("-DPR_PREVIEW_PATHS=4 ", "-DPR_PREVIEW_PATHS=8 ", 1),
        ],
        "source/_shaders/preview_sparse_common.glsl": [(OLD_PROFILE, PROFILE, 1)],
        "source/_shaders/resident.frag": [(OLD_PROFILE, PROFILE, 2)],
        "source/_shaders/resident_spatial.comp": [(OLD_PROFILE, PROFILE, 2)],
        "source/_shaders/preview_sparse_ratio.glsl": [
            ("layout(constant_id=84) const uint PR_PREVIEW_SKY_VISIBILITY_RATIO_ABI=1u;",
             "#if PR_PREVIEW_PATHS != 8\n#error Eight-proposal profile requires exactly eight proposals\n#endif\n"
             "layout(constant_id=84) const uint PR_PREVIEW_SKY_VISIBILITY_RATIO_ABI=1u;", 1),
            ("const uint nx=PR_PREVIEW_PATHS==16?4u:2u;", "const uint nx=4u;", 1),
            ("const uint mx=halfCount<4u?halfCount:4u;", "const uint mx=2u;", 1),
        ],
        "source/_shaders/resident_indirect.glsl": [
            ("Stratification is fixed 4x4", "This isolated profile uses eight 4x2 cosine strata", 1),
            ("Primary proposals cover all eight CDF strata once; secondary uses one.",
             "This profile covers all four CDF strata once; unused secondary code uses one.", 1),
            ("Balance MIS: 8+8 first-hop proposals, or one equiprobable mixture draw at",
             "Balance MIS: 4+4 first-hop proposals, or one equiprobable mixture draw at", 1),
            ("for(uint i=0u;i<4u;++i)", "for(uint i=0u;i<8u;++i)", 1),
            ("vec2 q=(vec2(i%2u,i/2u)+jitter)/vec2(2.0,2.0);",
             "vec2 q=(vec2(i%4u,i/4u)+jitter)/vec2(4.0,2.0);", 1),
            ("if(environmentMIS)q=(vec2((i%2u)%2u,(i%2u)/2u)+jitter)/vec2(2.0,1.0);",
             "if(environmentMIS)q=(vec2((i%4u)%2u,(i%4u)/2u)+jitter)/vec2(2.0,2.0);", 1),
            ("bool fromEnvironment=environmentMIS && i>=2u;",
             "bool fromEnvironment=environmentMIS && i>=4u;", 1),
            ("importanceSample(key,i-2u,2u,direction,sampledPdf)",
             "importanceSample(key,i-4u,4u,direction,sampledPdf)", 1),
            ("return sum*(1.0/4.0);", "return sum*(1.0/8.0);", 1),
        ],
    }


def patch_text(relative, text):
    for old, new, count in source_edits()[relative]:
        if text.count(old) != count:
            raise ValueError(f"eight-proposal source drift: {relative}: {old[:100]}")
        text = text.replace(old, new)
    return text


def safe_relative(value):
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in value:
        raise ValueError("unsafe generated file path")
    if not relative.parts or relative.parts[0] not in ("source", "runtime"):
        raise ValueError("only pinned source/runtime files may be copied")
    return Path(*relative.parts)


def write(path, value):
    Path(path).write_text(value, encoding="utf-8", newline="\n")


def verify_profile_bindings(root):
    """Check consumers independently of the list of files being patched.

    The first build found an old symbol in resident.frag after its declaration
    had changed. Scanning source_edits alone could not detect that omission.
    Every source file is now scanned, and each expected live binding is required.
    """
    root = Path(root)
    for path in sorted((root / "source").rglob("*")):
        if not path.is_file() or path.suffix not in (".cpp", ".h", ".inc", ".glsl", ".frag", ".vert", ".comp", ".txt"):
            continue
        text = path.read_text(encoding="utf-8")
        names = re.findall(r"\bPR_PREVIEW_SPARSE_PROFILE_[A-Za-z0-9_]+\b", text)
        if any(name != PROFILE for name in names):
            raise ValueError("orphan/incompatible profile symbol: " + str(path.relative_to(root)))
    expected = {
        "source/_native/renderer.cpp": 3,
        "source/_shaders/preview_sparse_common.glsl": 1,
        "source/_shaders/resident.frag": 2,
        "source/_shaders/resident_spatial.comp": 2,
    }
    texts = {}
    for name, count in expected.items():
        path = root / name
        if not path.is_file():
            raise ValueError("profile consumer missing: " + name)
        texts[name] = path.read_text(encoding="utf-8")
        if texts[name].count(PROFILE) != count:
            raise ValueError("profile forward binding count differs: " + name)
    native = texts["source/_native/renderer.cpp"]
    for value in (f'named_shader_marker(frag,frag_bytes,83,"{PROFILE}")',
                  f'named_shader_marker(code,bytes,83,"{PROFILE}",5)',
                  f'named_shader_marker(guide,guide_bytes,83,"{PROFILE}")'):
        if value not in native:
            raise ValueError("native profile stage binding differs: " + value)
    declaration = f"layout(constant_id=83) const uint {PROFILE}=1u;"
    for name in ("source/_shaders/preview_sparse_common.glsl", "source/_shaders/resident_spatial.comp"):
        if declaration not in texts[name]:
            raise ValueError("profile declaration differs: " + name)
    fragment = texts["source/_shaders/resident.frag"]
    for abi in ("PR_PREVIEW_SPARSE_GUIDE_ABI", "PR_PREVIEW_SPARSE_PRIMARY_ABI"):
        if not any(abi + "!=1u" in line and PROFILE + "!=1u" in line for line in fragment.splitlines()):
            raise ValueError("fragment profile guard missing: " + abi)
    if PROFILE + "!=1u" not in texts["source/_shaders/resident_spatial.comp"]:
        raise ValueError("compute profile guard missing")
    return expected


def verify_sources(root):
    """Read actual output; fail before compilation if a budget seam was missed."""
    root = Path(root)
    profile_bindings = verify_profile_bindings(root)
    texts = {key: (root / key).read_text(encoding="utf-8") for key in source_edits()}
    for relative, text in texts.items():
        if OLD_PROFILE in text:
            raise ValueError("old proposal profile survived: " + relative)
    indirect = texts["source/_shaders/resident_indirect.glsl"]
    primary = indirect.split("vec3 residentDiffuseIncoming(", 1)[1]
    required = ("for(uint i=0u;i<8u;++i)", "vec2(4.0,2.0)",
                "i>=4u", "importanceSample(key,i-4u,4u,", "return sum*(1.0/8.0);")
    if not all(value in primary for value in required):
        raise ValueError("ordinary eight-proposal loop/split/normalization mismatch")
    if any(value in primary for value in ("i<4u", "i<16u", "1.0/4.0", "1.0/16.0")):
        raise ValueError("stale ordinary sample budget")
    ratio = texts["source/_shaders/preview_sparse_ratio.glsl"]
    for value in ("#if PR_PREVIEW_PATHS != 8", "const uint nx=4u;", "const uint mx=2u;",
                  "halfCount=uint(PR_PREVIEW_PATHS)/2u", "i<uint(PR_PREVIEW_PATHS)",
                  "importanceSample(key,i-halfCount,halfCount,", "hitSum*(1.0/float(PR_PREVIEW_PATHS))"):
        if value not in ratio:
            raise ValueError("ratio budget/split/normalization mismatch: " + value)
    if "return cosinePdf/(0.5*(cosinePdf+environmentPdf));" not in indirect:
        raise ValueError("equal-count balance PDF changed")
    if "-DPR_PREVIEW_PATHS=8 " not in texts["source/_native/CMakeLists.txt"]:
        raise ValueError("build macro differs")
    if "c.sparse_shader),4,8," not in texts["source/_native/renderer.cpp"]:
        raise ValueError("device info does not report the actual eight-proposal budget")
    if "atomicAdd(status.primaryProposals,uint(PR_PREVIEW_PATHS))" not in texts["source/_shaders/resident_spatial.comp"]:
        raise ValueError("actual proposal counter differs")
    return dict(profile=PROFILE, samples=8, cosine_grid=[4, 2], mis_cosine_grid=[2, 2],
                mis_counts=[4, 4], hit_normalization=.125, proposal_counter="valid traced anchors * 8",
                profile_bindings=profile_bindings,
                ray_count_claim=False, source_contract_checked=True, device_execution=False)


def prepare_compact(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.is_relative_to(parent):
        raise ValueError("new package must be outside the immutable parent package")
    raw = (parent / "source-manifest.json").read_bytes()
    manifest = json.loads(raw)
    validate_parent(manifest["contract"])
    if manifest.get("profile_marker") != OLD_PROFILE:
        raise ValueError("parent shader profile differs")
    pinned = {}
    for relative, expected in manifest["generated_sha256"].items():
        local = safe_relative(relative)
        if local in pinned:
            raise ValueError("duplicate normalized source path")
        path = parent / local
        if not path.resolve().is_relative_to(parent) or path.is_symlink():
            raise ValueError("source symlinks/outside files are not admitted")
        if digest(path) != expected:
            raise ValueError("parent source hash differs: " + relative)
        pinned[local] = path.read_bytes()
    if not pinned or any(Path(key) not in pinned for key in source_edits()):
        raise ValueError("parent manifest omits a required source")
    patched = {Path(key): patch_text(key, pinned[Path(key)].decode("utf-8")) for key in source_edits()}
    output.mkdir(parents=True, exist_ok=False)
    for relative, data in pinned.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in patched:
            write(target, patched[relative])
        else:
            target.write_bytes(data)
    checked = verify_sources(output)
    policy = deepcopy(manifest["contract"])
    policy["paths_per_anchor"] = 8
    policy["proposal_budget"] = dict(schema="phymeo-preview-compact-eight-v1", **checked,
        distance="unchanged full-distance terminal queries; no local-radius approximation",
        zero_contribution="below shading hemisphere or Ng-clipped proposals remain in the eight-trial normalization",
        ratio="same finite-sample visibility ratio and SH approximation; black environment reduces to hit/8",
        history="composed separately by the unchanged rigid age<=4 implementation")
    result = deepcopy(manifest)
    result.update(contract=policy, profile_marker=PROFILE,
                  eight_parent_prepared=str(parent), eight_parent_manifest_sha256=sha256(raw).hexdigest(),
                  eight_builder_sha256=digest(__file__),
                  generated_sha256={str(p.relative_to(output)): digest(p)
                                    for p in sorted(output.rglob("*")) if p.is_file()},
                  state="eight-proposal sources only; no compiled/device/performance result implied")
    write(output / "source-manifest.json", json.dumps(result, indent=2) + "\n")
    return result


def prepare(parent, output):
    # Import composers only when generating an explicitly requested new package.
    from preview_compact_history import build_source as history
    from preview_compact_rigid_history import build_source as rigid

    output = Path(output).resolve()
    if output.is_relative_to(Path(parent).resolve()):
        raise ValueError("new candidate must be outside the immutable parent package")
    output.mkdir(parents=True, exist_ok=False)
    prepare_compact(parent, output / "compact")
    history.prepare(output / "compact", output / "history")
    rigid.prepare(output / "history", output / "rigid")
    verify_sources(output / "rigid")
    receipt = dict(schema="phymeo-preview-compact-eight-recipe-v1", parent=str(Path(parent).resolve()),
                   parent_manifest_sha256=digest(Path(parent) / "source-manifest.json"),
                   packages={name: dict(path=str(output / name), manifest_sha256=digest(output / name / "source-manifest.json"))
                             for name in ("compact", "history", "rigid")},
                   composers=[dict(path=str(Path(module.__file__).resolve()), sha256=digest(module.__file__))
                              for module in (history, rigid)],
                   state="source composition only; final rigid package still requires build and device checks")
    write(output / "recipe.json", json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.compact_prepared, args.output)


if __name__ == "__main__":
    main()
