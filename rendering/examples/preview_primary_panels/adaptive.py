"""Build a separate 2x2-pilot/8x8-penumbra shader set for fixed opaque panels.

The public fixed-grid artifacts remain untouched. All terminal hits always use
the original 2x2 quadrature. The bounded pilot can miss a fine blocker.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess


def pilot_grid(mask, *, primary):
    if type(primary) is not bool or type(mask) is not int:
        raise TypeError("expected an integer mask and a boolean receiver")
    if not 0 <= mask < 16:
        raise ValueError("four visibility bits required")
    return 8 if primary and mask not in (0, 15) else 2


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("unsupported upstream area-light implementation: " + old[:55])
    return text.replace(old, new)


PILOT = """
        int sampleGrid = areaGrid;
        uint pilotVisible = 0u;
        if (areaGrid == 8) {
            for (int k=0; k<4; ++k) {
                vec2 cell=vec2(k&1,k>>1)-vec2(.5);
                vec3 d=panel.position.xyz+cell.x*U+cell.y*W-biasP;
                float distanceToLight=length(d);
                vec3 lightDirection=d/distanceToLight;
                if (dot(N,lightDirection)>0.0 && -dot(emitterNormal,lightDirection)>0.0 &&
                    !occluded(biasP,lightDirection,distanceToLight))
                    pilotVisible |= 1u<<uint(k);
            }
            if (pilotVisible == 15u || pilotVisible == 0u) sampleGrid=2;
        }
        int areaSamples=sampleGrid*sampleGrid;
        samplesByPanel[j]=areaSamples;
        if (areaGrid==8 && pilotVisible==0u) continue;
"""


def adapt_studio_source(text):
    text = replace_once(text, "int studioAreaSamples=areaGrid*areaGrid;",
                        "int samplesByPanel[2]=int[2](areaGrid*areaGrid,areaGrid*areaGrid);")
    text = replace_once(text, "        for (int i=0; i<studioAreaSamples; ++i) {",
                        PILOT + "        for (int i=0; i<areaSamples; ++i) {")
    text = replace_once(text, "if(areaGrid==2) ab=", "if(sampleGrid==2) ab=")
    text = replace_once(text, "i%areaGrid,i/areaGrid", "i%sampleGrid,i/sampleGrid")
    text = replace_once(text, "float(areaGrid)-vec2(1.0);", "float(sampleGrid)-vec2(1.0);")
    # A fully clear pilot has already tested exactly these four segments.
    text = replace_once(text, "nl<=0.0 || emit<=0.0 || occluded(biasP,L,dist)",
                        "nl<=0.0 || emit<=0.0 || ((areaGrid!=8 || sampleGrid!=2) && occluded(biasP,L,dist))")
    if text.count("float(studioAreaSamples)") != 3:
        raise ValueError("unexpected MIS or area normalization")
    at = text.index("    for (int i=0; i<studioBrdfSamples; ++i)")
    text = (text[:at].replace("float(studioAreaSamples)", "float(areaSamples)") +
            text[at:].replace("float(studioAreaSamples)", "float(samplesByPanel[j])"))
    text = replace_once(text, "layout(constant_id=73)", "layout(constant_id=91) const uint PR_PRIMARY_PANEL_PILOT=8u;\nlayout(constant_id=73)")
    return replace_once(text, "vec3 result = diffuseEnvironment*diffuse*(1.0-f0);",
                        "if(PR_PRIMARY_PANEL_PILOT!=8u)return vec3(0.0);\n    vec3 result = diffuseEnvironment*diffuse*(1.0-f0);")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def build(source, baseline, compiler, output):
    source, baseline, compiler = [p.resolve(strict=True) for p in (source, baseline, compiler)]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shaders = output / "sources"
    shutil.copytree(source, shaders)
    integrator = shaders / "resident_studio.glsl"
    integrator.write_text(adapt_studio_source(integrator.read_text(encoding="utf-8")), encoding="utf-8")
    fragments = sorted(baseline.glob("*_area8_secondary2.frag.spv"))
    if len(fragments) != 12:
        raise ValueError("expected the twelve fixed 8/2 primary/guide/secondary shader variants")
    for item in baseline.glob("*.spv"):
        if item not in fragments:
            shutil.copyfile(item, output / item.name)
    report = dict(schema="phymeo-primary-panel-pilot-v1", basis="four midpoint visibility probes per emitting panel on a primary hit; 8x8 on mixed visibility, otherwise 2x2; secondary always 2x2",
                  original_integrator=digest(source / "resident_studio.glsl"), compiled_integrator=digest(integrator),
                  compiler=digest(compiler), fragments={}, status="building")
    manifest = output / "adaptive.json"
    try:
        for old in fragments:
            flags = ["PR_RESIDENT_RAY_QUERY", "PR_STUDIO_AREA_GRID=8", "PR_STUDIO_SECONDARY_GRID=2"]
            for token, define in (("_normal_", "PR_RESIDENT_NORMAL_MAPS"), ("_reflection_", "PR_RESIDENT_REFLECTION"),
                                  ("_indirect_", "PR_RESIDENT_INDIRECT"), ("_progressive_", "PR_RESIDENT_PROGRESSIVE")):
                if token in old.name:
                    flags.append(define)
            cmd = [str(compiler), "-V", "--target-env", "vulkan1.2", "--glsl-version", "460",
                   *["-D" + value if "=" in value else "-D" + value + "=1" for value in flags],
                   str(shaders / "resident.frag"), "-o", str(output / old.name)]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (output / (old.name + ".log")).write_bytes(result.stdout)
            result.check_returncode()
            if b"PR_PRIMARY_PANEL_PILOT" not in (output / old.name).read_bytes():
                raise ValueError("shader did not retain the pilot contract")
            report["fragments"][old.name] = dict(original=digest(old), candidate=digest(output / old.name))
        report["status"] = "complete"
    finally:
        manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "baseline", "compiler", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.baseline, args.compiler, args.output)
