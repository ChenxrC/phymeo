"""Derive stable per-pixel IBL shaders inside a copied surface-quality pack."""
from math import isfinite
from pathlib import Path


MARKER = "layout(constant_id=108) const uint PR_PREVIEW_STABLE_LIGHTING_ABI=1u;"
RADIUS_MARKER = "layout(constant_id=109) const uint PR_PREVIEW_AO_RADIUS_UM="


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"unexpected surface-quality shader at {old.splitlines()[0]!r}")
    return source.replace(old, new)


def configure_shaders(shader_directory, ao_radius_m):
    """Patch the parent's primary, guide and reconstruction; return their contract.

    Radius is compiled as an integer micrometre specialization default which
    the pack seal can independently verify. Neither descriptor layouts nor the
    contour pass change. All changes are composed before the first write.
    """
    if isinstance(ao_radius_m, bool) or not isinstance(ao_radius_m, (int, float)):
        raise ValueError("AO radius must be a distance in metres")
    if not isfinite(ao_radius_m) or not 0 <= ao_radius_m <= (2**32-1)/1e6:
        raise ValueError("AO radius is outside the unsigned micrometre range")
    radius_um = round(ao_radius_m * 1e6)
    if ao_radius_m > 0 and radius_um == 0:
        raise ValueError("positive AO radius rounds to zero micrometres")
    root = Path(shader_directory)
    frag_path = root / "resident.frag"
    comp_path = root / "resident_spatial.comp"
    low_path = root / "preview_lowres.glsl"
    ao_path = root / "preview_stable_ao.glsl"
    if ao_path.exists():
        raise ValueError("stable lighting shaders are already present")
    ao = Path(__file__).with_name("stable_ao.glsl").read_text(encoding="utf-8")
    frag = frag_path.read_text(encoding="utf-8")
    comp = comp_path.read_text(encoding="utf-8")
    low = low_path.read_text(encoding="utf-8")

    anchor = '#include "preview_sparse_common.glsl"'
    frag = _replace(frag, anchor, f'''{anchor}
#if defined(PR_PREVIEW_SPARSE_MAIN) || defined(PR_PREVIEW_SPARSE_GUIDE)
{MARKER}
#endif
#ifdef PR_PREVIEW_SPARSE_MAIN
{RADIUS_MARKER}{radius_um}u;
#include "preview_stable_ao.glsl"
#endif''')
    # The frozen native interface verifies the original ray-capable guide ABI.
    # The new specialization disables the trace at its existing eligibility
    # gate and publishes a guide with no traced flags or proposal counts.
    frag = _replace(frag, "bool traced=u.counts.z!=0u",
                    "bool traced=PR_PREVIEW_STABLE_LIGHTING_ABI==0u&&u.counts.z!=0u")
    old = "compactAttributes=vec4(shadingFacesView?sparseFallback(N):vec3(0.0),float(sparsePackNormal(N,false)));"
    frag = _replace(frag, old, '''vec3 environmentIncoming=shadingFacesView?sparseFallback(N):vec3(0.0);
    if(PR_PREVIEW_STABLE_LIGHTING_ABI==1u && PR_PREVIEW_AO_RADIUS_UM>0u &&
       indirectActive && reflectionFaceValid && sparsePackRho(rho)!=0u &&
       any(greaterThan(environmentIncoming,vec3(0.0)))) {
        environmentIncoming *= stableEnvironmentVisibility(
            worldPosition,N,reflectionFace,mat3(motion.transforms[ids.y].world));
    }
    compactAttributes=vec4(environmentIncoming,float(sparsePackNormal(N,false)));''')

    comp = _replace(comp, '#include "preview_lowres.glsl"', MARKER+'\n#include "preview_lowres.glsl"')
    start = "            vec3 normal=decodeNormal(uint(own.a));"
    end = "            linear+=rho*incoming;"
    if comp.count(start) != 1 or comp.count(end) != 1:
        raise ValueError("unexpected surface-quality reconstruction")
    first = comp.index(start)
    last = comp.index(end, first) + len(end)
    previous = comp[first:last]
    # Keep descriptor bindings for the sealed parent's native setup. Native
    # code uses the compiled defaults; no specialization override is supplied.
    updated = '''            if(PR_PREVIEW_STABLE_LIGHTING_ABI==1u) {
                historyValue=vec4(own.rgb,0.0);
                linear+=rho*own.rgb;
                atomicAdd(status.fallback,1u);
            } else {
''' + "\n".join("    "+line for line in previous.splitlines()) + "\n            }"
    comp = comp[:first] + updated + comp[last:]
    # Keep the readback view defined even though its data are no longer used.
    old = "lowFiltered.values[offset]=center;"
    low = _replace(low, old, old+"\n    if(PR_PREVIEW_STABLE_LIGHTING_ABI==1u)return;")
    history = (root / "preview_history.comp").read_text(encoding="utf-8")
    contour = (root / "preview_contour.glsl").read_text(encoding="utf-8")
    if not all(token in history for token in ("bool supported=current.a==1.0",
            "vec3 filtered=current.rgb;float age=0.0;", "nextHistory.pixels[index].incomingAge=vec4(filtered,age);")):
        raise ValueError("parent history no longer publishes unsupported current incoming unchanged")
    if ("HistoryPixel sampleValue=nextHistory.pixels[" not in contour or
            "radiance+=rho*sampleValue.incomingAge.rgb;" not in contour):
        raise ValueError("parent contour no longer composes the current frame's local incoming")
    for path, source in ((ao_path, ao), (frag_path, frag), (comp_path, comp), (low_path, low)):
        path.write_text(source, encoding="utf-8", newline="\n")
    return dict(ao_radius_m=radius_um/1e6, ao_radius_um=radius_um,
                max_ao_rays_per_diffuse_pixel=10 if radius_um else 0,
                ao_distribution="rigid-link dodecahedral axes, geometric hemisphere, cosine weights",
                ao_attenuation="smoothstep from geometric bias to radius at nearest opaque TLAS hit",
                indirect="per-receiver HDR diffuse SH or constant environment times local visibility",
                sparse_gi_proposals=0, temporal_history=False)
