// Copyright 2024 The Khronos Group, Inc.
// SPDX-License-Identifier: Apache-2.0
// Pinned upstream: KhronosGroup/ToneMapping b5a2eed5ddf6c2227090449399de9c7affb9e4c9
// PBR_Neutral/pbrNeutral.glsl. Full license: phymeo_rendering/_panda_assets/LICENSE.
// Phymeo adaptation: explicit opt-in, nonnegative input and bundle ABI marker.
layout(constant_id=58) const uint PR_DISPLAY_TRANSFORM_ABI=1u;

vec3 PBRNeutralToneMapping(vec3 color) {
    const float startCompression=0.8-0.04;
    const float desaturation=0.15;
    float x=min(color.r,min(color.g,color.b));
    float offset=x<0.08 ? x-6.25*x*x : 0.04;
    color-=offset;
    float peak=max(color.r,max(color.g,color.b));
    if (peak<startCompression) return color;
    // Overflow of finite nonnegative exposed radiance approaches white under
    // this operator. This limit also avoids inf/inf; ordinary values unchanged.
    if (isinf(peak)) return vec3(1.0);
    const float d=1.0-startCompression;
    float newPeak=1.0-d*d/(peak+d-startCompression);
    color*=newPeak/peak;
    float g=1.0-1.0/(desaturation*(peak-newPeak)+1.0);
    return mix(color,newPeak*vec3(1.0),g);
}
vec3 residentDisplayLinear(vec3 exposed, float mode) {
    if (mode==1.0 && PR_DISPLAY_TRANSFORM_ABI==1u)
        return PBRNeutralToneMapping(max(exposed,vec3(0.0)));
    return exposed; // Original clip/OETF is kept at the original call site.
}
