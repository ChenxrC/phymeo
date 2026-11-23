#version 450
#extension GL_GOOGLE_include_directive : require
layout(set=0,binding=1,std140) uniform Parameters {
    mat4 view; vec4 intrinsics; vec4 clipping; vec4 camera;
    vec4 lightDirections[4]; vec4 lightColors[4]; uvec4 counts; vec4 options;
} u;
#include "resident_environment.glsl"
layout(location=0) out vec4 rgb;
#include "resident_display.glsl"
#include "resident_msaa.glsl"
vec3 encodeSky(vec3 linearColor) {
    if (PR_MSAA_COLOR_ABI==1u && PR_MSAA_LINEAR_OUTPUT==1u) return clamp(linearColor,0.0,1.0);
    vec3 x=clamp(linearColor,0.0,1.0);
    return mix(12.92*x,1.055*pow(x,vec3(1.0/2.4))-0.055,greaterThan(x,vec3(0.0031308)));
}
void main() {
    // Pixel centers and off-axis principal point use the same camera convention
    // as resident.vert: camera forward -Z, up +Y, image rows downwards.
    vec2 sensor=(gl_FragCoord.xy-u.intrinsics.zw)/u.intrinsics.xy;
    vec3 ray=transpose(mat3(u.view))*normalize(vec3(sensor.x,-sensor.y,-1.0));
    vec3 radiance=textureLod(environmentCube,ray.xzy,0.0).rgb;
    // Reading enabled makes a mismatched sky UBO visible to interface checks.
    rgb=vec4(encodeSky(residentDisplayLinear((residentEnvironmentEnabled()?radiance:vec3(0.0))*u.options.x,u.options.z)),1.0);
}
