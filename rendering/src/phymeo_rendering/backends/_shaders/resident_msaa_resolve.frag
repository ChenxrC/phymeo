#version 450
layout(constant_id=61) const uint PR_MSAA_RESOLVE_ABI=1u;
layout(set=0,binding=0) uniform sampler2D displayLinearResolved;
layout(location=0) out vec4 rgb;
void main() {
    if (PR_MSAA_RESOLVE_ABI!=1u) discard;
    // Vulkan averaged the four covered display-linear samples into RGBA16F.
    // Exactly one IEC sRGB transfer follows that average, never before it.
    vec3 x=clamp(texelFetch(displayLinearResolved,ivec2(gl_FragCoord.xy),0).rgb,0.0,1.0);
    vec3 encoded=mix(12.92*x,1.055*pow(x,vec3(1.0/2.4))-0.055,
                     greaterThan(x,vec3(0.0031308)));
    rgb=vec4(encoded,1.0);
}
