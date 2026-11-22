#version 450
layout(location=0) in vec4 inPosition;
layout(location=1) in vec4 inNormalMetallic;
layout(location=2) in vec4 inColorRoughness;
layout(location=3) in uvec4 inIds;
layout(set=0,binding=1,std140) uniform Parameters {
    mat4 view;
    vec4 intrinsics;
    vec4 clipping;
    vec4 camera;
    vec4 lightDirections[4];
    vec4 lightColors[4];
    uvec4 counts;
    vec4 options;
} u;
layout(location=0) out vec3 worldPosition;
layout(location=1) out vec3 worldNormal;
layout(location=2) out vec3 baseColor;
layout(location=3) out vec2 metalRough;
layout(location=4) flat out uvec2 ids;
void main() {
    vec3 p = (u.view * vec4(inPosition.xyz, 1.0)).xyz;
    float depth = -p.z;
    float nearPlane = u.clipping.z;
    float farPlane = u.clipping.w;
    // Positive-height Vulkan viewport: NDC -Y is the top row.
    gl_Position = vec4(
        2.0*u.intrinsics.x/u.clipping.x*p.x + (2.0*u.intrinsics.z/u.clipping.x-1.0)*depth,
        -2.0*u.intrinsics.y/u.clipping.y*p.y + (2.0*u.intrinsics.w/u.clipping.y-1.0)*depth,
        farPlane/(farPlane-nearPlane)*depth - nearPlane*farPlane/(farPlane-nearPlane),
        depth);
    worldPosition = inPosition.xyz;
    worldNormal = inNormalMetallic.xyz;
    baseColor = inColorRoughness.xyz;
    metalRough = vec2(inNormalMetallic.w, inColorRoughness.w);
    ids = inIds.xy;
}
