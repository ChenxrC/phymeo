#version 450
layout(location=0) in vec4 inPosition;
layout(location=1) in vec4 inNormal;
struct Transform { mat4 world; mat4 local; };
struct Instance { uvec4 binding; vec4 colorRoughness; vec4 material; };
layout(set=0,binding=2,std430) readonly buffer Transforms { Transform transforms[]; } motion;
layout(set=0,binding=3,std430) readonly buffer Instances { Instance instances[]; } scene;
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
// Default smooth interpolation is perspective-correct, matching gl_Position.w.
layout(location=5) out vec2 meshUV;
layout(location=6) flat out uint textureSlot;
layout(set=0,binding=10,std430) readonly buffer TextureMetadata { uint words[]; } textureData;
void main() {
    uint slot = uint(gl_InstanceIndex);
    Instance instance = scene.instances[slot];
    vec4 world = motion.transforms[slot].world * inPosition;
    vec3 p = (u.view * world).xyz;
    float depth = -p.z;
    float nearPlane = u.clipping.z;
    float farPlane = u.clipping.w;
    gl_Position = vec4(
        2.0*u.intrinsics.x/u.clipping.x*p.x + (2.0*u.intrinsics.z/u.clipping.x-1.0)*depth,
        -2.0*u.intrinsics.y/u.clipping.y*p.y + (2.0*u.intrinsics.w/u.clipping.y-1.0)*depth,
        farPlane/(farPlane-nearPlane)*depth - nearPlane*farPlane/(farPlane-nearPlane),
        depth);
    worldPosition = world.xyz;
    worldNormal = transpose(mat3(motion.transforms[slot].local)) * inNormal.xyz;
    baseColor = instance.colorRoughness.xyz;
    metalRough = vec2(instance.material.x, instance.colorRoughness.w);
    ids = uvec2(instance.binding.y, slot);
    textureSlot=textureData.words[4u+slot];
    meshUV=vec2(0.0);
    if (textureSlot!=0xffffffffu) {
        uint address=textureData.words[3]+2u*uint(gl_VertexIndex);
        meshUV=uintBitsToFloat(uvec2(textureData.words[address],textureData.words[address+1u]));
    }
}
