#version 450
#extension GL_GOOGLE_include_directive : require
// Opaque shader: no discard, depth replacement or storage side effects.
layout(early_fragment_tests) in;
struct Vertex { vec4 position; vec4 normal; };
struct Transform { mat4 world; mat4 local; };
struct Instance { uvec4 binding; vec4 colorRoughness; vec4 material; };
struct Node { vec4 minimum; vec4 maximum; uvec4 data; };
struct Mesh { uvec4 range; uvec4 bvh; };
layout(set=0,binding=0,std430) readonly buffer SceneVertices { Vertex vertices[]; } geometry;
layout(set=0,binding=2,std430) readonly buffer Transforms { Transform transforms[]; } motion;
layout(set=0,binding=3,std430) readonly buffer Instances { Instance instances[]; } scene;
layout(set=0,binding=4,std430) readonly buffer Nodes { Node nodes[]; } tree;
layout(set=0,binding=5,std430) readonly buffer Indices { uint indices[]; } topology;
layout(set=0,binding=6,std430) readonly buffer Meshes { Mesh meshes[]; } assets;
struct StudioPanel { vec4 position; vec4 halfU; vec4 halfV; vec4 radiance; };
layout(set=0,binding=7,std140) uniform Studio { vec4 environment; StudioPanel panels[2]; } studio;
layout(set=0,binding=8,std430) readonly buffer InstanceTree { Node nodes[]; } topTree;
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
layout(location=0) in vec3 worldPosition;
layout(location=1) in vec3 worldNormal;
layout(location=2) in vec3 baseColor;
layout(location=3) in vec2 metalRough;
layout(location=4) flat in uvec2 ids;
layout(location=0) out vec4 rgb;
layout(location=1) out float forwardDepth;
layout(location=2) out uint instanceId;
layout(location=3) out uint valid;
const float PI = 3.141592653589793;

// Static per-mesh threaded BVHs, rigid instance transforms per frame.
// Every escape index increases. Traversal is bounded by the actual node count,
// never by an arbitrary work cutoff that could silently discard an occluder.
bool hitsBox(vec3 origin, vec3 direction, vec3 low, vec3 high, float tMax) {
    float first = u.options.y;
    float last = tMax;
    for (int axis=0; axis<3; ++axis) {
        if (abs(direction[axis]) < 1e-30) {
            if (origin[axis] < low[axis] || origin[axis] > high[axis]) return false;
        } else {
            float a = (low[axis]-origin[axis])/direction[axis];
            float b = (high[axis]-origin[axis])/direction[axis];
            first = max(first,min(a,b));
            last = min(last,max(a,b));
            if (last < first) return false;
        }
    }
    return true;
}
bool hitsTriangle(uint first, uint vertexBase, vec3 origin, vec3 direction, float tMax) {
    vec3 a = geometry.vertices[vertexBase+topology.indices[first]].position.xyz;
    vec3 e1 = geometry.vertices[vertexBase+topology.indices[first+1u]].position.xyz-a;
    vec3 e2 = geometry.vertices[vertexBase+topology.indices[first+2u]].position.xyz-a;
    vec3 p = cross(direction,e2);
    float determinant = dot(e1,p);
    // Relative parallel rejection does not discard small authored triangles.
    if (abs(determinant) <= 1e-7*length(e1)*length(e2)) return false;
    float inverse = 1.0/determinant;
    vec3 displacement = origin-a;
    float b1 = dot(displacement,p)*inverse;
    if (b1 < 0.0 || b1 > 1.0) return false;
    vec3 q = cross(displacement,e1);
    float b2 = dot(direction,q)*inverse;
    if (b2 < 0.0 || b1+b2 > 1.0) return false;
    float distance = dot(e2,q)*inverse;
    return distance > u.options.y && distance < tMax;
}
bool hitsInstance(uint slot, vec3 worldOrigin, vec3 worldDirection, float tMax) {
    Instance instance = scene.instances[slot];
    Mesh mesh = assets.meshes[instance.binding.x];
    vec3 origin = (motion.transforms[slot].local*vec4(worldOrigin,1.0)).xyz;
    // Do not normalize: affine inversion preserves the world ray parameter t.
    vec3 direction = (motion.transforms[slot].local*vec4(worldDirection,0.0)).xyz;
    uint node = mesh.bvh.x;
    uint end = node+mesh.bvh.y;
    for (uint visited=0u; visited<mesh.bvh.y && node<end; ++visited) {
        Node box = tree.nodes[node];
        if (!hitsBox(origin,direction,box.minimum.xyz,box.maximum.xyz,tMax)) {
            node = box.data.z;
        } else if (box.data.y != 0u) {
            for (uint t=0u; t<box.data.y; ++t)
                if (hitsTriangle(3u*(box.data.x+t),mesh.range.x,origin,direction,tMax))
                    return true;
            node = box.data.z;
        } else ++node;
    }
    return false;
}
bool occluded(vec3 worldOrigin, vec3 worldDirection, float tMax) {
    if (u.counts.w == 0u || tMax <= u.options.y) return false;
    uint count = 2u*u.counts.w-1u;
    uint node = 0u;
    for (uint visited=0u; visited<count && node<count; ++visited) {
        Node box = topTree.nodes[node];
        // Conservative culling margin only. The original local triangle test
        // still decides the hit; this cannot create geometry or a shadow.
        vec3 margin = 3.814697265625e-6 *
            (abs(worldOrigin)+max(abs(box.minimum.xyz),abs(box.maximum.xyz))+vec3(1e-30));
        if (!hitsBox(worldOrigin,worldDirection,box.minimum.xyz-margin,box.maximum.xyz+margin,tMax)) {
            node = box.data.z;
        } else if (box.data.y != 0u) {
            if (hitsInstance(box.data.x,worldOrigin,worldDirection,tMax)) return true;
            node = box.data.z;
        } else ++node;
    }
    return false;
}

#include "resident_studio.glsl"

vec3 encodeSRGB(vec3 linearColor) {
    vec3 x = clamp(linearColor, 0.0, 1.0);
    return mix(12.92*x, 1.055*pow(x, vec3(1.0/2.4))-0.055,
               greaterThan(x, vec3(0.0031308)));
}
void main() {
    forwardDepth = -(u.view*vec4(worldPosition, 1.0)).z;
    instanceId = ids.x;
    valid = 1u;
    vec3 color = vec3(0.0);
    if (u.counts.z != 0u) {
        vec3 N = normalize(worldNormal);
        vec3 V = normalize(u.camera.xyz-worldPosition);
        if (dot(N,V) < 0.0) N = -N; // Opaque, two-sided authored triangles.
        float nv = max(dot(N,V), 1e-5);
        float metallic = metalRough.x;
        // Explicit finite GGX lobe: perceptual roughness minimum 0.045.
        float alpha = max(metalRough.y*metalRough.y, 0.045*0.045);
        float a2 = alpha*alpha;
        vec3 f0 = mix(vec3(0.04), baseColor, metallic);
        for (uint i=0u; i<u.counts.x; ++i) {
            vec3 L = normalize(-u.lightDirections[i].xyz);
            float nl = max(dot(N,L), 0.0);
            if (nl <= 0.0 || occluded(worldPosition+N*u.options.y, L, 3.402823466e38)) continue;
            vec3 H = normalize(V+L);
            float nh = max(dot(N,H), 0.0);
            float vh = max(dot(V,H), 0.0);
            float denominator = nh*nh*(a2-1.0)+1.0;
            float D = a2/(PI*denominator*denominator);
            float Gv = 2.0*nv/(nv+sqrt(a2+(1.0-a2)*nv*nv));
            float Gl = 2.0*nl/(nl+sqrt(a2+(1.0-a2)*nl*nl));
            vec3 F = f0+(1.0-f0)*pow(1.0-vh,5.0);
            vec3 specular = D*Gv*Gl*F/max(4.0*nv*nl,1e-8);
            vec3 diffuse = (1.0-F)*(1.0-metallic)*baseColor/PI;
            color += (diffuse+specular)*u.lightColors[i].rgb*nl;
        }
    }
    if (u.counts.z != 0u && studio.environment.w != 0.0) {
        vec3 N = normalize(worldNormal);
        vec3 V = normalize(u.camera.xyz-worldPosition);
        if (dot(N,V) < 0.0) N = -N;
        color += residentStudioRadiance(worldPosition,N,V,baseColor,metalRough.x,metalRough.y);
    }
    rgb = vec4(encodeSRGB(color*u.options.x), 1.0);
}
