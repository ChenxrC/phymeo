#version 450
struct Vertex { vec4 position; vec4 normalMetallic; vec4 colorRoughness; uvec4 ids; };
layout(set=0,binding=0,std430) readonly buffer SceneVertices { Vertex vertices[]; } geometry;
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

// Bounded primary raster with GPU hard-shadow rays. This is not a PT integrator.
bool occluded(vec3 origin, vec3 direction) {
    for (uint t=0u; t<u.counts.y; ++t) {
        if (t == ids.y) continue;
        vec3 a = geometry.vertices[3u*t].position.xyz;
        vec3 e1 = geometry.vertices[3u*t+1u].position.xyz-a;
        vec3 e2 = geometry.vertices[3u*t+2u].position.xyz-a;
        vec3 p = cross(direction, e2);
        float determinant = dot(e1, p);
        if (abs(determinant) < 1e-8) continue;
        float inverse = 1.0/determinant;
        vec3 displacement = origin-a;
        float b1 = dot(displacement, p)*inverse;
        if (b1 < 0.0 || b1 > 1.0) continue;
        vec3 q = cross(displacement, e1);
        float b2 = dot(direction, q)*inverse;
        if (b2 < 0.0 || b1+b2 > 1.0) continue;
        if (dot(e2, q)*inverse > u.options.y) return true;
    }
    return false;
}
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
            if (nl <= 0.0 || occluded(worldPosition+N*u.options.y, L)) continue;
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
    rgb = vec4(encodeSRGB(color*u.options.x), 1.0);
}
