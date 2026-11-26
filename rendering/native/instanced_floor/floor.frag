#version 450
#extension GL_GOOGLE_include_directive : require
#include "floor_light.glsl"
layout(location=0) in vec3 worldPosition;
layout(location=1) in vec3 worldNormal;
layout(location=2) in vec3 albedo;
layout(location=3) flat in vec3 towardCamera;
layout(set=0,binding=0) uniform sampler2D shadowMap;
layout(location=0) out vec4 pixel;

float visibility(vec3 normal, float facing)
{
    vec3 q = lightClip(worldPosition + 0.0012 * normal).xyz;
    vec2 uv = q.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 1.0;
    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
    float depth = q.z - (0.00018 + 0.00055 * (1.0 - facing));
    float sum = 0.0;
    for (int y=-1; y<=1; ++y)
        for (int x=-1; x<=1; ++x)
            sum += float(depth <= texture(shadowMap, uv + vec2(x, y) * texel).r) *
                   float((2-abs(x)) * (2-abs(y)));
    return sum / 16.0;
}
void main()
{
    vec3 n = normalize(worldNormal);
    vec3 l = normalize(lightVector);
    float facing = max(dot(n,l), 0.0);
    float visible = visibility(n, facing);
    float hemi = clamp(n.z * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky = mix(vec3(0.22,0.205,0.18), vec3(0.59,0.65,0.72), hemi);
    vec3 bounce = vec3(0.12,0.125,0.13) * max(0.0,dot(n,normalize(vec3(0.8,0.6,0.36))));
    vec3 lit = albedo * (sky + bounce + visible * facing * vec3(0.88,0.82,0.71));
    float highlight = pow(max(0.0,dot(n,normalize(l + towardCamera))), 36.0);
    lit += visible * facing * (0.045 + 0.07 * albedo) * highlight;
    // A gentle highlight shoulder. The sRGB attachment handles the display transfer.
    pixel = vec4(lit / (vec3(1.0) + 0.23 * lit), 1.0);
}
