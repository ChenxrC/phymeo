// A fixed directional key covers the entire 16 by 16 laboratory in world metres.
// Project along the rays onto z=0; larger world z is closer to this light.
const vec3 lightVector = vec3(-0.56, -0.40, 1.0);
vec4 lightClip(vec3 p)
{
    vec2 footprint = p.xy - lightVector.xy * p.z;
    return vec4(footprint / vec2(17.0, 12.0), (5.0 - p.z) / 8.0, 1.0);
}
