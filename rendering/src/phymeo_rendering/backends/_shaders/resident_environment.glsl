// Directional environment v1. SH uses world XYZ, cube sampling uses XZY.
layout(set=0,binding=11) uniform samplerCube environmentCube;
layout(set=0,binding=12,std140) uniform Environment {
    uvec4 config; // side, mip levels, enabled, reserved
    vec4 diffuseSH[9];
} environmentData;
bool residentEnvironmentEnabled() { return environmentData.config.z!=0u; }
vec3 residentEnvironmentDiffuse(vec3 n) {
    float x=n.x,y=n.y,z=n.z;
    float basis[9]=float[9](0.28209479177387814,
        0.4886025119029199*y,0.4886025119029199*z,0.4886025119029199*x,
        1.0925484305920792*x*y,1.0925484305920792*y*z,
        0.31539156525252005*(3.0*z*z-1.0),1.0925484305920792*x*z,
        0.5462742152960396*(x*x-y*y));
    vec3 result=vec3(0.0);
    for (int i=0;i<9;++i) result+=environmentData.diffuseSH[i].rgb*basis[i];
    return max(result,vec3(0.0)); // finite SH approximation may undershoot zero
}
vec3 residentEnvironmentSpecular(vec3 reflection,float roughness) {
    // Input levels are prefiltered at PERCEPTUAL roughness i/(levels-1).
    // A single split-sum lookup avoids a second directional convolution.
    float lod=clamp(roughness,0.0,1.0)*float(environmentData.config.y-1u);
    return textureLod(environmentCube,reflection.xzy,lod).rgb;
}
