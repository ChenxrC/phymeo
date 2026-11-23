// Private experiment ABI. Numeric float encodings stay in the exact uint24
// integer range: no NaN payloads and no reinterpretation as float bit patterns.
#if defined(PR_PREVIEW_SPARSE_MAIN)
layout(constant_id=78) const uint PR_PREVIEW_SPARSE_PRIMARY_ABI=1u;
#endif
#if defined(PR_PREVIEW_SPARSE_GUIDE)
layout(constant_id=79) const uint PR_PREVIEW_SPARSE_GUIDE_ABI=1u;
#endif
uint sparsePackRho(vec3 rho) {
    uvec3 code=uvec3(round(clamp(rho,vec3(0.0),vec3(1.0))*255.0));
    return code.x|(code.y<<8u)|(code.z<<16u);
}
vec2 sparseSign(vec2 v) {return mix(vec2(-1.0),vec2(1.0),greaterThanEqual(v,vec2(0.0)));}
uint sparsePackNormal(vec3 n,bool traced) {
    n/=abs(n.x)+abs(n.y)+abs(n.z);
    vec2 oct=n.z>=0.0?n.xy:(vec2(1.0)-abs(n.yx))*sparseSign(n.xy);
    uvec2 code=uvec2(round(clamp(oct*.5+.5,0.0,1.0)*1023.0));
    return code.x|(code.y<<10u)|(traced?(1u<<20u):0u);
}
bool sparseAnchor(ivec2 pixel) {
    ivec2 extent=ivec2(u.clipping.xy);
    ivec2 anchor=min((pixel/PR_PREVIEW_TILE)*PR_PREVIEW_TILE+ivec2(PR_PREVIEW_TILE/2),extent-ivec2(1));
    return all(equal(pixel,anchor));
}
vec3 sparseFallback(vec3 normal) {
    if(residentEnvironmentEnabled())return residentEnvironmentDiffuse(normal);
    return studio.environment.w!=0.0?studio.environment.rgb:vec3(0.0);
}
