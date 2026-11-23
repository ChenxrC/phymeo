// Explicit biased finite-sample ratio estimator. Same proposals as radiance
// mode; no additional rays, ray truncation, radiance clamps or extra IBL lobe.
layout(constant_id=84) const uint PR_PREVIEW_SKY_VISIBILITY_RATIO_ABI=1u;
vec3 sparseRatioIncoming(vec3 P,vec3 N,vec3 faceN,out uint outcomes) {
    outcomes=0u;
    if(PR_PREVIEW_SKY_VISIBILITY_RATIO_ABI!=1u||PR_INDIRECT_ABI!=1u)
        return vec3(importanceInvalid());
    uvec2 pixel=uvec2(gl_FragCoord.xy);
    uint base=indirectHash(u.counts.y^indirectHash(pixel.x+0x9e3779b9u)^indirectHash(pixel.y+0x85ebca6bu));
    vec3 axis=abs(N.z)<.9?vec3(0,0,1):vec3(0,1,0);
    vec3 T=normalize(cross(axis,N));mat3 basis=mat3(T,cross(N,T),N);
    const uint nx=PR_PREVIEW_PATHS==16?4u:2u;
    const uint ny=uint(PR_PREVIEW_PATHS)/nx;
    const uint halfCount=uint(PR_PREVIEW_PATHS)/2u;
    const uint mx=halfCount<4u?halfCount:4u;
    const uint my=halfCount/mx;
    bool environmentMIS=importanceActive();
    vec3 skyTotal=vec3(0.0),skyVisible=vec3(0.0),hitSum=vec3(0.0);
    float totalWeight=0.0,visibleWeight=0.0;
    for(uint i=0u;i<uint(PR_PREVIEW_PATHS);++i) {
        uint key=indirectHash(base^(i*0x9e3779b9u));
        vec2 jitter=vec2(indirectOpen(indirectHash(key^0x68bc21ebu)),indirectOpen(indirectHash(key^0x02e5be93u)));
        vec2 q=(vec2(i%nx,i/nx)+jitter)/vec2(nx,ny);
        if(environmentMIS)q=(vec2((i%halfCount)%mx,(i%halfCount)/mx)+jitter)/vec2(mx,my);
        float r=sqrt(q.x),phi=2.0*PI*q.y;
        vec3 direction=basis*vec3(r*cos(phi),r*sin(phi),sqrt(max(0.0,1.0-q.x)));
        bool fromEnvironment=environmentMIS&&i>=halfCount;
        float sampledPdf=0.0;
        if(fromEnvironment&&!importanceSample(key,i-halfCount,halfCount,direction,sampledPdf))
            return vec3(importanceInvalid());
        if(environmentMIS&&dot(direction,N)<=0.0)continue;
        float weight=environmentMIS?(fromEnvironment?
            importanceWeightPdf(N,direction,sampledPdf):importanceWeight(N,direction)):1.0;
        vec3 sky=progressiveMiss(direction);
        // Denominator is the unobstructed *shading* hemisphere. An Ng-clipped
        // direction remains in it, with zero visibility and no terminal hit.
        skyTotal+=sky*weight;totalWeight+=weight;
        if(dot(direction,faceN)<=0.0){outcomes|=1u<<23u;continue;}
        vec3 terminal;
        if(residentTraceTerminal(P,faceN,direction,terminal)) {
            outcomes|=1u<<22u;hitSum+=terminal*weight;
        } else {
            outcomes|=1u<<21u;skyVisible+=sky*weight;visibleWeight+=weight;
        }
    }
    float visibility=totalWeight>0.0?visibleWeight/totalWeight:0.0;
    vec3 fraction;
    for(int c=0;c<3;++c)fraction[c]=skyTotal[c]>0.0?skyVisible[c]/skyTotal[c]:visibility;
    // The fallback is the existing normalized SH or constant IBL value.
    // Fully open -> exactly that value; no environment -> original hit/N.
    return sparseFallback(N)*fraction+hitSum*(1.0/float(PR_PREVIEW_PATHS));
}
