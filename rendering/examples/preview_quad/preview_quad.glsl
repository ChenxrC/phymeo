// Experimental 2/4-primary-proposal incoming-radiance reuse, NOT a public ABI.
// Included only after resident_indirect.glsl in the registered progressive path.
// Shared values are demodulated incoming: original residentStudioRadiance applies
// this receiver's diffuse coefficient exactly once. Direct/specular are untouched.
#ifndef PR_PREVIEW_SAMPLES
#error PR_PREVIEW_SAMPLES must be explicitly 2 or 4
#endif
#if PR_PREVIEW_SAMPLES != 2 && PR_PREVIEW_SAMPLES != 4
#error This experiment supports only 2 or 4 proposals
#endif
#ifndef PR_PREVIEW_SHARE
#define PR_PREVIEW_SHARE 1
#endif
layout(constant_id=120) const uint PR_PREVIEW_QUAD_ABI=1u;
layout(constant_id=121) const uint PR_PREVIEW_PRIMARY_PROPOSALS=PR_PREVIEW_SAMPLES;
layout(constant_id=122) const uint PR_PREVIEW_SPATIAL_REUSE=PR_PREVIEW_SHARE;

vec3 previewLocalIncoming(vec3 P,vec3 N,vec3 faceN) {
    uvec2 pixel=uvec2(gl_FragCoord.xy);
    uint base=indirectHash(u.counts.y^indirectHash(pixel.x+0x9e3779b9u)^indirectHash(pixel.y+0x85ebca6bu));
    uint batch=uint(u.options.w)>>1u;
    if(batch!=0u)base=indirectHash(base^indirectHash(batch^0xa511e9b3u));
    vec3 axis=abs(N.z)<.9?vec3(0,0,1):vec3(0,1,0);
    vec3 T=normalize(cross(axis,N));
    mat3 basis=mat3(T,cross(N,T),N);
    bool mis=importanceActive();
    uint halfCount=PR_PREVIEW_PRIMARY_PROPOSALS/2u;
    vec3 sum=vec3(0.0);
    for(uint i=0u;i<PR_PREVIEW_PRIMARY_PROPOSALS;++i) {
        uint key=indirectHash(base^(i*0x9e3779b9u));
        vec2 jitter=vec2(indirectOpen(indirectHash(key^0x68bc21ebu)),
                         indirectOpen(indirectHash(key^0x02e5be93u)));
        // Radial equal-probability strata, full azimuth per stratum.
        // 2 total -> 1 cosine + 1 CDF; 4 total -> 2 cosine + 2 CDF.
        uint cosineStrata=mis?halfCount:PR_PREVIEW_PRIMARY_PROPOSALS;
        vec2 q=vec2((float(i%cosineStrata)+jitter.x)/float(cosineStrata),jitter.y);
        float r=sqrt(q.x),phi=2.0*PI*q.y;
        vec3 direction=basis*vec3(r*cos(phi),r*sin(phi),sqrt(max(0.0,1.0-q.x)));
        bool fromEnvironment=mis && i>=halfCount;
        float sampledPdf=0.0;
        if(fromEnvironment && !importanceSample(key,i-halfCount,halfCount,direction,sampledPdf))
            return vec3(importanceInvalid());
        // Rejected hemisphere samples remain zero in the FIXED denominator.
        if(dot(direction,faceN)<=0.0 || (mis && dot(direction,N)<=0.0))continue;
        float weight=mis?(fromEnvironment?importanceWeightPdf(N,direction,sampledPdf):
                         importanceWeight(N,direction)):1.0;
        vec3 incoming;
        if(!residentTraceTerminal(P,faceN,direction,incoming))incoming=progressiveMiss(direction);
        sum+=incoming*weight;
    }
    return sum/float(PR_PREVIEW_PRIMARY_PROPOSALS);
}

vec3 previewQuadIncoming(vec3 P,vec3 N,vec3 faceN,float depth,uvec2 identity,bool receiver) {
    // Uniform entry from main: never call this under a material/receiver branch.
    bool finiteGeometry=!any(isnan(N))&&!any(isinf(N))&&!any(isnan(faceN))&&
                        !any(isinf(faceN))&&!isnan(depth)&&!isinf(depth)&&depth>0.0;
    bool covered=!gl_HelperInvocation && (gl_SampleMaskIn[0]&1)!=0;
    bool live=covered && receiver && finiteGeometry;
    vec3 localIncoming=vec3(0.0);
    if(live)localIncoming=previewLocalIncoming(P,N,faceN);
    // All participating invocations reach this ballot after local estimation.
    // Helpers may be inactive for ballot; they are NEVER admitted as sources.
    uvec4 liveMask=subgroupBallot(live);
    uint quadBase=gl_SubgroupInvocationID&~3u;
    vec3 sum=vec3(0.0);
    float count=0.0;

    // Each guard is identical across its quad. The source is checked BEFORE any
    // broadcast, so no inactive/helper output is fetched and then masked later.
    // All exchanges precede the per-receiver compatibility branch. Literal IDs
    // also support SPIR-V <=1.4's constant QuadBroadcast Index restriction.
#define PREVIEW_TAP(LANE) \
    if(subgroupBallotBitExtract(liveMask,quadBase+LANE)) { \
        uvec2 otherID=subgroupQuadBroadcast(identity,LANE); \
        float otherDepth=subgroupQuadBroadcast(depth,LANE); \
        vec3 otherN=subgroupQuadBroadcast(N,LANE); \
        vec3 otherFace=subgroupQuadBroadcast(faceN,LANE); \
        vec3 otherIncoming=subgroupQuadBroadcast(localIncoming,LANE); \
        bool compatible=all(equal(identity,otherID)) && \
            abs(depth-otherDepth)<=0.003*max(depth,otherDepth) && \
            dot(N,otherN)>=0.995 && dot(faceN,otherFace)>=0.999; \
        if(live && compatible) { sum+=otherIncoming; count+=1.0; } \
    }
    PREVIEW_TAP(0u)
    PREVIEW_TAP(1u)
    PREVIEW_TAP(2u)
    PREVIEW_TAP(3u)
#undef PREVIEW_TAP
    vec3 reused=count>0.0?sum/count:localIncoming;
    // Keep named specialization markers live for independent SPIR-V admission.
    // This limited first-hop prototype must never silently accept a two-hop run.
    bool contract=PR_INDIRECT_ABI==1u && PR_PREVIEW_QUAD_ABI==1u &&
                  (PR_PREVIEW_PRIMARY_PROPOSALS==2u || PR_PREVIEW_PRIMARY_PROPOSALS==4u) &&
                  (PR_PREVIEW_SPATIAL_REUSE==0u || PR_PREVIEW_SPATIAL_REUSE==1u) &&
                  (uint(u.options.w)&1u)==0u;
    if(!contract)return vec3(importanceInvalid());
    return live?(PR_PREVIEW_SPATIAL_REUSE!=0u?reused:localIncoming):vec3(0.0);
}

