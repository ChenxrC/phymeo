#ifdef PR_PREVIEW_SPARSE_GUIDE
#error Edge rescue must use the full-distance branch, never the low-resolution guide branch.
#endif
// Full-rate selective incoming estimator. No ROI or object-specific policy.
layout(early_fragment_tests) in;
layout(constant_id=99) const uint PR_PREVIEW_EDGE_FRAGMENT_ABI=1u;
layout(set=0,binding=16) uniform sampler2D edgeSelection;
layout(set=0,binding=17,std430) buffer EdgeStatus {
    uint invalid,visible,anchors,reconstructed,fallback,nondiffuse,acceptedNeighbors,primaryProposals;
    uint selected,invocations,traced,merged,neighbors,missingRescue,budgetExhausted,maxInvocations;
} edgeStatus;
layout(set=0,binding=18) uniform usampler2D edgeIds;
layout(set=0,binding=19) uniform sampler2D edgeAttributes;
void edgeRescue(vec3 N,vec3 reflectionFace,bool faceValid,bool facesView,vec3 albedo) {
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    if(PR_PREVIEW_EDGE_FRAGMENT_ABI!=1u){atomicOr(edgeStatus.invalid,256u);discard;}
    if(texelFetch(edgeSelection,pixel,0).a!=1.0)discard;
    // Depth EQUAL plus identity and exact quantized normal binds the rasterized
    // surface to the primary pass. No claim of a primitive-ID gate is made.
    if(texelFetch(edgeIds,pixel,0).r!=ids.x)discard;
    uint code=sparsePackNormal(N,false);
    if(uint(texelFetch(edgeAttributes,pixel,0).a)!=code)discard;
    uint ticket=atomicAdd(edgeStatus.invocations,1u);
    if(ticket>=edgeStatus.maxInvocations){
        atomicAdd(edgeStatus.budgetExhausted,1u);atomicOr(edgeStatus.invalid,512u);discard;
    }
    vec3 incoming=vec3(0.0);
    if(faceValid&&facesView) {
        incoming=residentDiffuseIncoming(worldPosition,N,reflectionFace);
        atomicAdd(edgeStatus.traced,1u);
    }
    if(any(isnan(incoming))||any(isinf(incoming))||any(lessThan(incoming,vec3(0.0)))) {
        atomicOr(edgeStatus.invalid,1024u);discard;
    }
    // Valid bit means a genuine full-rate result, including physically zero
    // contribution for a rejected geometric hemisphere; not a fake traced flag.
    rgb=vec4(incoming,float(code|(1u<<20u)));
}