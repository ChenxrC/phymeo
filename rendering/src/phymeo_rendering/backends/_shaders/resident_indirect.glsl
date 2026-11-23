// One diffuse bounce: cosine PDF cancels cos(theta)/pi, so mean Li is multiplied
// by rho_d exactly once by the studio lobe. Stratification is fixed 4x4; seed
// randomizes each stratum independently. No frame/time/camera history exists.
layout(constant_id=65) const uint PR_INDIRECT_ABI=1u;
uint indirectHash(uint x) {
    x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;x^=x>>16u;return x;
}
float indirectOpen(uint x) {
    // 23 bits plus .5 is exactly representable in float, strictly in (0,1).
    return (float(x>>9u)+0.5)*1.1920928955078125e-7;
}
vec3 residentDiffuseIncoming(vec3 P,vec3 N,vec3 faceN) {
    if(PR_INDIRECT_ABI!=1u)return vec3(0.0);
    uvec2 pixel=uvec2(gl_FragCoord.xy);
    uint base=indirectHash(u.counts.y^indirectHash(pixel.x+0x9e3779b9u)^indirectHash(pixel.y+0x85ebca6bu));
    vec3 axis=abs(N.z)<.9?vec3(0,0,1):vec3(0,1,0);
    vec3 T=normalize(cross(axis,N));mat3 basis=mat3(T,cross(N,T),N);
    vec3 sum=vec3(0.0);
    for(uint i=0u;i<16u;++i) {
        uint key=indirectHash(base^(i*0x9e3779b9u));
        vec2 jitter=vec2(indirectOpen(indirectHash(key^0x68bc21ebu)),indirectOpen(indirectHash(key^0x02e5be93u)));
        vec2 q=(vec2(i&3u,i>>2u)+jitter)*.25;
        float r=sqrt(q.x),phi=2.0*PI*q.y;
        vec3 direction=basis*vec3(r*cos(phi),r*sin(phi),sqrt(max(0.0,1.0-q.x)));
        // Do not renormalize surviving samples: clipping means zero contribution
        // from directions below the actual triangle's geometric hemisphere.
        if(dot(direction,faceN)<=0.0)continue;
        vec3 incoming;
        if(!residentTraceTerminal(P,faceN,direction,incoming)) {
            incoming=studio.environment.w!=0.0?studio.environment.rgb:vec3(0.0);
            if(residentEnvironmentEnabled())incoming=textureLod(environmentCube,direction.xzy,0.0).rgb;
        }
        sum+=incoming;
    }
    return sum*(1.0/16.0);
}
