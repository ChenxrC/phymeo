layout(constant_id=98) const uint PR_PREVIEW_EDGE_COMPUTE_ABI=1u;
layout(set=0,binding=12) uniform sampler2D edgeIncoming;
vec3 edgeCode(vec3 linear) {
    vec3 c=clamp(residentDisplayLinear(linear*p.exposure,float(p.mode)),0.0,1.0);
    c=mix(12.92*c,1.055*pow(c,vec3(1.0/2.4))-.055,greaterThan(c,vec3(.0031308)));
    return floor(c*255.0+vec3(.5));
}
void edgeMerge(ivec2 pixel,ivec2 extent,vec4 primary) {
    vec4 saved=imageLoad(meanLinear,pixel);
    vec3 linear=saved.rgb;
    if(PR_PREVIEW_EDGE_COMPUTE_ABI!=1u){atomicOr(status.invalid,2048u);return;}
    if(saved.a==1.0) {
        vec4 own=texelFetch(edgeIncoming,pixel,0);
        if(!finiteRadiance(own.rgb)||!finiteCode(own.a,2097151.0)||(uint(own.a)&(1u<<20u))==0u) {
            atomicAdd(status.missingRescue,1u);atomicOr(status.invalid,4096u);return;
        }
        uint id=texelFetch(instanceIds,pixel,0).r;
        float z=texelFetch(forwardDepth,pixel,0).r;
        vec3 normal=decodeNormal(uint(own.a)),guide=edgeCode(primary.rgb);
        const float kernel[5]=float[5](1.,4.,6.,4.,1.);
        vec3 sum=vec3(0.);float weights=0.;uint neighbors=0u;
        // Filter incoming only. Direct/specular and this pixel's rho stay local.
        // Read only the complete extra-pass image; never the in-place mean.
        for(int y=-2;y<=2;++y)for(int x=-2;x<=2;++x) {
            ivec2 q=pixel+ivec2(x,y);
            if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,extent)))continue;
            if(texelFetch(coverage,q,0).r==0u||texelFetch(instanceIds,q,0).r!=id)continue;
            vec4 sampleValue=texelFetch(edgeIncoming,q,0);
            if(!finiteCode(sampleValue.a,2097151.0)||(uint(sampleValue.a)&(1u<<20u))==0u)continue;
            if(!depthMatches(z,texelFetch(forwardDepth,q,0).r))continue;
            if(dot(normal,decodeNormal(uint(sampleValue.a)))<.98)continue;
            if(any(greaterThan(abs(edgeCode(imageLoad(currentLinear,q).rgb)-guide),vec3(2.))))continue;
            if(!finiteRadiance(sampleValue.rgb)){atomicOr(status.invalid,8192u);continue;}
            float weight=kernel[x+2]*kernel[y+2]*(1.0/256.0);sum+=weight*sampleValue.rgb;weights+=weight;++neighbors;
        }
        if(!(weights>0.)){atomicOr(status.invalid,16384u);return;}
        vec3 incoming=sum/weights;
        uint code=uint(primary.a);
        vec3 rho=vec3(code&255u,(code>>8u)&255u,(code>>16u)&255u)*(1.0/255.0);
        linear=primary.rgb+rho*incoming;
        atomicAdd(status.merged,1u);atomicAdd(status.neighbors,neighbors);
    }
    if(!finiteRadiance(linear)){atomicOr(status.invalid,32768u);return;}
    imageStore(meanLinear,pixel,vec4(linear,1.));
    vec3 mapped=residentDisplayLinear(linear*p.exposure,float(p.mode));
    if(!finiteRadiance(mapped)){atomicOr(status.invalid,65536u);return;}
    vec3 color=clamp(mapped,0.,1.);
    color=mix(12.92*color,1.055*pow(color,vec3(1./2.4))-.055,greaterThan(color,vec3(.0031308)));
    imageStore(displayColor,pixel,vec4(color,1.));
}