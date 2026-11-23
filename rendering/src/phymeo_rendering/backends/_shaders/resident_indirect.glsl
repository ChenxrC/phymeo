// Legacy cosine sampling cancels cos(theta)/pi, so mean Li is multiplied
// by rho_d exactly once by the studio lobe. Stratification is fixed 4x4; seed
// randomizes each stratum independently. Progressive HDR opt-in uses balance MIS.
layout(constant_id=65) const uint PR_INDIRECT_ABI=1u;
uint indirectHash(uint x) {
    x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;x^=x>>16u;return x;
}
float indirectOpen(uint x) {
    // 23 bits plus .5 is exactly representable in float, strictly in (0,1).
    return (float(x>>9u)+0.5)*1.1920928955078125e-7;
}
#ifdef PR_RESIDENT_PROGRESSIVE
layout(constant_id=72) const uint PR_ENVIRONMENT_CDF_ABI=1u;
// Private packed alpha: L0 is a global inclusive CDF; L1 is its inverse directory.
// Only explicitly marked, nonconstant maps opt in; RGB and the old path stay intact.
bool importanceActive() {
    return PR_ENVIRONMENT_CDF_ABI==1u && residentEnvironmentEnabled() && environmentData.config.w==1u;
}
float importanceInvalid() { return uintBitsToFloat(0x7fc00000u); }
vec3 importanceFaceVector(int face,vec2 uv) {
    if(face==0)return vec3(1.0,-uv.y,-uv.x);
    if(face==1)return vec3(-1.0,-uv.y,uv.x);
    if(face==2)return vec3(uv.x,1.0,uv.y);
    if(face==3)return vec3(uv.x,-1.0,-uv.y);
    if(face==4)return vec3(uv.x,-uv.y,1.0);
    return vec3(-uv.x,-uv.y,-1.0);
}
void importanceCoordinates(vec3 world,out int face,out vec2 uv,out ivec2 cell) {
    vec3 d=world.xzy,a=abs(d);
    // Deterministic tie order. Interior proposals never depend on cube seam ties.
    if(a.x>=a.y && a.x>=a.z) {
        face=d.x>=0.0?0:1;
        uv=d.x>=0.0?vec2(-d.z,-d.y)/a.x:vec2(d.z,-d.y)/a.x;
    } else if(a.y>=a.z) {
        face=d.y>=0.0?2:3;
        uv=d.y>=0.0?vec2(d.x,d.z)/a.y:vec2(d.x,-d.z)/a.y;
    } else {
        face=d.z>=0.0?4:5;
        uv=d.z>=0.0?vec2(d.x,-d.y)/a.z:vec2(-d.x,-d.y)/a.z;
    }
    int side=int(environmentData.config.x);
    cell=clamp(ivec2(floor((uv+vec2(1.0))*(0.5*float(side)))),ivec2(0),ivec2(side-1));
}
float importancePackedAlpha(int index,int level) {
    int side=int(environmentData.config.x)>>level,faceSize=side*side;
    int face=index/faceSize,local=index-face*faceSize;
    ivec2 cell=ivec2(local%side,local/side);
    // Power-of-two exact texel centres; major coordinate stays exactly 1.
    // Never normalize this metadata lookup or interpolate adjacent CDF entries.
    vec2 uv=(vec2(cell)+vec2(0.5))*(2.0/float(side))-vec2(1.0);
    return textureLod(environmentCube,importanceFaceVector(face,uv),float(level)).a;
}
float importanceCdf(int index) {
    return index<0?0.0:importancePackedAlpha(index,0);
}
float importanceLeafProbability(int face,ivec2 cell) {
    int side=int(environmentData.config.x),index=face*side*side+cell.y*side+cell.x;
    float upper=importanceCdf(index),lower=importanceCdf(index-1);
    if(isnan(upper)||isinf(upper)||isnan(lower)||isinf(lower)||lower<0.0||upper<lower||upper>1.0)
        return importanceInvalid();
    return upper-lower;
}
int importanceDirectory(int bucket,int size) {
    float value=importancePackedAlpha(bucket,1);
    // All indices <=6*1024^2-1 are exactly representable in FP32.
    if(isnan(value)||isinf(value)||value<0.0||value>=float(size)||floor(value)!=value)return -1;
    return int(value);
}
int importanceSearch(float selector,int lo,int hi) {
    // ceil(log2(6*1024^2))=23; normally the directory narrows this to a few bins.
    for(int step=0;step<23 && lo<hi;++step) {
        int mid=lo+(hi-lo)/2;
        float value=importanceCdf(mid);
        if(isnan(value)||isinf(value)||value<0.0||value>1.0)return -1;
        if(value>selector)hi=mid;else lo=mid+1;
    }
    return lo==hi?lo:-1;
}
int importanceInverse(float selector,out float probability) {
    int side=int(environmentData.config.x),size=6*side*side,buckets=size/4;
    int bucket=clamp(int(floor(selector*float(buckets))),0,buckets-1);
    // Directory j was built at rounded FP32(j/B), whereas u*B can round onto
    // either side. Correct that bucket before looking up its two exact indices.
    for(int correction=0;correction<2;++correction) {
        if(bucket>0 && selector<float(bucket)/float(buckets))--bucket;
        else if(bucket+1<buckets && selector>=float(bucket+1)/float(buckets))++bucket;
        else break;
    }
    int lo=0,hi=size-1;
    bool bucketValid=selector>=float(bucket)/float(buckets) &&
        (bucket+1==buckets || selector<float(bucket+1)/float(buckets));
    if(bucketValid) {
        lo=importanceDirectory(bucket,size);
        hi=bucket+1<buckets?importanceDirectory(bucket+1,size):size-1;
        if(lo<0||hi<lo) {lo=0;hi=size-1;}
    }
    // No redundant bracket texture reads. The two final reads needed for PMF
    // also establish the strict inverse invariant; only failure takes fallback.
    for(int attempt=0;attempt<2;++attempt) {
        int index=importanceSearch(selector,lo,hi);
        if(index<0)return -1;
        float upper=importanceCdf(index),lower=importanceCdf(index-1);
        if(!isnan(upper)&&!isinf(upper)&&!isnan(lower)&&!isinf(lower)&&
            lower>=0.0 && upper<=1.0 && lower<=selector && selector<upper) {
            probability=upper-lower;
            return index;
        }
        lo=0;hi=size-1;
    }
    return -1; // Malformed data fails the progressive finite-value publication gate.
}
float importanceUvPdf(float probability,vec2 uv) {
    float side=float(environmentData.config.x);
    // dOmega/du/dv=(1+u*u+v*v)^(-3/2), texel UV area=4/side^2.
    return probability*(side*side*0.25)*pow(1.0+dot(uv,uv),1.5);
}
float importancePdf(vec3 direction) {
    int face;vec2 uv;ivec2 cell;
    importanceCoordinates(direction,face,uv,cell);
    return importanceUvPdf(importanceLeafProbability(face,cell),uv);
}
float importanceNextUp(float value) {
    if(value==0.0)return uintBitsToFloat(1u);
    uint bits=floatBitsToUint(value);
    return uintBitsToFloat(value>0.0?bits+1u:bits-1u);
}
float importanceInterior(float lower,float upper,float jitter) {
    float lo=importanceNextUp(lower),hi=-importanceNextUp(-upper);
    return clamp(lower+(upper-lower)*jitter,lo,hi);
}
bool importanceSample(uint key,uint stratum,uint strata,out vec3 direction,out float sampledPdf) {
    // Primary proposals cover all eight CDF strata once; secondary uses one.
    // The PDF remains the aggregate proposal, as required by balance MIS.
    float jitterSelector=indirectOpen(indirectHash(key^0xe3a916d5u));
    float selector=(float(stratum)+jitterSelector)/float(strata);
    // Rounding must not move the last value of one stratum into the next.
    float stratumLo=float(stratum)/float(strata),stratumHi=float(stratum+1u)/float(strata);
    selector=clamp(selector,importanceNextUp(stratumLo),-importanceNextUp(-stratumHi));
    float probability;
    int index=importanceInverse(selector,probability);
    if(index<0)return false;
    int n=int(environmentData.config.x),faceSize=n*n;
    int face=index/faceSize,local=index-face*faceSize;
    ivec2 cell=ivec2(local%n,local/n);
    float side=float(environmentData.config.x);
    vec2 lower=vec2(cell)*(2.0/side)-vec2(1.0),upper=vec2(cell+ivec2(1))*(2.0/side)-vec2(1.0);
    vec2 jitter=vec2(indirectOpen(indirectHash(key^0x5c2b7a91u)),indirectOpen(indirectHash(key^0xb713e4cdu)));
    vec2 uv=vec2(importanceInterior(lower.x,upper.x,jitter.x),importanceInterior(lower.y,upper.y,jitter.y));
    direction=normalize(importanceFaceVector(face,uv)).xzy;
    int actualFace;vec2 actualUV;ivec2 actualCell;
    importanceCoordinates(direction,actualFace,actualUV,actualCell);
    if(actualFace!=face || any(notEqual(actualCell,cell))) {
        // Only unrepresentable edge samples: normalization/UV inversion can
        // round across a leaf by one ULP. Keep the selected leaf, never wrap it.
        uv=(vec2(cell)+vec2(0.5))*(2.0/side)-vec2(1.0);
        direction=normalize(importanceFaceVector(face,uv)).xzy;
        importanceCoordinates(direction,actualFace,actualUV,actualCell);
    }
    // Reuse the selected adjacent-CDF difference; no second inverse search.
    sampledPdf=importanceUvPdf(probability,actualUV);
    return true;
}
float importanceWeightPdf(vec3 N,vec3 direction,float environmentPdf) {
    float cosinePdf=max(dot(N,direction),0.0)/PI;
    if(cosinePdf==0.0)return 0.0;
    if(isnan(environmentPdf)||isinf(environmentPdf)||environmentPdf<0.0)return importanceInvalid();
    // Balance MIS: 8+8 first-hop proposals, or one equiprobable mixture draw at
    // the second hop. Real surface hits and environment misses share this PDF.
    return cosinePdf/(0.5*(cosinePdf+environmentPdf));
}
float importanceWeight(vec3 N,vec3 direction) {
    return importanceWeightPdf(N,direction,importancePdf(direction));
}


vec3 progressiveMiss(vec3 direction) {
    if(residentEnvironmentEnabled())return textureLod(environmentCube,direction.xzy,0.0).rgb;
    return studio.environment.w!=0.0?studio.environment.rgb:vec3(0.0);
}
vec3 progressiveFirstHit(vec3 P,vec3 faceN,vec3 direction,uint key) {
    vec3 radiance;BounceSurface hit;
    if(!residentTraceSurface(P,faceN,direction,false,radiance,hit))return progressiveMiss(direction);
    if(!hit.valid)return vec3(0.0);
    vec3 incoming=vec3(0.0);
    vec3 rho=(1.0-hit.metal)*hit.albedo*(1.0-mix(vec3(.04),hit.albedo,hit.metal));
    if(any(greaterThan(rho,vec3(0.0)))) {
        vec3 axis=abs(hit.N.z)<.9?vec3(0,0,1):vec3(0,1,0);
        vec3 T=normalize(cross(axis,hit.N));
        vec2 q=vec2(indirectOpen(indirectHash(key^0x94d049bbu)),indirectOpen(indirectHash(key^0x369dea0fu)));
        float r=sqrt(q.x),phi=2.0*PI*q.y;
        vec3 nextDirection=mat3(T,cross(hit.N,T),hit.N)*vec3(r*cos(phi),r*sin(phi),sqrt(1.0-q.x));
        bool environmentMIS=importanceActive();
        bool fromEnvironment=environmentMIS && indirectOpen(indirectHash(key^0x6f4d2c83u))>=0.5;
        float sampledPdf=0.0;
        if(fromEnvironment) {
            if(!importanceSample(indirectHash(key^0xd28a37f1u),0u,1u,nextDirection,sampledPdf))return vec3(importanceInvalid());
        }
        if(dot(nextDirection,hit.Ng)>0.0 && (!environmentMIS || dot(nextDirection,hit.N)>0.0)) {
            if(!residentTraceTerminal(hit.P,hit.Ng,nextDirection,incoming))incoming=progressiveMiss(nextDirection);
            if(environmentMIS)incoming*=fromEnvironment?
                importanceWeightPdf(hit.N,nextDirection,sampledPdf):importanceWeight(hit.N,nextDirection);
        }
    }
    radiance=reflectionDirect(hit.P,hit.N,hit.Ng,hit.V,hit.albedo,hit.metal,hit.roughness);
    // Replace this first hit's diffuse IBL with one continuation, retaining its
    // local direct lighting and specular environment. Terminal hit never recurses.
    radiance+=residentStudioRadianceSecondary(hit.P,hit.N,hit.Ng,hit.V,hit.albedo,hit.metal,hit.roughness
#ifdef PR_RESIDENT_REFLECTION
        ,false,vec3(0.0)
#endif
        ,true,incoming);
    return radiance;
}
#endif
vec3 residentDiffuseIncoming(vec3 P,vec3 N,vec3 faceN) {
    if(PR_INDIRECT_ABI!=1u)return vec3(0.0);
    uvec2 pixel=uvec2(gl_FragCoord.xy);
    uint base=indirectHash(u.counts.y^indirectHash(pixel.x+0x9e3779b9u)^indirectHash(pixel.y+0x85ebca6bu));
#ifdef PR_RESIDENT_PROGRESSIVE
    uint batch=uint(u.options.w)>>1u;
    if(batch!=0u)base=indirectHash(base^indirectHash(batch^0xa511e9b3u));
#endif
    vec3 axis=abs(N.z)<.9?vec3(0,0,1):vec3(0,1,0);
    vec3 T=normalize(cross(axis,N));mat3 basis=mat3(T,cross(N,T),N);
    vec3 sum=vec3(0.0);
#ifdef PR_RESIDENT_PROGRESSIVE
    bool environmentMIS=importanceActive();
#endif
    for(uint i=0u;i<16u;++i) {
        uint key=indirectHash(base^(i*0x9e3779b9u));
        vec2 jitter=vec2(indirectOpen(indirectHash(key^0x68bc21ebu)),indirectOpen(indirectHash(key^0x02e5be93u)));
        vec2 q=(vec2(i&3u,i>>2u)+jitter)*.25;
#ifdef PR_RESIDENT_PROGRESSIVE
        if(environmentMIS)q=(vec2(i&3u,(i&7u)>>2u)+jitter)*vec2(0.25,0.5);
#endif
        float r=sqrt(q.x),phi=2.0*PI*q.y;
        vec3 direction=basis*vec3(r*cos(phi),r*sin(phi),sqrt(max(0.0,1.0-q.x)));
#ifdef PR_RESIDENT_PROGRESSIVE
        float sampledPdf=0.0;
        bool fromEnvironment=environmentMIS && i>=8u;
        if(fromEnvironment && !importanceSample(key,i-8u,8u,direction,sampledPdf))return vec3(importanceInvalid());
        if(environmentMIS && dot(direction,N)<=0.0)continue;
        float misWeight=environmentMIS?(fromEnvironment?
            importanceWeightPdf(N,direction,sampledPdf):importanceWeight(N,direction)):1.0;
#endif
        // Do not renormalize surviving samples: clipping means zero contribution
        // from directions below the actual triangle's geometric hemisphere.
        if(dot(direction,faceN)<=0.0)continue;
#ifdef PR_RESIDENT_PROGRESSIVE
        if((uint(u.options.w)&1u)!=0u) {
            vec3 value=progressiveFirstHit(P,faceN,direction,key);
            sum+=environmentMIS?value*misWeight:value;continue;
        }
#endif
        vec3 incoming;
        if(!residentTraceTerminal(P,faceN,direction,incoming)) {
            incoming=studio.environment.w!=0.0?studio.environment.rgb:vec3(0.0);
            if(residentEnvironmentEnabled())incoming=textureLod(environmentCube,direction.xzy,0.0).rgb;
        }
#ifdef PR_RESIDENT_PROGRESSIVE
        sum+=environmentMIS?incoming*misWeight:incoming;
#else
        sum+=incoming;
#endif
    }
    return sum*(1.0/16.0);
}
