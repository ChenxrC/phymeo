// The first dispatch shades every pixel and publishes its immutable history
// record. The second dispatch only replaces pixels with mixed primary coverage.
layout(constant_id=104) const uint PR_PREVIEW_CONTOUR_AA_ABI=1u;
layout(set=0,binding=18) uniform accelerationStructureEXT contourScene;
struct ContourInstance { uvec4 binding; vec4 colorRoughness; vec4 material; };
layout(set=0,binding=19,std430) readonly buffer ContourInstances { ContourInstance slots[]; } contourInstances;

// Vulkan's standard rotated four-sample positions in top-left pixel space.
const vec2 contourOffsets[4]=vec2[](vec2(3,1),vec2(7,3),vec2(1,5),vec2(5,7));

vec3 historyColor(ivec2 pixel) {
    HistoryPixel sampleValue=nextHistory.pixels[uint(pixel.y)*p.width+uint(pixel.x)];
    vec3 radiance=sampleValue.lighting.rgb;
    if(sampleValue.geometry.z!=0u) {
        uint packedRho=uint(sampleValue.lighting.a);
        vec3 rho=vec3(packedRho&255u,(packedRho>>8u)&255u,(packedRho>>16u)&255u)/255.0;
        radiance+=rho*sampleValue.incomingAge.rgb;
    }
    return radiance;
}

uint nearestSurface(vec2 position,out float forwardZ) {
    vec3 origin=transpose(mat3(h.view))*(-h.view[3].xyz);
    vec2 viewXY=(position-h.intrinsics.zw)/h.intrinsics.xy;
    // Leave the direction unnormalized. Parameter t is forward depth in
    // metres, so both ends of the ray use the raster camera's clip planes.
    vec3 direction=transpose(mat3(h.view))*vec3(viewXY.x,-viewXY.y,-1.0);
    rayQueryEXT query;
    rayQueryInitializeEXT(query,contourScene,gl_RayFlagsOpaqueEXT,0xff,
                          origin,h.clipping.z,direction,h.clipping.w);
    while(rayQueryProceedEXT(query)) {}
    forwardZ=0.0;
    if(rayQueryGetIntersectionTypeEXT(query,true)==gl_RayQueryCommittedIntersectionNoneEXT)return 0u;
    forwardZ=rayQueryGetIntersectionTEXT(query,true);
    uint slot=uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query,true));
    return contourInstances.slots[slot].binding.y;
}

bool hasContour(ivec2 pixel,ivec2 extent,uint centerId) {
    for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x) {
        ivec2 neighbor=pixel+ivec2(x,y);
        if(any(lessThan(neighbor,ivec2(0)))||any(greaterThanEqual(neighbor,extent)))continue;
        if(texelFetch(instanceIds,neighbor,0).r!=centerId)return true;
    }
    return false;
}

bool donorPixel(ivec2 pixel,ivec2 extent,vec2 position,uint id,float z,out ivec2 donor) {
    float best=16.0;
    for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x) {
        ivec2 q=pixel+ivec2(x,y);
        if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,extent)))continue;
        if(texelFetch(instanceIds,q,0).r!=id)continue;
        if(id!=0u) {
            float sampleZ=texelFetch(forwardDepth,q,0).r;
            if(abs(z-sampleZ)>.03*min(z,sampleZ))continue;
        }
        vec2 delta=(vec2(q)+.5)-position;
        float distance2=dot(delta,delta);
        if(distance2<best){best=distance2;donor=q;}
    }
    return best<16.0;
}

void resolveContour(ivec2 pixel,ivec2 extent) {
    uint centerId=texelFetch(instanceIds,pixel,0).r;
    if(!hasContour(pixel,extent,centerId))return;
    atomicAdd(status.contourCandidates,1u);
    vec3 center=historyColor(pixel),sum=vec3(0.0);
    bool mixed=false;
    for(int i=0;i<4;++i) {
        vec2 position=vec2(pixel)+contourOffsets[i]*.125;
        float z;
        uint id=nearestSurface(position,z);
        if(id==centerId){sum+=center;continue;}
        ivec2 donor;
        if(!donorPixel(pixel,extent,position,id,z,donor)) {
            // No measured shading for this subpixel surface. Publish the
            // existing center unchanged for the whole pixel.
            atomicAdd(status.contourUnsupported,1u);
            atomicAdd(status.contourRays,uint(i+1));
            return;
        }
        mixed=true;
        sum+=historyColor(donor);
    }
    atomicAdd(status.contourRays,4u);
    if(!mixed)return;
    atomicAdd(status.contourResolved,1u);
    publishColor(pixel,sum*.25);
}
