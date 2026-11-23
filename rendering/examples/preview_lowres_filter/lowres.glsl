// One fixed low-grid pass. Original raster MRT remains immutable.
layout(constant_id=100) const uint PR_PREVIEW_LOWRES_FILTER_ABI=1u;
layout(set=0,binding=17,std430) buffer LowFiltered {vec4 values[];} lowFiltered;

bool lowresSupport(ivec2 q,ivec2 size,uint id,float z,vec3 n) {
    if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,size)))return false;
    ivec2 anchor=q*4+ivec2(2);
    if(any(greaterThanEqual(anchor,ivec2(p.width,p.height))))return false;
    if(texelFetch(guideCoverage,q,0).r==0u||texelFetch(guideIds,q,0).r!=id||
       texelFetch(coverage,anchor,0).r==0u||texelFetch(instanceIds,anchor,0).r!=id)return false;
    vec4 own=texelFetch(incomingGuide,q,0),full=texelFetch(fullAttributes,anchor,0);
    if(!finiteRadiance(own.rgb)||!finiteCode(own.a,16777215.0)||!finiteCode(full.a,1048575.0))return false;
    if((uint(own.a)&(1u<<20u))==0u)return false;
    float lowZ=texelFetch(guideDepth,q,0).r,fullZ=texelFetch(forwardDepth,anchor,0).r;
    vec3 lowN=decodeNormal(uint(own.a));
    return depthMatches(lowZ,fullZ)&&depthMatches(z,lowZ)&&
           dot(lowN,decodeNormal(uint(full.a)))>=.98&&dot(n,lowN)>=.98;
}
void lowresPass() {
    ivec2 q=ivec2(gl_GlobalInvocationID.xy),size=(ivec2(p.width,p.height)+ivec2(3))/4;
    if(any(greaterThanEqual(q,size)))return;
    uint offset=uint(q.y*size.x+q.x);
    vec4 center=texelFetch(incomingGuide,q,0);
    lowFiltered.values[offset]=center; // alpha and unsupported center are exact source
    uint id=texelFetch(guideIds,q,0).r;
    if(id==0u||texelFetch(guideCoverage,q,0).r==0u||!finiteCode(center.a,16777215.0))return;
    float z=texelFetch(guideDepth,q,0).r;vec3 n=decodeNormal(uint(center.a));
    if(!lowresSupport(q,size,id,z,n))return;
    // Evaluate each of the 25 local guide positions once, rather than
    // refetching overlapping corridors for every one of the nine weights.
    uint supportMask=1u<<12u;
    for(int y=-2;y<=2;++y)for(int x=-2;x<=2;++x) {
        if(x==0&&y==0)continue;
        if(lowresSupport(q+ivec2(x,y),size,id,z,n))supportMask|=1u<<uint((y+2)*5+x+2);
    }
    vec3 sum=vec3(0.0);float weightSum=0.0;
    for(int ky=-1;ky<=1;++ky)for(int kx=-1;kx<=1;++kx) {
        ivec2 delta=ivec2(kx,ky)*2,neighbor=q+delta;
        bool supported=true;
        // Tiny conservative supercover rectangle: no stride jumps across
        // missing support, other IDs, depth seams or inconsistent normals.
        for(int y=min(0,delta.y);y<=max(0,delta.y)&&supported;++y)
            for(int x=min(0,delta.x);x<=max(0,delta.x);++x)
                if((supportMask&(1u<<uint((y+2)*5+x+2)))==0u){supported=false;break;}
        if(!supported)continue;
        float weight=float((kx==0?2:1)*(ky==0?2:1))*(1.0/16.0);
        sum+=texelFetch(incomingGuide,neighbor,0).rgb*weight;weightSum+=weight;
    }
    if(weightSum>0.0)lowFiltered.values[offset]=vec4(sum/weightSum,center.a);
}