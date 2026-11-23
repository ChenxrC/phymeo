// One closest triangle, then terminal direct/environment shading; no recursion.
// Query semantics: KhronosGroup/GLSL extensions/ext/GLSL_EXT_ray_query.txt.
#ifdef PR_RESIDENT_REFLECTION
layout(constant_id=64) const uint PR_REFLECTION_ABI=1u;
#endif

vec3 reflectionDirect(vec3 P,vec3 N,vec3 Ng,vec3 V,vec3 albedo,float metallic,float roughness) {
    vec3 color=vec3(0.0);
    float nv=max(dot(N,V),1e-5);
    float alpha=max(roughness*roughness,0.045*0.045),a2=alpha*alpha;
    vec3 f0=mix(vec3(.04),albedo,metallic);
    for(uint i=0u;i<u.counts.x;++i) {
        vec3 L=normalize(-u.lightDirections[i].xyz);
        float nl=max(dot(N,L),0.0);
        if(nl<=0.0 || occluded(P+Ng*u.options.y,L,3.402823466e38)) continue;
        vec3 H=normalize(V+L);
        float nh=max(dot(N,H),0.0),vh=max(dot(V,H),0.0);
        float denominator=nh*nh*(a2-1.0)+1.0;
        float D=a2/(PI*denominator*denominator);
        float Gv=2.0*nv/(nv+sqrt(a2+(1.0-a2)*nv*nv));
        float Gl=2.0*nl/(nl+sqrt(a2+(1.0-a2)*nl*nl));
        vec3 F=f0+(1.0-f0)*pow(1.0-vh,5.0);
        vec3 specular=D*Gv*Gl*F/max(4.0*nv*nl,1e-8);
        vec3 diffuse=(1.0-F)*(1.0-metallic)*albedo/PI;
        color+=(diffuse+specular)*u.lightColors[i].rgb*nl;
    }
    return color;
}
vec2 reflectionUV(uint index,bool normals) {
    uint address;
#ifdef PR_RESIDENT_NORMAL_MAPS
    if(normals) {
        address=normalData.words[3]+2u*index;
        return uintBitsToFloat(uvec2(normalData.words[address],normalData.words[address+1u]));
    }
#endif
    address=textureData.words[3]+2u*index;
    return uintBitsToFloat(uvec2(textureData.words[address],textureData.words[address+1u]));
}
#ifdef PR_RESIDENT_NORMAL_MAPS
vec3 reflectionHitNormal(vec3 N,uint slot,vec2 uv,vec3 e1,vec3 e2,vec2 t1,vec2 t2) {
    float determinant=t1.x*t2.y-t1.y*t2.x;
    float scale=abs(t1.x*t2.y)+abs(t1.y*t2.x);
    if(abs(determinant)<=max(1e-30,3.814697265625e-6*scale)) return N;
    vec3 tangent=(e1*t2.y-e2*t1.y)*sign(determinant);
    vec3 bitangent=(-e1*t2.x+e2*t1.x)*sign(determinant);
    if(!normalFinite(tangent)||!normalFinite(bitangent)) return N;
    float maximum=max(abs(tangent.x),max(abs(tangent.y),abs(tangent.z)));
    float bmaximum=max(abs(bitangent.x),max(abs(bitangent.y),abs(bitangent.z)));
    if(maximum<=1e-30||bmaximum<=1e-30) return N;
    tangent/=maximum;bitangent/=bmaximum;tangent-=N*dot(N,tangent);
    float length2=dot(tangent,tangent);if(length2<=1e-20) return N;
    tangent*=inversesqrt(length2);
    float handedness=dot(cross(N,tangent),bitangent);if(abs(handedness)<=1e-20) return N;
    bitangent=sign(handedness)*cross(N,tangent);
    uint dimensions=4u+u.counts.w+2u*slot;
    vec2 size=vec2(normalData.words[dimensions],normalData.words[dimensions+1u]);
    vec2 coordinate=(fract(uv)*size+vec2(1.0))/vec2(textureSize(normalMaps,0).xy);
    vec3 sampled=2.0*textureLod(normalMaps,vec3(coordinate,float(slot)),0.0).rgb-vec3(1.0);
    float sampled2=dot(sampled,sampled);if(sampled2<=1e-20) return N;
    sampled*=inversesqrt(sampled2);
    vec3 result=tangent*sampled.x+bitangent*sampled.y+N*sampled.z;
    float result2=dot(result,result);
    return normalFinite(result)&&result2>1e-20?result*inversesqrt(result2):N;
}
#endif
bool residentTraceTerminal(vec3 P,vec3 Ng,vec3 direction,out vec3 radiance) {
    radiance=vec3(0.0);
    if(u.counts.w==0u) return false;
    rayQueryEXT query;
    rayQueryInitializeEXT(query,sceneAcceleration,gl_RayFlagsNoOpaqueEXT,0xffu,
        P+Ng*u.options.y,u.options.y,direction,3.402823466e38);
    while(rayQueryProceedEXT(query)) {
        if(rayQueryGetIntersectionTypeEXT(query,false)==gl_RayQueryCandidateIntersectionTriangleEXT &&
            rayQueryGetIntersectionTEXT(query,false)>u.options.y) rayQueryConfirmIntersectionEXT(query);
    }
    if(rayQueryGetIntersectionTypeEXT(query,true)!=gl_RayQueryCommittedIntersectionTriangleEXT) return false;
    uint slot=uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query,true));
    uint primitive=uint(rayQueryGetIntersectionPrimitiveIndexEXT(query,true));
    Instance instance=scene.instances[slot]; Mesh mesh=assets.meshes[instance.binding.x];
    // The BLAS consumes exactly this same reordered index buffer, one geometry
    // per mesh. PrimitiveIndex therefore indexes mesh.range.z, not source order.
    uint first=mesh.range.z+3u*primitive;
    uvec3 vi=mesh.range.x+uvec3(topology.indices[first],topology.indices[first+1u],topology.indices[first+2u]);
    vec2 b=rayQueryGetIntersectionBarycentricsEXT(query,true);
    vec3 weights=vec3(1.0-b.x-b.y,b.x,b.y);
    vec3 p0=geometry.vertices[vi.x].position.xyz,p1=geometry.vertices[vi.y].position.xyz,p2=geometry.vertices[vi.z].position.xyz;
    mat4 world=motion.transforms[slot].world;
    vec3 hitP=(world*vec4(weights.x*p0+weights.y*p1+weights.z*p2,1.0)).xyz;
    vec3 hitNg=transpose(mat3(motion.transforms[slot].local))*(weights.x*geometry.vertices[vi.x].normal.xyz+
        weights.y*geometry.vertices[vi.y].normal.xyz+weights.z*geometry.vertices[vi.z].normal.xyz);
    if(dot(hitNg,hitNg)<=1e-20) hitNg=mat3(world)*cross(p1-p0,p2-p0);
    vec3 faceN=cross(mat3(world)*(p1-p0),mat3(world)*(p2-p0));
    float faceScale=max(abs(faceN.x),max(abs(faceN.y),abs(faceN.z)));
    if(any(isnan(faceN))||any(isinf(faceN))||faceScale<=0.0) return true;
    faceN=normalize(faceN/faceScale);
    hitNg=normalize(hitNg);vec3 hitN=hitNg,hitV=-direction;
    if(dot(faceN,hitV)<0.0) faceN=-faceN;
    vec3 albedo=instance.colorRoughness.rgb;
    float roughness=instance.colorRoughness.w,metal=instance.material.x;
    uint colorSlot=textureData.words[4u+slot];
    if(colorSlot!=0xffffffffu) {
        vec2 uv=weights.x*reflectionUV(vi.x,false)+weights.y*reflectionUV(vi.y,false)+weights.z*reflectionUV(vi.z,false);
        uint dimensions=4u+u.counts.w+2u*colorSlot;
        vec2 size=vec2(textureData.words[dimensions],textureData.words[dimensions+1u]);
        vec2 coordinate=(fract(uv)*size+vec2(1.0))/vec2(textureSize(baseColorTextures,0).xy);
        vec4 texel=textureLod(baseColorTextures,vec3(coordinate,float(colorSlot)),0.0);
        albedo*=texel.rgb;if(PR_SURFACE_MAPS_ABI==1u)roughness*=texel.a;
    }
#ifdef PR_RESIDENT_NORMAL_MAPS
    uint normalSlot=normalData.words[4u+slot];
    if(normalSlot!=0xffffffffu) {
        vec2 t0=reflectionUV(vi.x,true),t1=reflectionUV(vi.y,true),t2=reflectionUV(vi.z,true);
        hitN=reflectionHitNormal(hitNg,normalSlot,weights.x*t0+weights.y*t1+weights.z*t2,
            mat3(world)*(p1-p0),mat3(world)*(p2-p0),t1-t0,t2-t0);
    }
#endif
    if(dot(hitNg,hitV)<0.0){hitNg=-hitNg;hitN=-hitN;}
    if(dot(hitN,hitV)>0.0) {
        radiance=reflectionDirect(hitP,hitN,faceN,hitV,albedo,metal,roughness);
        if(studio.environment.w!=0.0 || residentEnvironmentEnabled())
            radiance+=residentStudioRadiance(hitP,hitN,faceN,hitV,albedo,metal,roughness);
    }
    return true;
}

#ifdef PR_RESIDENT_REFLECTION
bool residentNearReflection(vec3 P,vec3 N,vec3 Ng,vec3 V,out vec3 radiance) {
    vec3 direction=normalize(reflect(-V,N));radiance=vec3(0.0);
    if(dot(direction,Ng)<=0.0||PR_REFLECTION_ABI!=1u)return false;
    return residentTraceTerminal(P,Ng,direction,radiance);
}
#endif