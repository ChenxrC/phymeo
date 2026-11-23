// Linear glTF +Y normal data. No geometric displacement, discard or depth write.
layout(constant_id=62) const uint PR_NORMAL_MAPS_ABI=1u;
layout(set=0,binding=14) uniform sampler2DArray normalMaps;
layout(set=0,binding=15,std430) readonly buffer NormalMetadata { uint words[]; } normalData;

bool normalFinite(vec3 value) { return !any(isnan(value)) && !any(isinf(value)); }
vec3 residentShadingNormal(vec3 geometricNormal) {
    if(PR_NORMAL_MAPS_ABI!=1u || u.counts.z==0u) return geometricNormal;
    uint slot=normalData.words[4u+ids.y];
    if(slot==0xffffffffu) return geometricNormal;
    // The flat slot is uniform within each primitive; evaluate derivatives
    // before per-pixel degeneracy branches. No implicit image LOD derivatives.
    vec3 px=dFdx(worldPosition),py=dFdy(worldPosition);
    vec2 ux=dFdx(meshUV),uy=dFdy(meshUV);
    float determinant=ux.x*uy.y-ux.y*uy.x;
    float scale=abs(ux.x*uy.y)+abs(ux.y*uy.x);
    if(isnan(determinant) || isinf(determinant) || abs(determinant)<=max(1e-30,3.814697265625e-6*scale)) return geometricNormal;
    vec3 tangent=(px*uy.y-py*ux.y)*sign(determinant);
    vec3 bitangent=(-px*uy.x+py*ux.x)*sign(determinant);
    if(!normalFinite(tangent) || !normalFinite(bitangent)) return geometricNormal;
    float maximum=max(abs(tangent.x),max(abs(tangent.y),abs(tangent.z)));
    float bmaximum=max(abs(bitangent.x),max(abs(bitangent.y),abs(bitangent.z)));
    if(maximum<=1e-30 || bmaximum<=1e-30) return geometricNormal;
    tangent/=maximum; bitangent/=bmaximum;
    tangent-=geometricNormal*dot(geometricNormal,tangent);
    float length2=dot(tangent,tangent);
    if(length2<=1e-20) return geometricNormal;
    tangent*=inversesqrt(length2);
    float handedness=dot(cross(geometricNormal,tangent),bitangent);
    if(abs(handedness)<=1e-20) return geometricNormal;
    bitangent=sign(handedness)*cross(geometricNormal,tangent);
    uint dimensions=4u+u.counts.w+2u*slot;
    vec2 size=vec2(normalData.words[dimensions],normalData.words[dimensions+1u]);
    vec2 coordinate=(fract(meshUV)*size+vec2(1.0))/vec2(textureSize(normalMaps,0).xy);
    vec3 sampled=2.0*textureLod(normalMaps,vec3(coordinate,float(slot)),0.0).rgb-vec3(1.0);
    float sampled2=dot(sampled,sampled);
    if(sampled2<=1e-20) return geometricNormal;
    sampled*=inversesqrt(sampled2);
    vec3 result=tangent*sampled.x+bitangent*sampled.y+geometricNormal*sampled.z;
    float result2=dot(result,result);
    return normalFinite(result) && result2>1e-20 ? result*inversesqrt(result2) : geometricNormal;
}
