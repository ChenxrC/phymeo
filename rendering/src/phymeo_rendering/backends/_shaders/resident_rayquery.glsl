layout(set=0,binding=13) uniform accelerationStructureEXT sceneAcceleration;
bool occluded(vec3 origin,vec3 direction,float tMax) {
    if (u.counts.w==0u || tMax<=u.options.y) return false;
    rayQueryEXT query;
    // Force candidates instead of automatic opaque acceptance so the existing
    // OPEN interval (bias,tMax) stays explicit, including finite panel segments.
    rayQueryInitializeEXT(query,sceneAcceleration,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsNoOpaqueEXT,
        0xffu,origin,u.options.y,direction,tMax);
    while (rayQueryProceedEXT(query)) {
        if (rayQueryGetIntersectionTypeEXT(query,false)==gl_RayQueryCandidateIntersectionTriangleEXT) {
            float t=rayQueryGetIntersectionTEXT(query,false);
            if (t>u.options.y && t<tMax) rayQueryConfirmIntersectionEXT(query);
        }
    }
    return rayQueryGetIntersectionTypeEXT(query,true)!=gl_RayQueryCommittedIntersectionNoneEXT;
}
