// The copied surface-quality package supplies an opaque TLAS and rigid model
// transforms. Directions are attached to the link, independent of the camera,
// screen pixel, frame seed and history. The nearest blocker determines the
// falloff, so traversal must not terminate at the first candidate.
float stableEnvironmentVisibility(vec3 p, vec3 n, vec3 faceN, mat3 world) {
    float radius = float(PR_PREVIEW_AO_RADIUS_UM) * 1.0e-6;
    float bias = u.options.y;
    if (radius <= bias || u.counts.w == 0u) return 1.0;
    // Ten antipodal pairs of regular dodecahedron vertices. For each pair,
    // only the direction in the shading hemisphere can have positive weight.
    const float a = 0.61803398875, b = 1.61803398875;
    const vec3 axes[10] = vec3[10](
        vec3(1,1,1), vec3(1,1,-1), vec3(1,-1,1), vec3(1,-1,-1),
        vec3(0,a,b), vec3(0,a,-b), vec3(a,b,0), vec3(a,-b,0),
        vec3(b,0,a), vec3(b,0,-a)
    );
    vec3 origin = p + faceN * bias;
    float sum = 0.0, weightSum = 0.0;
    for (int i = 0; i < 10; ++i) {
        vec3 d = normalize(world * axes[i]);
        float cosine = dot(n, d);
        if (cosine < 0.0) d = -d;
        // Fade out directions near the triangle's horizon. Normalizing the
        // accepted weights gives exactly unit visibility on an open surface.
        float weight = min(abs(cosine), dot(faceN, d));
        if (weight <= 0.0) continue;
        weightSum += weight;
        rayQueryEXT ray;
        rayQueryInitializeEXT(ray, sceneAcceleration, gl_RayFlagsOpaqueEXT,
                              0xffu, origin, bias, d, radius);
        while (rayQueryProceedEXT(ray)) {}
        float visibility = 1.0;
        if (rayQueryGetIntersectionTypeEXT(ray, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
            float t = rayQueryGetIntersectionTEXT(ray, true);
            visibility = smoothstep(bias, radius, t);
        }
        sum += visibility * weight;
    }
    return weightSum > 0.0 ? sum / weightSum : 1.0;
}
