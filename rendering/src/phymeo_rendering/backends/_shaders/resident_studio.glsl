// Finite one-sided parallelograms. Geometry, BRDF and radiance are in world
// metres and linear RGB. Direct lighting uses deterministic area and visible
// GGX normal samples with MIS. All shadow rays end at the sampled emitter.
// The uniform environment is distant, unoccluded fill; indirect bounces and
// interreflection are provided by the separate path-tracing reference.
const int studioAreaSamples = 4;
const int studioBrdfSamples = 8;

float studioG1(float cosine, float alpha2) {
    return 2.0 * cosine / (cosine + sqrt(alpha2 + (1.0-alpha2)*cosine*cosine));
}

float studioD(float nh, float a2) {
    // This form avoids losing the small alpha term at the mirror direction.
    float d = max(0.0, 1.0-nh*nh) + a2*nh*nh;
    return a2 / (PI*d*d);
}

float studioMIS(float sampled, float other) {
    if (other <= 0.0) return 1.0;
    float ratio = other / sampled;
    return 1.0 / (1.0 + ratio*ratio);
}

// Sample the distribution of visible microfacet normals (Heitz, JCGT 2018).
// These fixed stratified directions are identical on replay and after reset.
vec3 studioVisibleNormal(vec3 eye, float alpha, int i) {
    vec3 ve = normalize(vec3(alpha*eye.xy, eye.z));
    float xy = dot(ve.xy, ve.xy);
    vec3 t = xy > 1e-12 ? vec3(-ve.y,ve.x,0.0)*inversesqrt(xy) : vec3(1,0,0);
    vec3 b = cross(ve,t);
    float r = sqrt((float(i)+0.5)/float(studioBrdfSamples));
    float angle = 2.0*PI*(0.0625 + float(bitfieldReverse(uint(i))) * 2.3283064365386963e-10);
    vec2 p = r*vec2(cos(angle),sin(angle));
    p.y = mix(sqrt(max(0.0,1.0-p.x*p.x)), p.y, 0.5*(1.0+ve.z));
    vec3 h = t*p.x + b*p.y + ve*sqrt(max(0.0,1.0-dot(p,p)));
    return normalize(vec3(alpha*h.xy, max(0.0,h.z)));
}

vec3 residentStudioRadiance(vec3 P, vec3 N, vec3 V, vec3 albedo,
                            float metal, float roughness) {
    vec3 f0 = mix(vec3(0.04), albedo, metal);
    float nv = max(dot(N,V),1e-5);
    float alpha = max(roughness*roughness, 0.045*0.045);
    float a2 = alpha*alpha;
    float gv = studioG1(nv,a2);
    vec3 diffuse = (1.0-metal)*albedo;
    vec3 diffuseEnvironment = studio.environment.rgb;
    vec3 specularEnvironment = studio.environment.rgb;
#ifdef PR_RESIDENT_DIRECTIONAL_ENVIRONMENT
    if (residentEnvironmentEnabled()) {
        diffuseEnvironment = residentEnvironmentDiffuse(N);
        specularEnvironment = residentEnvironmentSpecular(reflect(-V,N),roughness);
    }
#endif
    vec3 result = diffuseEnvironment*diffuse*(1.0-f0);
    vec3 biasP = P + N*u.options.y;
    vec3 axis = abs(N.z) < 0.9 ? vec3(0,0,1) : vec3(0,1,0);
    vec3 T = normalize(cross(axis,N));
    mat3 basis = mat3(T,cross(N,T),N);
    vec3 localEye = transpose(basis)*V;

    for (int j=0; j<2; ++j) {
        StudioPanel panel = studio.panels[j];
        if (all(lessThanEqual(panel.radiance.rgb,vec3(0.0)))) continue;
        vec3 U = panel.halfU.xyz, W = panel.halfV.xyz;
        vec3 crossUW = cross(U,W);
        float area = 4.0*length(crossUW);
        vec3 emitterNormal = normalize(crossUW);
        if (dot(emitterNormal,biasP-panel.position.xyz) <= 0.0) continue;
        for (int i=0; i<studioAreaSamples; ++i) {
            vec2 ab = vec2(i&1,i>>1)-vec2(0.5);
            vec3 delta = panel.position.xyz + ab.x*U + ab.y*W - biasP;
            float d2 = dot(delta,delta), dist = sqrt(d2);
            vec3 L = delta/dist;
            float nl = dot(N,L), emit = -dot(emitterNormal,L);
            if (nl<=0.0 || emit<=0.0 || occluded(biasP,L,dist)) continue;
            float pdfArea = d2/(area*emit);
            vec3 H = normalize(V+L);
            vec3 F = f0+(1.0-f0)*pow(1.0-clamp(dot(V,H),0.0,1.0),5.0);
            float pdfGGX = studioD(clamp(dot(N,H),0.0,1.0),a2)*gv/(4.0*nv);
            float w = studioMIS(float(studioAreaSamples)*pdfArea,float(studioBrdfSamples)*pdfGGX);
            vec3 brdfCos = (1.0-F)*diffuse*(nl/PI) + F*(w*pdfGGX*studioG1(nl,a2));
            result += panel.radiance.rgb*brdfCos/(float(studioAreaSamples)*pdfArea);
        }
    }
    for (int i=0; i<studioBrdfSamples; ++i) {
        vec3 H = basis*studioVisibleNormal(localEye,alpha,i);
        vec3 L = reflect(-V,H);
        float nl = dot(N,L);
        if (nl<=0.0) continue;
        vec3 F = f0+(1.0-f0)*pow(1.0-clamp(dot(V,H),0.0,1.0),5.0);
        vec3 response = F*(studioG1(nl,a2)/float(studioBrdfSamples));
        result += response*specularEnvironment;
        float pdfGGX = studioD(clamp(dot(N,H),0.0,1.0),a2)*gv/(4.0*nv);
        for (int j=0; j<2; ++j) {
            StudioPanel panel = studio.panels[j];
            if (all(lessThanEqual(panel.radiance.rgb,vec3(0.0)))) continue;
            vec3 U = panel.halfU.xyz, W = panel.halfV.xyz;
            vec3 crossUW = cross(U,W);
            float size = length(crossUW);
            vec3 emitterNormal = crossUW/size;
            float emit = -dot(emitterNormal,L);
            if (emit<=0.0) continue;
            float dist = dot(biasP-panel.position.xyz,emitterNormal)/emit;
            if (dist<=u.options.y) continue;
            vec3 offset = biasP+dist*L-panel.position.xyz;
            if (abs(dot(offset,U))>dot(U,U) || abs(dot(offset,W))>dot(W,W)) continue;
            if (occluded(biasP,L,dist)) continue;
            float pdfArea = dist*dist/(4.0*size*emit);
            float w = studioMIS(float(studioBrdfSamples)*pdfGGX,float(studioAreaSamples)*pdfArea);
            result += response*panel.radiance.rgb*w;
        }
    }
    return result;
}
