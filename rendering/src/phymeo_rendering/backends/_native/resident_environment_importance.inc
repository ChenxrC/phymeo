// CPU-only preparation of the PRIVATE environment staging image. Included by
// resident_environment.inc and by its standalone C++17 CPU regression. The
// includer supplies <algorithm>, <array>, <cmath>, <cstddef>, <cstdint>,
// <limits>, and <stdexcept>; this file introduces no Vulkan or file-I/O API.
//
// Layout stays level-major, six face-major RGBA32F images at each level. RGB
// (including the independently prefiltered higher mips) is never written.
// Private alpha layout (the public source still requires alpha=1):
//   L0 = global inclusive FP32 CDF, index=face*side*side+y*side+x;
//   L1 = inverse-directory indices at FP32 bucket thresholds j/(6*side*side/4);
//   higher levels = zero. No probability hierarchy remains.
// Masses retain footprint-averaged peak RGB * exact texel solid angle. The
// ACTUAL adjacent stored CDF difference is the proposal PMF; the shader uses
// it in both sampling and PDF evaluation, including bins rounded to zero.
namespace pr_environment_importance {

inline double rectangle_primitive(double u, double v) noexcept {
    return std::atan2(u*v, std::sqrt(1.0 + u*u + v*v));
}

inline double texel_solid_angle(uint32_t side, uint32_t x, uint32_t y) noexcept {
    const double step = 2.0 / static_cast<double>(side);
    const double u0 = step*x - 1.0, u1 = step*(x+1u) - 1.0;
    const double v0 = step*y - 1.0, v1 = step*(y+1u) - 1.0;
    return rectangle_primitive(u1,v1) - rectangle_primitive(u0,v1)
         - rectangle_primitive(u1,v0) + rectangle_primitive(u0,v0);
}

// Fetch an L0 texel centre. Cross-face taps use the same face axes as the
// Vulkan cube lookup. At a diagonal corner either tied face is an admissible
// approximation for the proposal; shading always samples the actual cube.
inline double peak_at(const float* rgba, int n, int face, int x, int y) noexcept {
    if (x<0 || x>=n || y<0 || y>=n) {
        const int s=2*x+1-n, t=2*y+1-n;
        const int axes[6][3]={{n,-t,-s},{-n,-t,s},{s,n,t},
                              {s,-n,-t},{s,-t,n},{-s,-t,-n}};
        const int dx=axes[face][0],dy=axes[face][1],dz=axes[face][2];
        int m=std::max({std::abs(dx),std::abs(dy),std::abs(dz)}),u,v;
        if (std::abs(dx)==m) {face=dx>0?0:1; u=dx>0?-dz:dz; v=-dy;}
        else if (std::abs(dy)==m) {face=dy>0?2:3; u=dx; v=dy>0?dz:-dz;}
        else {face=dz>0?4:5; u=dz>0?dx:-dx; v=-dy;}
        x=std::clamp((m+u)*n/(2*m),0,n-1);
        y=std::clamp((m+v)*n/(2*m),0,n-1);
    }
    const size_t i=((static_cast<size_t>(face)*n+y)*n+x)*4u;
    return std::max({rgba[i],rgba[i+1],rgba[i+2]});
}

inline bool populate(uint32_t side, uint32_t levels, float* rgba, size_t rgba_bytes) {
    static_assert(sizeof(float)==4, "environment staging requires IEEE-sized float32");
    // Match the existing native maximum; this does not narrow its public input
    // gate. Check layout before touching memory or doing side-dependent loops.
    if (!rgba || side<2u || side>1024u || (side&(side-1u))!=0u)
        throw std::invalid_argument("environment importance requires side 2..1024 power of two");
    std::array<size_t,11> offsets{};
    size_t expected_floats=0;
    uint32_t expected_levels=0;
    for (uint32_t dim=side; dim; dim>>=1u) {
        offsets[expected_levels++]=expected_floats;
        expected_floats+=static_cast<size_t>(dim)*dim*6u*4u;
    }
    if (levels!=expected_levels || rgba_bytes!=expected_floats*sizeof(float))
        throw std::invalid_argument("environment importance requires the complete RGBA32F mip chain");
    // The caller already enforces these public requirements. Keep this pure
    // helper independently safe and atomic: all validation precedes alpha writes.
    for (size_t i=0; i<expected_floats; ++i) {
        const float value=rgba[i];
        if (!std::isfinite(value) || (i%4u==3u ? value!=1.0f : value<0.0f || value>100.0f))
            throw std::invalid_argument("environment importance requires RGB in [0,100] and source alpha=1");
    }

    const size_t face_pixels=static_cast<size_t>(side)*side;
    double max_peak=0.0;
    bool constant=true;
    for (size_t i=0; i<face_pixels*6u*4u; i+=4u) {
        max_peak=std::max(max_peak,static_cast<double>(std::max({rgba[i],rgba[i+1],rgba[i+2]})));
        constant=constant && rgba[i]==rgba[0] && rgba[i+1]==rgba[1] && rgba[i+2]==rgba[2];
    }
    // No energy epsilon: even a single positive subnormal source texel is
    // normalized in double before storing mass in float. Truly black is exact.
    if (max_peak==0.0) {
        for (size_t i=3; i<expected_floats; i+=4u) rgba[i]=0.0f;
        return false;
    }
    double total=0.0;
    for (uint32_t y=0; y<side; ++y) for (uint32_t x=0; x<side; ++x) {
        const double omega=texel_solid_angle(side,x,y);
        for (uint32_t face=0; face<6u; ++face) {
            const size_t i=(face*face_pixels+static_cast<size_t>(y)*side+x)*4u;
            double peak=max_peak;
            if (!constant) {
                // Integrate the bilinear tent over a texel in UV. A hot texel
                // contributes to nine cells, including across face edges.
                // This changes only the sampling PDF; RGB remains untouched.
                peak=0.;
                for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx)
                    peak+=(dx==0?.75:.125)*(dy==0?.75:.125)*
                        peak_at(rgba,static_cast<int>(side),face,static_cast<int>(x)+dx,static_cast<int>(y)+dy);
            }
            rgba[i+3u]=static_cast<float>((peak/max_peak)*omega);
            total+=rgba[i+3u];
        }
    }
    const size_t leaf_count=face_pixels*6u;
    double cumulative=0.0;
    float previous=0.0f;
    for (size_t leaf=0; leaf<leaf_count; ++leaf) {
        const size_t alpha=4u*leaf+3u;
        cumulative+=static_cast<double>(rgba[alpha]);
        // Accumulate in double; only the final inclusive CDF is rounded.
        // The total and this prefix use different summation orders, so bound
        // the last-ULP discrepancy without ever decreasing a stored prefix.
        const float value=static_cast<float>(cumulative/total);
        rgba[alpha]=std::clamp(value,previous,1.0f);
        previous=rgba[alpha];
    }
    rgba[4u*(leaf_count-1u)+3u]=1.0f;

    const size_t bucket_count=leaf_count/4u;
    size_t leaf=0;
    for (size_t bucket=0; bucket<bucket_count; ++bucket) {
        // B=6*(side/2)^2 is not a power of two. Match shader FP32 division;
        // double division followed by a cast or multiplication by 1/B can
        // pick a different side of an exactly represented CDF boundary.
        const float threshold=static_cast<float>(bucket)/static_cast<float>(bucket_count);
        while (leaf+1u<leaf_count && rgba[4u*leaf+3u]<=threshold) ++leaf;
        // Maximum index 6*1024^2-1 is below 2^24: every index is exact in FP32.
        // Directory storage itself is the existing face-major L1 alpha plane.
        rgba[offsets[1]+4u*bucket+3u]=static_cast<float>(leaf);
    }
    if (levels>2u) {
        for (size_t i=offsets[2]+3u; i<expected_floats; i+=4u) rgba[i]=0.0f;
    }
    // The original cosine estimator is already noise-free for a constant map.
    // Compare the full RGB vector, not just its peak or luminance.
    return !constant;
}

} // namespace pr_environment_importance
