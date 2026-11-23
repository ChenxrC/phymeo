// Standalone C++17 CPU regression for the exact native upload-side producer.
// No Vulkan loader, GPU, or Python package is needed. Example:
// cl /nologo /std:c++17 /EHsc /W4 test_native_environment_importance.cpp /Fe:importance.exe
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../src/phymeo_rendering/backends/_native/resident_environment_importance.inc"

namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance, const char* message) {
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}
size_t offset(uint32_t side, uint32_t level) {
    size_t value = 0;
    for (uint32_t i = 0; i < level; ++i) value += size_t(side >> i) * (side >> i) * 6 * 4;
    return value;
}
size_t index(uint32_t side, uint32_t level, uint32_t face, uint32_t x, uint32_t y) {
    const size_t n = side >> level;
    return offset(side, level) + ((size_t(face) * n + y) * n + x) * 4;
}
uint32_t levels(uint32_t side) {
    uint32_t count = 0;
    for (; side; side >>= 1) ++count;
    return count;
}
std::vector<float> input(uint32_t side, float red = 0.f, float green = 0.f, float blue = 0.f) {
    std::vector<float> values(offset(side, levels(side)));
    for (size_t i = 0; i < values.size(); i += 4) {
        values[i] = red; values[i + 1] = green; values[i + 2] = blue; values[i + 3] = 1.f;
    }
    return values;
}
using V = std::array<double, 3>;
V unit(V v) {
    const double length = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    for (auto& value : v) value /= length;
    return v;
}
double dot(V a, V b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
double spherical_triangle(V a, V b, V c) {
    // Independent spherical-triangle formula, not the production rectangular primitive.
    const V cross{b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0]};
    return 2. * std::atan2(std::abs(dot(a, cross)), 1. + dot(a,b) + dot(b,c) + dot(c,a));
}
double solid_angle(uint32_t side, uint32_t x, uint32_t y) {
    const double lo_x = -1. + 2.*x/side, hi_x = -1. + 2.*(x+1)/side;
    const double lo_y = -1. + 2.*y/side, hi_y = -1. + 2.*(y+1)/side;
    const V a=unit({lo_x,lo_y,1.}), b=unit({hi_x,lo_y,1.});
    const V c=unit({hi_x,hi_y,1.}), d=unit({lo_x,hi_y,1.});
    return spherical_triangle(a,b,c) + spherical_triangle(a,c,d);
}
void rgb_unchanged(const std::vector<float>& before, const std::vector<float>& after) {
    check(before.size() == after.size(), "payload size changed");
    for (size_t i = 0; i < before.size(); i += 4)
        check(std::memcmp(before.data()+i, after.data()+i, 3*sizeof(float)) == 0,
              "source RGB bytes changed at some face/mip");
}
float leaf_mass(uint32_t side, const std::vector<float>& values, uint32_t face, uint32_t x, uint32_t y) {
    const size_t k=(static_cast<size_t>(face)*side+y)*side+x;
    return values[4*k+3]-(k==0?0.f:values[4*(k-1)+3]);
}
void cdf_and_hints(uint32_t side, const std::vector<float>& values) {
    const size_t count=static_cast<size_t>(side)*side*6u, buckets=count/4u;
    std::vector<float> cdf(count);
    float previous=0.f;
    for (size_t k=0; k<count; ++k) {
        const float value=values[4*k+3];
        check(std::isfinite(value) && value>=previous && value<=1.f,"L0 is not an inclusive monotone FP32 CDF");
        cdf[k]=value; previous=value;
    }
    check(cdf.back()==1.f,"CDF must end at exact one");
    const size_t directory=offset(side,1);
    for (size_t j=0; j<buckets; ++j) {
        const float threshold=static_cast<float>(j)/static_cast<float>(buckets);
        // Independent binary search, not the producer's forward scan.
        const auto found=std::upper_bound(cdf.begin(),cdf.end(),threshold);
        const size_t expected=static_cast<size_t>(found-cdf.begin());
        const float actual=values[directory+4*j+3];
        check(expected<count,"directory threshold has no greater leaf");
        check(actual==static_cast<float>(expected),"L1 is not first CDF strictly above the exact FP32 bucket threshold");
        check(actual>=0.f && actual<static_cast<float>(count) && std::floor(actual)==actual,"directory index is invalid");
        check(static_cast<size_t>(actual)==expected,"directory integer lost precision");
    }
    for (size_t i=offset(side,2)+3; i<values.size(); i+=4)
        check(values[i]==0.f,"higher private mips must not masquerade as a probability tree");
}
void uniform_and_rgb_preservation() {
    for (uint32_t side : {2u,4u,16u,256u}) {
        auto original=input(side,.25f,.5f,.125f);
        // Higher RGB mips deliberately do NOT equal the base map. They must not
        // affect probabilities, or be overwritten while constructing the alpha CDF.
        for (size_t i=offset(side,1); i<original.size(); i+=4) {
            original[i]=99.f; original[i+1]=.75f; original[i+2]=-.0f;
        }
        auto staged=original;
        check(!pr_environment_importance::populate(side,levels(side),staged.data(),staged.size()*sizeof(float)),
              "direction-constant map must retain the noiseless old path");
        rgb_unchanged(original,staged); cdf_and_hints(side,staged);
        const double sphere=4.*std::acos(-1.);
        double cumulative=0.;
        for (uint32_t face=0;face<6;++face) for (uint32_t y=0;y<side;++y) for (uint32_t x=0;x<side;++x) {
            cumulative+=solid_angle(side,x,y)/sphere;
            near(staged[index(side,0,face,x,y)+3],cumulative,1.5e-7,
                 "uniform CDF is not cumulative exact solid-angle mass in global face-major order");
        }
        if (side>=4) check(leaf_mass(side,staged,0,side/2,side/2)>leaf_mass(side,staged,0,0,0),
                           "uniform solid angle confused with uniform texel probability");
    }
}
void sparse_rgb_max_face_order() {
    const uint32_t side=8;
    auto original=input(side);
    const size_t a=index(side,0,1,2,2), b=index(side,0,4,4,4);
    original[a+1]=2.f; original[b]=4.f; original[b+2]=1.f;
    auto staged=original;
    check(pr_environment_importance::populate(side,levels(side),staged.data(),staged.size()*sizeof(float)), "sparse map disabled");
    // Independent cell integral of a tent centred at the bright texel centre:
    // its 1D integral over its own texel is .75, over each neighbour .125.
    auto tent=[](int d){return d==0?.75:std::abs(d)==1?.125:0.;};
    auto expected=[&](uint32_t f,int x,int y) {
        const double peak=f==1?2.*tent(x-2)*tent(y-2):f==4?4.*tent(x-4)*tent(y-4):0.;
        return peak*solid_angle(side,x,y);
    };
    double total=0.;
    for (uint32_t f=0;f<6;++f) for(int y=0;y<8;++y) for(int x=0;x<8;++x) total+=expected(f,x,y);
    double cumulative=0.;
    for (uint32_t f=0;f<6;++f) for(int y=0;y<8;++y) for(int x=0;x<8;++x) {
        cumulative+=expected(f,x,y)/total;
        near(staged[index(side,0,f,x,y)+3],cumulative,2.e-7,"CDF does not preserve bilinear RGB footprint masses");
    }
    rgb_unchanged(original,staged); cdf_and_hints(side,staged);
    for (uint32_t face : {0u,2u,3u,5u}) for(uint32_t y=0;y<side;++y) for(uint32_t x=0;x<side;++x)
        check(leaf_mass(side,staged,face,x,y)==0.f,"CDF difference leaked mass across faces");
    auto seam=input(side);
    seam[index(side,0,0,7,3)]=1.f;
    pr_environment_importance::populate(side,levels(side),seam.data(),seam.size()*sizeof(float));
    check(leaf_mass(side,seam,5,0,3)>0.f,"bright +X edge omitted on touching -Z face");
    cdf_and_hints(side,seam);
}
void zero_base_and_subnormal() {
    const uint32_t side=4;
    auto black=input(side);
    for (size_t i=offset(side,1);i<black.size();i+=4) black[i]=100.f;
    const auto original=black;
    check(!pr_environment_importance::populate(side,levels(side),black.data(),black.size()*sizeof(float)), "zero base must keep old sampling path");
    rgb_unchanged(original,black);
    for (size_t i=3;i<black.size();i+=4) check(black[i]==0.f,"zero base must clear every alpha level");
    auto tiny=input(side);
    tiny[index(side,0,5,3,2)+2]=std::numeric_limits<float>::denorm_min();
    check(pr_environment_importance::populate(side,levels(side),tiny.data(),tiny.size()*sizeof(float)), "positive subnormal total treated as black");
    check(leaf_mass(side,tiny,5,3,2)>.4f,"subnormal footprint lost its central probability");
    cdf_and_hints(side,tiny);
}
void malformed_payload_is_atomic() {
    for (int variant=0;variant<8;++variant) {
        auto staged=input(4,.5f);
        uint32_t side=4, count=3; size_t bytes=staged.size()*sizeof(float);
        if (variant==0) side=0;
        if (variant==1) side=3;
        if (variant==2) side=2048;
        if (variant==3) count=2;
        if (variant==4) --bytes;
        if (variant==5) staged.back()=.5f;
        if (variant==6) staged[staged.size()-4]=std::numeric_limits<float>::quiet_NaN();
        if (variant==7) staged[staged.size()-4]=-1.f;
        const auto original=staged;
        bool refused=false;
        try { pr_environment_importance::populate(side,count,staged.data(),bytes); }
        catch (const std::invalid_argument&) { refused=true; }
        check(refused,"malformed private payload accepted");
        check(std::memcmp(staged.data(),original.data(),original.size()*sizeof(float))==0,
              "rejection partially rewrote input");
    }
    bool refused=false;
    try { pr_environment_importance::populate(4,3,nullptr,0); }
    catch (const std::invalid_argument&) { refused=true; }
    check(refused,"null payload accepted");
}
}
int main() {
    try {
        uniform_and_rgb_preservation();
        sparse_rgb_max_face_order();
        zero_base_and_subnormal();
        malformed_payload_is_atomic();
        std::cout << "PASS native environment CDF: exact solid-angle footprints, FP32 strict-bucket directory, RGB/mip identity, black/subnormal, atomic rejects\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
