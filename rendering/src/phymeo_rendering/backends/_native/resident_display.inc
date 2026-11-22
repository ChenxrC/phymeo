// Khronos PBR Neutral CPU clear-color companion. Same pinned Apache-2.0 source
// and equations as resident_display.glsl. Only the opt-in clear path uses this;
// default clip continues through its original double-precision expression.
std::array<double,3> neutral_clear(const float* radiance,float exposure) {
    std::array<double,3> color{};
    for (uint32_t k=0;k<3;++k) color[k]=std::max(0.0,static_cast<double>(radiance[k])*exposure);
    const double x=std::min(color[0],std::min(color[1],color[2]));
    const double offset=x<0.08 ? x-6.25*x*x : 0.04;
    for (auto& value:color) value-=offset;
    const double peak=std::max(color[0],std::max(color[1],color[2]));
    if (peak>=0.76) {
        const double new_peak=1.0-0.24*0.24/(peak+0.24-0.76);
        const double g=1.0-1.0/(0.15*(peak-new_peak)+1.0);
        for (auto& value:color) value=value*(new_peak/peak)*(1.0-g)+new_peak*g;
    }
    return color;
}
