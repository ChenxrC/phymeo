// Bounded interface guard against mixing the shipped optional shader bundles.
// This is not a general SPIR-V verifier or an untrusted-code sandbox. Vulkan
// still validates modules. No allocation uses the module's declared ID bound.
struct SpirvInterface { bool valid=false, main=false; uint64_t bindings=0, locations=0; };
SpirvInterface spirv_interface(const uint32_t* code,size_t bytes,uint32_t model) {
    SpirvInterface result;
    if (!code || bytes<20 || bytes%4 || bytes>16u*1024u*1024u || code[0]!=0x07230203) return result;
    const size_t words=bytes/4;
    std::array<uint32_t,64> binding_ids{};
    uint64_t present=0;
    for (size_t p=5;p<words;) {
        const uint32_t count=code[p]>>16, opcode=code[p]&0xffffu;
        if (!count || count>words-p) return result;
        if (opcode==15 && count>=5 && code[p+1]==model &&
            code[p+3]==0x6e69616du && (code[p+4]&255u)==0) result.main=true;
        if (opcode==71 && count>=4) {
            if (code[p+2]==33) {
                uint32_t binding=code[p+3];
                if (binding>=64 || (present & (1ull<<binding))) return result;
                present|=1ull<<binding; binding_ids[binding]=code[p+1];
            }
            if (code[p+2]==30) {
                if (code[p+3]>=64) return result;
                result.locations|=1ull<<code[p+3];
            }
        }
        p+=count;
    }
    for (size_t p=5;p<words;) {
        const uint32_t count=code[p]>>16,opcode=code[p]&0xffffu;
        if (opcode==71 && count>=4 && code[p+2]==34 && code[p+3]==0)
            for (uint32_t b=0;b<64;++b)
                if ((present&(1ull<<b)) && binding_ids[b]==code[p+1]) result.bindings|=1ull<<b;
        p+=count;
    }
    result.valid=true;
    return result;
}

// This named uint specialization constant is a declared bundle ABI, not a
// general proof of arbitrary shader behavior. No pipeline overrides SpecId 57.
// Validate stage, exact OpName, SpecId, uint32 type and literal default before
// accepting alpha as a roughness channel. Old main fragments fail closed.
bool surface_maps_shader_marker(const uint32_t* code,size_t bytes) {
    const auto interface=spirv_interface(code,bytes,4);
    if (!interface.valid || !interface.main) return false;
    const size_t words=bytes/4;
    uint32_t marker=0,type=0;
    bool named=false,constant=false,unsigned32=false;
    for (size_t p=5;p<words;p+=code[p]>>16) {
        uint32_t count=code[p]>>16,opcode=code[p]&0xffffu;
        if (opcode==71 && count==4 && code[p+2]==1 && code[p+3]==57) {
            if (marker) return false;
            marker=code[p+1];
        }
    }
    if (!marker) return false;
    constexpr char marker_name[]="PR_SURFACE_MAPS_ABI";
    for (size_t p=5;p<words;p+=code[p]>>16) {
        uint32_t count=code[p]>>16,opcode=code[p]&0xffffu;
        if (opcode==5 && count==2+(sizeof(marker_name)+3)/4 && code[p+1]==marker)
            named=std::memcmp(code+p+2,marker_name,sizeof(marker_name))==0;
        if (opcode==50 && count==4 && code[p+2]==marker) {
            if (constant || code[p+3]!=1) return false;
            constant=true; type=code[p+1];
        }
    }
    for (size_t p=5;p<words;p+=code[p]>>16)
        if ((code[p]&0xffffu)==21 && (code[p]>>16)==4 && code[p+1]==type)
            unsigned32=code[p+2]==32 && code[p+3]==0;
    return named && constant && unsigned32;
}
