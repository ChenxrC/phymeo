PR_API uint32_t pr_preview_edge_abi_version(){return 1;}
PR_API int pr_preview_edge_set(void* handle,const uint32_t* code,size_t bytes,uint32_t limit){
    try{require(handle,"null edge context");edge_setup(*static_cast<Context*>(handle),code,bytes,limit);return 0;}
    catch(const std::exception& e){last_error=e.what();return -1;}
}
PR_API int pr_preview_edge_info(void* handle,uint32_t* output,size_t bytes){
    if(!handle||!output||bytes!=64){last_error="edge info requires 64 bytes";return -1;}
    auto& c=*static_cast<Context*>(handle);std::memcpy(output,c.edge_counts.data(),64);return 0;
}