
PR_API uint32_t pr_preview_history_abi_version(){return 1;}
PR_API int pr_preview_history_setup(void* handle,const uint32_t* code,size_t bytes){
    try{require(handle,"null history context");history_setup(*static_cast<Context*>(handle),code,bytes);return 0;}
    catch(const std::exception& e){last_error=e.what();return -1;}
}
PR_API int pr_preview_history_arm(void* handle,uint32_t reset,uint32_t sequence){
    try{require(handle&&reset<=1,"invalid history arm");auto& c=*static_cast<Context*>(handle);
        require(c.history_enabled&&!c.poisoned&&!c.history_armed,"history context unavailable or already armed");
        c.history_reset=reset!=0;c.history_sequence=sequence;c.history_armed=true;c.history_rendered=false;return 0;
    }catch(const std::exception& e){last_error=e.what();return -1;}
}
PR_API int pr_preview_history_cancel(void* handle){
    if(!handle){last_error="null history context";return -1;}
    auto& c=*static_cast<Context*>(handle);c.history_armed=false;c.history_rendered=false;return 0;
}
PR_API int pr_preview_history_publish(void* handle){
    try{require(handle,"null history context");auto& c=*static_cast<Context*>(handle);
        require(!c.poisoned,"poisoned history cannot publish");history_commit(c);return 0;
    }catch(const std::exception& e){last_error=e.what();return -1;}
}
PR_API int pr_preview_history_info(void* handle,uint32_t* out,size_t bytes){
    try{require(handle&&out&&bytes==128,"history info extent");auto& c=*static_cast<Context*>(handle);
        auto* s=c.progressive_state.get();const uint64_t allocated=s&&s->history_current.handle?static_cast<uint64_t>(s->width)*s->height*80+176:0;
        const uint32_t values[16]={1,c.history_enabled,c.history_captures,c.history_last_sequence,c.history_last_reset,
            s?s->width:0,s?s->height:0,8,s?s->history_read_slot:0,s&&s->history_valid,c.history_armed,c.history_resets,
            static_cast<uint32_t>(allocated),static_cast<uint32_t>(allocated>>32),static_cast<uint32_t>(c.history_reads),0};
        std::memset(out,0,bytes);std::memcpy(out,values,sizeof(values));std::memcpy(out+16,c.history_counts.data(),32);
        const uint64_t totals[4]={c.history_uniform_bytes,c.history_reads,c.history_read_bytes,allocated};
        std::memcpy(out+24,totals,32);return 0;
    }catch(const std::exception& e){last_error=e.what();return -1;}
}
PR_API int pr_preview_history_read(void* handle,uint32_t plane,void* output,size_t bytes){
    try{require(handle,"null history context");history_read(*static_cast<Context*>(handle),plane,output,bytes);return 0;}
    catch(const std::exception& e){last_error=e.what();return -1;}
}
