// Actual native transforms are the same validated FP32 values uploaded to TLAS
// and primary raster, including Instance.local_pose. IDs need not be contiguous.
struct GpuRigidPair {
    GpuTransform current,previous;
    uint32_t identity[4];
};
static_assert(sizeof(GpuRigidPair)==272,"rigid history table ABI");

void require_history_rigid(const GpuTransform& transform) {
    const auto* a=transform.world;
    for(uint32_t i=0;i<3;++i)for(uint32_t j=0;j<3;++j){
        double dot=0;for(uint32_t k=0;k<3;++k)dot+=static_cast<double>(a[4*i+k])*a[4*j+k];
        require(std::abs(dot-(i==j?1.:0.))<=2e-5,"rigid history excludes scaled/sheared transforms");
    }
    const double determinant=static_cast<double>(a[0])*(a[5]*a[10]-a[6]*a[9])-
        static_cast<double>(a[4])*(a[1]*a[10]-a[2]*a[9])+static_cast<double>(a[8])*(a[1]*a[6]-a[2]*a[5]);
    require(determinant>0,"rigid history excludes mirrored transforms");
}

void rigid_upload(Context& c,const std::vector<GpuTransform>& transforms) {
    auto& s=*c.progressive_state;const auto& instances=c.resident_state->instances;
    require(transforms.size()==instances.size(),"rigid history instance catalog mismatch");
    const uint32_t count=static_cast<uint32_t>(transforms.size());
    require(count<=1024,"rigid history binary-ID table supports at most 1024 instances");
    const size_t bytes=16+static_cast<size_t>(count)*sizeof(GpuRigidPair);
    require(bytes<=c.storage_buffer_limit,"rigid history table capacity exceeded");
    if(!s.rigid_motion.handle)c.make_buffer(s.rigid_motion,bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::vector<uint32_t> order(count);for(uint32_t i=0;i<count;++i)order[i]=i;
    std::sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){return instances[a].instance_id<instances[b].instance_id;});
    std::vector<GpuRigidPair> pairs(count);
    const bool previous_valid=s.history_valid&&!c.history_reset&&s.rigid_previous.size()==transforms.size();
    c.rigid_moved_instances=0;
    for(uint32_t j=0;j<count;++j){
        const uint32_t i=order[j];auto& pair=pairs[j];
        require_history_rigid(transforms[i]);
        pair.current=transforms[i];pair.previous=previous_valid?s.rigid_previous[i]:transforms[i];
        pair.identity[0]=instances[i].instance_id;
        pair.identity[1]=std::memcmp(pair.current.world,pair.previous.world,64)!=0;
        pair.identity[2]=0;pair.identity[3]=0;
        if(pair.identity[1]){
            ++c.rigid_moved_instances;
            double translation=0,trace=0;
            for(uint32_t k=0;k<3;++k){double d=static_cast<double>(pair.current.world[12+k])-pair.previous.world[12+k];translation+=d*d;}
            for(uint32_t k=0;k<3;++k)for(uint32_t i2=0;i2<3;++i2)trace+=static_cast<double>(pair.current.world[4*k+i2])*pair.previous.world[4*k+i2];
            pair.identity[2]=translation>.25*.25||(trace-1.)*.5<std::cos(20.*3.14159265358979323846/180.);
        }
    }
    std::vector<uint8_t> packed(bytes,0);std::memcpy(packed.data(),&count,4);
    if(count)std::memcpy(packed.data()+16,pairs.data(),count*sizeof(GpuRigidPair));
    c.upload(s.rigid_motion,packed.data(),bytes);c.rigid_upload_bytes+=bytes;
    c.rigid_instance_count=count;s.rigid_pending=transforms;
    VkDescriptorBufferInfo info{s.rigid_motion.handle,0,bytes};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};write.dstSet=s.set;write.dstBinding=16;
    write.descriptorCount=1;write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;write.pBufferInfo=&info;
    vkUpdateDescriptorSets(c.device,1,&write,0,nullptr);
}
