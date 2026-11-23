struct RigidPair {mat4 world,local,previousWorld,previousLocal;uvec4 identity;};
layout(set=0,binding=16,std430) readonly buffer RigidTable {uvec4 metadata;RigidPair pairs[];} rigid;
int findRigid(uint id) {
    uint lo=0u,hi=rigid.metadata.x;
    // Native admission is <=1024 instances; binary lookup handles arbitrary IDs.
    for(uint i=0u;i<12u&&lo<hi;++i){
        uint middle=lo+(hi-lo)/2u;
        if(rigid.pairs[middle].identity.x<id)lo=middle+1u;else hi=middle;
    }
    if(lo>=rigid.metadata.x||rigid.pairs[lo].identity.x!=id)return -1;
    return int(lo);
}
bool guideMatches(vec3 a,vec3 b) {
    return all(lessThanEqual(abs(a-b),.2*max(a,b)+vec3(1e-5)));
}
bool incomingMatches(vec3 previous,vec3 low,vec3 high) {
    vec3 margin=.5*(high-low)+.1*max(high,vec3(1e-5));
    return all(greaterThanEqual(previous,low-margin))&&all(lessThanEqual(previous,high+margin));
}
