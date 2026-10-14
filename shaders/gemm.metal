#include <metal_stdlib>
using namespace metal;
kernel void gemm(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                 device float* c [[buffer(2)]], constant uint3& p [[buffer(3)]],
                 uint2 gid [[thread_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float ta[16][16];
    threadgroup float tb[16][16];
    float sum=0;
    for(uint base=0;base<p.z;base+=16) {
        ta[lid.y][lid.x]=(gid.y<p.x && base+lid.x<p.z) ? a[gid.y*p.z+base+lid.x] : 0;
        tb[lid.y][lid.x]=(gid.x<p.y && base+lid.y<p.z) ? b[(base+lid.y)*p.y+gid.x] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;j++) sum+=ta[lid.y][j]*tb[j][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(gid.y<p.x && gid.x<p.y) c[gid.y*p.y+gid.x]=sum;
}
kernel void convolution(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* c [[buffer(2)]],
                        device const float* bias [[buffer(3)]], constant uint* p [[buffer(4)]],
                        uint3 gid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float ta[16][16];
    threadgroup float tb[16][16];
    uint ic=p[1],ih=p[2],iw=p[3],oc=p[4],oh=p[5],ow=p[6],kh=p[7],kw=p[8],groups=p[15];
    uint group=gid.z%groups,batch=gid.z/groups,cig=ic/groups,cog=oc/groups,spatial=oh*ow,k=cig*kh*kw,co=group*cog+gid.y;
    float sum=0;
    for(uint base=0;base<k;base+=16) {
        ta[lid.y][lid.x]=(gid.y<cog && base+lid.x<k) ? a[co*k+base+lid.x] : 0;
        uint ki=base+lid.y;
        int iy=int(gid.x/ow*p[9]+(ki/kw%kh)*p[13])-int(p[11]);
        int ix=int(gid.x%ow*p[10]+ki%kw*p[14])-int(p[12]);
        uint ci=group*cig+ki/(kh*kw);
        tb[lid.y][lid.x]=(gid.x<spatial && ki<k && iy>=0 && ix>=0 && iy<int(ih) && ix<int(iw)) ? b[((batch*ic+ci)*ih+uint(iy))*iw+uint(ix)] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;j++) sum+=ta[lid.y][j]*tb[j][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(gid.y<cog && gid.x<spatial) { float v=sum+bias[co]; c[(batch*oc+co)*spatial+gid.x]=(p[0]==1) ? v/(1+exp(-v)) : v; }
}
