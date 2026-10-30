#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "metadata.cuh"

// Different lanes request different records. Only one lane composes each frame;
// there is no whole-block participation contract in this thread-scope interface.
template<class T> __global__ void probe(T *result) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane < grid::model::nj) {
        grid::model::joint_record j{};
        if (!grid::model::joint(lane, j)) return;
        T *out = result + lane * 45;
        out[0]=j.id; out[1]=j.parent_body; out[2]=j.child_body;
        out[3]=j.q_start; out[4]=j.v_start; out[5]=j.local_nq; out[6]=j.local_nv;
        out[7]=static_cast<int>(j.kind); out[8]=j.owns_coordinates;
        out[9]=j.has_axis; out[10]=j.has_pitch; out[11]=j.has_mimic;
        out[12]=j.leader_joint; out[13]=j.multiplier; out[14]=j.offset;
        for(int k=0;k<3;++k) out[15+k]=j.axis[k];
        out[18]=j.pitch;
        for(int k=0;k<16;++k) out[19+k]=j.T_parent_joint_zero[k];
        out[35]=j.has_position_limit[0]; out[36]=j.has_position_limit[1];
        out[37]=j.has_velocity_limit; out[38]=j.has_effort_limit;
        out[39]=j.position_limit[0]; out[40]=j.position_limit[1];
        out[41]=j.velocity_limit; out[42]=j.effort_limit;
        grid::model::joint_record sentinel{}; sentinel.id=71;
        out[43]=grid::model::joint(-1, sentinel);
        out[44]=sentinel.id;
    }
    if (lane < grid::model::frame_count) {
        grid::model::frame_record f{};
        if (!grid::model::frame(lane, f)) return;
        T *out=result+45*grid::model::nj+lane*38;
        out[0]=f.id; out[1]=f.attached_body; out[2]=f.joint_id;
        out[3]=static_cast<int>(f.kind);
        for(int k=0;k<16;++k) out[4+k]=f.T_body_frame[k];
        T body[16]={0,1,0,0,-1,0,0,0,0,0,1,0,T(.3),T(-.4),T(.2),1};
        T composed[16]; for (int k=0;k<16;++k) composed[k]=T(91);
        out[20]=grid::model::compose_frame_from_body(lane,f.attached_body,body,composed);
        for(int k=0;k<16;++k) out[21+k]=composed[k];
        // A wrong body must not overwrite output or run hidden synchronization.
        out[37]=grid::model::compose_frame_from_body(lane,f.attached_body+1,body,composed);
        for(int k=0;k<16;++k) if(out[21+k]!=composed[k]) out[37]=T(99);
    }
}

template<class T> int run(int threads) {
    const size_t n=45*grid::model::nj+38*grid::model::frame_count;
    T *allocation=nullptr;
    if(cudaMalloc(&allocation,(n+2)*sizeof(T))!=cudaSuccess) return 2;
    std::vector<T> values(n+2,T(-123));
    if(cudaMemcpy(allocation,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice)!=cudaSuccess) return 3;
    int records=grid::model::nj>grid::model::frame_count?grid::model::nj:grid::model::frame_count;
    probe<<<(records+threads-1)/threads,threads>>>(allocation+1);
    if(cudaGetLastError()!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess) return 4;
    if(cudaMemcpy(values.data(),allocation,values.size()*sizeof(T),cudaMemcpyDeviceToHost)!=cudaSuccess) return 5;
    if(cudaFree(allocation)!=cudaSuccess || values.front()!=T(-123) || values.back()!=T(-123)) return 6;
    for(size_t k=1;k<=n;++k) std::printf("%.17g\n",double(values[k]));
    return 0;
}
int main(int argc,char **argv) {
    if(argc!=3) return 1;
    return std::atoi(argv[2])?run<double>(std::atoi(argv[1])):run<float>(std::atoi(argv[1]));
}
