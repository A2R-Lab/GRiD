// Pinocchio-convention full base force: eight independent rows, no actuator map.
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using T = TEST_SCALAR;
constexpr int NQ=7, NV=6, NX=13, NJ=12*18, BATCH=8;
static_assert(grid::NUM_POS == NQ && grid::NUM_VEL == NV);
#define CK(x) do {auto err=(x);if(err!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(err));exit(2);}}while(0)

template<grid::IntegratorType IT>
void run(T *out, T *jac, T *x, T *force, grid::robotModel<T> *model, int threads, T dt) {
    const size_t value_bytes=grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>();
    const size_t jac_bytes=grid_plant::PLANT_STEP_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    CK(cudaFuncSetAttribute(grid_plant::plant_step_kernel<T,IT>,cudaFuncAttributeMaxDynamicSharedMemorySize,value_bytes));
    CK(cudaFuncSetAttribute(grid_plant::plant_step_gradient_kernel<T,IT>,cudaFuncAttributeMaxDynamicSharedMemorySize,jac_bytes));
    grid_plant::plant_step_kernel<T,IT><<<BATCH,threads,value_bytes>>>(out,x,force,NX,NV,model,T(-9.81),dt,BATCH);
    CK(cudaGetLastError());
    grid_plant::plant_step_gradient_kernel<T,IT><<<BATCH,threads,jac_bytes>>>(jac,x,force,NX,NV,model,T(-9.81),dt,BATCH);
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());
}

int main(int argc, char **argv) {
    if(argc!=6)return 1;
    const int threads=atoi(argv[3]),semi=atoi(argv[4]);
    if(threads>grid::MAX_PERF_LEVEL_THREADS)return 3;
    const T dt=T(atof(argv[5]));
    std::vector<T> input(BATCH*(NX+NV)),x(BATCH*NX),force(BATCH*NV),result(BATCH*(NX+NJ)),first;
    FILE *file=fopen(argv[1],"rb");if(!file||fread(input.data(),sizeof(T),input.size(),file)!=input.size())return 4;fclose(file);
    for(int b=0;b<BATCH;++b){
        for(int i=0;i<NX;++i)x[b*NX+i]=input[b*(NX+NV)+i];
        for(int i=0;i<NV;++i)force[b*NV+i]=input[b*(NX+NV)+NX+i];
    }
    T *dx,*du,*out,*jac;grid::robotModel<T> *model=nullptr;
    CK(grid::init_robotModel_checked<T>(&model));
    CK(cudaMalloc(&dx,x.size()*sizeof(T)));CK(cudaMalloc(&du,force.size()*sizeof(T)));
    CK(cudaMalloc(&out,BATCH*NX*sizeof(T)));CK(cudaMalloc(&jac,BATCH*NJ*sizeof(T)));
    CK(cudaMemcpy(dx,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(du,force.data(),force.size()*sizeof(T),cudaMemcpyHostToDevice));
    for(int repeat=0;repeat<3;++repeat){
        CK(cudaMemset(out,0xff,BATCH*NX*sizeof(T)));CK(cudaMemset(jac,0xff,BATCH*NJ*sizeof(T)));
        if(semi)run<grid::IntegratorType::SEMI_IMPLICIT_EULER>(out,jac,dx,du,model,threads,dt);
        else run<grid::IntegratorType::EULER>(out,jac,dx,du,model,threads,dt);
        CK(cudaMemcpy(result.data(),out,BATCH*NX*sizeof(T),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(result.data()+BATCH*NX,jac,BATCH*NJ*sizeof(T),cudaMemcpyDeviceToHost));
        if(repeat && std::memcmp(result.data(),first.data(),result.size()*sizeof(T)))return 5;
        first=result;
    }
    file=fopen(argv[2],"wb");if(!file||fwrite(result.data(),sizeof(T),result.size(),file)!=result.size())return 6;fclose(file);
    CK(cudaFree(dx));CK(cudaFree(du));CK(cudaFree(out));CK(cudaFree(jac));CK(grid::free_robotModel_checked<T>(model));
    return 0;
}
