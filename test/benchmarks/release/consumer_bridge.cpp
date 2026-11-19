// Native synchronous boundaries. The CPU backend's D2H, conversion/evaluation
// callback and H2D all belong to the sample. Allocation and validation downloads
// are outside it. No Python callback is used in production.
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstring>
#include <new>

using Evaluate = int (*)(void *, const float *, float *, int, bool);
struct Consumer {
    Evaluate evaluate;
    void *context;
    bool gpu;
    int batch;
    size_t input_bytes, output_bytes;
    float *host_in=nullptr, *host_out=nullptr, *device_in=nullptr, *device_out=nullptr;
};
static int error(cudaError_t e) { return e == cudaSuccess ? 0 : 100+int(e); }
static int dispose(Consumer *c) {
    int first=0;
    auto keep=[&](cudaError_t e) { if (!first) first=error(e); };
    if(c->device_in) keep(cudaFree(c->device_in));
    if(c->device_out) keep(cudaFree(c->device_out));
    if(c->host_in) keep(cudaFreeHost(c->host_in));
    if(c->host_out) keep(cudaFreeHost(c->host_out));
    delete c; return first;
}
extern "C" int consumer_create(void **out, Evaluate evaluate, void *context, int gpu,
                               int batch, int input_width, int output_width, const float *input) {
    if(!out || !evaluate || !context || !input || batch<1 || batch>1024 || input_width<1 || output_width<1) return -1;
    *out=nullptr;
    auto *c=new(std::nothrow) Consumer{evaluate,context,gpu!=0,batch,
        size_t(batch)*input_width*sizeof(float),size_t(batch)*output_width*sizeof(float)};
    if(!c) return -2;
    int rc=0;
    if(!(rc=error(cudaMallocHost(reinterpret_cast<void**>(&c->host_in),c->input_bytes))) &&
       !(rc=error(cudaMallocHost(reinterpret_cast<void**>(&c->host_out),c->output_bytes))) &&
       !(rc=error(cudaMalloc(reinterpret_cast<void**>(&c->device_in),c->input_bytes))) &&
       !(rc=error(cudaMalloc(reinterpret_cast<void**>(&c->device_out),c->output_bytes)))) {
        std::memcpy(c->host_in,input,c->input_bytes);
        rc=error(cudaMemcpy(c->device_in,input,c->input_bytes,cudaMemcpyHostToDevice));
        if(!rc) rc=error(cudaDeviceSynchronize());
    }
    if(rc) { const int cleanup=dispose(c); return rc ? rc : cleanup; }
    *out=c; return 0;
}
static int call(Consumer &c, bool on_gpu) {
    if(c.gpu) {
        int rc=c.evaluate(c.context,on_gpu?c.device_in:c.host_in,
                          on_gpu?c.device_out:c.host_out,c.batch,on_gpu);
        if(rc) return rc;
        return error(cudaDeviceSynchronize());
    }
    if(on_gpu) {
        int rc=error(cudaMemcpy(c.host_in,c.device_in,c.input_bytes,cudaMemcpyDeviceToHost));
        if(rc) return rc;
        // Host memory must be ready before CPU computation even for runtimes
        // where a transfer enqueue alone does not imply host readability.
        rc=error(cudaDeviceSynchronize()); if(rc) return rc;
    }
    int rc=c.evaluate(c.context,c.host_in,c.host_out,c.batch,false);
    if(rc || !on_gpu) return rc;
    rc=error(cudaMemcpy(c.device_out,c.host_out,c.output_bytes,cudaMemcpyHostToDevice));
    return rc ? rc : error(cudaDeviceSynchronize());
}
static int snapshot(Consumer &c, bool on_gpu, float *output) {
    if(on_gpu) return error(cudaMemcpy(output,c.device_out,c.output_bytes,cudaMemcpyDeviceToHost));
    std::memcpy(output,c.host_out,c.output_bytes); return 0;
}
extern "C" int consumer_once(void *context, int on_gpu, float *output) {
    if(!context || !output || (on_gpu!=0 && on_gpu!=1)) return -1;
    auto &c=*static_cast<Consumer*>(context);
    int rc=call(c,on_gpu); return rc ? rc : snapshot(c,on_gpu,output);
}
static double now_us() {
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
extern "C" int consumer_measure(void *context, int on_gpu, int warmups, double warm_seconds,
                                int iterations, double *samples, float *last) {
    if(!context || !samples || !last || iterations<1 || warmups<1 || warm_seconds<0 || (on_gpu!=0 && on_gpu!=1)) return -1;
    auto &c=*static_cast<Consumer*>(context);
    const double start=now_us(); int done=0,rc=0;
    do { rc=call(c,on_gpu); if(rc) return rc; ++done; }
    while(done<warmups || now_us()-start<warm_seconds*1e6);
    for(int i=0;i<iterations;++i) {
        const double t=now_us(); rc=call(c,on_gpu); if(rc) return rc;
        samples[i]=now_us()-t;
    }
    // Return the actual final timed output, not a replacement evaluation.
    return snapshot(c,on_gpu,last);
}
extern "C" int consumer_close(void *context) {
    return context ? dispose(static_cast<Consumer*>(context)) : 0;
}
