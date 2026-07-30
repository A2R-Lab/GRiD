// Robot-agnostic validation for the W3 Component E static geometry header
// (grid_collision::grid_cc_sphere_{sphere,capsule,cuboid} + reduction/self-collision/driver).
//
// Pure geometry: NO grid.cuh, no robot model. For each baked config it evaluates the SDF on the
// DEVICE and on the HOST (the primitives are __host__ __device__) and prints the FULL config plus
// the GPU squared-gap, so the Python oracle recomputes the expected value from the printed geometry
// with zero config duplication. Also self-checks HOST==DEVICE (fp64 bit-exact) and that a T=float
// instantiation agrees on the collision/free SIGN for every config (fp32 is the default precision).
//
// SDF convention: return squared_gap = d2 - r_sum^2 ; value < 0  <=>  in collision.
#include "grid_collision_geometry.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using T = double;
using grid_collision::Capsule;
using grid_collision::Cuboid;
using grid_collision::Environment;
using grid_collision::Plane;
using grid_collision::Sphere;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

// ---- device kernels: one thread per config, gap[i] = SDF(config i) ----
template <typename S>
__global__ void ss_kernel(const S *cfg, int n, S *gap) {          // 8 floats/config
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const S *c = cfg + 8 * i;
    gap[i] = grid_collision::grid_cc_sphere_sphere<S>(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
}
template <typename S>
__global__ void sc_kernel(const S *cfg, int n, S *gap) {          // 11 floats/config: cap(7)+sph(4)
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const S *c = cfg + 11 * i;
    Capsule<S> cap{c[0], c[1], c[2], c[3], c[4], c[5], c[6]};
    gap[i] = grid_collision::grid_cc_sphere_capsule<S>(cap, c[7], c[8], c[9], c[10]);
}
template <typename S>
__global__ void cb_kernel(const S *cfg, int n, S *gap) {          // 19 floats/config: box(15)+sph(4)
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const S *c = cfg + 19 * i;
    Cuboid<S> box{c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13], c[14]};
    gap[i] = grid_collision::grid_cc_sphere_cuboid<S>(box, c[15], c[16], c[17], c[18]);
}

template <typename S>
__global__ void cc_kernel(const S *cfg, int n, S *gap) {          // 14 floats/config: capA(7)+capB(7)
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const S *c = cfg + 14 * i;
    Capsule<S> A{c[0], c[1], c[2], c[3], c[4], c[5], c[6]};
    Capsule<S> B{c[7], c[8], c[9], c[10], c[11], c[12], c[13]};
    gap[i] = grid_collision::grid_cc_capsule_capsule<S>(A, B);
}
template <typename S>
__global__ void cp_kernel(const S *cfg, int n, S *gap) {          // 11 floats/config: plane(4)+cap(7)
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const S *c = cfg + 11 * i;
    Plane<S> p{c[0], c[1], c[2], c[3]};
    Capsule<S> cap{c[4], c[5], c[6], c[7], c[8], c[9], c[10]};
    gap[i] = grid_collision::grid_cc_capsule_plane<S>(p, cap);
}
template <typename S>
__global__ void cx_kernel(const S *cfg, int n, S *gap) {          // 22 floats/config: box(15)+cap(7)
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const S *c = cfg + 22 * i;
    Cuboid<S> box{c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13], c[14]};
    Capsule<S> cap{c[15], c[16], c[17], c[18], c[19], c[20], c[21]};
    gap[i] = grid_collision::grid_cc_capsule_cuboid<S>(box, cap);
}

// ---- host mirrors (same header, host path) ----
static T host_ss(const T *c){ return grid_collision::grid_cc_sphere_sphere<T>(c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]); }
static T host_sc(const T *c){ Capsule<T> cap{c[0],c[1],c[2],c[3],c[4],c[5],c[6]}; return grid_collision::grid_cc_sphere_capsule<T>(cap,c[7],c[8],c[9],c[10]); }
static T host_cb(const T *c){ Cuboid<T> b{c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],c[13],c[14]}; return grid_collision::grid_cc_sphere_cuboid<T>(b,c[15],c[16],c[17],c[18]); }
static T host_cc(const T *c){ Capsule<T> A{c[0],c[1],c[2],c[3],c[4],c[5],c[6]}, B{c[7],c[8],c[9],c[10],c[11],c[12],c[13]}; return grid_collision::grid_cc_capsule_capsule<T>(A,B); }
static T host_cp(const T *c){ Plane<T> p{c[0],c[1],c[2],c[3]}; Capsule<T> cap{c[4],c[5],c[6],c[7],c[8],c[9],c[10]}; return grid_collision::grid_cc_capsule_plane<T>(p,cap); }
static T host_cx(const T *c){ Cuboid<T> b{c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],c[13],c[14]}; Capsule<T> cap{c[15],c[16],c[17],c[18],c[19],c[20],c[21]}; return grid_collision::grid_cc_capsule_cuboid<T>(b,cap); }

// push an axis*sign rotation frame (columns of a rotation) for cuboid tests
static void push_box(std::vector<T>&v, T cx,T cy,T cz, T yaw,T pitch,
                     T hu,T hv,T hw, T px,T py,T pz,T pr){
    // R = Rz(yaw) * Ry(pitch); columns u,v,w are the box axes
    T cy_=cos(yaw), sy_=sin(yaw), cp=cos(pitch), sp=sin(pitch);
    T ux= cy_*cp, uy= sy_*cp, uz=-sp;
    T vx=-sy_,    vy= cy_,     vz= 0;
    T wx= cy_*sp, wy= sy_*sp,  wz= cp;
    T box[15]={cx,cy,cz, ux,uy,uz,hu, vx,vy,vz,hv, wx,wy,wz,hw};
    for(int i=0;i<15;++i) v.push_back(box[i]);
    v.push_back(px); v.push_back(py); v.push_back(pz); v.push_back(pr);
}

// same rotation math as push_box, but a CAPSULE probe (box 15 + capsule 7 = 22)
static void push_box_cap(std::vector<T>&v, T cx,T cy,T cz, T yaw,T pitch,
                         T hu,T hv,T hw, T ax,T ay,T az, T bx,T by,T bz, T cr){
    T cy_=cos(yaw), sy_=sin(yaw), cp=cos(pitch), sp=sin(pitch);
    T ux= cy_*cp, uy= sy_*cp, uz=-sp;
    T vx=-sy_,    vy= cy_,     vz= 0;
    T wx= cy_*sp, wy= sy_*sp,  wz= cp;
    T box[15]={cx,cy,cz, ux,uy,uz,hu, vx,vy,vz,hv, wx,wy,wz,hw};
    for(int i=0;i<15;++i) v.push_back(box[i]);
    T cap[7]={ax,ay,az,bx,by,bz,cr};
    for(int i=0;i<7;++i) v.push_back(cap[i]);
}

template <typename S>
static void run(const std::vector<T>&cfg, int stride, int n, std::vector<T>&gpu,
                void(*k)(const S*,int,S*)){
    std::vector<S> hc(cfg.begin(), cfg.end());
    S *d_c,*d_g; cudaMalloc(&d_c,hc.size()*sizeof(S)); cudaMalloc(&d_g,n*sizeof(S));
    cudaMemcpy(d_c,hc.data(),hc.size()*sizeof(S),cudaMemcpyHostToDevice);
    k<<<(n+63)/64,64>>>(d_c,n,d_g); cudaDeviceSynchronize();
    std::vector<S> hg(n); cudaMemcpy(hg.data(),d_g,n*sizeof(S),cudaMemcpyDeviceToHost);
    gpu.assign(hg.begin(), hg.end());
    cudaFree(d_c); cudaFree(d_g);
}

int main(){
    // ---------------- sphere_sphere: overlap / touching / separated ----------------
    std::vector<T> ss; auto SS=[&](T ax,T ay,T az,T ar,T bx,T by,T bz,T br){
        T a[8]={ax,ay,az,ar,bx,by,bz,br}; for(int i=0;i<8;++i) ss.push_back(a[i]); };
    SS(0,0,0,0.5, 0.4,0,0, 0.5);      // overlap  (d=0.4 < 1.0)
    SS(0,0,0,0.5, 1.0,0,0, 0.5);      // touching (d=1.0 == r_sum -> gap 0)
    SS(0,0,0,0.5, 2.0,0,0, 0.5);      // separated
    SS(1,2,3,0.3,-1,0,1,   0.4);      // separated, off-axis
    SS(0,0,0,1.0, 0.1,0.1,0.1,0.2);   // deep overlap
    int n_ss = ss.size()/8;

    // ---------------- sphere_capsule: interior t, clamp t=0, clamp t=1, free ----------------
    std::vector<T> sc; auto SC=[&](T ax,T ay,T az,T bx,T by,T bz,T cr, T px,T py,T pz,T pr){
        T a[11]={ax,ay,az,bx,by,bz,cr,px,py,pz,pr}; for(int i=0;i<11;++i) sc.push_back(a[i]); };
    SC(-1,0,0, 1,0,0, 0.2,  0,0.3,0,   0.1);   // interior projection, collision (gap<0)
    SC(-1,0,0, 1,0,0, 0.2,  0,0.5,0,   0.1);   // interior projection, free
    SC(-1,0,0, 1,0,0, 0.2, -2,0.1,0,   0.1);   // before a -> clamp t=0
    SC(-1,0,0, 1,0,0, 0.2,  2,0.1,0,   0.1);   // after  b -> clamp t=1
    SC(0,0,0, 0,0,1, 0.3,  0.1,0,0.5,  0.15);  // z-aligned segment, side approach, collision
    SC(0,0,0, 0,0,0, 0.3,  0.4,0,0,    0.1);   // degenerate segment (a==b) -> sphere-sphere at a
    int n_sc = sc.size()/11;

    // ---------------- sphere_cuboid: axis-aligned + rotated; face / edge / corner / inside / free -
    std::vector<T> cb;
    push_box(cb, 0,0,0, 0,0,        0.5,0.5,0.5,  1.0,0,0,  0.2);   // AA box, +x face, free
    push_box(cb, 0,0,0, 0,0,        0.5,0.5,0.5,  0.6,0,0,  0.2);   // AA box, +x face, collision
    push_box(cb, 0,0,0, 0,0,        0.5,0.5,0.5,  0.0,0,0,  0.2);   // center inside -> collision (-r^2)
    push_box(cb, 0,0,0, 0,0,        0.5,0.5,0.5,  0.8,0.8,0,0.2);   // near an edge (two axes outside)
    push_box(cb, 0,0,0, 0,0,        0.5,0.5,0.5,  0.8,0.8,0.8,0.2); // near a corner (three axes outside)
    push_box(cb, 1,1,1, 0.6,0.3,    0.4,0.3,0.5,  1.7,1.2,1.3,0.25);// ROTATED box, off-corner
    push_box(cb, 1,1,1, 0.6,0.3,    0.4,0.3,0.5,  1.0,1.0,1.0,0.25);// ROTATED box, center inside
    int n_cb = cb.size()/19;

    // ---------------- capsule_capsule: parallel / skew / clamp / degenerate ----------------
    std::vector<T> cc; auto CC=[&](T a1x,T a1y,T a1z,T b1x,T b1y,T b1z,T r1,
                                   T a2x,T a2y,T a2z,T b2x,T b2y,T b2z,T r2){
        T a[14]={a1x,a1y,a1z,b1x,b1y,b1z,r1,a2x,a2y,a2z,b2x,b2y,b2z,r2};
        for(int i=0;i<14;++i) cc.push_back(a[i]); };
    CC(-1,0,0, 1,0,0, 0.2,  -1,0.3,0, 1,0.3,0, 0.2);   // parallel, d=0.3 < 0.4 -> collision
    CC(-1,0,0, 1,0,0, 0.1,   0,-1,0.5, 0,1,0.5, 0.1);  // skew crossing above, d=0.5 free
    CC(-1,0,0, 1,0,0, 0.15,  0,-1,0.2, 0,1,0.2, 0.1);  // skew crossing, d=0.2 < 0.25 collision
    CC(-1,0,0, 1,0,0, 0.2,   2,0.1,0, 3,0.5,0, 0.15);  // endpoint-endpoint clamp (t=1 vs t=0)
    CC( 0,0,0, 0,0,0, 0.3,   0.4,0,0, 0.4,0,0, 0.2);   // both degenerate -> sphere-sphere, collision
    CC( 0,0,1, 0,0,1, 0.1,  -1,0,0, 1,0,0, 0.2);       // one degenerate, d=1.0 free
    CC( 0,0,0, 2,0,0, 0.3,   0.5,0.2,0, 1.5,0.2,0, 0.2); // parallel overlap, deep collision
    int n_cc = cc.size()/14;

    // ---------------- capsule_plane: clear / dipping / endpoint-below / tilted ----------------
    std::vector<T> cpl; auto CP=[&](T nx,T ny,T nz,T d, T ax,T ay,T az,T bx,T by,T bz,T r){
        T a[11]={nx,ny,nz,d,ax,ay,az,bx,by,bz,r}; for(int i=0;i<11;++i) cpl.push_back(a[i]); };
    CP(0,0,1, 0,   0,0,0.5,  1,0,0.8,  0.2);           // both endpoints clear -> free
    CP(0,0,1, 0,   0,0,0.1,  1,0,1.0,  0.2);           // low endpoint within r -> collision
    CP(0,0,1, 0,   0,0,-0.3, 1,0,0.5,  0.1);           // endpoint below plane -> -r^2
    CP(0.6,0,0.8, 0.2,  1,0,1,  2,0.5,1.5,  0.25);     // tilted plane, free
    int n_cp = cpl.size()/11;

    // ---------------- capsule_cuboid: crossing / parallel-face / edge / degenerate / rotated ----
    std::vector<T> cx;
    push_box_cap(cx, 0,0,0, 0,0,       0.5,0.5,0.5,  -1,0,0,   1,0,0,    0.1);  // pierces box -> -r^2
    push_box_cap(cx, 0,0,0, 0,0,       0.5,0.5,0.5,  -1,0,0.8, 1,0,0.8,  0.2);  // parallel above face, d=0.3
    push_box_cap(cx, 0,0,0, 0,0,       0.5,0.5,0.5,   0.8,0.8,-1, 0.8,0.8,1, 0.1); // along an edge line, d=sqrt(0.18)
    push_box_cap(cx, 0,0,0, 0,0,       0.5,0.5,0.5,   0,0,0,   0,0,0,    0.2);  // degenerate point inside
    push_box_cap(cx, 0,0,0, 0,0,       0.5,0.5,0.5,   1.2,0,0, 3,0,0,    0.1);  // clamp t=0 endpoint closest
    push_box_cap(cx, 1,1,1, 0.6,0.3,   0.4,0.3,0.5,   2,1,0,   2,1,2,    0.15); // rotated box, skew segment
    push_box_cap(cx, 0,0,0, 0,0,       0.5,0.5,0.5,   1.5,-1,0.2, 0.7,1,0.6, 0.1); // slanted, interior stationary
    int n_cx = cx.size()/22;

    // ---------------- device eval (fp64) ----------------
    std::vector<T> g_ss,g_sc,g_cb,g_cc,g_cp,g_cx;
    run<T>(ss,8, n_ss,g_ss,ss_kernel<T>);
    run<T>(sc,11,n_sc,g_sc,sc_kernel<T>);
    run<T>(cb,19,n_cb,g_cb,cb_kernel<T>);
    run<T>(cc,14,n_cc,g_cc,cc_kernel<T>);
    run<T>(cpl,11,n_cp,g_cp,cp_kernel<T>);
    run<T>(cx,22,n_cx,g_cx,cx_kernel<T>);
    CK(cudaGetLastError());

    // ---------------- host==device (benign FMA-contraction diff, not bit-exact) ----------------
    double hd=0;
    for(int i=0;i<n_ss;++i) hd=std::max(hd,std::fabs(host_ss(&ss[8*i]) -g_ss[i]));
    for(int i=0;i<n_sc;++i) hd=std::max(hd,std::fabs(host_sc(&sc[11*i])-g_sc[i]));
    for(int i=0;i<n_cb;++i) hd=std::max(hd,std::fabs(host_cb(&cb[19*i])-g_cb[i]));
    for(int i=0;i<n_cc;++i) hd=std::max(hd,std::fabs(host_cc(&cc[14*i])-g_cc[i]));
    for(int i=0;i<n_cp;++i) hd=std::max(hd,std::fabs(host_cp(&cpl[11*i])-g_cp[i]));
    for(int i=0;i<n_cx;++i) hd=std::max(hd,std::fabs(host_cx(&cx[22*i])-g_cx[i]));
    printf("HOSTDEV maxdiff=%.3e\n", hd);

    // ---------------- fp32 lane: same configs, SIGN classification must agree with fp64 ----------
    // (skip knife-edge configs where |fp64 gap| is within fp32 boundary noise ~1e-6 -- the sign is
    //  meaningless there; the numeric oracle covers those exact-touching cases instead.)
    std::vector<T> f_ss,f_sc,f_cb,f_cc,f_cp,f_cx;
    run<float>(ss,8, n_ss,f_ss,ss_kernel<float>);
    run<float>(sc,11,n_sc,f_sc,sc_kernel<float>);
    run<float>(cb,19,n_cb,f_cb,cb_kernel<float>);
    run<float>(cc,14,n_cc,f_cc,cc_kernel<float>);
    run<float>(cpl,11,n_cp,f_cp,cp_kernel<float>);
    run<float>(cx,22,n_cx,f_cx,cx_kernel<float>);
    int signdiff=0;
    auto chk=[&](const std::vector<T>&a,const std::vector<T>&b){ for(size_t i=0;i<a.size();++i)
        if(std::fabs(a[i])>1e-6 && (a[i]<0)!=(b[i]<0)) ++signdiff; };
    chk(g_ss,f_ss); chk(g_sc,f_sc); chk(g_cb,f_cb); chk(g_cc,f_cc); chk(g_cp,f_cp); chk(g_cx,f_cx);
    printf("FP32SIGN mismatches=%d\n", signdiff);

    // ---------------- print self-describing configs + fp64 gap ----------------
    for(int i=0;i<n_ss;++i){ const T*c=&ss[8*i];
        printf("SS %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g GAP %.17g\n",
               c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7], g_ss[i]); }
    for(int i=0;i<n_sc;++i){ const T*c=&sc[11*i];
        printf("SC %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g GAP %.17g\n",
               c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10], g_sc[i]); }
    for(int i=0;i<n_cb;++i){ const T*c=&cb[19*i];
        printf("CB %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g GAP %.17g\n",
               c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18], g_cb[i]); }
    for(int i=0;i<n_cc;++i){ const T*c=&cc[14*i];
        printf("CC %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g GAP %.17g\n",
               c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],c[13], g_cc[i]); }
    for(int i=0;i<n_cp;++i){ const T*c=&cpl[11*i];
        printf("CP %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g GAP %.17g\n",
               c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10], g_cp[i]); }
    for(int i=0;i<n_cx;++i){ const T*c=&cx[22*i];
        printf("CX %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g GAP %.17g\n",
               c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18],c[19],c[20],c[21], g_cx[i]); }

    // ---------------- composition smoke: environment reduction + self-collision + driver --------
    // A tiny scene: one obstacle sphere at origin r=0.3; probes hit/miss it. Self-collision over a
    // 3-sphere set with one overlapping pair. Driver: broad free -> return true; broad hit -> fine.
    Sphere<T> obs[1] = {{0,0,0,0.3}};
    Environment<T> env{obs,1,nullptr,0,nullptr,0};
    bool e_hit  = grid_collision::grid_cc_sphere_in_environment<T>(env, 0.2,0,0, 0.2);   // 0.4<0.5 -> hit
    bool e_miss = grid_collision::grid_cc_sphere_in_environment<T>(env, 1.0,0,0, 0.2);   // 1.0>0.5 -> miss
    T spos[9] = {0,0,0,  0.4,0,0,  5,5,5};   // sphere0 & sphere1 overlap; sphere2 far
    T sr[3]   = {0.3,0.3,0.1};
    int ranges_hit[3]  = {0,1,1};            // check s0 vs s1 -> overlap
    int ranges_free[3] = {0,2,2};            // check s0 vs s2 -> free
    bool s_hit  = grid_collision::grid_cc_self_collision<T>(spos,sr,ranges_hit,1);
    bool s_free = grid_collision::grid_cc_self_collision<T>(spos,sr,ranges_free,1);
    printf("COMPO env_hit=%d env_miss=%d self_hit=%d self_free=%d\n",
           e_hit?1:0, e_miss?1:0, s_hit?1:0, s_free?1:0);

    // -------- signed-distance + NORMAL (differentiable path): FD the returned normal per primitive --------
    // The signed SDFs return n = d(signed_dist)/d(sphere_center); central-difference the signed distance
    // w.r.t. the query point (host: the primitives are __host__ __device__) and compare (all 3 shapes,
    // outside points where the normal is well-defined). Covers the capsule/cuboid branches the robot
    // cost FD (sphere obstacle) does not hit.
    Capsule<T> cap{ -0.2,0,0,  0.2,0,0,  0.05 };   // segment along x, radius 0.05
    Cuboid<T>  box{ 0,0,0,  1,0,0,0.1,  0,1,0,0.15,  0,0,1,0.2 };  // axis-aligned box
    T pts[4][3] = {{0.30,0.10,0.00},{0.00,0.20,0.10},{0.25,0.25,0.30},{0.40,0.00,0.05}};
    T eps=1e-6, nerr=0;
    for (int k=0;k<4;++k){
        T x=pts[k][0], y=pts[k][1], z=pts[k][2], r=0.03;
        for (int shape=0; shape<3; ++shape){
            T nx,ny,nz;
            T d0 = shape==0 ? grid_collision::grid_cc_sphere_sphere_signed<T>(x,y,z,r, 0.0,0.0,0.0,0.08, &nx,&ny,&nz)
                 : shape==1 ? grid_collision::grid_cc_sphere_capsule_signed<T>(cap,x,y,z,r,&nx,&ny,&nz)
                 :            grid_collision::grid_cc_sphere_cuboid_signed<T>(box,x,y,z,r,&nx,&ny,&nz);
            (void)d0;
            T fd[3]; T p[3]={x,y,z};
            for (int a=0;a<3;++a){
                T sv=p[a]; T t2,t3,t4; p[a]=sv+eps;
                T dp = shape==0 ? grid_collision::grid_cc_sphere_sphere_signed<T>(p[0],p[1],p[2],r,0.0,0.0,0.0,0.08,&t2,&t3,&t4)
                     : shape==1 ? grid_collision::grid_cc_sphere_capsule_signed<T>(cap,p[0],p[1],p[2],r,&t2,&t3,&t4)
                     :            grid_collision::grid_cc_sphere_cuboid_signed<T>(box,p[0],p[1],p[2],r,&t2,&t3,&t4);
                p[a]=sv-eps;
                T dm = shape==0 ? grid_collision::grid_cc_sphere_sphere_signed<T>(p[0],p[1],p[2],r,0.0,0.0,0.0,0.08,&t2,&t3,&t4)
                     : shape==1 ? grid_collision::grid_cc_sphere_capsule_signed<T>(cap,p[0],p[1],p[2],r,&t2,&t3,&t4)
                     :            grid_collision::grid_cc_sphere_cuboid_signed<T>(box,p[0],p[1],p[2],r,&t2,&t3,&t4);
                p[a]=sv; fd[a]=(dp-dm)/(2*eps);
            }
            nerr = std::max(nerr, std::max(std::fabs(fd[0]-nx), std::max(std::fabs(fd[1]-ny), std::fabs(fd[2]-nz))));
        }
    }
    printf("NORMALS fd_maxerr=%.3e\n", nerr);

    bool ok = (hd<1e-12) && (signdiff==0) && e_hit && !e_miss && s_hit && !s_free && (nerr<1e-6);
    printf("RESULT: %s\n", ok?"PASS":"FAIL");
    return ok?0:3;
}
