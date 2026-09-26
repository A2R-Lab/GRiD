// Reuse the existing analytical FDSVA synthesis and codegen getters, not its
// historical random inputs, timing boundaries, or hard-coded batch sweep.
#define GRID_RELEASE_PIN_HELPERS_ONLY
#include "../baselines/pinocchio/timePinocchio.cpp"
#include "pin_codegen_init.h"
#include "release_pool.h"
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

struct ReleasePin {
    pinocchio::Model model;
    std::unique_ptr<pinocchio::Data> data;
    std::unique_ptr<CodeGenRNEAWithGetRes<float>> rnea;
    std::unique_ptr<DerivedCodeGenRNEADerivatives<float>> grad;
    std::unique_ptr<pinocchio::CodeGenMinv<float>> minv;
    FdsvaSoScratch scratch;
    int op;
    pinocchio::FrameIndex frame;
    ReleasePin(const char *urdf, bool floating, int operation, const char *target): op(operation) {
        if (floating) pinocchio::urdf::buildModel(urdf, pinocchio::JointModelFreeFlyer(), model);
        else pinocchio::urdf::buildModel(urdf, model);
        model.gravity.linear(Eigen::Vector3d(0,0,-9.81));
        data.reset(new pinocchio::Data(model));
        frame = model.getFrameId(target);
        if (op == 7 && frame >= model.frames.size()) throw std::runtime_error("FK target frame missing");
        if (op == 0 || op == 4 || op == 5) {
            rnea.reset(new CodeGenRNEAWithGetRes<float>(model.cast<float>()));
            init_release_codegen(*rnea);
        }
        if (op == 1 || op == 5) {
            grad.reset(new DerivedCodeGenRNEADerivatives<float>(model.cast<float>()));
            init_release_codegen(*grad);
        }
        if (op == 3 || op == 4 || op == 5) {
            minv.reset(new pinocchio::CodeGenMinv<float>(model.cast<float>()));
            init_release_codegen(*minv);
        }
    }
};
static thread_local std::string release_error;
extern "C" const char *pin_release_error() { return release_error.c_str(); }
extern "C" void *pin_release_create(const char *urdf, int floating, int op, const char *target) {
    try { return new ReleasePin(urdf, floating, op, target); }
    catch(const std::exception &e) { release_error=e.what(); return nullptr; }
}
extern "C" void pin_release_close(void *p) { delete static_cast<ReleasePin*>(p); }
template<class M> void copy_matrix(const M &m, double *out) {
    for (int i=0; i<m.rows(); ++i) for(int j=0; j<m.cols(); ++j) *out++ = m(i,j);
}
template<class Tensor> void copy_tensor(const Tensor &t, int n, double *out, bool transpose=false) {
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) for(int k=0;k<n;++k)
        *out++ = transpose ? t(i,k,j) : t(i,j,k);
}

// Doubles written per sample by eval_one for this context's operation.
static int sample_size(const ReleasePin &c) {
    const int n = c.model.nv;
    switch (c.op) {
        case 0: case 4: return n;
        case 1: case 5: return 2*n*n;
        case 2: case 6: return 4*n*n*n;
        case 3: return n*n;
        case 7: return 6;
        default: return -1;
    }
}

// One sample of the context's operation into `output` (sample_size doubles).
// Shared by the single-context ABI and the pool so both paths compute the
// same thing; returns 0, or -2 for an unknown operation.
static int eval_one(ReleasePin &c, const float *q_in, const float *v_in, const float *t_in, double *output) {
    const int nq=c.model.nq, n=c.model.nv;
    Eigen::VectorXf q=Eigen::Map<const Eigen::VectorXf>(q_in,nq);
    Eigen::VectorXf v=Eigen::Map<const Eigen::VectorXf>(v_in,n);
    Eigen::VectorXf t=Eigen::Map<const Eigen::VectorXf>(t_in,n);
    if(c.op==0) { c.rnea->evalFunction(q,v,t); copy_matrix(c.rnea->getRes(),output); }
    else if(c.op==1) {
        c.grad->evalFunction(q,v,t);
        Eigen::MatrixXf J(n,2*n); J << c.grad->getDtauDq(), c.grad->getDtauDv();
        copy_matrix(J,output);
    } else if(c.op==2) {
        c.data->d2tau_dqdq.setZero(); c.data->d2tau_dvdv.setZero();
        c.data->d2tau_dqdv.setZero(); c.data->d2tau_dadq.setZero();
        pinocchio::ComputeRNEASecondOrderDerivatives(c.model,*c.data,q.cast<double>(),v.cast<double>(),t.cast<double>());
        copy_tensor(c.data->d2tau_dqdq,n,output); output+=n*n*n;
        copy_tensor(c.data->d2tau_dvdv,n,output); output+=n*n*n;
        copy_tensor(c.data->d2tau_dqdv,n,output,true); output+=n*n*n;
        copy_tensor(c.data->d2tau_dadq,n,output);
    } else if(c.op==3 || c.op==4 || c.op==5) {
        c.minv->evalFunction(q);
        // Installed CodeGenMinv allocates nv x nq, even though only
        // its leading nv x nv triangle is defined. Free bases have
        // nq=nv+1: copying/multiplying the full buffer is incorrect.
        Eigen::MatrixXf mi=c.minv->Minv.topLeftCorner(n,n);
        mi.triangularView<Eigen::StrictlyLower>()=mi.transpose().triangularView<Eigen::StrictlyLower>();
        if(c.op==3) { copy_matrix(mi,output); return 0; }
        c.rnea->evalFunction(q,v,Eigen::VectorXf::Zero(n));
        Eigen::VectorXf acc=mi*(t-c.rnea->getRes());
        if(c.op==4) { copy_matrix(acc,output); }
        else {
            c.grad->evalFunction(q,v,acc);
            Eigen::MatrixXf J(n,2*n); J << -mi*c.grad->getDtauDq(), -mi*c.grad->getDtauDv();
            copy_matrix(J,output);
        }
    } else if(c.op==6) {
        fdsvaSoSynth_one<float>(c.model,*c.data,q,v,t,c.scratch);
        copy_tensor(c.scratch.daba_dqdq,n,output); output+=n*n*n;
        copy_tensor(c.scratch.daba_dvdq,n,output); output+=n*n*n;
        copy_tensor(c.scratch.daba_dvdv,n,output); output+=n*n*n;
        copy_tensor(c.scratch.daba_dtdq,n,output);
    } else if(c.op==7) {
        pinocchio::forwardKinematics(c.model,*c.data,q.cast<double>());
        pinocchio::updateFramePlacements(c.model,*c.data);
        const auto &p=c.data->oMf[c.frame]; const auto &r=p.rotation();
        for(int i=0;i<3;++i) *output++=p.translation()[i];
        *output++=std::atan2(r(2,1),r(2,2));
        *output++=std::atan2(-r(2,0),std::sqrt(r(2,2)*r(2,2)+r(2,1)*r(2,1)));
        *output++=std::atan2(r(1,0),r(0,0));
    } else return -2;
    return 0;
}

static int eval_range(ReleasePin &c, const float *qs, const float *vs, const float *ts, int start, int stop, double *output) {
    const int nq=c.model.nq, n=c.model.nv, size=sample_size(c);
    for (int b=start; b<stop; ++b) {
        int rc = eval_one(c, qs+b*nq, vs+b*n, ts+b*n, output+(size_t)b*size);
        if (rc) return rc;
    }
    return 0;
}

extern "C" int pin_release_eval(void *ptr, const float *qs, const float *vs,
                                  const float *ts, int batch, double *output) {
    if(!ptr || !qs || !vs || !ts || !output || batch < 1) return -1;
    try { return eval_range(*static_cast<ReleasePin*>(ptr), qs, vs, ts, 0, batch, output); }
    catch(const std::exception &e) { release_error=e.what(); return -3; }
}

// ── Persistent pool: N independent contexts, batch split into contiguous
// slices, slice 0 on the caller, the rest on already-running threads. ──
struct ReleasePinPool {
    std::vector<std::unique_ptr<ReleasePin>> contexts;
    std::unique_ptr<ReleasePool> pool;
};

extern "C" void *pin_release_pool_create(const char *urdf, int floating, int op, const char *target, int threads) {
    if (threads < 1) { release_error = "pool needs at least one thread"; return nullptr; }
    try {
        auto *p = new ReleasePinPool;
        for (int k = 0; k < threads; ++k) p->contexts.emplace_back(new ReleasePin(urdf, floating, op, target));
        p->pool.reset(new ReleasePool((std::size_t)threads - 1));
        return p;
    } catch(const std::exception &e) { release_error=e.what(); return nullptr; }
}
extern "C" int pin_release_pool_threads(void *ptr) {
    return ptr ? (int)static_cast<ReleasePinPool*>(ptr)->contexts.size() : 0;
}
extern "C" void pin_release_pool_close(void *ptr) { delete static_cast<ReleasePinPool*>(ptr); }

// Evaluate `batch` samples on `active` threads (clamped to the pool size and
// to the batch). Contiguous slices of near-equal size; every slice has its own
// model/data/codegen context, so nothing is shared between threads but the
// input and output arrays at disjoint offsets.
extern "C" int pin_release_pool_eval(void *ptr, const float *qs, const float *vs, const float *ts,
                                     int batch, double *output, int active) {
    if(!ptr || !qs || !vs || !ts || !output || batch < 1 || active < 1) return -1;
    auto &p = *static_cast<ReleasePinPool*>(ptr);
    const int count = std::min<int>(active, std::min<int>((int)p.contexts.size(), batch));
    std::vector<int> rcs((size_t)count, 0);
    std::vector<std::string> errors((size_t)count);
    p.pool->run((std::size_t)count, [&](std::size_t slot) {
        const int start = (int)((long long)batch * (long long)slot / count);
        const int stop = (int)((long long)batch * (long long)(slot + 1) / count);
        try { rcs[slot] = eval_range(*p.contexts[slot], qs, vs, ts, start, stop, output); }
        catch(const std::exception &e) { rcs[slot] = -3; errors[slot] = e.what(); }
    });
    for (int k = 0; k < count; ++k) {
        if (rcs[(size_t)k]) { release_error = errors[(size_t)k]; return rcs[(size_t)k]; }
    }
    return 0;
}
