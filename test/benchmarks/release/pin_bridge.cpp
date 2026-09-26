// Reuse the existing analytical FDSVA synthesis and codegen getters, not its
// historical random inputs, timing boundaries, or hard-coded batch sweep.
#define GRID_RELEASE_PIN_HELPERS_ONLY
#include "../baselines/pinocchio/timePinocchio.cpp"
#include "pin_codegen_init.h"
#include "release_pool.h"
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// Operations (index = Python OPS order):
//   0 inverse_dynamics  1 inverse_dynamics_gradient  2 idsva_so  3 minv
//   4 forward_dynamics  5 forward_dynamics_gradient  6 fdsva_so  7 end_effector_pose
//   8 crba  9 nonlinear_effects  10 generalized_gravity  11 ccrba  12 coriolis_matrix
// Two modes share every input, output layout and precision policy:
//   codegen = CppADCodeGen libraries (RNEA, its derivatives, Minv, CRBA) in fp32,
//             FD / grad FD / bias / gravity composed from them;
//   plain   = Pinocchio's standard templated algorithms instantiated in fp32
//             (the API most users call), no code generation.
// The second-order operations (2, 6) and FK (7) run the same analytical fp64
// path in both modes (there is no codegen for them).
struct ReleasePin {
    pinocchio::Model model;
    pinocchio::ModelTpl<float> modelf;
    std::unique_ptr<pinocchio::Data> data;
    std::unique_ptr<pinocchio::DataTpl<float>> dataf;
    std::unique_ptr<CodeGenRNEAWithGetRes<float>> rnea;
    std::unique_ptr<DerivedCodeGenRNEADerivatives<float>> grad;
    std::unique_ptr<pinocchio::CodeGenMinv<float>> minv;
    std::unique_ptr<pinocchio::CodeGenCRBA<float>> crba_gen;
    FdsvaSoScratch scratch;
    int op;
    bool plain;
    pinocchio::FrameIndex frame;
    ReleasePin(const char *urdf, bool floating, int operation, const char *target, bool plain_mode)
        : op(operation), plain(plain_mode) {
        if (floating) pinocchio::urdf::buildModel(urdf, pinocchio::JointModelFreeFlyer(), model);
        else pinocchio::urdf::buildModel(urdf, model);
        model.gravity.linear(Eigen::Vector3d(0,0,-9.81));
        data.reset(new pinocchio::Data(model));
        modelf = model.cast<float>();
        dataf.reset(new pinocchio::DataTpl<float>(modelf));
        frame = model.getFrameId(target);
        if (op == 7 && frame >= model.frames.size()) throw std::runtime_error("FK target frame missing");
        if (op == 11 || op == 12) {
            if (!plain) throw std::runtime_error("no CppADCodeGen class for this operation; use pinocchio_plain");
        }
        if (plain) return;
        if (op == 0 || op == 4 || op == 5 || op == 9 || op == 10) {
            rnea.reset(new CodeGenRNEAWithGetRes<float>(modelf));
            init_release_codegen(*rnea);
        }
        if (op == 1 || op == 5) {
            grad.reset(new DerivedCodeGenRNEADerivatives<float>(modelf));
            init_release_codegen(*grad);
        }
        if (op == 3 || op == 4 || op == 5) {
            minv.reset(new pinocchio::CodeGenMinv<float>(modelf));
            init_release_codegen(*minv);
        }
        if (op == 8) {
            crba_gen.reset(new pinocchio::CodeGenCRBA<float>(modelf));
            init_release_codegen(*crba_gen);
        }
    }
};
static thread_local std::string release_error;
extern "C" const char *pin_release_error() { return release_error.c_str(); }
extern "C" void *pin_release_create(const char *urdf, int floating, int op, const char *target) {
    try { return new ReleasePin(urdf, floating, op, target, false); }
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
        case 0: case 4: case 9: case 10: return n;
        case 1: case 5: return 2*n*n;
        case 2: case 6: return 4*n*n*n;
        case 3: case 8: case 12: return n*n;
        case 7: return 6;
        case 11: return 6*n + 6;   // centroidal momentum matrix Ag (6 x nv) then the momentum hg (6)
        default: return -1;
    }
}

// Pinocchio's CRBA / Minv fill the upper triangle; mirror it before use.
static Eigen::MatrixXf symmetrized_upper(const Eigen::MatrixXf &m) {
    Eigen::MatrixXf out = m;
    out.triangularView<Eigen::StrictlyLower>() = out.transpose().triangularView<Eigen::StrictlyLower>();
    return out;
}

// Standard (non-codegen) Pinocchio algorithms in fp32; same outputs as the
// codegen path. Returns 0, or -2 for an operation this mode does not cover.
static int eval_plain(ReleasePin &c, const Eigen::VectorXf &q, const Eigen::VectorXf &v, const Eigen::VectorXf &t, double *output) {
    const int n=c.model.nv;
    auto &m = c.modelf; auto &d = *c.dataf;
    if(c.op==0) { copy_matrix(pinocchio::rnea(m,d,q,v,t),output); }
    else if(c.op==1) {
        pinocchio::computeRNEADerivatives(m,d,q,v,t);
        Eigen::MatrixXf J(n,2*n); J << d.dtau_dq, d.dtau_dv;
        copy_matrix(J,output);
    } else if(c.op==3) {
        pinocchio::computeMinverse(m,d,q);
        copy_matrix(symmetrized_upper(d.Minv),output);
    } else if(c.op==4) { copy_matrix(pinocchio::aba(m,d,q,v,t),output); }
    else if(c.op==5) {
        pinocchio::computeABADerivatives(m,d,q,v,t);
        Eigen::MatrixXf J(n,2*n); J << d.ddq_dq, d.ddq_dv;
        copy_matrix(J,output);
    } else if(c.op==8) { pinocchio::crba(m,d,q); copy_matrix(symmetrized_upper(d.M),output); }
    else if(c.op==9) { copy_matrix(pinocchio::nonLinearEffects(m,d,q,v),output); }
    else if(c.op==10) { copy_matrix(pinocchio::computeGeneralizedGravity(m,d,q),output); }
    else if(c.op==11) {
        // Ag from ccrba; the momentum is h = Ag * v by definition (the float
        // instantiation did not expose a filled data.hg).
        pinocchio::ccrba(m,d,q,v); copy_matrix(d.Ag,output); output+=6*n;
        Eigen::VectorXf h = d.Ag * v; copy_matrix(h,output);
    }
    else if(c.op==12) { pinocchio::computeCoriolisMatrix(m,d,q,v); copy_matrix(d.C,output); }
    else return -2;
    return 0;
}

// One sample of the context's operation into `output` (sample_size doubles).
// Shared by the single-context ABI and the pool so both paths compute the
// same thing; returns 0, or -2 for an unknown operation.
static int eval_one(ReleasePin &c, const float *q_in, const float *v_in, const float *t_in, double *output) {
    const int nq=c.model.nq, n=c.model.nv;
    Eigen::VectorXf q=Eigen::Map<const Eigen::VectorXf>(q_in,nq);
    Eigen::VectorXf v=Eigen::Map<const Eigen::VectorXf>(v_in,n);
    Eigen::VectorXf t=Eigen::Map<const Eigen::VectorXf>(t_in,n);
    const bool analytical = (c.op==2 || c.op==6 || c.op==7);
    if(c.plain && !analytical) return eval_plain(c,q,v,t,output);
    if(c.op==0) { c.rnea->evalFunction(q,v,t); copy_matrix(c.rnea->getRes(),output); }
    else if(c.op==9) { c.rnea->evalFunction(q,v,Eigen::VectorXf::Zero(n)); copy_matrix(c.rnea->getRes(),output); }
    else if(c.op==10) { c.rnea->evalFunction(q,Eigen::VectorXf::Zero(n),Eigen::VectorXf::Zero(n)); copy_matrix(c.rnea->getRes(),output); }
    else if(c.op==8) { c.crba_gen->evalFunction(q); copy_matrix(symmetrized_upper(c.crba_gen->M.topLeftCorner(n,n)),output); }
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

extern "C" void *pin_release_pool_create(const char *urdf, int floating, int op, const char *target, int threads, int plain) {
    if (threads < 1) { release_error = "pool needs at least one thread"; return nullptr; }
    try {
        auto *p = new ReleasePinPool;
        for (int k = 0; k < threads; ++k) p->contexts.emplace_back(new ReleasePin(urdf, floating, op, target, plain != 0));
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
