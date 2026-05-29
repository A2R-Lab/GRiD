/*
 * This timing code is based on the benchmarking code as written in the Pinocchio repository
 * g++ -std=c++11 timePinocchio.cpp -o timePinocchio -O3 $(pkg-config --cflags --libs pinocchio cppadcg)
 * example usage: ./timePinocchio urdfs/atlas.urdf True iiwa_link_ee
 */
#include "../util/experiment_helpers.h"
#include "ReusableThreads/ReusableThreads.h"

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/rnea-second-order-derivatives.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"

#ifdef HAVE_CPPADCG
#include "pinocchio/codegen/cppadcg.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"
#endif // HAVE_CPPADCG

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/multibody/sample-models.hpp"

#include "pinocchio/container/aligned-vector.hpp"

#include <Eigen/StdVector>

#ifdef HAVE_CPPADCG
#include "../util/getters/GetResRNEA.hpp"
#include "../util/getters/GettersDerivatives.hpp"
#endif

using namespace Eigen;
using namespace pinocchio;

#define time_delta_us_timespec(start,end) (1e6*static_cast<double>(end.tv_sec - start.tv_sec)+1e-3*static_cast<double>(end.tv_nsec - start.tv_nsec))

// ---------------------------------------------------------------------------
// Threading helpers — codegen algorithms (require CppADCodeGen)
// ---------------------------------------------------------------------------

#ifdef HAVE_CPPADCG

template<typename T>
void inverseDynamicsThreaded_codegen_inner(CodeGenRNEAWithGetRes<T> *rnea_code_gen, int nq, int nv, \
                                           Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds, int tid, int kStart, int kMax){
    Matrix<T, Eigen::Dynamic, 1> zeros = Matrix<T, Eigen::Dynamic, 1>::Zero(nv);
    for(int k = kStart; k < kMax; k++){
        rnea_code_gen->evalFunction(qs[k],qds[k],zeros);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void inverseDynamicsThreaded_codegen(CodeGenRNEAWithGetRes<T> **rnea_code_gen_arr, int nq, int nv, \
                                     Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds, ReusableThreads<NUM_THREADS> *threads){
        constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
        for (int tid = 0; tid < ET; tid++){
            int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
            if(tid == ET-1){kMax = NUM_TIME_STEPS;}
            threads->addTask(tid, &inverseDynamicsThreaded_codegen_inner<T>, std::ref(rnea_code_gen_arr[tid]), nq, nv,
                                                                             std::ref(qs), std::ref(qds), tid, kStart, kMax);
        }
        threads->sync();
}

template<typename T>
void minvThreaded_codegen_inner(CodeGenMinv<T> *minv_code_gen, int nq, int nv, Matrix<T, Eigen::Dynamic, 1> *qs, int tid, int kStart, int kMax){
    for(int k = kStart; k < kMax; k++){
        minv_code_gen->evalFunction(qs[k]);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void minvThreaded_codegen(CodeGenMinv<T> **minv_code_gen_arr, int nq, int nv, Matrix<T, Eigen::Dynamic, 1> *qs, ReusableThreads<NUM_THREADS> *threads){
        constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
        for (int tid = 0; tid < ET; tid++){
            int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
            if(tid == ET-1){kMax = NUM_TIME_STEPS;}
            threads->addTask(tid, &minvThreaded_codegen_inner<T>, std::ref(minv_code_gen_arr[tid]), nq, nv, std::ref(qs), tid, kStart, kMax);
        }
        threads->sync();
}

template<typename T>
void forwardDynamicsThreaded_codegen_inner(CodeGenMinv<T> *minv_code_gen, CodeGenRNEAWithGetRes<T> *rnea_code_gen, int nq, int nv, \
                                           Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds, Matrix<T, Eigen::Dynamic, 1> *qdds, \
                                           Matrix<T, Eigen::Dynamic, 1> *us, int tid, int kStart, int kMax){
    Matrix<T, Eigen::Dynamic, 1> zeros = Matrix<T, Eigen::Dynamic, 1>::Zero(nv);
    for(int k = kStart; k < kMax; k++){
        minv_code_gen->evalFunction(qs[k]);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> minv = minv_code_gen->Minv.block(0,0,nv,nv);
        minv.template triangularView<Eigen::StrictlyLower>() =
            minv.transpose().template triangularView<Eigen::StrictlyLower>();
        rnea_code_gen->evalFunction(qs[k],qds[k],zeros);
        qdds[k].noalias() = minv*(us[k] - rnea_code_gen->getRes());
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void forwardDynamicsThreaded_codegen(CodeGenMinv<T> **minv_code_gen_arr, CodeGenRNEAWithGetRes<T> **rnea_code_gen_arr, int nq, int nv, \
                                     Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds, Matrix<T, Eigen::Dynamic, 1> *qdds, \
                                     Matrix<T, Eigen::Dynamic, 1> *us, ReusableThreads<NUM_THREADS> *threads){
        constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
        for (int tid = 0; tid < ET; tid++){
            int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
            if(tid == ET-1){kMax = NUM_TIME_STEPS;}
            threads->addTask(tid, &forwardDynamicsThreaded_codegen_inner<T>, std::ref(minv_code_gen_arr[tid]),
                                                                             std::ref(rnea_code_gen_arr[tid]), nq, nv,
                                                                             std::ref(qs), std::ref(qds), std::ref(qdds), std::ref(us),
                                                                             tid, kStart, kMax);
        }
        threads->sync();
}

template<typename T>
void inverseDynamicsGradientThreaded_codegen_inner(DerivedCodeGenRNEADerivatives<T> *rnea_derivatives_code_gen, \
                                                   int nq, int nv, Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds, \
                                                   int tid, int kStart, int kMax){
    Matrix<T, Eigen::Dynamic, 1> zeros = Matrix<T, Eigen::Dynamic, 1>::Zero(nv);
    for(int k = kStart; k < kMax; k++){
        rnea_derivatives_code_gen->evalFunction(qs[k],qds[k],zeros);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void inverseDynamicsGradientThreaded_codegen(DerivedCodeGenRNEADerivatives<T> **rnea_derivatives_code_gen_arr, \
                                             int nq, int nv, Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                                             ReusableThreads<NUM_THREADS> *threads){
        constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
        for (int tid = 0; tid < ET; tid++){
            int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
            if(tid == ET-1){kMax = NUM_TIME_STEPS;}
            threads->addTask(tid, &inverseDynamicsGradientThreaded_codegen_inner<T>, std::ref(rnea_derivatives_code_gen_arr[tid]),
                                                                                     nq, nv, std::ref(qs), std::ref(qds), tid, kStart, kMax);
        }
        threads->sync();
}

template<typename T>
void forwardDynamicsGradientThreaded_codegen_inner(DerivedCodeGenRNEADerivatives<T> *rnea_derivatives_code_gen, \
                                                   CodeGenMinv<T> *minv_code_gen, CodeGenRNEAWithGetRes<T> *rnea_code_gen, \
                                                   int nq, int nv, Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *dqdd_dqs, Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *dqdd_dvs, \
                                                   Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds, Matrix<T, Eigen::Dynamic, 1> *us, \
                                                   int tid, int kStart, int kMax){
    Matrix<T, Eigen::Dynamic, 1> zeros = Matrix<T, Eigen::Dynamic, 1>::Zero(nv);
    for(int k = kStart; k < kMax; k++){
        minv_code_gen->evalFunction(qs[k]);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> minv = minv_code_gen->Minv.block(0,0,nv,nv);
        minv.template triangularView<Eigen::StrictlyLower>() =
            minv.transpose().template triangularView<Eigen::StrictlyLower>();
        rnea_code_gen->evalFunction(qs[k],qds[k],zeros);
        Matrix<T, Eigen::Dynamic, 1> qdd = minv*(us[k] - rnea_code_gen->getRes());
        rnea_derivatives_code_gen->evalFunction(qs[k],qds[k],qdd);
        dqdd_dqs[k].noalias() = -(minv)*(rnea_derivatives_code_gen->getDtauDq());
        dqdd_dvs[k].noalias() = -(minv)*(rnea_derivatives_code_gen->getDtauDv());
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void forwardDynamicsGradientThreaded_codegen(DerivedCodeGenRNEADerivatives<T> **rnea_derivatives_code_gen_arr, \
                                             CodeGenMinv<T> **minv_code_gen_arr, CodeGenRNEAWithGetRes<T> **rnea_code_gen_arr, \
                                             int nq, int nv, Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *dqdd_dqs, Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *dqdd_dvs, \
                                             Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds, Matrix<T, Eigen::Dynamic, 1> *us, \
                                             ReusableThreads<NUM_THREADS> *threads){
        constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
        for (int tid = 0; tid < ET; tid++){
            int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
            if(tid == ET-1){kMax = NUM_TIME_STEPS;}
            threads->addTask(tid, &forwardDynamicsGradientThreaded_codegen_inner<T>, std::ref(rnea_derivatives_code_gen_arr[tid]),
                                                                                     std::ref(minv_code_gen_arr[tid]),
                                                                                     std::ref(rnea_code_gen_arr[tid]), nq, nv,
                                                                                     std::ref(dqdd_dqs), std::ref(dqdd_dvs),
                                                                                     std::ref(qs), std::ref(qds), std::ref(us),
                                                                                     tid, kStart, kMax);
    }
        threads->sync();
}

template<typename T>
void abaThreaded_codegen_inner(CodeGenABA<T> *aba_code_gen, int nq, int nv, \
                                           Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds, int tid, int kStart, int kMax){
    Matrix<T, Eigen::Dynamic, 1> zeros = Matrix<T, Eigen::Dynamic, 1>::Zero(nv);
    for(int k = kStart; k < kMax; k++){
        aba_code_gen->evalFunction(qs[k],qds[k],zeros);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void abaThreaded_codegen(CodeGenABA<T> **aba_code_gen_arr, int nq, int nv, \
                                     Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds, ReusableThreads<NUM_THREADS> *threads){
        constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
        for (int tid = 0; tid < ET; tid++){
            int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
            if(tid == ET-1){kMax = NUM_TIME_STEPS;}
            threads->addTask(tid, &abaThreaded_codegen_inner<T>, std::ref(aba_code_gen_arr[tid]), nq, nv,
                                                                             std::ref(qs), std::ref(qds), tid, kStart, kMax);
        }
        threads->sync();
}

// ---------------------------------------------------------------------------
// Threading helpers — more codegen algorithms
// ---------------------------------------------------------------------------

template<typename T>
void crbaThreaded_codegen_inner(CodeGenCRBA<T> *crba_code_gen, int nq, int nv,
                                Matrix<T, Eigen::Dynamic, 1> *qs, int tid, int kStart, int kMax){
    for(int k = kStart; k < kMax; k++){
        crba_code_gen->evalFunction(qs[k]);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void crbaThreaded_codegen(CodeGenCRBA<T> **crba_code_gen_arr, int nq, int nv,
                           Matrix<T, Eigen::Dynamic, 1> *qs, ReusableThreads<NUM_THREADS> *threads){
    constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
    for(int tid = 0; tid < ET; tid++){
        int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
        if(tid == ET-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &crbaThreaded_codegen_inner<T>, std::ref(crba_code_gen_arr[tid]), nq, nv,
                                                              std::ref(qs), tid, kStart, kMax);
    }
    threads->sync();
}

#endif // HAVE_CPPADCG

// ---------------------------------------------------------------------------
// Threading helpers — direct API core dynamics (no CppADCodeGen needed)
// ---------------------------------------------------------------------------

template<typename T>
void idDirectThreaded_inner(const pinocchio::Model *model, pinocchio::Data *data,
                             Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                             Matrix<T, Eigen::Dynamic, 1> *qdds, int tid, int kStart, int kMax){
    for(int k = kStart; k < kMax; k++){
        pinocchio::rnea(*model, *data, qs[k].template cast<double>(), qds[k].template cast<double>(), qdds[k].template cast<double>());
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void idDirectThreaded(const pinocchio::Model *model, pinocchio::Data *datas,
                       Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                       Matrix<T, Eigen::Dynamic, 1> *qdds, ReusableThreads<NUM_THREADS> *threads){
    constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
    for(int tid = 0; tid < ET; tid++){
        int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
        if(tid == ET-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &idDirectThreaded_inner<T>, model, &datas[tid],
                         std::ref(qs), std::ref(qds), std::ref(qdds), tid, kStart, kMax);
    }
    threads->sync();
}

template<typename T>
void abaDirectThreaded_inner(const pinocchio::Model *model, pinocchio::Data *data,
                              Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                              Matrix<T, Eigen::Dynamic, 1> *us, int tid, int kStart, int kMax){
    for(int k = kStart; k < kMax; k++){
        pinocchio::aba(*model, *data, qs[k].template cast<double>(), qds[k].template cast<double>(), us[k].template cast<double>());
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void abaDirectThreaded(const pinocchio::Model *model, pinocchio::Data *datas,
                        Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                        Matrix<T, Eigen::Dynamic, 1> *us, ReusableThreads<NUM_THREADS> *threads){
    constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
    for(int tid = 0; tid < ET; tid++){
        int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
        if(tid == ET-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &abaDirectThreaded_inner<T>, model, &datas[tid],
                         std::ref(qs), std::ref(qds), std::ref(us), tid, kStart, kMax);
    }
    threads->sync();
}

template<typename T>
void crbaDirectThreaded_inner(const pinocchio::Model *model, pinocchio::Data *data,
                               Matrix<T, Eigen::Dynamic, 1> *qs, int tid, int kStart, int kMax){
    for(int k = kStart; k < kMax; k++){
        pinocchio::crba(*model, *data, qs[k].template cast<double>());
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void crbaDirectThreaded(const pinocchio::Model *model, pinocchio::Data *datas,
                         Matrix<T, Eigen::Dynamic, 1> *qs, ReusableThreads<NUM_THREADS> *threads){
    constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
    for(int tid = 0; tid < ET; tid++){
        int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
        if(tid == ET-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &crbaDirectThreaded_inner<T>, model, &datas[tid],
                         std::ref(qs), tid, kStart, kMax);
    }
    threads->sync();
}

template<typename T>
void minvDirectThreaded_inner(const pinocchio::Model *model, pinocchio::Data *data,
                               Matrix<T, Eigen::Dynamic, 1> *qs, int tid, int kStart, int kMax){
    for(int k = kStart; k < kMax; k++){
        pinocchio::crba(*model, *data, qs[k].template cast<double>());
        pinocchio::cholesky::decompose(*model, *data);
        pinocchio::cholesky::computeMinv(*model, *data);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void minvDirectThreaded(const pinocchio::Model *model, pinocchio::Data *datas,
                         Matrix<T, Eigen::Dynamic, 1> *qs, ReusableThreads<NUM_THREADS> *threads){
    constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
    for(int tid = 0; tid < ET; tid++){
        int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
        if(tid == ET-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &minvDirectThreaded_inner<T>, model, &datas[tid],
                         std::ref(qs), tid, kStart, kMax);
    }
    threads->sync();
}

template<typename T>
void idDuDirectThreaded_inner(const pinocchio::Model *model, pinocchio::Data *data,
                               Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                               Matrix<T, Eigen::Dynamic, 1> *qdds, int tid, int kStart, int kMax){
    Eigen::MatrixXd dtau_dq = Eigen::MatrixXd::Zero(model->nv, model->nv);
    Eigen::MatrixXd dtau_dv = Eigen::MatrixXd::Zero(model->nv, model->nv);
    Eigen::MatrixXd dtau_da = Eigen::MatrixXd::Zero(model->nv, model->nv);
    for(int k = kStart; k < kMax; k++){
        pinocchio::computeRNEADerivatives(*model, *data,
            qs[k].template cast<double>(), qds[k].template cast<double>(), qdds[k].template cast<double>(),
            dtau_dq, dtau_dv, dtau_da);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void idDuDirectThreaded(const pinocchio::Model *model, pinocchio::Data *datas,
                         Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                         Matrix<T, Eigen::Dynamic, 1> *qdds, ReusableThreads<NUM_THREADS> *threads){
    constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
    for(int tid = 0; tid < ET; tid++){
        int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
        if(tid == ET-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &idDuDirectThreaded_inner<T>, model, &datas[tid],
                         std::ref(qs), std::ref(qds), std::ref(qdds), tid, kStart, kMax);
    }
    threads->sync();
}

template<typename T>
void fdDuDirectThreaded_inner(const pinocchio::Model *model, pinocchio::Data *data,
                               Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                               Matrix<T, Eigen::Dynamic, 1> *us, int tid, int kStart, int kMax){
    Eigen::MatrixXd ddq_dq = Eigen::MatrixXd::Zero(model->nv, model->nv);
    Eigen::MatrixXd ddq_dv = Eigen::MatrixXd::Zero(model->nv, model->nv);
    Eigen::MatrixXd ddq_dtau = Eigen::MatrixXd::Zero(model->nv, model->nv);
    for(int k = kStart; k < kMax; k++){
        pinocchio::computeABADerivatives(*model, *data,
            qs[k].template cast<double>(), qds[k].template cast<double>(), us[k].template cast<double>(),
            ddq_dq, ddq_dv, ddq_dtau);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void fdDuDirectThreaded(const pinocchio::Model *model, pinocchio::Data *datas,
                         Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                         Matrix<T, Eigen::Dynamic, 1> *us, ReusableThreads<NUM_THREADS> *threads){
    constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
    for(int tid = 0; tid < ET; tid++){
        int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
        if(tid == ET-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &fdDuDirectThreaded_inner<T>, model, &datas[tid],
                         std::ref(qs), std::ref(qds), std::ref(us), tid, kStart, kMax);
    }
    threads->sync();
}

// ---------------------------------------------------------------------------
// Threading helpers — direct API algorithms (no CppADCodeGen needed)
// ---------------------------------------------------------------------------

template<typename T>
void eePoseThreaded_inner(const pinocchio::Model *model, pinocchio::Data *data,
                           pinocchio::FrameIndex frame_id,
                           Matrix<T, Eigen::Dynamic, 1> *qs, int tid, int kStart, int kMax){
    for(int k = kStart; k < kMax; k++){
        pinocchio::forwardKinematics(*model, *data, qs[k].template cast<double>());
        pinocchio::updateFramePlacements(*model, *data);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void eePoseThreaded(const pinocchio::Model *model, pinocchio::Data *datas,
                     pinocchio::FrameIndex frame_id,
                     Matrix<T, Eigen::Dynamic, 1> *qs, ReusableThreads<NUM_THREADS> *threads){
    constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
    for(int tid = 0; tid < ET; tid++){
        int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
        if(tid == ET-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &eePoseThreaded_inner<T>, model, &datas[tid], frame_id,
                                                        std::ref(qs), tid, kStart, kMax);
    }
    threads->sync();
}

template<typename T>
void eePoseGradientThreaded_inner(const pinocchio::Model *model, pinocchio::Data *data,
                                   pinocchio::FrameIndex frame_id,
                                   Matrix<T, Eigen::Dynamic, 1> *qs, int tid, int kStart, int kMax){
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, model->nv);
    for(int k = kStart; k < kMax; k++){
        pinocchio::computeJointJacobians(*model, *data, qs[k].template cast<double>());
        pinocchio::getFrameJacobian(*model, *data, frame_id, pinocchio::LOCAL, J);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void eePoseGradientThreaded(const pinocchio::Model *model, pinocchio::Data *datas,
                              pinocchio::FrameIndex frame_id,
                              Matrix<T, Eigen::Dynamic, 1> *qs, ReusableThreads<NUM_THREADS> *threads){
    constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
    for(int tid = 0; tid < ET; tid++){
        int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
        if(tid == ET-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &eePoseGradientThreaded_inner<T>, model, &datas[tid], frame_id,
                                                                 std::ref(qs), tid, kStart, kMax);
    }
    threads->sync();
}

// ee_pose_hessian: pinocchio's analytic kinematic Hessian per joint, then
// returned as a (6, nv, nv) Tensor in LOCAL_WORLD_ALIGNED frame to match
// GRiD's d/dv tangent convention. computeForwardKinematicsDerivatives must
// run first (it sets up the data state computeJointKinematicHessians needs);
// then getJointKinematicHessian(model, data, joint_id, ref_frame, H_out)
// extracts the analytic 3D tensor for a chosen joint.
template<typename T>
void eePoseHessianThreaded_inner(const pinocchio::Model *model, pinocchio::Data *data,
                                  pinocchio::FrameIndex frame_id,
                                  Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                                  int tid, int kStart, int kMax){
    Eigen::VectorXd a_zero = Eigen::VectorXd::Zero(model->nv);
    pinocchio::JointIndex joint_id = model->frames[frame_id].parent;
    Eigen::Tensor<double, 3> H(6, model->nv, model->nv);
    for(int k = kStart; k < kMax; k++){
        pinocchio::computeForwardKinematicsDerivatives(*model, *data,
            qs[k].template cast<double>(), qds[k].template cast<double>(), a_zero);
        pinocchio::computeJointKinematicHessians(*model, *data);
        pinocchio::getJointKinematicHessian(*model, *data, joint_id,
            pinocchio::LOCAL_WORLD_ALIGNED, H);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void eePoseHessianThreaded(const pinocchio::Model *model, pinocchio::Data *datas,
                            pinocchio::FrameIndex frame_id,
                            Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                            ReusableThreads<NUM_THREADS> *threads){
    constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
    for(int tid = 0; tid < ET; tid++){
        int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
        if(tid == ET-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &eePoseHessianThreaded_inner<T>, model, &datas[tid], frame_id,
                                                                 std::ref(qs), std::ref(qds), tid, kStart, kMax);
    }
    threads->sync();
}

template<typename T>
void idsvaSoThreaded_inner(const pinocchio::Model *model, pinocchio::Data *data,
                            Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                            int tid, int kStart, int kMax){
    Eigen::VectorXd a_zero = Eigen::VectorXd::Zero(model->nv);
    for(int k = kStart; k < kMax; k++){
        pinocchio::ComputeRNEASecondOrderDerivatives(*model, *data,
            qs[k].template cast<double>(), qds[k].template cast<double>(), a_zero);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void idsvaSoThreaded(const pinocchio::Model *model, pinocchio::Data *datas,
                      Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                      ReusableThreads<NUM_THREADS> *threads){
    constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
    for(int tid = 0; tid < ET; tid++){
        int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
        if(tid == ET-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &idsvaSoThreaded_inner<T>, model, &datas[tid],
                                                          std::ref(qs), std::ref(qds), tid, kStart, kMax);
    }
    threads->sync();
}

// ---------------------------------------------------------------------------
// FDSVA_SO synthesis: pinocchio has no direct fdsva_so. We compose it from
// pinocchio primitives following Singh/Carpentier (Second Order Derivatives of
// Rigid Body Dynamics, 2022) and the formulation already used by the
// equivalence harness in RBDReference/equivalents/pinocchio_backend.py::fdsva_so.
//
// Inputs:  q, qd, u    (qdd is computed via ABA)
// Outputs: daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq (rank-3 (nv,nv,nv) tensors)
//
// Pipeline per sample:
//   1. ABA(q, qd, u)                                   -> qdd                (data.ddq)
//   2. ComputeRNEASecondOrderDerivatives(q, qd, qdd)   -> d2tau_d{qq,vv,qv}, dM/dq (data.d2tau_dadq)
//   3. computeABADerivatives(q, qd, u)                 -> fd_dq, fd_dqd, Minv (upper)
//      Symmetrize Minv.
//   4. Contractions (Python einsum notation, ":" = sum):
//        T1[i,j,k] = sum_l dM_dq[i,l,k] * fd_dq[l,j]
//        daba_dqdq[i,j,k] = -sum_l Minv[i,l] * ( d2tau_dqdq[l,j,k] + T1[l,j,k] + T1[l,k,j] )
//        T2[i,j,k] = sum_l dM_dq[i,l,k] * fd_dqd[l,j]
//        daba_dvdq[i,j,k] = -sum_l Minv[i,l] * ( d2tau_dvdq[l,j,k] + T2[l,j,k] )
//          where d2tau_dvdq = d2tau_dqdv.transpose(1,2)
//        daba_dvdv[i,j,k] = -sum_l Minv[i,l] * d2tau_dvdv[l,j,k]
//        T3[i,j,k] = sum_l dM_dq[i,l,k] * Minv[l,j]
//        daba_dtdq[i,j,k] = -sum_l Minv[i,l] * T3[l,j,k]
//
// We allocate scratch buffers once per thread and reuse them across the
// timed loop. The synthesis dominates the SO RNEA cost for large robots
// (multiple nv^4 contractions) but is the only way to get a pinocchio-based
// fdsva_so baseline — no library function exposes it directly. This is what
// any downstream pinocchio user would write.
// ---------------------------------------------------------------------------

struct FdsvaSoScratch {
    int nv = 0;
    Eigen::MatrixXd Minv;        // nv x nv (symmetric)
    Eigen::MatrixXd fd_dq;       // nv x nv
    Eigen::MatrixXd fd_dqd;      // nv x nv
    Eigen::MatrixXd ddq_dtau;    // nv x nv (unused output of computeABADerivatives)
    // Per-page (n0=nv) row-major scratch slabs nv*nv for one fixed k page.
    std::vector<double> page_a;  // nv*nv
    std::vector<double> page_b;  // nv*nv
    // Output tensors (column-major, like pinocchio's Tensor3x).
    Eigen::Tensor<double, 3> daba_dqdq;
    Eigen::Tensor<double, 3> daba_dvdq;
    Eigen::Tensor<double, 3> daba_dvdv;
    Eigen::Tensor<double, 3> daba_dtdq;

    void resize(int nv_in){
        if (nv == nv_in) return;
        nv = nv_in;
        Minv.setZero(nv, nv);
        fd_dq.setZero(nv, nv);
        fd_dqd.setZero(nv, nv);
        ddq_dtau.setZero(nv, nv);
        page_a.assign(nv*nv, 0.0);
        page_b.assign(nv*nv, 0.0);
        daba_dqdq.resize(nv, nv, nv);
        daba_dvdq.resize(nv, nv, nv);
        daba_dvdv.resize(nv, nv, nv);
        daba_dtdq.resize(nv, nv, nv);
    }
};

// Contract: out[i,j,k] = sum_l dM_dq[i,l,k] * A[l,j]
// dM_dq is Tensor3x (column-major Eigen tensor); A is Eigen::MatrixXd (col-major).
// Result stored in `out` (Tensor3x). Computed page-by-page in k for cache locality.
static inline void contract_ilk_lj_ijk(
    const Eigen::Tensor<double, 3> &dM_dq,
    const Eigen::MatrixXd &A,
    Eigen::Tensor<double, 3> &out)
{
    const int nv = static_cast<int>(A.rows());
    // For each fixed k, dM_dq[:,:,k] is an nv x nv matrix (slice). Map it,
    // multiply by A on the right, write into out[:,:,k].
    for (int k = 0; k < nv; ++k) {
        Eigen::Map<const Eigen::MatrixXd> dMk(dM_dq.data() + static_cast<ptrdiff_t>(k)*nv*nv, nv, nv);
        Eigen::Map<Eigen::MatrixXd>       Ok (out.data()  + static_cast<ptrdiff_t>(k)*nv*nv, nv, nv);
        Ok.noalias() = dMk * A;
    }
}

// In-place add: dst[i,j,k] += src[i,k,j]  (transpose pages 1<->2 elementwise add)
static inline void add_transpose_jk(
    const Eigen::Tensor<double, 3> &src,
    Eigen::Tensor<double, 3> &dst)
{
    const auto &dims = src.dimensions();
    const int n0 = static_cast<int>(dims[0]);
    const int n1 = static_cast<int>(dims[1]);
    const int n2 = static_cast<int>(dims[2]);
    // src(i,j,k) -> dst(i,k,j) means dst index (i,k_dst=j_src,j_dst=k_src)
    // Use raw column-major layout: t(i,j,k) at i + j*n0 + k*n0*n1
    const double *S = src.data();
    double       *D = dst.data();
    for (int k = 0; k < n2; ++k) {
        for (int j = 0; j < n1; ++j) {
            for (int i = 0; i < n0; ++i) {
                // dst(i,j,k) += src(i,k,j)
                D[i + j*n0 + k*n0*n1] += S[i + k*n0 + j*n0*n1];
            }
        }
    }
}

// Compute: out[i,j,k] = -sum_l Minv[i,l] * A[l,j,k]
// (i.e. apply Minv along axis 0 of A and negate).
static inline void apply_minv_neg(
    const Eigen::MatrixXd &Minv,
    const Eigen::Tensor<double, 3> &A,
    Eigen::Tensor<double, 3> &out)
{
    const int nv = static_cast<int>(Minv.rows());
    for (int k = 0; k < nv; ++k) {
        Eigen::Map<const Eigen::MatrixXd> Ak(A.data() + static_cast<ptrdiff_t>(k)*nv*nv, nv, nv);
        Eigen::Map<Eigen::MatrixXd>       Ok(out.data() + static_cast<ptrdiff_t>(k)*nv*nv, nv, nv);
        Ok.noalias() = -Minv * Ak;
    }
}

// Build d2tau_dvdq from d2tau_dqdv by transposing pages 1<->2 (per the
// pinocchio_backend convention: d2tau_dqdv[i,j,k] = d²τ_i/(dq_j dv_k);
// we want d2tau_dvdq[i,j,k] = d²τ_i/(dv_j dq_k) = d2tau_dqdv[i,k,j]).
static inline void transpose_jk_into(
    const Eigen::Tensor<double, 3> &src,
    Eigen::Tensor<double, 3> &dst)
{
    const auto &dims = src.dimensions();
    const int n0 = static_cast<int>(dims[0]);
    const int n1 = static_cast<int>(dims[1]);
    const int n2 = static_cast<int>(dims[2]);
    const double *S = src.data();
    double       *D = dst.data();
    for (int k = 0; k < n2; ++k) {
        for (int j = 0; j < n1; ++j) {
            for (int i = 0; i < n0; ++i) {
                D[i + j*n0 + k*n0*n1] = S[i + k*n0 + j*n0*n1];
            }
        }
    }
}

// Single sample: synthesize fdsva_so for one (q, qd, u). Reuses scratch.
template<typename T>
inline void fdsvaSoSynth_one(const pinocchio::Model &model, pinocchio::Data &data,
                              const Matrix<T, Eigen::Dynamic, 1> &q,
                              const Matrix<T, Eigen::Dynamic, 1> &qd,
                              const Matrix<T, Eigen::Dynamic, 1> &u,
                              FdsvaSoScratch &S)
{
    const int nv = model.nv;
    S.resize(nv);

    // Cast inputs once.
    Eigen::VectorXd q_d  = q.template  cast<double>();
    Eigen::VectorXd qd_d = qd.template cast<double>();
    Eigen::VectorXd u_d  = u.template  cast<double>();

    // (1) ABA to get qdd in data.ddq.
    pinocchio::aba(model, data, q_d, qd_d, u_d);
    Eigen::VectorXd qdd_d = data.ddq;

    // (2) Second-order RNEA derivatives. Pinocchio requires the four Tensor3x
    // in data to be zeroed before this call.
    data.d2tau_dqdq.setZero();
    data.d2tau_dvdv.setZero();
    data.d2tau_dqdv.setZero();
    data.d2tau_dadq.setZero();
    pinocchio::ComputeRNEASecondOrderDerivatives(model, data, q_d, qd_d, qdd_d);

    // (3) First-order ABA derivatives: fills data.Minv (upper), data.ddq_dq,
    // data.ddq_dv. Note this internally redoes some RNEA work, but matches what
    // a real pinocchio fdsva_so synthesis user would do.
    pinocchio::computeABADerivatives(model, data, q_d, qd_d, u_d,
                                     S.fd_dq, S.fd_dqd, S.ddq_dtau);
    // Symmetrize Minv.
    S.Minv = data.Minv;
    S.Minv.triangularView<Eigen::StrictlyLower>() =
        S.Minv.transpose().triangularView<Eigen::StrictlyLower>();

    // (4) Tensor contractions.
    //   d2tau_dvdq = transpose_jk(d2tau_dqdv)
    Eigen::Tensor<double, 3> d2tau_dvdq(nv, nv, nv);
    transpose_jk_into(data.d2tau_dqdv, d2tau_dvdq);

    //   work[i,j,k] = sum_l dM_dq[i,l,k] * fd_dq[l,j]    (dM_dq == data.d2tau_dadq)
    Eigen::Tensor<double, 3> work(nv, nv, nv);
    contract_ilk_lj_ijk(data.d2tau_dadq, S.fd_dq, work);

    //   tmp_qq = d2tau_dqdq + work + transpose_jk(work)
    Eigen::Tensor<double, 3> tmp(nv, nv, nv);
    tmp = data.d2tau_dqdq + work;
    add_transpose_jk(work, tmp);
    apply_minv_neg(S.Minv, tmp, S.daba_dqdq);

    //   tmp_vq = d2tau_dvdq + (work = dM_dq <ilk,lj> fd_dqd)
    contract_ilk_lj_ijk(data.d2tau_dadq, S.fd_dqd, work);
    tmp = d2tau_dvdq + work;
    apply_minv_neg(S.Minv, tmp, S.daba_dvdq);

    //   daba_dvdv = -Minv * d2tau_dvdv
    apply_minv_neg(S.Minv, data.d2tau_dvdv, S.daba_dvdv);

    //   daba_dtdq = -Minv * (dM_dq <ilk,lj> Minv)
    contract_ilk_lj_ijk(data.d2tau_dadq, S.Minv, work);
    apply_minv_neg(S.Minv, work, S.daba_dtdq);
}

template<typename T>
void fdsvaSoThreaded_inner(const pinocchio::Model *model, pinocchio::Data *data,
                            Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                            Matrix<T, Eigen::Dynamic, 1> *us, int tid, int kStart, int kMax){
    FdsvaSoScratch scratch;
    for(int k = kStart; k < kMax; k++){
        fdsvaSoSynth_one<T>(*model, *data, qs[k], qds[k], us[k], scratch);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void fdsvaSoThreaded(const pinocchio::Model *model, pinocchio::Data *datas,
                      Matrix<T, Eigen::Dynamic, 1> *qs, Matrix<T, Eigen::Dynamic, 1> *qds,
                      Matrix<T, Eigen::Dynamic, 1> *us, ReusableThreads<NUM_THREADS> *threads){
    constexpr int ET = effective_thread_count(NUM_TIME_STEPS, NUM_THREADS);
    for(int tid = 0; tid < ET; tid++){
        int kStart = NUM_TIME_STEPS/ET*tid; int kMax = NUM_TIME_STEPS/ET*(tid+1);
        if(tid == ET-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &fdsvaSoThreaded_inner<T>, model, &datas[tid],
                          std::ref(qs), std::ref(qds), std::ref(us), tid, kStart, kMax);
    }
    threads->sync();
}

// ---------------------------------------------------------------------------
// Main test function
// ---------------------------------------------------------------------------

// Algorithm gating for parallel per-algo subprocess runs. `enabled_algo` is the
// CLI --algo value: "all" (default) runs every algorithm; "id", "minv", "fd",
// "aba", "crba", "id_du", "fd_du", "ee_pose", "ee_pose_gradient", "idsva_so"
// run only that algorithm and skip the codegen JIT for any algorithms whose
// underlying CodeGen* objects are not needed (RNEA, Minv, ABA, CRBA,
// RNEADerivatives). This is the lever for sidestepping the >25 min cppadcg
// JIT wall on g1_floating: by farming algos out to parallel subprocesses in
// run.py, the wall time becomes max(per_algo_time) instead of sum.
inline bool is_algo_active(const std::string &enabled, const char *algo) {
    return enabled == "all" || enabled == algo;
}
inline bool needs_codegen(const std::string &enabled, const char *cg) {
    if (enabled == "all") return true;
    // Map algos to the cppadcg CodeGen* objects each depends on.
    // RNEA codegen used by: id, fd, fd_du
    if (std::string(cg) == "rnea")   return enabled == "id" || enabled == "fd" || enabled == "fd_du";
    // Minv codegen used by: minv, fd, fd_du
    if (std::string(cg) == "minv")   return enabled == "minv" || enabled == "fd" || enabled == "fd_du";
    // ABA codegen used by: aba
    if (std::string(cg) == "aba")    return enabled == "aba";
    // CRBA codegen used by: crba
    if (std::string(cg) == "crba")   return enabled == "crba";
    // RNEADerivatives codegen used by: id_du, fd_du
    if (std::string(cg) == "rnea_d") return enabled == "id_du" || enabled == "fd_du";
    return false;
}

template<typename T, int TEST_ITERS, int NUM_THREADS, int NUM_TIME_STEPS>
void test(std::string urdf_filepath, bool floating_base, std::string frame_name = "", std::string enabled_algo = "all"){
    struct timespec start, end;

    typedef Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXT;
    typedef Matrix<T, Eigen::Dynamic, 1> VectorXT;

    Model model;
    if (floating_base) {pinocchio::urdf::buildModel(urdf_filepath,pinocchio::JointModelFreeFlyer(), model);}
    else {pinocchio::urdf::buildModel(urdf_filepath,model);}
    model.gravity.linear(Eigen::Vector3d(0,0,-9.81));
    Data datas[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){datas[i] = Data(model);}

    // Resolve EE frame
    pinocchio::FrameIndex frame_id = 0;
    bool have_frame = !frame_name.empty();
    if(have_frame){
        if(model.existFrame(frame_name)){
            frame_id = model.getFrameId(frame_name);
        } else {
            printf("Warning: frame '%s' not found, skipping EE timing\n", frame_name.c_str());
            have_frame = false;
        }
    }

    // Initialize codegen objects (requires CppADCodeGen). Each block is gated
    // on `needs_codegen()` so per-algo CLI subprocesses only pay the cppadcg
    // JIT cost for the codegens actually needed. The CodeGen* solo objects
    // are stack-allocated (constructor is cheap; JIT happens in initLib);
    // the threaded arrays are zero-initialized so deletes on the unused
    // entries are safe nullptrs at the end of the function.
#ifdef HAVE_CPPADCG
    CodeGenRNEAWithGetRes<T> rnea_code_gen(model.cast<T>());
    if(needs_codegen(enabled_algo, "rnea")){ rnea_code_gen.initLib(); rnea_code_gen.loadLib(); }

    CodeGenRNEAWithGetRes<T> *rnea_code_gen_arr[NUM_THREADS] = {nullptr};
    if(needs_codegen(enabled_algo, "rnea")){
        for(int i = 0; i < NUM_THREADS; i++){
            rnea_code_gen_arr[i] = new CodeGenRNEAWithGetRes<T>(model.cast<T>());
            rnea_code_gen_arr[i]->initLib(); rnea_code_gen_arr[i]->loadLib();
        }
    }

    CodeGenMinv<T> minv_code_gen(model.cast<T>());
    if(needs_codegen(enabled_algo, "minv")){ minv_code_gen.initLib(); minv_code_gen.loadLib(); }

    CodeGenMinv<T> *minv_code_gen_arr[NUM_THREADS] = {nullptr};
    if(needs_codegen(enabled_algo, "minv")){
        for(int i = 0; i < NUM_THREADS; i++){
            minv_code_gen_arr[i] = new CodeGenMinv<T>(model.cast<T>());
            minv_code_gen_arr[i]->initLib(); minv_code_gen_arr[i]->loadLib();
        }
    }

    DerivedCodeGenRNEADerivatives<T> rnea_derivatives_code_gen(model.cast<T>());
    if(needs_codegen(enabled_algo, "rnea_d")){ rnea_derivatives_code_gen.initLib(); rnea_derivatives_code_gen.loadLib(); }

    DerivedCodeGenRNEADerivatives<T> *rnea_derivatives_code_gen_arr[NUM_THREADS] = {nullptr};
    if(needs_codegen(enabled_algo, "rnea_d")){
        for(int i = 0; i < NUM_THREADS; i++){
            rnea_derivatives_code_gen_arr[i] = new DerivedCodeGenRNEADerivatives<T>(model.cast<T>());
            rnea_derivatives_code_gen_arr[i]->initLib(); rnea_derivatives_code_gen_arr[i]->loadLib();
        }
    }

    CodeGenABA<T> aba_code_gen(model.cast<T>());
    if(needs_codegen(enabled_algo, "aba")){ aba_code_gen.initLib(); aba_code_gen.loadLib(); }

    CodeGenABA<T> *aba_code_gen_arr[NUM_THREADS] = {nullptr};
    if(needs_codegen(enabled_algo, "aba")){
        for(int i = 0; i < NUM_THREADS; i++){
            aba_code_gen_arr[i] = new CodeGenABA<T>(model.cast<T>());
            aba_code_gen_arr[i]->initLib(); aba_code_gen_arr[i]->loadLib();
        }
    }

    CodeGenCRBA<T> crba_code_gen(model.cast<T>());
    if(needs_codegen(enabled_algo, "crba")){ crba_code_gen.initLib(); crba_code_gen.loadLib(); }

    CodeGenCRBA<T> *crba_code_gen_arr[NUM_THREADS] = {nullptr};
    if(needs_codegen(enabled_algo, "crba")){
        for(int i = 0; i < NUM_THREADS; i++){
            crba_code_gen_arr[i] = new CodeGenCRBA<T>(model.cast<T>());
            crba_code_gen_arr[i]->initLib(); crba_code_gen_arr[i]->loadLib();
        }
    }
#endif // HAVE_CPPADCG

    // Allocate state arrays
    VectorXT qs[NUM_TIME_STEPS];
    VectorXT qds[NUM_TIME_STEPS];
    VectorXT qdds[NUM_TIME_STEPS];
    VectorXT us[NUM_TIME_STEPS];
    MatrixXT dqdd_dqs[NUM_TIME_STEPS];
    MatrixXT dqdd_dvs[NUM_TIME_STEPS];
    for(int i = 0; i < NUM_TIME_STEPS; i++){
        qs[i] = VectorXT::Zero(model.nq);
        qds[i] = VectorXT::Zero(model.nv);
        qdds[i] = VectorXT::Zero(model.nv);
        us[i] = VectorXT::Zero(model.nv);
        dqdd_dqs[i] = MatrixXT::Zero(model.nv,model.nq);
        dqdd_dvs[i] = MatrixXT::Zero(model.nv,model.nv);
        for(int j = 0; j < model.nq; j++){qs[i][j] = getRand<T>();}
        for(int j = 0; j < model.nv; j++){qds[i][j] = getRand<T>(); us[i][j] = getRand<T>();}
        // Normalize quaternion for floating-base (free-flyer: q[3:7] = xyzw)
        if(floating_base && model.nq >= 7){qs[i].segment(3,4).normalize();}
    }

    #if TEST_FOR_EQUIVALENCE
        std::cout << "q,qd,u" << std::endl;
    #else
        if(NUM_TIME_STEPS == 1){
            // Print algorithm metadata once before single-call timing
            printf("=== BEGIN PINOCCHIO METADATA ===\n");
#ifdef HAVE_CPPADCG
            printf("ID codegen: true\n");
            printf("Minv codegen: true\n");
            printf("ABA codegen: true\n");
            printf("FD codegen: true\n");
            printf("CRBA codegen: true\n");
            printf("ID_DU codegen: true\n");
            printf("FD_DU codegen: true\n");
#else
            printf("ID codegen: false\n");
            printf("Minv codegen: false\n");
            printf("ABA codegen: false\n");
            printf("FD codegen: false\n");
            printf("CRBA codegen: false\n");
            printf("ID_DU codegen: false\n");
            printf("FD_DU codegen: false\n");
            printf("ID direct: true\n");
            printf("Minv direct: true\n");
            printf("ABA direct: true\n");
            printf("FD direct: false\n");
            printf("CRBA direct: true\n");
            printf("ID_DU direct: true\n");
            printf("FD_DU direct: true\n");
#endif
            printf("EE_POSE codegen: false\n");
            printf("EE_POSE_GRADIENT codegen: false\n");
            printf("IDSVA_SO codegen: false\n");
            printf("FDSVA_SO codegen: null\n");
            printf("=== END PINOCCHIO METADATA ===\n");

            VectorXT zeros = VectorXT::Zero(model.nv);
            Eigen::VectorXd zeros_d = Eigen::VectorXd::Zero(model.nv);
            Eigen::MatrixXd J_single = Eigen::MatrixXd::Zero(6, model.nv);

#ifdef HAVE_CPPADCG
            if(is_algo_active(enabled_algo, "id")){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    rnea_code_gen.evalFunction(qs[0],qds[0],qdds[0]);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("ID codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }

            if(is_algo_active(enabled_algo, "minv")){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    minv_code_gen.evalFunction(qs[0]);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("Minv codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }

            if(is_algo_active(enabled_algo, "aba")){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    aba_code_gen.evalFunction(qs[0],qds[0],us[0]);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("ABA codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }

            if(is_algo_active(enabled_algo, "fd")){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    minv_code_gen.evalFunction(qs[0]);
                    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> minv = minv_code_gen.Minv.block(0,0,model.nv,model.nv);
                    minv.template triangularView<Eigen::StrictlyLower>() =
                        minv.transpose().template triangularView<Eigen::StrictlyLower>();
                    rnea_code_gen.evalFunction(qs[0],qds[0],zeros);
                    qdds[0].noalias() = minv*(us[0] - rnea_code_gen.getRes());
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("FD codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }

            if(is_algo_active(enabled_algo, "crba")){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    crba_code_gen.evalFunction(qs[0]);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("CRBA codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }

            if(is_algo_active(enabled_algo, "id_du")){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    rnea_derivatives_code_gen.evalFunction(qs[0],qds[0],qdds[0]);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("ID_DU codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }

            if(is_algo_active(enabled_algo, "fd_du")){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    minv_code_gen.evalFunction(qs[0]);
                    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> minv = minv_code_gen.Minv.block(0,0,model.nv,model.nv);
                    minv.template triangularView<Eigen::StrictlyLower>() =
                        minv.transpose().template triangularView<Eigen::StrictlyLower>();
                    rnea_code_gen.evalFunction(qs[0],qds[0],zeros);
                    VectorXT qdd = minv*(us[0] - rnea_code_gen.getRes());
                    rnea_derivatives_code_gen.evalFunction(qs[0],qds[0],qdd);
                    dqdd_dqs[0].noalias() = -minv*rnea_derivatives_code_gen.getDtauDq();
                    dqdd_dvs[0].noalias() = -minv*rnea_derivatives_code_gen.getDtauDv();
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("FD_DU codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }
#else
            // cppadcg not available — codegen variants null; run direct library instead
            printf("ID codegen null\n");
            printf("Minv codegen null\n");
            printf("ABA codegen null\n");
            printf("FD codegen null\n");
            printf("CRBA codegen null\n");
            printf("ID_DU codegen null\n");
            printf("FD_DU codegen null\n");

            if(is_algo_active(enabled_algo, "id")){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    pinocchio::rnea(model, datas[0], qs[0].template cast<double>(), qds[0].template cast<double>(), qdds[0].template cast<double>());
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("ID direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }

            if(is_algo_active(enabled_algo, "minv")){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    pinocchio::crba(model, datas[0], qs[0].template cast<double>());
                    pinocchio::cholesky::decompose(model, datas[0]);
                    pinocchio::cholesky::computeMinv(model, datas[0]);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("Minv direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }

            if(is_algo_active(enabled_algo, "aba")){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    pinocchio::aba(model, datas[0], qs[0].template cast<double>(), qds[0].template cast<double>(), us[0].template cast<double>());
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("ABA direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }

            if(is_algo_active(enabled_algo, "fd")){
                printf("FD direct null\n");
            }

            if(is_algo_active(enabled_algo, "crba")){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    pinocchio::crba(model, datas[0], qs[0].template cast<double>());
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("CRBA direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }

            if(is_algo_active(enabled_algo, "id_du")){
                Eigen::MatrixXd dtau_dq_s = Eigen::MatrixXd::Zero(model.nv, model.nv);
                Eigen::MatrixXd dtau_dv_s = Eigen::MatrixXd::Zero(model.nv, model.nv);
                Eigen::MatrixXd dtau_da_s = Eigen::MatrixXd::Zero(model.nv, model.nv);
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    pinocchio::computeRNEADerivatives(model, datas[0],
                        qs[0].template cast<double>(), qds[0].template cast<double>(), qdds[0].template cast<double>(),
                        dtau_dq_s, dtau_dv_s, dtau_da_s);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("ID_DU direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }

            if(is_algo_active(enabled_algo, "fd_du")){
                Eigen::MatrixXd ddq_dq_s = Eigen::MatrixXd::Zero(model.nv, model.nv);
                Eigen::MatrixXd ddq_dv_s = Eigen::MatrixXd::Zero(model.nv, model.nv);
                Eigen::MatrixXd ddq_dtau_s = Eigen::MatrixXd::Zero(model.nv, model.nv);
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    pinocchio::computeABADerivatives(model, datas[0],
                        qs[0].template cast<double>(), qds[0].template cast<double>(), us[0].template cast<double>(),
                        ddq_dq_s, ddq_dv_s, ddq_dtau_s);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("FD_DU direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }
#endif // HAVE_CPPADCG

            if(have_frame && is_algo_active(enabled_algo, "ee_pose")){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    pinocchio::forwardKinematics(model, datas[0], qs[0].template cast<double>());
                    pinocchio::updateFramePlacements(model, datas[0]);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("EE_POSE direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }
            if(have_frame && is_algo_active(enabled_algo, "ee_pose_gradient")){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    pinocchio::computeJointJacobians(model, datas[0], qs[0].template cast<double>());
                    pinocchio::getFrameJacobian(model, datas[0], frame_id, pinocchio::LOCAL, J_single);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("EE_POSE_GRADIENT direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }
            if(have_frame && is_algo_active(enabled_algo, "ee_pose_hessian")){
                pinocchio::JointIndex joint_id = model.frames[frame_id].parent;
                Eigen::Tensor<double, 3> H_single(6, model.nv, model.nv);
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    pinocchio::computeForwardKinematicsDerivatives(model, datas[0],
                        qs[0].template cast<double>(), qds[0].template cast<double>(), zeros_d);
                    pinocchio::computeJointKinematicHessians(model, datas[0]);
                    pinocchio::getJointKinematicHessian(model, datas[0], joint_id,
                        pinocchio::LOCAL_WORLD_ALIGNED, H_single);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("EE_POSE_HESSIAN direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }

            if(is_algo_active(enabled_algo, "idsva_so_body_frame")){
                // IDSVA_SO is expensive — use fewer iterations
                int idsva_so_iters = std::max(1, TEST_ITERS/10);
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < idsva_so_iters; i++){
                    pinocchio::ComputeRNEASecondOrderDerivatives(model, datas[0],
                        qs[0].template cast<double>(), qds[0].template cast<double>(), zeros_d);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("idsva_so_body_frame direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(idsva_so_iters));
            }

            // FDSVA_SO: pinocchio has no direct equivalent — synthesize via the
            // Singh/Carpentier chain rule (RNEA SO + ABA derivatives + Minv).
            if(is_algo_active(enabled_algo, "fdsva_so")){
                // FDSVA_SO is expensive — use the same iters/10 budget as IDSVA_SO.
                int fdsva_so_iters = std::max(1, TEST_ITERS/10);
                FdsvaSoScratch scratch;
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < fdsva_so_iters; i++){
                    fdsvaSoSynth_one<T>(model, datas[0], qs[0], qds[0], us[0], scratch);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("fdsva_so direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(fdsva_so_iters));
            }
        }
        else{
            ReusableThreads<NUM_THREADS> threads;
            std::vector<double> times = {};

#ifdef HAVE_CPPADCG
            if(is_algo_active(enabled_algo, "id")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    inverseDynamicsThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(rnea_code_gen_arr,
                                                                                  model.nq,model.nv,qs,qds,&threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: ID codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(is_algo_active(enabled_algo, "minv")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    minvThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(minv_code_gen_arr,model.nq,model.nv,qs,&threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: Minv codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(is_algo_active(enabled_algo, "aba")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    abaThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(aba_code_gen_arr,
                                                                      model.nq,model.nv,qs,qds,&threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: ABA codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(is_algo_active(enabled_algo, "fd")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    forwardDynamicsThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(minv_code_gen_arr,rnea_code_gen_arr,
                                                                                  model.nq,model.nv,qs,qds,qdds,us,&threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: FD codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(is_algo_active(enabled_algo, "crba")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    crbaThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(crba_code_gen_arr,model.nq,model.nv,qs,&threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: CRBA codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(is_algo_active(enabled_algo, "id_du")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    inverseDynamicsGradientThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(rnea_derivatives_code_gen_arr,
                                                                                          model.nq,model.nv,qs,qds,&threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: ID_DU codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(is_algo_active(enabled_algo, "fd_du")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    forwardDynamicsGradientThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(rnea_derivatives_code_gen_arr,
                                                                                        minv_code_gen_arr,rnea_code_gen_arr,
                                                                                        model.nq,model.nv,dqdd_dqs,dqdd_dvs,
                                                                                        qs,qds,us,&threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: FD_DU codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }
#endif // HAVE_CPPADCG

            if(is_algo_active(enabled_algo, "id")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    idDirectThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, qs, qds, qdds, &threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: ID direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(is_algo_active(enabled_algo, "minv")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    minvDirectThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, qs, &threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: Minv direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(is_algo_active(enabled_algo, "aba")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    abaDirectThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, qs, qds, us, &threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: ABA direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(is_algo_active(enabled_algo, "crba")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    crbaDirectThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, qs, &threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: CRBA direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(is_algo_active(enabled_algo, "id_du")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    idDuDirectThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, qs, qds, qdds, &threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: ID_DU direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(is_algo_active(enabled_algo, "fd_du")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    fdDuDirectThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, qs, qds, us, &threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: FD_DU direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(have_frame && is_algo_active(enabled_algo, "ee_pose")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    eePoseThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, frame_id, qs, &threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: EE_POSE direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }
            if(have_frame && is_algo_active(enabled_algo, "ee_pose_gradient")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    eePoseGradientThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, frame_id, qs, &threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: EE_POSE_GRADIENT direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }
            if(have_frame && is_algo_active(enabled_algo, "ee_pose_hessian")){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    eePoseHessianThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, frame_id, qs, qds, &threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: EE_POSE_HESSIAN direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(is_algo_active(enabled_algo, "idsva_so_body_frame")){
                // IDSVA_SO uses fewer TEST_ITERS due to high cost (especially for large robots)
                int idsva_so_iters = std::max(1, TEST_ITERS/10);
                for(int iter = 0; iter < idsva_so_iters; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    idsvaSoThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, qs, qds, &threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: idsva_so_body_frame direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            if(is_algo_active(enabled_algo, "fdsva_so")){
                // FDSVA_SO synthesized from pinocchio primitives — same budget as IDSVA_SO.
                int fdsva_so_iters = std::max(1, TEST_ITERS/10);
                for(int iter = 0; iter < fdsva_so_iters; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    fdsvaSoThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, qs, qds, us, &threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: fdsva_so direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }
        }
    #endif

#ifdef HAVE_CPPADCG
    for(int i = 0; i < NUM_THREADS; i++){
        delete rnea_derivatives_code_gen_arr[i];
        delete crba_code_gen_arr[i];
    }
#endif
}

template<typename T, int TEST_ITERS, int CPU_THREADS>
void run_all_tests(std::string urdf_filepath, bool floating_base, std::string frame_name = "", std::string enabled_algo = "all"){
    test<T,10*TEST_ITERS,CPU_THREADS,1>(urdf_filepath, floating_base, frame_name, enabled_algo);
    #if !TEST_FOR_EQUIVALENCE
        test<T,TEST_ITERS,CPU_THREADS,16>(urdf_filepath, floating_base, frame_name, enabled_algo);
        test<T,TEST_ITERS,CPU_THREADS,32>(urdf_filepath, floating_base, frame_name, enabled_algo);
        test<T,TEST_ITERS,CPU_THREADS,64>(urdf_filepath, floating_base, frame_name, enabled_algo);
        test<T,TEST_ITERS,CPU_THREADS,128>(urdf_filepath, floating_base, frame_name, enabled_algo);
        test<T,TEST_ITERS,CPU_THREADS,256>(urdf_filepath, floating_base, frame_name, enabled_algo);
    #endif
}

int main(int argc, const char ** argv){
    std::string urdf_filepath;
    std::string frame_name = "";
    std::string enabled_algo = "all";
    bool floating_base = false;
    if(argc > 1){
        urdf_filepath = argv[1];
        if(argc > 2 && argv[2][0] == 'T'){floating_base = true; printf("Floating Base = True\n");}
        if(argc > 3){frame_name = argv[3];}
        if(argc > 4){enabled_algo = argv[4];}
    }
    else{printf("Usage is: urdf_filepath [T/F floating_base] [frame_name] [algo|all]\n"); return 1;}
    if(!floating_base){printf("Floating Base = False\n");}
    if(!frame_name.empty()){printf("EE Frame: %s\n", frame_name.c_str());}
    if(enabled_algo != "all"){printf("Algo: %s\n", enabled_algo.c_str());}
    run_all_tests<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL>(urdf_filepath, floating_base, frame_name, enabled_algo);
    return 0;
}
