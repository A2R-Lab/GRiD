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
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/rnea-second-order-derivatives.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "pinocchio/codegen/cppadcg.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/multibody/sample-models.hpp"

#include "pinocchio/container/aligned-vector.hpp"

#include <Eigen/StdVector>

#include "../util/getters/GetResRNEA.hpp"
#include "../util/getters/GettersDerivatives.hpp"

using namespace Eigen;
using namespace pinocchio;

#define time_delta_us_timespec(start,end) (1e6*static_cast<double>(end.tv_sec - start.tv_sec)+1e-3*static_cast<double>(end.tv_nsec - start.tv_nsec))

// ---------------------------------------------------------------------------
// Threading helpers — existing algorithms
// ---------------------------------------------------------------------------

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
        for (int tid = 0; tid < NUM_THREADS; tid++){
            int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1);
            if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;}
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
        for (int tid = 0; tid < NUM_THREADS; tid++){
            int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1);
            if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;}
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
        for (int tid = 0; tid < NUM_THREADS; tid++){
            int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1);
            if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;}
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
        for (int tid = 0; tid < NUM_THREADS; tid++){
            int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1);
            if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;}
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
        for (int tid = 0; tid < NUM_THREADS; tid++){
            int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1);
            if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;}
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
        for (int tid = 0; tid < NUM_THREADS; tid++){
            int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1);
            if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;}
            threads->addTask(tid, &abaThreaded_codegen_inner<T>, std::ref(aba_code_gen_arr[tid]), nq, nv,
                                                                             std::ref(qs), std::ref(qds), tid, kStart, kMax);
        }
        threads->sync();
}

// ---------------------------------------------------------------------------
// Threading helpers — new algorithms
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
    for(int tid = 0; tid < NUM_THREADS; tid++){
        int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1);
        if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &crbaThreaded_codegen_inner<T>, std::ref(crba_code_gen_arr[tid]), nq, nv,
                                                              std::ref(qs), tid, kStart, kMax);
    }
    threads->sync();
}

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
    for(int tid = 0; tid < NUM_THREADS; tid++){
        int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1);
        if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;}
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
    for(int tid = 0; tid < NUM_THREADS; tid++){
        int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1);
        if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &eePoseGradientThreaded_inner<T>, model, &datas[tid], frame_id,
                                                                 std::ref(qs), tid, kStart, kMax);
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
    for(int tid = 0; tid < NUM_THREADS; tid++){
        int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1);
        if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;}
        threads->addTask(tid, &idsvaSoThreaded_inner<T>, model, &datas[tid],
                                                          std::ref(qs), std::ref(qds), tid, kStart, kMax);
    }
    threads->sync();
}

// ---------------------------------------------------------------------------
// Main test function
// ---------------------------------------------------------------------------

template<typename T, int TEST_ITERS, int NUM_THREADS, int NUM_TIME_STEPS>
void test(std::string urdf_filepath, bool floating_base, std::string frame_name = ""){
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

    // Initialize codegen objects
    CodeGenRNEAWithGetRes<T> rnea_code_gen(model.cast<T>());
    rnea_code_gen.initLib(); rnea_code_gen.loadLib();

    CodeGenRNEAWithGetRes<T> *rnea_code_gen_arr[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){
        rnea_code_gen_arr[i] = new CodeGenRNEAWithGetRes<T>(model.cast<T>());
        rnea_code_gen_arr[i]->initLib(); rnea_code_gen_arr[i]->loadLib();
    }

    CodeGenMinv<T> minv_code_gen(model.cast<T>());
    minv_code_gen.initLib(); minv_code_gen.loadLib();

    CodeGenMinv<T> *minv_code_gen_arr[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){
        minv_code_gen_arr[i] = new CodeGenMinv<T>(model.cast<T>());
        minv_code_gen_arr[i]->initLib(); minv_code_gen_arr[i]->loadLib();
    }

    DerivedCodeGenRNEADerivatives<T> rnea_derivatives_code_gen(model.cast<T>());
    rnea_derivatives_code_gen.initLib(); rnea_derivatives_code_gen.loadLib();

    DerivedCodeGenRNEADerivatives<T> *rnea_derivatives_code_gen_arr[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){
        rnea_derivatives_code_gen_arr[i] = new DerivedCodeGenRNEADerivatives<T>(model.cast<T>());
        rnea_derivatives_code_gen_arr[i]->initLib(); rnea_derivatives_code_gen_arr[i]->loadLib();
    }

    CodeGenABA<T> aba_code_gen(model.cast<T>());
    aba_code_gen.initLib(); aba_code_gen.loadLib();

    CodeGenABA<T> *aba_code_gen_arr[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){
        aba_code_gen_arr[i] = new CodeGenABA<T>(model.cast<T>());
        aba_code_gen_arr[i]->initLib(); aba_code_gen_arr[i]->loadLib();
    }

    CodeGenCRBA<T> crba_code_gen(model.cast<T>());
    crba_code_gen.initLib(); crba_code_gen.loadLib();

    CodeGenCRBA<T> *crba_code_gen_arr[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){
        crba_code_gen_arr[i] = new CodeGenCRBA<T>(model.cast<T>());
        crba_code_gen_arr[i]->initLib(); crba_code_gen_arr[i]->loadLib();
    }

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
            printf("ID codegen: true\n");
            printf("Minv codegen: true\n");
            printf("ABA codegen: true\n");
            printf("FD codegen: true\n");
            printf("CRBA codegen: true\n");
            printf("ID_DU codegen: true\n");
            printf("FD_DU codegen: true\n");
            printf("EE_POSE codegen: false\n");
            printf("EE_POSE_GRADIENT codegen: false\n");
            printf("IDSVA_SO codegen: false\n");
            printf("FDSVA_SO codegen: null\n");
            printf("=== END PINOCCHIO METADATA ===\n");

            VectorXT zeros = VectorXT::Zero(model.nv);
            Eigen::VectorXd zeros_d = Eigen::VectorXd::Zero(model.nv);
            Eigen::MatrixXd J_single = Eigen::MatrixXd::Zero(6, model.nv);

            clock_gettime(CLOCK_MONOTONIC,&start);
            for(int i = 0; i < TEST_ITERS; i++){
                rnea_code_gen.evalFunction(qs[0],qds[0],qdds[0]);
            }
            clock_gettime(CLOCK_MONOTONIC,&end);
            printf("ID codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));

            clock_gettime(CLOCK_MONOTONIC,&start);
            for(int i = 0; i < TEST_ITERS; i++){
                minv_code_gen.evalFunction(qs[0]);
            }
            clock_gettime(CLOCK_MONOTONIC,&end);
            printf("Minv codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));

            clock_gettime(CLOCK_MONOTONIC,&start);
            for(int i = 0; i < TEST_ITERS; i++){
                aba_code_gen.evalFunction(qs[0],qds[0],us[0]);
            }
            clock_gettime(CLOCK_MONOTONIC,&end);
            printf("ABA codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));

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

            clock_gettime(CLOCK_MONOTONIC,&start);
            for(int i = 0; i < TEST_ITERS; i++){
                crba_code_gen.evalFunction(qs[0]);
            }
            clock_gettime(CLOCK_MONOTONIC,&end);
            printf("CRBA codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));

            clock_gettime(CLOCK_MONOTONIC,&start);
            for(int i = 0; i < TEST_ITERS; i++){
                rnea_derivatives_code_gen.evalFunction(qs[0],qds[0],qdds[0]);
            }
            clock_gettime(CLOCK_MONOTONIC,&end);
            printf("ID_DU codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));

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

            if(have_frame){
                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    pinocchio::forwardKinematics(model, datas[0], qs[0].template cast<double>());
                    pinocchio::updateFramePlacements(model, datas[0]);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("EE_POSE direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));

                clock_gettime(CLOCK_MONOTONIC,&start);
                for(int i = 0; i < TEST_ITERS; i++){
                    pinocchio::computeJointJacobians(model, datas[0], qs[0].template cast<double>());
                    pinocchio::getFrameJacobian(model, datas[0], frame_id, pinocchio::LOCAL, J_single);
                }
                clock_gettime(CLOCK_MONOTONIC,&end);
                printf("EE_POSE_GRADIENT direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));
            }

            // IDSVA_SO is expensive — use fewer iterations
            int idsva_so_iters = std::max(1, TEST_ITERS/10);
            clock_gettime(CLOCK_MONOTONIC,&start);
            for(int i = 0; i < idsva_so_iters; i++){
                pinocchio::ComputeRNEASecondOrderDerivatives(model, datas[0],
                    qs[0].template cast<double>(), qds[0].template cast<double>(), zeros_d);
            }
            clock_gettime(CLOCK_MONOTONIC,&end);
            printf("IDSVA_SO direct %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(idsva_so_iters));

            // FDSVA_SO: no Pinocchio equivalent
            printf("FDSVA_SO direct null\n");
        }
        else{
            ReusableThreads<NUM_THREADS> threads;
            std::vector<double> times = {};

            for(int iter = 0; iter < TEST_ITERS; iter++){
                clock_gettime(CLOCK_MONOTONIC,&start);
                inverseDynamicsThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(rnea_code_gen_arr,
                                                                              model.nq,model.nv,qs,qds,&threads);
                clock_gettime(CLOCK_MONOTONIC,&end);
                times.push_back(time_delta_us_timespec(start,end));
            }
            printf("[N:%d]: ID codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
            printf("----------------------------------------\n");

            for(int iter = 0; iter < TEST_ITERS; iter++){
                clock_gettime(CLOCK_MONOTONIC,&start);
                minvThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(minv_code_gen_arr,model.nq,model.nv,qs,&threads);
                clock_gettime(CLOCK_MONOTONIC,&end);
                times.push_back(time_delta_us_timespec(start,end));
            }
            printf("[N:%d]: Minv codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
            printf("----------------------------------------\n");

            for(int iter = 0; iter < TEST_ITERS; iter++){
                clock_gettime(CLOCK_MONOTONIC,&start);
                abaThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(aba_code_gen_arr,
                                                                  model.nq,model.nv,qs,qds,&threads);
                clock_gettime(CLOCK_MONOTONIC,&end);
                times.push_back(time_delta_us_timespec(start,end));
            }
            printf("[N:%d]: ABA codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
            printf("----------------------------------------\n");

            for(int iter = 0; iter < TEST_ITERS; iter++){
                clock_gettime(CLOCK_MONOTONIC,&start);
                forwardDynamicsThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(minv_code_gen_arr,rnea_code_gen_arr,
                                                                              model.nq,model.nv,qs,qds,qdds,us,&threads);
                clock_gettime(CLOCK_MONOTONIC,&end);
                times.push_back(time_delta_us_timespec(start,end));
            }
            printf("[N:%d]: FD codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
            printf("----------------------------------------\n");

            for(int iter = 0; iter < TEST_ITERS; iter++){
                clock_gettime(CLOCK_MONOTONIC,&start);
                crbaThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(crba_code_gen_arr,model.nq,model.nv,qs,&threads);
                clock_gettime(CLOCK_MONOTONIC,&end);
                times.push_back(time_delta_us_timespec(start,end));
            }
            printf("[N:%d]: CRBA codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
            printf("----------------------------------------\n");

            for(int iter = 0; iter < TEST_ITERS; iter++){
                clock_gettime(CLOCK_MONOTONIC,&start);
                inverseDynamicsGradientThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(rnea_derivatives_code_gen_arr,
                                                                                      model.nq,model.nv,qs,qds,&threads);
                clock_gettime(CLOCK_MONOTONIC,&end);
                times.push_back(time_delta_us_timespec(start,end));
            }
            printf("[N:%d]: ID_DU codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
            printf("----------------------------------------\n");

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

            if(have_frame){
                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    eePoseThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, frame_id, qs, &threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: EE_POSE direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");

                for(int iter = 0; iter < TEST_ITERS; iter++){
                    clock_gettime(CLOCK_MONOTONIC,&start);
                    eePoseGradientThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, frame_id, qs, &threads);
                    clock_gettime(CLOCK_MONOTONIC,&end);
                    times.push_back(time_delta_us_timespec(start,end));
                }
                printf("[N:%d]: EE_POSE_GRADIENT direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
                printf("----------------------------------------\n");
            }

            // IDSVA_SO uses fewer TEST_ITERS due to high cost (especially for large robots)
            int idsva_so_iters = std::max(1, TEST_ITERS/10);
            for(int iter = 0; iter < idsva_so_iters; iter++){
                clock_gettime(CLOCK_MONOTONIC,&start);
                idsvaSoThreaded<T,NUM_THREADS,NUM_TIME_STEPS>(&model, datas, qs, qds, &threads);
                clock_gettime(CLOCK_MONOTONIC,&end);
                times.push_back(time_delta_us_timespec(start,end));
            }
            printf("[N:%d]: IDSVA_SO direct: ",NUM_TIME_STEPS); printStats(&times); times.clear();
            printf("----------------------------------------\n");

            // FDSVA_SO: no Pinocchio equivalent — not timed
        }
    #endif

    for(int i = 0; i < NUM_THREADS; i++){
        delete rnea_derivatives_code_gen_arr[i];
        delete crba_code_gen_arr[i];
    }
}

template<typename T, int TEST_ITERS, int CPU_THREADS>
void run_all_tests(std::string urdf_filepath, bool floating_base, std::string frame_name = ""){
    test<T,10*TEST_ITERS,CPU_THREADS,1>(urdf_filepath, floating_base, frame_name);
    #if !TEST_FOR_EQUIVALENCE
        test<T,TEST_ITERS,CPU_THREADS,16>(urdf_filepath, floating_base, frame_name);
        test<T,TEST_ITERS,CPU_THREADS,32>(urdf_filepath, floating_base, frame_name);
        test<T,TEST_ITERS,CPU_THREADS,64>(urdf_filepath, floating_base, frame_name);
        test<T,TEST_ITERS,CPU_THREADS,128>(urdf_filepath, floating_base, frame_name);
        test<T,TEST_ITERS,CPU_THREADS,256>(urdf_filepath, floating_base, frame_name);
    #endif
}

int main(int argc, const char ** argv){
    std::string urdf_filepath;
    std::string frame_name = "";
    bool floating_base = false;
    if(argc > 1){
        urdf_filepath = argv[1];
        if(argc > 2 && argv[2][0] == 'T'){floating_base = true; printf("Floating Base = True\n");}
        if(argc > 3){frame_name = argv[3];}
    }
    else{printf("Usage is: urdf_filepath [T/F floating_base] [frame_name]\n"); return 1;}
    if(!floating_base){printf("Floating Base = False\n");}
    if(!frame_name.empty()){printf("EE Frame: %s\n", frame_name.c_str());}
    run_all_tests<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL>(urdf_filepath, floating_base, frame_name);
    return 0;
}
