"""Compact armature identification, reusing the FD inner's inverse once.

Only requested identification builds emit these helpers. The FD inner retains
its symmetric-upper inverse in the first nv*nv entries of its scratch arena.
The sensitivity consumes that inverse after a block barrier, before reuse.
"""
from grid_codegen.armature import supported_slots
from grid_codegen.helpers._code_generation_helpers import gen_workspace_repoint_line


def gen_armature_identification(self):
    nv, nq = self.robot.get_num_vel(), self.robot.get_num_pos()
    slots = supported_slots(self.robot)
    na = len(slots)
    # A switch works on host and device without a device-global table or loads.
    self.gen_add_code_line("__host__ __device__ constexpr int armature_parameter_slot(int column) {", True)
    self.gen_add_code_line("switch (column) {", True)
    for c, slot in enumerate(slots):
        self.gen_add_code_line(f"case {c}: return {slot};")
    self.gen_add_code_line("default: return -1;")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()
    self.gen_add_code_lines([
        "// Row-major nv x NUM_ARMATURE_PARAMS; all block threads participate.",
        "// Caller synchronizes before reading the output. No q/qd dependence.",
        "template <typename T> __device__ void armature_regressor_device(T *out, const T *qdd) {",
        "    for (int i = threadIdx.x + threadIdx.y*blockDim.x; i < NUM_VEL * NUM_ARMATURE_PARAMS; i += blockDim.x*blockDim.y) {",
        f"        const int row = i / {max(1, na)};",
        f"        const int slot = armature_parameter_slot(i % {max(1, na)});",
        "        out[i] = row == slot ? qdd[row] : T(0);",
        "    }",
        "}",
        "// FD inner scratch prefix is symmetric-upper Minv, column-major.",
        "template <typename T> __device__ void armature_gradient_finish(T *out, const T *qdd, const T *Minv) {",
        "    for (int i = threadIdx.x + threadIdx.y*blockDim.x; i < NUM_VEL * NUM_ARMATURE_PARAMS; i += blockDim.x*blockDim.y) {",
        f"        const int row = i / {max(1, na)};",
        f"        const int slot = armature_parameter_slot(i % {max(1, na)});",
        "        const int index = row <= slot ? slot * NUM_VEL + row : row * NUM_VEL + slot;",
        "        out[i] = -Minv[index] * qdd[slot];",
        "    }",
        "}",
    ])
    # Device composition: same FD scratch, no dense regressor or second inverse.
    def body():
        self.gen_forward_dynamics_inner_function_call()
        self.gen_add_sync()
        self.gen_add_code_line("armature_gradient_finish<T>(out, s_qdd, s_temp);")
        self.gen_add_sync()
    self.gen_add_code_lines([
        "// Collective shared-tier helper; all threads call with identical pointers.",
        "// Reserve FORWARD_DYNAMICS_ARMATURE_GRADIENT_DEVICE_SHARED_MEM_BYTES<T>().",
        "// Inputs/output must not alias this dynamic arena. Output is synchronized.",
        "// Use the kernel/host surface for workspace-backed spill execution.",
    ])
    self.gen_device_wrapper(
        "FD armature sensitivity at fixed q, qd and torque, pinocchio convention",
        "void forward_dynamics_armature_gradient_device(T *out, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity) {",
        self.gen_forward_dynamics_inner_temp_mem_size(), body,
        extra_t_buffers=[("s_qdd", nv)])
    self.gen_add_code_lines([
        "template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>",
        "__global__ __launch_bounds__(tier_max_threads<RESOURCE_TIER>())",
        "void forward_dynamics_armature_gradient_kernel(T *out, unsigned char *d_workspace, const T *d_q_qd_u, int stride_q_qd_u, T *d_f_ext, const robotModel<T> *d_robotModel, T gravity, int NUM_TIMESTEPS) {",
    ], True)
    # Exactly the FD kernel's arena, including its existing tiered F-band spill.
    picks = self.fd_spill_tier_3way
    condition = f"if constexpr ((RESOURCE_TIER == TIER_SHARED ? {picks[0]} : RESOURCE_TIER == TIER_LITE ? {picks[1]} : {picks[2]}) == 0) {{"
    for spill, condition in [(False, condition),
                             (True, "else {")]:
        self.gen_add_code_line(condition, True)
        self.gen_XImats_helpers_temp_shared_memory_code(
            self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=not spill),
            extra_t_buffers=[("s_q_qd_u", 3*nq), ("s_qdd", nv)])
        self.gen_add_code_line(f"T *s_q = s_q_qd_u; T *s_qd = s_q_qd_u + {nq}; T *s_u = s_q_qd_u + {2*nq};")
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q_qd_u", str(3*nq), stride="stride_q_qd_u")
        self.gen_add_code_line("T *d_f_ext_k = d_f_ext ? d_f_ext + size_t(k) * 6 * NUM_BODIES : nullptr;")
        if spill:
            self.gen_add_code_line(gen_workspace_repoint_line("fd_d_workspace", "GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()", batch_indexed=True, declare=True))
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_forward_dynamics_inner_function_call(
            updated_var_names=dict(d_f_ext_name="d_f_ext_k", d_workspace_name="fd_d_workspace" if spill else "nullptr"),
            minv_f_in_smem_expr="false" if spill else "true")
        self.gen_add_sync()
        self.gen_add_code_line("armature_gradient_finish<T>(out + size_t(k) * NUM_VEL * NUM_ARMATURE_PARAMS, s_qdd, s_temp);")
        self.gen_add_sync()
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
    self.gen_add_end_function()
    self.gen_add_code_lines([
        "template <typename T, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>",
        "__host__ void forward_dynamics_armature_gradient(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, T gravity, int num_timesteps, dim3 block_dimms, dim3 thread_dimms, cudaStream_t *streams) {",
        "    gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u, hd_data->h_q_qd_u, size_t(3)*NUM_JOINTS*num_timesteps*sizeof(T), cudaMemcpyHostToDevice, streams[0]));",
        "    gpuErrchk(cudaStreamSynchronize(streams[0]));",
        "    block_dimms.x = min(block_dimms.x, static_cast<unsigned>(hd_data->workspace_timestep_slots));",
        "    forward_dynamics_armature_gradient_kernel<T, RESOURCE_TIER><<<block_dimms, thread_dimms, FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dqdd_da, hd_data->d_workspace, hd_data->d_q_qd_u, 3*NUM_JOINTS, hd_data->d_f_ext, d_robotModel, gravity, num_timesteps);",
        "    gpuErrchkKernel();",
        "    gpuErrchk(cudaMemcpy(hd_data->h_dqdd_da, hd_data->d_dqdd_da, size_t(NUM_VEL)*NUM_ARMATURE_PARAMS*num_timesteps*sizeof(T), cudaMemcpyDeviceToHost));",
        "}",
    ])
