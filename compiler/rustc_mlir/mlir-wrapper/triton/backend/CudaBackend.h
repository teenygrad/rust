/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef TRITON_CUDA_BACKEND_H
#define TRITON_CUDA_BACKEND_H

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"

#include "Backend.h"

namespace mlir {
namespace triton {

namespace ttng = mlir::triton::nvidia_gpu;

struct CudaOptions {
  int num_warps = 4;
  int num_ctas = 1;
  int num_stages = 3;
  std::optional<int> maxnreg = std::nullopt;
  std::tuple<int, int, int> cluster_dims = {1, 1, 1};
  std::optional<int> ptx_version = std::nullopt;
  std::optional<std::string> ptx_options = std::nullopt;
  // filename of a user-defined IR (*.{ttir|ttgir|llir|ptx})
  std::optional<std::string> ir_override = std::nullopt;
  bool enable_fp_fusion = true;
  bool launch_cooperative_grid = false;
  bool launch_pdl = false;
  std::vector<std::string> supported_fp8_dtypes = {"fp8e5", "fp8e4b15"};
  std::vector<std::string> deprecated_fp8_dot_operand_dtypes = {};
  std::string default_dot_input_precision = "tf32";
  std::vector<std::string> allowed_dot_input_precisions = {"tf32", "tf32x3",
                                                           "ieee"};
  std::optional<bool> max_num_imprecise_acc_default = std::nullopt;
  std::map<std::string, std::string> extern_libs = {};
  bool debug = false;
  std::string backend_name = "cuda";
  bool sanitize_overflow = true;
  std::optional<std::string> arch = std::nullopt;
  bool dump_enabled = false;
  bool enable_experimental_consan = false;
  bool instrumentation = false;
  bool disable_line_info = false;
};

enum Capability {
  Sm80 = 80,
  Sm86 = 86,
  Sm87 = 87,
  Sm89 = 89,
  Sm90 = 90,
  Sm100 = 100,
  Sm103 = 103,
  Sm110 = 110,
  Sm120 = 120,
};

enum CudaPass {
  // ttgpuir
  allocate_shared_memory_nv,
  ttgpuir_to_llvmir,

  // ttnvgpuir
  ttnvgpuir_plan_cta,
  ttnvgpuir_fence_insertion,
  ttnvgpuir_proxy_fence_insertion,
  ttnvgpuir_tma_lowering,
  ttnvgpuir_promote_lhs_to_tmem,
  ttnvgpuir_remove_tmem_tokens,
  ttnvgpuir_nvgpu_to_llvm,
  ttnvgpuir_warp_specialize_to_llvm,
  ttnvgpuir_allocate_tensor_memory,
  ttnvgpuir_lower_mma,
  ttnvgpuir_optimize_descriptor_encoding,
  ttnvgpuir_optimize_tmem_layouts,
  ttnvgpuir_interleave_tmem,

  // nvws
  nvws_lower_warp_group,
  nvws_lower_aref,
  nvws_assign_stage_phase,

  // hopper
  hopper_warpspec
};

class CudaBackend : public Backend {
public:
  CudaBackend(std::string target, CudaOptions options);

  virtual ~CudaBackend();

  virtual void loadDialects(MLIRContext &context);

  virtual LogicalResult applyPasses(MLIRContext &context, ModuleOp module,
                                    Language language) override;

  LogicalResult make_ttir(MLIRContext &context, ModuleOp module);

  LogicalResult make_ttgir(MLIRContext &context, ModuleOp module);

  LogicalResult gluon_to_ttgir(MLIRContext &context, ModuleOp module);

  LogicalResult make_llir(MLIRContext &context, ModuleOp module);

private:
  std::optional<Error> addCudaPass(PassManager &pm, CudaPass pass);

  std::optional<Error> addCudaPass(PassManager &pm, CudaPass pass, int arg0);

  std::optional<Error> addCudaPass(PassManager &pm, CudaPass pass, int arg0,
                                   int arg1);

  std::optional<Error> addCudaPass(PassManager &pm, CudaPass pass, int arg0,
                                   bool arg1);

  std::unique_ptr<mlir::Pass>
  createTritonGPUProxyFenceInsertionWrapper(int32_t capability);

  CudaOptions m_options;
  Capability m_capability;

  std::unordered_map<CudaPass, std::unique_ptr<Pass> (*)()> m_nvidia_pass_fns =
      {
          {ttnvgpuir_fence_insertion, ttng::createTritonGPUFenceInsertion},
          {ttnvgpuir_tma_lowering, ttng::createTritonNvidiaGPUTMALoweringPass},
          {ttnvgpuir_promote_lhs_to_tmem,
           ttng::createTritonNvidiaGPUPromoteLHSToTMemPass},
          {ttnvgpuir_remove_tmem_tokens,
           ttng::createTritonNvidiaGPURemoveTMEMTokensPass},
          {ttnvgpuir_nvgpu_to_llvm, mlir::triton::createConvertNVGPUToLLVM},
          {ttnvgpuir_warp_specialize_to_llvm,
           mlir::triton::createConvertWarpSpecializeToLLVM},
          {ttnvgpuir_allocate_tensor_memory,
           ttng::createTritonTensorMemoryAllocationPass},
          {ttnvgpuir_lower_mma, ttng::createTritonNvidiaGPUMMALoweringPass},
          {ttnvgpuir_optimize_descriptor_encoding,
           ttng::createTritonNvidiaGPUOptimizeDescriptorEncodingPass},
          {ttnvgpuir_optimize_tmem_layouts,
           ttng::createTritonNvidiaGPUOptimizeTMemLayoutsPass},
          {ttnvgpuir_interleave_tmem,
           ttng::createTritonNvidiaGPUInterleaveTMemPass},
          {ttnvgpuir_plan_cta, ttng::createTritonNvidiaGPUPlanCTAPass},

          // nvws
          {nvws_lower_warp_group, mlir::triton::createNVWSLowerWarpGroup},
          {nvws_lower_aref, mlir::triton::createNVWSLowerAref},
      };

  Capability getCapability() const;
};

} // namespace triton
} // namespace mlir

#endif /*! TRITON_CUDA_BACKEND_H */
