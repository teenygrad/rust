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

#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "llvm/IR/Constants.h"

#include "CudaBackend.h"

using namespace mlir;
using namespace triton;
using namespace nvidia_gpu;

CudaBackend::CudaBackend(std::string target, CudaOptions options)
    : Backend(target), m_options(options) {
  m_capability = Capability::Sm120; // AXM FIXME: Get capability from target
}

CudaBackend::~CudaBackend() {
  // nop
}

void CudaBackend::loadDialects(MLIRContext &context) {
  DialectRegistry registry;

  registry.insert<mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                  mlir::triton::nvgpu::NVGPUDialect,
                  mlir::triton::nvws::NVWSDialect>();

  registerNVVMDialectTranslation(registry);

  context.appendDialectRegistry(registry);
}

Capability CudaBackend::getCapability() const { return m_capability; }

LogicalResult CudaBackend::applyPasses(MLIRContext &context, ModuleOp module,
                                       Language language) {
  auto m_result = LogicalResult::success();
  printIR("TTIR_BEFORE", module);

  if (language == Language::TRITON) {
    m_result = make_ttir(context, module);
    CHECK_RESULT(m_result, "Failed to make TTIR module. Aborting translation.");
    printIR("TTIR", module);

    m_result = make_ttgir(context, module);
    CHECK_RESULT(m_result,
                 "Failed to make TTGIR module. Aborting translation.");
    printIR("TTGIR", module);
  } else {
    m_result = gluon_to_ttgir(context, module);
    CHECK_RESULT(m_result, "Failed to convert GLUON module to TTGIR module. "
                           "Aborting translation.");
    printIR("TTGLUONIR", module);
  }

  m_result = make_llir(context, module);
  CHECK_RESULT(m_result, "Failed to make LLIR module. Aborting translation.");
  printIR("LLIR", module);

  return LogicalResult::success();
}

std::optional<Error> CudaBackend::addCudaPass(PassManager &pm, CudaPass pass) {
  auto pass_fn = m_nvidia_pass_fns.find(pass);
  if (pass_fn == m_nvidia_pass_fns.end()) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid nvidia pass";
    return m_last_error;
  }

  pm.addPass(pass_fn->second());
  return std::nullopt;
}

std::optional<Error> CudaBackend::addCudaPass(PassManager &pm, CudaPass pass,
                                              int arg0) {
  if (pass != CudaPass::ttnvgpuir_proxy_fence_insertion) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid nvidia pass";
    return m_last_error;
  }

  pm.addPass(CudaBackend::createTritonGPUProxyFenceInsertionWrapper(arg0));
  return std::nullopt;
}

std::optional<Error> CudaBackend::addCudaPass(PassManager &pm, CudaPass pass,
                                              int arg0, int arg1) {
  if (pass != CudaPass::ttnvgpuir_to_llvmir) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid nvidia pass";
    return m_last_error;
  }

  pm.addPass(createConvertTritonGPUToLLVMPass(arg0, arg1));
  return std::nullopt;
}

std::optional<Error> CudaBackend::addCudaPass(PassManager &pm, CudaPass pass,
                                              int arg0, bool arg1) {
  if (pass != CudaPass::hopper_warpspec) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid nvidia pass";
    return m_last_error;
  }

  pm.addPass(createNVGPUWarpSpecialization({arg0, arg1}));
  return std::nullopt;
}

LogicalResult CudaBackend::make_ttir(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  auto capability = getCapability();
  auto op = module.getOperation();

  addPass(pm, MlirPass::inliner);
  addPass(pm, MlirPass::ttir_rewrite_tensor_pointer);
  if (capability < 90) {
    addPass(pm, MlirPass::ttir_rewrite_tensor_descriptor_to_pointer);
  }
  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::ttir_combine);
  addPass(pm, MlirPass::ttir_reorder_broadcast);
  addPass(pm, MlirPass::cse);
  addPass(pm, MlirPass::symbol_dce);
  addPass(pm, MlirPass::ttir_loop_unroll);

  return pm.run(op);
}

LogicalResult CudaBackend::make_ttgir(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  auto capability = getCapability();
  auto capability_major = static_cast<int>(capability) / 10;
  auto op = module.getOperation();
  auto emuTF32 = (capability_major >= 8);

  if (m_options.maxnreg.has_value()) {
    auto maxnreg = m_options.maxnreg.value();
    OpBuilder builder(&context);

    op->setAttr("ttg.maxnreg", builder.getI32IntegerAttr(maxnreg));
  }

  std::string capability_str =
      std::string("cuda:").append(std::to_string(static_cast<int>(capability)));

  addPass(pm, MlirPass::ttir_convert_to_ttgpuir, capability_str,
          m_options.num_warps, 32, m_options.num_ctas);

  // optimize TTGIR
  addPass(pm, MlirPass::ttgpuir_coalesce);
  addPass(pm, MlirPass::ttgpuir_f32_dot_tc, emuTF32);

  addCudaPass(pm, CudaPass::ttnvgpuir_plan_cta);
  addPass(pm, MlirPass::ttgpuir_remove_layout_conversions);
  addPass(pm, MlirPass::ttgpuir_optimize_thread_locality);
  addPass(pm, MlirPass::ttgpuir_accelerate_matmul);
  addPass(pm, MlirPass::ttgpuir_remove_layout_conversions);
  addPass(pm, MlirPass::ttgpuir_optimize_dot_operands, capability_major >= 8);

  addCudaPass(pm, CudaPass::ttnvgpuir_optimize_descriptor_encoding);
  addPass(pm, MlirPass::ttir_loop_aware_cse);

  if (capability_major == 8 || capability_major == 9) {
    addPass(pm, MlirPass::ttgpuir_fuse_nested_loops);
    addPass(pm, MlirPass::canonicalizer);
    addPass(pm, MlirPass::ttir_triton_licm);
    addPass(pm, MlirPass::canonicalizer);
    addPass(pm, MlirPass::ttgpuir_combine_tensor_select_and_if);
    addCudaPass(pm, CudaPass::hopper_warpspec, m_options.num_stages,
                m_options.dump_enabled);
    addPass(pm, MlirPass::ttgpuir_assign_latencies, m_options.num_stages);
    addPass(pm, MlirPass::ttgpuir_schedule_loops);
    addPass(pm, MlirPass::ttgpuir_pipeline, m_options.num_stages,
            m_options.dump_enabled);
  } else if (capability_major >= 10) {
    addPass(pm, MlirPass::ttgpuir_fuse_nested_loops);
    addPass(pm, MlirPass::canonicalizer);
    addPass(pm, MlirPass::ttir_triton_licm);
    addPass(pm, MlirPass::ttgpuir_optimize_accumulator_init);
    addPass(pm, MlirPass::ttgpuir_hoist_tmem_alloc, false);

    addCudaPass(pm, CudaPass::ttnvgpuir_promote_lhs_to_tmem);
    addPass(pm, MlirPass::ttgpuir_assign_latencies, m_options.num_stages);
    addPass(pm, MlirPass::ttgpuir_schedule_loops);

    addPass(pm, MlirPass::ttgpuir_warp_specialize, m_options.num_stages);
    addPass(pm, MlirPass::ttgpuir_pipeline, m_options.num_stages,
            m_options.dump_enabled);
    addPass(pm, MlirPass::ttgpuir_optimize_partition_warps);
    addPass(pm, MlirPass::ttgpuir_combine_tensor_select_and_if);
    // hoist again and allow hoisting out of if statements
    addPass(pm, MlirPass::ttgpuir_hoist_tmem_alloc, true);
    addCudaPass(pm, CudaPass::ttnvgpuir_remove_tmem_tokens);
  } else {
    addPass(pm, MlirPass::ttir_triton_licm);
  }

  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::ttir_loop_aware_cse);
  addPass(pm, MlirPass::ttgpuir_prefetch);
  addPass(pm, MlirPass::ttgpuir_optimize_dot_operands, capability_major >= 8);

  addPass(pm, MlirPass::ttgpuir_coalesce_async_copy);
  addCudaPass(pm, CudaPass::ttnvgpuir_optimize_tmem_layouts);
  if (capability_major >= 9) {
    addCudaPass(pm, CudaPass::ttnvgpuir_tma_lowering);
  }
  addPass(pm, MlirPass::ttgpuir_remove_layout_conversions);
  addCudaPass(pm, CudaPass::ttnvgpuir_interleave_tmem);

  addPass(pm, MlirPass::ttgpuir_reduce_data_duplication);
  addPass(pm, MlirPass::ttgpuir_reorder_instructions);
  addPass(pm, MlirPass::ttir_loop_aware_cse);
  addPass(pm, MlirPass::symbol_dce);

  addCudaPass(pm, CudaPass::ttnvgpuir_fence_insertion, capability);
  addCudaPass(pm, CudaPass::ttnvgpuir_lower_mma);

  addPass(pm, MlirPass::sccp);
  addPass(pm, MlirPass::cse);
  addPass(pm, MlirPass::canonicalizer);

  return pm.run(op);
}

LogicalResult CudaBackend::gluon_to_ttgir(MLIRContext &context,
                                          ModuleOp module) {
  PassManager pm(&context);
  auto capability = getCapability();
  auto capability_major = static_cast<int>(capability) / 10;
  auto op = module.getOperation();

  addPass(pm, MlirPass::gluon_inliner);
  addPass(pm, MlirPass::gluon_infer_coalesced_encodings);
  addPass(pm, MlirPass::gluon_resolve_auto_encodings);
  addCudaPass(pm, CudaPass::ttnvgpuir_tma_lowering);
  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::sccp);
  addPass(pm, MlirPass::ttir_loop_aware_cse);
  addPass(pm, MlirPass::gluon_canonicalizer);
  addPass(pm, MlirPass::ttgpuir_combine_tensor_select_and_if);

  return pm.run(op);
}

LogicalResult CudaBackend::make_llir(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  auto capability = getCapability();
  auto capability_major = static_cast<int>(capability) / 10;
  auto ptx_version = m_options.ptx_version.value_or(90);
  auto op = module.getOperation();

  addPass(pm, MlirPass::ttgpuir_combine_tensor_select_and_if);
  addPass(pm, MlirPass::ttgpuir_allocate_warp_groups);
  addPass(pm, MlirPass::scf_to_cf);
  addPass(pm, MlirPass::gluon_inliner);
  addPass(pm, MlirPass::ttgpuir_allocate_shared_memory_nv, capability,
          ptx_version);
  addCudaPass(pm, CudaPass::ttnvgpuir_allocate_tensor_memory);
  addCudaPass(pm, CudaPass::ttnvgpuir_check_matmul_two_cta);
  if (m_options.enable_experimental_consan) {
    addPass(pm, MlirPass::ttgpuir_concurrency_sanitizer);
  }

  addPass(pm, MlirPass::ttgpuir_allocate_global_scratch_memory);
  addCudaPass(pm, CudaPass::ttnvgpuir_proxy_fence_insertion, capability);

  if (m_options.instrumentation) {
    // AXM TODO: Implement instrumentation
    // CUDABackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
  }

  addCudaPass(pm, CudaPass::ttnvgpuir_to_llvmir, capability, ptx_version);
  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::cse);
  addCudaPass(pm, CudaPass::ttnvgpuir_nvgpu_to_llvm);
  addCudaPass(pm, CudaPass::ttnvgpuir_warp_specialize_to_llvm);
  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::cse);
  addPass(pm, MlirPass::symbol_dce);
  addPass(pm, MlirPass::nvvm_to_llvm);
  if (!m_options.disable_line_info) {
    addPass(pm, MlirPass::llvmir_di_scope);
  }

  if (m_options.instrumentation) {
    // AXM TODO: Implement instrumentation
    // CUDABackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)
  }

  return pm.run(op);

  //  # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
  //       llvm.init_targets()
  //       context = llvm.context()
  //       if knobs.compilation.enable_asan:
  //           raise RuntimeError(
  //               "Address Sanitizer Error: Address sanitizer is currently only
  //               supported on the AMD backend")
  //       llvm_mod = llvm.to_module(mod, context)
  //       proc = sm_arch_from_capability(capability)
  //       features = get_features(options, self.target.arch)
  //       triple = 'nvptx64-nvidia-cuda'
  //       nvidia.set_short_ptr()
  //       llvm.attach_datalayout(llvm_mod, triple, proc, features)
  //       nvidia.set_nvvm_reflect_ftz(llvm_mod)

  //       if options.extern_libs and nvidia.has_extern_deps(llvm_mod):
  //           paths = [path for (name, path) in options.extern_libs]
  //           llvm.link_extern_libs(llvm_mod, paths)

  //       llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

  //       # Get some metadata
  //       # warp-specialization mutates num_warps
  //       total_num_warps = src.get_int_attr("ttg.total-num-warps")
  //       if total_num_warps is not None:
  //           metadata["num_warps"] = total_num_warps
  //       metadata["shared"] = src.get_int_attr("ttg.shared")
  //       metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
  //       metadata["global_scratch_size"] =
  //       src.get_int_attr("ttg.global_scratch_memory_size")
  //       metadata["global_scratch_align"] =
  //       src.get_int_attr("ttg.global_scratch_memory_alignment")
  //       metadata["profile_scratch_size"] =
  //       src.get_int_attr("ttg.profile_scratch_memory_size") or 0
  //       metadata["profile_scratch_align"] =
  //       src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1 ret =
  //       str(llvm_mod) del llvm_mod del context return ret
}

std::unique_ptr<mlir::Pass>
CudaBackend::createTritonGPUProxyFenceInsertionWrapper(int32_t capability) {
  ttng::TritonGPUProxyFenceInsertionOptions options;
  options.computeCapability = capability;
  return ttng::createTritonGPUProxyFenceInsertion(options);
}
