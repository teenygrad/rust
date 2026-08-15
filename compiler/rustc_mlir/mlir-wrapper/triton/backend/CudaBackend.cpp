/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"

#include "CudaBackend.h"

#include <cstdlib>
#include <regex>
#include <sstream>

using namespace mlir;
using namespace triton;
using namespace nvidia_gpu;

namespace {

/// Locates the `ptxas` executable: honors `TEENYC_PTXAS_PATH` (explicit
/// override) first, then `CUDA_HOME`/`CUDA_PATH`, then the standard CUDA
/// toolkit install location, then falls back to searching `PATH`.
std::string findPtxas() {
  if (const char *env = std::getenv("TEENYC_PTXAS_PATH")) {
    if (llvm::sys::fs::exists(env)) {
      return env;
    }
  }

  for (const char *envVar : {"CUDA_HOME", "CUDA_PATH"}) {
    if (const char *base = std::getenv(envVar)) {
      llvm::SmallString<256> candidate(base);
      llvm::sys::path::append(candidate, "bin", "ptxas");
      if (llvm::sys::fs::exists(candidate)) {
        return std::string(candidate.str());
      }
    }
  }

  static const char *defaultPath = "/usr/local/cuda/bin/ptxas";
  if (llvm::sys::fs::exists(defaultPath)) {
    return defaultPath;
  }

  auto found = llvm::sys::findProgramByName("ptxas");
  if (found) {
    return *found;
  }

  return {};
}

/// Parses the register/spill/stack-frame/constant-memory statistics that
/// `ptxas -v` writes to stderr, e.g.:
///
///   ptxas info    : Compiling entry function 'kernel' for 'sm_90a'
///   ptxas info    : Function properties for kernel
///       32 bytes stack frame, 16 bytes spill stores, 8 bytes spill loads
///   ptxas info    : Used 40 registers, 384 bytes cmem[0], 8 bytes cmem[2]
///
/// This backend compiles a single kernel per module (see the single-name
/// assumption in makeASM), so it's sufficient to search the whole log for
/// each field rather than associating blocks with individual kernel names.
/// Returns false (leaving `metadata` untouched) if the "Used N registers"
/// line isn't found, which means ptxas's output wasn't in the expected
/// format -- we'd rather report nothing than fabricate zeros.
bool parsePtxasStats(const std::string &log, KernelMetadata &metadata) {
  static const std::regex regsRe(R"(Used (\d+) registers)");
  static const std::regex stackRe(R"((\d+) bytes stack frame)");
  static const std::regex spillStoresRe(R"((\d+) bytes spill stores)");
  static const std::regex spillLoadsRe(R"((\d+) bytes spill loads)");
  static const std::regex cmemRe(R"((\d+) bytes cmem\[(\d+)\])");

  std::smatch match;
  if (!std::regex_search(log, match, regsRe)) {
    return false;
  }
  metadata.num_regs = std::stoi(match[1].str());

  if (std::regex_search(log, match, stackRe)) {
    metadata.stack_frame = std::stoi(match[1].str());
  }
  if (std::regex_search(log, match, spillStoresRe)) {
    metadata.spill_stores = std::stoi(match[1].str());
  }
  if (std::regex_search(log, match, spillLoadsRe)) {
    metadata.spill_loads = std::stoi(match[1].str());
  }

  for (auto it = std::sregex_iterator(log.begin(), log.end(), cmemRe);
       it != std::sregex_iterator(); ++it) {
    int32_t bytes = std::stoi((*it)[1].str());
    int32_t bank = std::stoi((*it)[2].str());
    metadata.cmem_banks.push_back({bank, bytes});
  }

  metadata.has_ptxas_stats = true;
  return true;
}

} // namespace

CudaBackend::CudaBackend(std::string target, CudaCompileOptions options)
    : Backend(target), m_options(options) {
  m_capability = static_cast<Capability>(options.capability);

  llvm::outs() << "CudaBackend: capability = " << m_capability << "\n";
  if (options.ptx_version.has_value) {
    llvm::outs() << "CudaBackend: ptx_version = " << options.ptx_version.value
                 << "\n";
  } else {
    llvm::outs() << "CudaBackend: ptx_version = not set\n";
  }
  llvm::outs() << "CudaBackend: ptx_version = " << options.ptx_version.value
               << "\n";
}

CudaBackend::~CudaBackend() {
  // nop
}

void CudaBackend::loadDialects(MLIRContext &context) {
  DialectRegistry registry;

  registry.insert<mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                  mlir::triton::nvgpu::NVGPUDialect,
                  mlir::triton::nvws::NVWSDialect>();

  // Register the LLVM-IR translation interfaces for every dialect that survives
  // into the post-`convert-triton-gpu-to-llvm` module. `translateModuleToLLVMIR`
  // (see makeLLVMIR) looks these up on the *module's* context, so without them
  // it fails with "missing LLVMTranslationDialectInterface registration ... for
  // op: builtin.module". NVVM alone is not enough — builtin/llvm/gpu are also
  // present in the lowered kernel.
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerGPUDialectTranslation(registry);
  registerNVVMDialectTranslation(registry);

  context.appendDialectRegistry(registry);
}

Capability CudaBackend::getCapability() const { return m_capability; }

LogicalResult CudaBackend::makeLLVMIR(MLIRContext &context, ModuleOp module) {
  llvm::LLVMContext llvmContext;

  // Initialize LLVM targets (required for NVPTX/codegen)
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Address Sanitizer is only supported on the AMD backend; not applicable
  // when using the base Backend (no knobs). Subclasses can override and check
  // enable_asan and return failure() for NVIDIA.

  // Translate MLIR module (LLVM dialect) to LLVM IR module
  auto llvmMod =
      mlir::translateModuleToLLVMIR(module.getOperation(), llvmContext);
  if (!llvmMod) {
    llvm::errs() << "Failed to translate MLIR module to LLVM IR\n";
    return LogicalResult::failure();
  }

  // Set target triple for NVIDIA PTX
  auto triple = llvm::Triple("nvptx64-nvidia-cuda");
  llvmMod->setTargetTriple(triple);

  // Attach data layout for NVPTX64 (matches triple/capability; proc/features
  // would require TargetMachine if layout varied per SM).
  static const char nvptx64DataLayout[] =
      "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-"
      "f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:"
      "64";
  llvmMod->setDataLayout(llvm::DataLayout(nvptx64DataLayout));

  if (m_options.enable_reflect_ftz) {
    llvmMod->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz", 1u);
  }

  // Collect user-specified extern libs.
  std::vector<std::string> libPaths;
  for (size_t i = 0; i < m_options.extern_libs_len; ++i) {
    libPaths.push_back(m_options.extern_lib_values[i]);
  }

  // Auto-link libdevice when the module references any __nv_* device functions.
  // MLIR lowers math.sqrt / math.rsqrt etc. to __nv_sqrtf / __nv_rsqrtf
  // declarations; the PTX JIT cannot resolve these without libdevice linked at
  // LLVM IR level.
  auto needsLibdevice = [&]() -> bool {
    for (const auto &F : llvmMod->functions()) {
      if (F.isDeclaration() && F.getName().starts_with("__nv_")) {
        return true;
      }
    }
    return false;
  };

  if (needsLibdevice()) {
    // Prefer the env-var override, then the standard CUDA toolkit path.
    const char *env_path = std::getenv("CUDA_LIBDEVICE_PATH");
    static const char *candidates[] = {
        env_path,
        "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
    };
    bool found = false;
    for (const char *path : candidates) {
      if (path && llvm::sys::fs::exists(path)) {
        libPaths.push_back(path);
        found = true;
        break;
      }
    }
    if (!found) {
      llvm::errs() << "Warning: module references __nv_* device functions but "
                      "libdevice.10.bc was not found; PTX JIT will likely fail. "
                      "Set CUDA_LIBDEVICE_PATH to override.\n";
    }
  }

  if (!libPaths.empty()) {
    auto result = linkExternLibs(llvmContext, *llvmMod, libPaths);
    if (failed(result)) {
      return result;
    }
  }

  // Do NOT run host O3 here: the pipeline has no TargetMachine for NVPTX, so
  // passes like the CGSCC devirt repeater mis-analyze GPU IR (and can loop on
  // infinite-loop / recursive functions that appear in Rust no_core stubs).
  // Optimization is done at the correct level by llvmTranslateToAsm via the
  // NVPTX TargetMachine's own CodeGen pipeline.

  // Serialize LLVM module to string and store in m_llvmir
  llvm::raw_string_ostream os(m_llvmir);
  llvmMod->print(os, nullptr);

  return LogicalResult::success();
}

LogicalResult CudaBackend::makeASM(MLIRContext &context, ModuleOp module) {
  int ptx_version = this->m_options.ptx_version.has_value
                        ? this->m_options.ptx_version.value
                        : 90;
  std::string features = ""; // AXM TODO - get_features

  std::string proc = "sm_" + std::to_string(this->m_capability);
  if (this->m_capability >= 90) {
    proc += "a";
  }

  std::string triple = "nvptx64-nvidia-cuda";
  std::vector<std::string> flags = {"nvptx-mad-wide-opt"};

  // 2. Translate LLVM module to assembly (PTX)
  // This part is pseudo-code, as actual translation will depend on LLVM API
  // presence.
  std::string src_asm = m_llvmir; // Assume m_llvmir contains the LLVM IR for
                                  // the module serialized right before
  std::string ret = llvmTranslateToAsm(src_asm, triple, proc, features, flags,
                                       m_options.enable_fp_fusion, false);
  if (ret.empty()) {
    llvm::errs() << "Failed to translate LLVM IR to PTX\n";
    llvm::errs() << "LLVM IR: " << src_asm << "\n";
    llvm::errs() << "Triple: " << triple << "\n";
    return LogicalResult::failure();
  }
  // 3. Find kernel name
  std::regex kernel_re(R"(\.visible \.entry ([a-zA-Z_][a-zA-Z0-9_]*))");
  std::smatch match;
  std::string kernel_name;
  if (std::regex_search(ret, match, kernel_re)) {
    kernel_name = match[1].str();
  } else {
    llvm::errs() << "Could not find kernel name in PTX output\n";
    return LogicalResult::failure();
  }
  m_metadata.name = kernel_name;

  // 4. Post-process version and target
  char ptx_major_minor[8];
  snprintf(ptx_major_minor, sizeof(ptx_major_minor), "%d.%d", ptx_version / 10,
           ptx_version % 10);
  ret = std::regex_replace(ret, std::regex(R"(\.version \d+\.\d+)"),
                           ".version " + std::string(ptx_major_minor));
  ret = std::regex_replace(ret, std::regex(R"(\.target sm_\d+)"),
                           ".target sm_" + std::to_string(m_capability));

  // 5. Remove debug flag if desired
  // No 'knobs' defined; always remove for now or leave as TODO.
  ret = std::regex_replace(ret, std::regex(R"(,\s*debug|debug,\s*)"), "");

  // 6. Append kernel metadata as PTX line comments. PTX comments are ignored
  //    by ptxas and the CUDA driver, so this is safe. The Rust side parses
  //    these lines to recover launch parameters without a separate FFI channel.
  ret += "\n// --- triton-metadata ---\n";
  ret += "// meta:name="                  + m_metadata.name + "\n";
  ret += "// meta:num_warps="             + std::to_string(m_metadata.num_warps) + "\n";
  ret += "// meta:num_ctas="              + std::to_string(m_metadata.num_ctas) + "\n";
  ret += "// meta:shared="               + std::to_string(m_metadata.shared) + "\n";
  ret += "// meta:tmem_size="             + std::to_string(m_metadata.tmem_size) + "\n";
  ret += "// meta:global_scratch_size="   + std::to_string(m_metadata.global_scratch_size) + "\n";
  ret += "// meta:global_scratch_align="  + std::to_string(m_metadata.global_scratch_align) + "\n";
  ret += "// meta:profile_scratch_size="  + std::to_string(m_metadata.profile_scratch_size) + "\n";
  ret += "// meta:profile_scratch_align=" + std::to_string(m_metadata.profile_scratch_align) + "\n";

  // 7. Save PTX (exposed via getASM())
  m_asm = std::move(ret);
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
  switch (pass) {
  case CudaPass::ttnvgpuir_to_llvmir:
    pm.addPass(createConvertTritonGPUToLLVMPass(arg0, arg1));
    return std::nullopt;
  case CudaPass::allocate_shared_memory_nv:
    pm.addPass(mlir::triton::createAllocateSharedMemoryNvPass(arg0, arg1));
    return std::nullopt;
  default:
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid nvidia pass";
    return m_last_error;
  }
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

LogicalResult CudaBackend::makeTTIR(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  auto capability = getCapability();
  auto op = module.getOperation();

  if (m_options.debug) {
    pm.enableIRPrinting();
  }

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

LogicalResult CudaBackend::makeTTGIR(MLIRContext &context, ModuleOp module) {
  auto capability = getCapability();
  auto capability_major = static_cast<int>(capability) / 10;
  auto op = module.getOperation();
  auto emuTF32 = (capability_major >= 8);

  if (m_options.maxnreg.has_value) {
    auto maxnreg = m_options.maxnreg.value;
    OpBuilder builder(&context);
    op->setAttr("ttg.maxnreg", builder.getI32IntegerAttr(maxnreg));
  }

  std::string capability_str =
      std::string("cuda:").append(std::to_string(static_cast<int>(capability)));

  // Run ttir_convert_to_ttgpuir as a separate pass to get early crash detection.
  {
    PassManager pm0(&context);
    if (m_options.debug) {
      pm0.enableIRPrinting();
    }
    addPass(pm0, MlirPass::ttir_convert_to_ttgpuir, capability_str,
            m_options.num_warps, 32, m_options.num_ctas);
    auto r = pm0.run(op);
    if (failed(r)) return r;
  }

  PassManager pm(&context);
  if (m_options.debug) {
    pm.enableIRPrinting();
  }

  // teenyc-6mv: stage any values a front-end/codegen marked with
  // `ttg.stage_shared` through shared memory. This runs immediately after
  // `convert-triton-to-tritongpu` (which just assigned a distributed encoding
  // to every tensor and preserved the marker) so the inserted
  // local_alloc/local_store/local_load reuse that encoding, avoiding both the
  // null-encoding crash (Gluon path) and unresolved encoded<->unencoded
  // materialization (direct path). A no-op when nothing is marked.
  addPass(pm, MlirPass::ttgpuir_stage_shared_memory);

  // teenyc-6mv / teenygrad-3w0.10: lower any `tt.shared_alloc`/
  // `shared_store_index`/`shared_barrier`/`shared_trans`/`shared_load_index`
  // marker ops (an indexed shared-memory buffer, e.g. a transpose staging
  // area -- something `ttg.stage_shared` can't express) into the real
  // `ttg.local_alloc`/`memdesc_index`/`local_store`/`barrier`/`memdesc_trans`/
  // `local_load` sequence. Must also run right after conversion, for the same
  // reason as the pass above. A no-op when nothing is marked.
  addPass(pm, MlirPass::ttgpuir_lower_indexed_shared_memory);

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

LogicalResult CudaBackend::gluonToTTGIR(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  auto capability = getCapability();
  auto capability_major = static_cast<int>(capability) / 10;
  auto op = module.getOperation();

  addPass(pm, MlirPass::gluon_inliner);
  addPass(pm, MlirPass::gluon_infer_coalesced_encodings);
  addPass(pm, MlirPass::gluon_resolve_auto_encodings);
  addCudaPass(pm, CudaPass::ttnvgpuir_tma_lowering);
  addPass(pm, MlirPass::gluon_canonicalizer);
  addPass(pm, MlirPass::sccp);
  addPass(pm, MlirPass::ttir_loop_aware_cse);
  addPass(pm, MlirPass::gluon_canonicalizer);
  addPass(pm, MlirPass::ttgpuir_combine_tensor_select_and_if);

  return pm.run(op);
}

LogicalResult CudaBackend::makeLLIR(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  auto capability = getCapability();
  auto capability_major = static_cast<int>(capability) / 10;
  auto ptx_version =
      m_options.ptx_version.has_value ? m_options.ptx_version.value : 90;
  auto op = module.getOperation();

  if (m_options.debug) {
    pm.enableIRPrinting();
  }

  addPass(pm, MlirPass::ttgpuir_combine_tensor_select_and_if);
  addPass(pm, MlirPass::ttgpuir_allocate_warp_groups);
  addPass(pm, MlirPass::scf_to_cf);
  addPass(pm, MlirPass::gluon_inliner);
  addCudaPass(pm, CudaPass::allocate_shared_memory_nv, capability,
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
  if (!m_options.disable_line_info && !m_options.dump_ir_extract_di_local_variables) {
    addPass(pm, MlirPass::llvmir_di_scope);
  }

  if (m_options.instrumentation) {
    // AXM TODO: Implement instrumentation
    // CUDABackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)
  }

  auto result = pm.run(op);

  if (succeeded(result) && m_options.dump_ir_extract_di_local_variables) {
    if (!m_options.disable_line_info) {
      PassManager diScopePm(&context);
      if (m_options.debug) {
        diScopePm.enableIRPrinting();
      }
      addPass(diScopePm, MlirPass::llvmir_di_scope);
      result = diScopePm.run(op);
    }

    if (succeeded(result)) {
      // Insert dbg intrinsics with several DI attributes (source var name,
      // type info). This pass and llvmir_di_scope must run in separate
      // PassManagers -- combining them into the main pipeline triggers a
      // segfault without any error message, possibly an MLIR/pybind11 bug.
      PassManager diLocalVarPm(&context);
      if (m_options.debug) {
        diLocalVarPm.enableIRPrinting();
      }
      addPass(diLocalVarPm, MlirPass::llvmir_di_local_variable);
      result = diLocalVarPm.run(op);
    }
  }

  if (succeeded(result)) {
    // Read resource metadata written by the allocation passes as MLIR module
    // attributes. These are later appended to the PTX as comments so Rust can
    // recover them without a separate FFI channel.
    auto getInt = [&](llvm::StringRef key, int32_t def = 0) -> int32_t {
      auto attr = op->getAttrOfType<mlir::IntegerAttr>(key);
      return attr ? static_cast<int32_t>(attr.getInt()) : def;
    };

    // Warp-specialization may have mutated the warp count, so prefer the
    // post-pipeline attribute over the original options value.
    auto totalWarps = op->getAttrOfType<mlir::IntegerAttr>("ttg.total-num-warps");
    m_metadata.num_warps   = totalWarps ? static_cast<int32_t>(totalWarps.getInt())
                                        : m_options.num_warps;
    m_metadata.num_ctas              = m_options.num_ctas;
    m_metadata.shared                = getInt("ttg.shared");
    m_metadata.tmem_size             = getInt("ttg.tensor_memory_size");
    m_metadata.global_scratch_size   = getInt("ttg.global_scratch_memory_size");
    m_metadata.global_scratch_align  = getInt("ttg.global_scratch_memory_alignment", 1);
    m_metadata.profile_scratch_size  = getInt("ttg.profile_scratch_memory_size");
    m_metadata.profile_scratch_align = getInt("ttg.profile_scratch_memory_alignment", 1);
  }
  return result;
}

std::unique_ptr<mlir::Pass>
CudaBackend::createTritonGPUProxyFenceInsertionWrapper(int32_t capability) {
  ttng::TritonGPUProxyFenceInsertionOptions options;
  options.computeCapability = capability;
  return ttng::createTritonGPUProxyFenceInsertion(options);
}

LogicalResult CudaBackend::makeBIN(MLIRContext &context, ModuleOp module) {
  if (!m_options.generate_bin) {
    // Not requested: stay a no-op, exactly as before. m_bin stays empty and
    // the ptxas-derived KernelMetadata fields stay at their zero defaults
    // (has_ptxas_stats stays false), since nothing was actually measured.
    return LogicalResult::success();
  }

  std::string ptxas = findPtxas();
  if (ptxas.empty()) {
    llvm::errs() << "CudaBackend: generate_bin was requested but `ptxas` "
                    "could not be found. Set TEENYC_PTXAS_PATH, CUDA_HOME, "
                    "or CUDA_PATH, or ensure `ptxas` is on PATH.\n";
    return LogicalResult::failure();
  }

  llvm::SmallString<128> ptxPath, binPath, logPath;
  int ptxFd;
  if (auto ec = llvm::sys::fs::createTemporaryFile("teenyc-triton", "ptx",
                                                    ptxFd, ptxPath)) {
    llvm::errs() << "CudaBackend: failed to create temp PTX file: "
                 << ec.message() << "\n";
    return LogicalResult::failure();
  }
  {
    llvm::raw_fd_ostream ptxOut(ptxFd, /*shouldClose=*/true);
    ptxOut << m_asm;
  }

  if (auto ec = llvm::sys::fs::createTemporaryFile("teenyc-triton", "cubin",
                                                    binPath)) {
    llvm::errs() << "CudaBackend: failed to create temp cubin path: "
                 << ec.message() << "\n";
    llvm::sys::fs::remove(ptxPath);
    return LogicalResult::failure();
  }
  if (auto ec =
          llvm::sys::fs::createTemporaryFile("teenyc-triton", "log", logPath)) {
    llvm::errs() << "CudaBackend: failed to create temp log path: "
                 << ec.message() << "\n";
    llvm::sys::fs::remove(ptxPath);
    llvm::sys::fs::remove(binPath);
    return LogicalResult::failure();
  }

  auto cleanup = [&]() {
    llvm::sys::fs::remove(ptxPath);
    llvm::sys::fs::remove(binPath);
    llvm::sys::fs::remove(logPath);
  };

  std::string arch = "sm_" + std::to_string(m_capability);
  if (m_capability >= 90) {
    arch += "a";
  }

  // Mirrors the ptxas invocation in the vendored Triton Python backend
  // (third_party/nvidia/backend/compiler.py, make_cubin).
  std::vector<std::string> args = {ptxas};
  if (!m_options.disable_line_info) {
    args.push_back("-lineinfo");
  }
  if (!m_options.enable_fp_fusion) {
    args.push_back("--fmad=false");
  }
  args.push_back("-v");
  args.push_back("--regAllocOptLevel=2");
  if (m_options.ptx_options) {
    std::istringstream extra(m_options.ptx_options);
    std::string opt;
    while (extra >> opt) {
      args.push_back(opt);
    }
  }
  args.push_back("--gpu-name=" + arch);
  args.push_back(std::string(ptxPath.str()));
  args.push_back("-o");
  args.push_back(std::string(binPath.str()));

  std::vector<llvm::StringRef> argRefs(args.begin(), args.end());
  std::optional<llvm::StringRef> redirects[] = {std::nullopt, std::nullopt,
                                                llvm::StringRef(logPath)};

  std::string execError;
  bool executionFailed = false;
  int rc = llvm::sys::ExecuteAndWait(
      ptxas, argRefs, /*Env=*/std::nullopt, redirects,
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, &execError, &executionFailed);

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> logBuf =
      llvm::MemoryBuffer::getFile(logPath);
  std::string log = logBuf ? (*logBuf)->getBuffer().str() : std::string();

  if (executionFailed || rc != 0) {
    llvm::errs() << "CudaBackend: `ptxas` failed (exit code " << rc << ")\n";
    if (!execError.empty()) {
      llvm::errs() << execError << "\n";
    }
    llvm::errs() << "ptxas stderr:\n" << log << "\n";
    cleanup();
    return LogicalResult::failure();
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> binBuf =
      llvm::MemoryBuffer::getFile(binPath, /*IsText=*/false);
  if (!binBuf) {
    llvm::errs()
        << "CudaBackend: ptxas reported success but produced no cubin at "
        << binPath << "\n";
    cleanup();
    return LogicalResult::failure();
  }
  // m_bin crosses the FFI boundary as a null-terminated C string (see
  // Backend::getBIN), which can't safely carry arbitrary binary data
  // (embedded NULs, invalid UTF-8), so the cubin is base64-encoded here and
  // must be decoded by the Rust caller.
  m_bin = llvm::encodeBase64((*binBuf)->getBuffer());

  if (parsePtxasStats(log, m_metadata)) {
    // Only append these once they were actually measured: their presence in
    // the PTX comment block is itself the "was a bin requested and did
    // ptxas run" signal the Rust-side parser keys off.
    m_asm += "// meta:num_regs=" + std::to_string(m_metadata.num_regs) + "\n";
    m_asm +=
        "// meta:spill_stores=" + std::to_string(m_metadata.spill_stores) + "\n";
    m_asm +=
        "// meta:spill_loads=" + std::to_string(m_metadata.spill_loads) + "\n";
    m_asm +=
        "// meta:stack_frame=" + std::to_string(m_metadata.stack_frame) + "\n";
    if (!m_metadata.cmem_banks.empty()) {
      std::string cmem;
      for (size_t i = 0; i < m_metadata.cmem_banks.size(); ++i) {
        if (i) cmem += ",";
        cmem += std::to_string(m_metadata.cmem_banks[i].bank) + ":" +
                std::to_string(m_metadata.cmem_banks[i].bytes);
      }
      m_asm += "// meta:cmem=" + cmem + "\n";
    }
  }

  cleanup();
  return LogicalResult::success();
}

LogicalResult
CudaBackend::linkExternLibs(llvm::LLVMContext &llvmContext,
                            llvm::Module &module,
                            const std::vector<std::string> &libPaths) {
  llvm::Linker linker(module);
  for (const auto &libPath : libPaths) {
    auto buf = llvm::MemoryBuffer::getFile(libPath);
    if (!buf) {
      llvm::errs() << "Failed to get memory buffer: " << libPath << "\n";
      return LogicalResult::failure();
    }

    // Use parseBitcodeFile (not getLazyBitcodeModule) so that all function
    // bodies from libdevice are fully materialized before linking. The lazy
    // variant leaves functions as declarations, so the linker cannot inline
    // them and __nv_rsqrtf / __nv_sqrtf etc. remain as .extern in PTX.
    // LinkOnlyNeeded ensures we only pull in symbols actually referenced by
    // our module, keeping PTX size reasonable.
    auto src = llvm::parseBitcodeFile((*buf)->getMemBufferRef(), llvmContext);
    if (!src) {
      llvm::errs() << "Failed to parse bitcode file: " << libPath << "\n";
      return LogicalResult::failure();
    }

    if (linker.linkInModule(std::move(*src),
                            llvm::Linker::Flags::LinkOnlyNeeded)) {
      llvm::errs() << "Failed to link extern library: " << libPath << "\n";
      return LogicalResult::failure();
    }
  }

  return LogicalResult::success();
}

/// Translates LLVM IR to NVPTX assembly (PTX) using the given triple, CPU,
/// and features. Returns the PTX string or empty on error.
std::string CudaBackend::llvmTranslateToAsm(
    const std::string &llvmIr, const std::string &tripleStr,
    const std::string &cpu, const std::string &features,
    const std::vector<std::string> & /*flags*/, bool /*enableFpFusion*/,
    bool /*verbose*/) {
  // Targets were already initialized in makeLLVMIR; no need to repeat.

  llvm::LLVMContext ctx;
  auto buf = llvm::MemoryBuffer::getMemBuffer(llvmIr, "<llvm-ir>");
  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::Module> mod =
      llvm::parseIR(buf->getMemBufferRef(), err, ctx);
  if (!mod) {
    err.print("CudaBackend", llvm::errs());
    return {};
  }

  std::string targetError;
  llvm::Triple triple(llvm::Triple::normalize(tripleStr));
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.getTriple(), targetError);
  if (!target) {
    llvm::errs() << targetError << "\n";
    return {};
  }

  llvm::TargetOptions opts;
  llvm::TargetMachine *tm = target->createTargetMachine(
      triple, cpu, features, opts, llvm::Reloc::Static, std::nullopt,
      llvm::CodeGenOptLevel::Default);
  if (!tm)
    return {};

  llvm::SmallVector<char, 0> asmBuf;
  {
    llvm::raw_svector_ostream os(asmBuf);
    llvm::legacy::PassManager pm;
    if (tm->addPassesToEmitFile(pm, os, nullptr,
                                llvm::CodeGenFileType::AssemblyFile)) {
      llvm::errs() << "Failed to add passes to emit file\n";
      delete tm;
      return {};
    }
    (void)pm.run(*mod);
  }
  delete tm;
  // Build return string after stream and pass manager are destroyed so
  // no shared state can cause use-after-free or hang when copying.
  return std::string(asmBuf.data(), asmBuf.size());
}
