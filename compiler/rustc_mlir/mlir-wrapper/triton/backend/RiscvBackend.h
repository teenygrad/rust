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

#ifndef TRITON_RISCV_BACKEND_H
#define TRITON_RISCV_BACKEND_H

#include <memory>
#include <stdint.h>
#include <string>

#include "llvm/Target/TargetMachine.h"

#include "Backend.h"

namespace mlir {
namespace triton {

// ---------------------------------------------------------------------------
// RISC-V backend compile options (FFI-safe / repr(C))
//
// Mirrors the FFI-safe struct conventions used by CudaCompileOptions in
// CudaBackend.h. RiscvBackend below does not yet lower the incoming
// Triton/MLIR module (see makeTTIR/makeTTGIR/makeLLIR): makeLLVMIR
// synthesizes a placeholder kernel function instead, which makeASM/makeBIN
// then compile for real through LLVM's RISC-V backend.
// ---------------------------------------------------------------------------

/// FFI-safe compilation options for the (stub) RISC-V backend.
struct RiscvCompileOptions {
  const char *target_triple; /// RISC-V target triple
  const char *cpu;           /// RISC-V CPU
  const char *features;      /// RISC-V features
  bool debug;
};

/// RISC-V backend. Does not yet lower the incoming Triton/MLIR module (see
/// makeTTIR/makeTTGIR/makeLLIR); makeLLVMIR instead synthesizes a minimal
/// placeholder `void @<name>()` kernel function, which makeASM/makeBIN
/// compile for real through LLVM's RISC-V backend -- makeBIN links the
/// result into a shared library via `ld.lld` so it can be dlopen'd and run.
class RiscvBackend : public Backend {
public:
  RiscvBackend(std::string target, RiscvCompileOptions options);

  virtual ~RiscvBackend();

  virtual void loadDialects(MLIRContext &context) override;

  virtual LogicalResult makeTTIR(MLIRContext &context,
                                 ModuleOp module) override;

  virtual LogicalResult makeTTGIR(MLIRContext &context,
                                  ModuleOp module) override;

  virtual LogicalResult gluonToTTGIR(MLIRContext &context,
                                     ModuleOp module) override;

  virtual LogicalResult makeLLIR(MLIRContext &context,
                                 ModuleOp module) override;

  virtual LogicalResult makeLLVMIR(MLIRContext &context,
                                   ModuleOp module) override;

  virtual LogicalResult makeASM(MLIRContext &context, ModuleOp module) override;

  virtual LogicalResult makeBIN(MLIRContext &context, ModuleOp module) override;

private:
  /// Creates an LLVM `TargetMachine` for this backend's target triple,
  /// logging why and returning nullptr on failure. Caller owns the
  /// returned pointer. Uses a real, generic LLVM cpu name matching the
  /// triple's width rather than `m_options.cpu` -- see the comment in the
  /// .cpp for why forwarding that Triton-side chip identifier directly
  /// would abort the process instead of failing gracefully.
  llvm::TargetMachine *createRiscvTargetMachine();

  /// Reparses `m_llvmir` (populated by makeLLVMIR) into `context`, logging
  /// why and returning nullptr on failure.
  std::unique_ptr<llvm::Module> parseStoredLLVMIR(llvm::LLVMContext &context);

  /// Locates the `ld.lld` binary used to link makeBIN's object file into a
  /// shared library: `$TEENYC_LLD_PATH` if set, else the first `ld.lld` on
  /// `PATH`. Returns an empty string if neither is found.
  std::string findLld();

  RiscvCompileOptions m_options;
};

} // namespace triton
} // namespace mlir

#endif /*! TRITON_RISCV_BACKEND_H */
