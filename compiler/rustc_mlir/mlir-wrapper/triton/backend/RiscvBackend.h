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

#include <stdint.h>
#include <string>

#include "Backend.h"

namespace mlir {
namespace triton {

// ---------------------------------------------------------------------------
// RISC-V backend compile options (FFI-safe / repr(C))
//
// Reserved for a future real RISC-V Triton backend. RiscvBackend below is a
// stub that reports Error::NotImplemented for every codegen stage; these
// fields are not yet consumed. Mirrors the FFI-safe struct conventions used
// by CudaCompileOptions in CudaBackend.h.
// ---------------------------------------------------------------------------

/// FFI-safe compilation options for the (stub) RISC-V backend.
struct RiscvCompileOptions {
  const char *target_triple; ///< NULL = backend default
  const char *cpu;           ///< NULL = backend default
  const char *features;      ///< NULL = backend default
  bool debug;
};

/// Stub RISC-V backend. Implements the Backend interface so
/// TargetBackend_Riscv can be dispatched without crashing, but every codegen
/// stage returns failure with Error::NotImplemented until a real Triton
/// RISC-V backend is implemented.
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
  LogicalResult notImplemented(const char *stage);

  RiscvCompileOptions m_options;
};

} // namespace triton
} // namespace mlir

#endif /*! TRITON_RISCV_BACKEND_H */
