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

#include "RiscvBackend.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace triton {

RiscvBackend::RiscvBackend(std::string target, RiscvCompileOptions options)
    : Backend(target), m_options(options) {}

RiscvBackend::~RiscvBackend() {}

void RiscvBackend::loadDialects(MLIRContext &context) {
  // No RISC-V-specific dialects to register yet; this backend is a stub.
}

LogicalResult RiscvBackend::notImplemented(const char *stage) {
  m_last_error = std::make_optional(Error::NotImplemented);
  m_last_error_string =
      std::string("RISC-V backend: ") + stage + " is not implemented yet";
  llvm::errs() << m_last_error_string << "\n";
  return LogicalResult::failure();
}

LogicalResult RiscvBackend::makeTTIR(MLIRContext &context, ModuleOp module) {
  return notImplemented("makeTTIR");
}

LogicalResult RiscvBackend::makeTTGIR(MLIRContext &context, ModuleOp module) {
  return notImplemented("makeTTGIR");
}

LogicalResult RiscvBackend::gluonToTTGIR(MLIRContext &context,
                                         ModuleOp module) {
  // NOP for RISC-V backend
  return success();
}

LogicalResult RiscvBackend::makeLLIR(MLIRContext &context, ModuleOp module) {
  return notImplemented("makeLLIR");
}

LogicalResult RiscvBackend::makeLLVMIR(MLIRContext &context, ModuleOp module) {
  return notImplemented("makeLLVMIR");
}

LogicalResult RiscvBackend::makeASM(MLIRContext &context, ModuleOp module) {
  return notImplemented("makeASM");
}

LogicalResult RiscvBackend::makeBIN(MLIRContext &context, ModuleOp module) {
  return notImplemented("makeBIN");
}

} // namespace triton
} // namespace mlir
