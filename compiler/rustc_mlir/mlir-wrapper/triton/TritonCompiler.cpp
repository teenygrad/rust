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

#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/MLIRContext.h"

#include "TritonCompiler.h"
#include "backend/CudaBackend.h"

using namespace std;

namespace mlir {
namespace triton {

TritonCompiler::TritonCompiler(MLIRContext *context, std::string target,
                               std::string options)
    : Compiler(context, target, options) {
  backend = new CudaBackend(target, CudaOptions());
  backend->loadDialects(*context);
}

TritonCompiler::~TritonCompiler() { delete backend; }

LogicalResult TritonCompiler::compile(ModuleOp mlir_module) {
  auto result = applyTritonPasses(mlir_module);
  if (failed(result)) {
    llvm::errs() << "Failed to apply Triton passes. Aborting translation.\n";
  }

  // The module is now in LLIR format, so we can generate the output from it.
  CudaBackend *cudaBackend = dynamic_cast<CudaBackend *>(backend);
  if (cudaBackend != nullptr) {
    auto ptxResult = cudaBackend->generatePtx(*context, mlir_module);
    if (failed(ptxResult)) {
      llvm::errs() << "Failed to generate PTX from CUDA backend.\n";
    }
  }

  if (failed(result)) {
    llvm::errs() << "Failed to generate PTX. Aborting translation.\n";
  }
}

LogicalResult TritonCompiler::applyTritonPasses(ModuleOp mlir_module) {
  auto result = backend->applyPasses(*context, mlir_module, Language::TRITON);
  if (failed(result)) {
    llvm::errs() << "Failed to apply Triton passes. Aborting translation.\n";
  }

  return result;
}

} // namespace triton
} // namespace mlir
