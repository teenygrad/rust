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

#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "TritonCompiler.h"
#include "backend/CudaBackend.h"

using namespace std;

namespace mlir {
namespace triton {

TritonCompiler::TritonCompiler(MLIRContext &context, std::string target)
    : context(context), target(target) {
  backend = new CudaBackend(target, CudaOptions());
  backend->loadDialects(context);
}

TritonCompiler::~TritonCompiler() { delete backend; }

LogicalResult TritonCompiler::applyTritonPasses(ModuleOp mlir_module) {
  auto result = backend->applyPasses(context, mlir_module, Language::TRITON);
  if (failed(result)) {
    llvm::errs() << "Failed to apply Triton passes. Aborting translation.\n";
  }
  return result;
}

} // namespace triton
} // namespace mlir
