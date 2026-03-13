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

#include <memory>

#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "TritonCompiler.h"
#include "backend/CudaBackend.h"

using namespace std;

namespace mlir {
namespace triton {

TritonCompiler::TritonCompiler(std::string target) {
  this->target = target;

  context.disableMultithreading();

  DialectRegistry registry;
  registry.insert<BuiltinDialect, DLTIDialect, LLVM::LLVMDialect,
                  func::FuncDialect>();
  registry.insert<TritonDialect>();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  backend = new CudaBackend(target, CudaOptions());
  backend->loadDialects(context);
}

TritonCompiler::~TritonCompiler() { delete backend; }

llvm::Module *TritonCompiler::applyTritonPasses(llvm::LLVMContext *llvm_ctx,
                                                llvm::Module *llvm_module) {
  auto result = LogicalResult::success();

  std::unique_ptr<llvm::Module> llvm_module_ptr(llvm_module);
  auto mlir_module =
      mlir::translateLLVMIRToModule(std::move(llvm_module_ptr), &context);

  result = backend->applyPasses(context, mlir_module, Language::TRITON);
  if (failed(result)) {
    llvm::errs() << "Failed to apply Triton passes. Aborting translation.\n";
    return nullptr;
  }

  mlir_module->print(llvm::outs());
  llvm::outs() << "\n";

  auto transformed_llvm_module =
      mlir::translateModuleToLLVMIR(mlir_module->getOperation(), *llvm_ctx);
  llvm::outs() << "After translateModuleToLLVMIR:\n";

  return transformed_llvm_module.release();
}

} // namespace triton
} // namespace mlir
