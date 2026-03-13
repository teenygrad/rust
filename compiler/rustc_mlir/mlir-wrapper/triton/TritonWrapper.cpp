
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

#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "TritonCompiler.h"

#include <iostream>

using namespace std;

using namespace mlir;
using namespace mlir::triton;

llvm::Module *convertModule(llvm::LLVMContext *llvm_ctx, llvm::Module *module) {
  TritonCompiler compiler("cuda");
  return compiler.applyTritonPasses(llvm_ctx, module);
}

extern "C" LLVMModuleRef LLVMRustApplyTritonPasses(LLVMContextRef ctx,
                                                   LLVMModuleRef module) {
  auto new_module = convertModule(llvm::unwrap(ctx), llvm::unwrap(module));
  return llvm::wrap(new_module);
}
