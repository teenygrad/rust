
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

// #include "mlir-c/IR.h"
// #include "mlir/CAPI/Wrap.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/IR/MLIRContext.h"

#include "../MLIRWrapper.h"

#include "TritonCompiler.h"

using namespace mlir;
using namespace mlir::triton;

extern "C" bool mlirApplyTritonPasses(MlirModule module) {
  Operation *op_ptr =
      const_cast<Operation *>(static_cast<const Operation *>(module.ptr));

  auto module_op = ModuleOp(op_ptr);
  MLIRContext *context = unwrap(mlirModuleGetContext(module));

  TritonCompiler compiler(*context, "cuda");
  return succeeded(compiler.applyTritonPasses(module_op));
}
