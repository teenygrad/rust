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

#ifndef TRITON_COMPILER_H
#define TRITON_COMPILER_H

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "backend/Backend.h"

namespace mlir {
namespace triton {

class TritonCompiler {
public:
  TritonCompiler(MLIRContext &context, std::string target);
  ~TritonCompiler();

  LogicalResult applyTritonPasses(ModuleOp mlir_module);

private:
  std::string target;

  Backend *backend;
  MLIRContext &context;
};

} // namespace triton
} // namespace mlir

#endif /* TRITON_COMPILER_H */
