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

#include "TritonWrapper.h"
#include "../MLIRWrapper.h"
#include "TritonCompiler.h"

#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {

struct TritonCompilerHandle {
  std::unique_ptr<TritonCompiler> compiler;
  std::string output;
};
} // namespace triton
} // namespace mlir

static ModuleOp moduleFromC(MlirModule module) {
  Operation *opPtr =
      const_cast<Operation *>(static_cast<const Operation *>(module.ptr));
  return ModuleOp(opPtr);
}

extern "C" ::MlirTritonCompiler mlirTritonCompilerCreate(MlirContext context,
                                                         const char *target,
                                                         const char *options) {
  if (!context.ptr || !target) {
    return ::MlirTritonCompiler{nullptr};
  }

  auto *ctx = unwrap(context);
  auto *handle = new TritonCompiler(ctx, target, options);
  return ::MlirTritonCompiler{handle};
}

extern "C" bool mlirTritonCompilerCompile(::MlirTritonCompiler compiler,
                                          MlirModule module) {
  auto *handle = static_cast<TritonCompilerHandle *>(compiler.ptr);
  if (!handle || !handle->compiler) {
    return false;
  }

  auto moduleOp = moduleFromC(module);
  auto ok = succeeded(handle->compiler->compile(moduleOp));
  if (!ok) {
    handle->output.clear();
    return false;
  }

  std::string printed;
  llvm::raw_string_ostream os(printed);
  moduleOp.print(os);
  os.flush();
  handle->output = std::move(printed);
  return true;
}

extern "C" const char *
mlirTritonCompilerGetOutput(::MlirTritonCompiler compiler) {
  auto *handle = static_cast<TritonCompilerHandle *>(compiler.ptr);
  if (!handle) {
    return nullptr;
  }
  return handle->output.c_str();
}

extern "C" void mlirTritonCompilerFree(::MlirTritonCompiler compiler) {
  delete static_cast<TritonCompilerHandle *>(compiler.ptr);
}
