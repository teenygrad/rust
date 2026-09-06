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

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "riscv/Dialect/RVV/IR/Dialect.h"
#include "riscv/Dialect/RVV/IR/RVVDialect.cpp.inc"
// clang-format on

#define GET_OP_CLASSES
#include "riscv/Dialect/RVV/IR/RVVOps.cpp.inc"

using namespace mlir;
using namespace mlir::rvv;

void RVVDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "riscv/Dialect/RVV/IR/RVVOps.cpp.inc"
      >();
}
