//===-- MLIRWrapper.cpp - C bindings for MLIR types -----------------------===//
//
// Implementation of C-compatible bindings for MLIR types.
//
//===----------------------------------------------------------------------===//

#include "MLIRWrapper.h"
#include "mlir/CAPI/Wrap.h"

#include "mlir/IR/DialectRegistry.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

DEFINE_C_API_PTR_METHODS(MlirContext, MLIRContext)

extern "C" void mlirLoadTritonDialect(MlirContext ctx) {
  MLIRContext *context = unwrap(ctx);
  DialectRegistry registry;

  registry.insert<triton::TritonDialect>();
  context->appendDialectRegistry(registry);

  context->loadDialect<triton::TritonDialect>();
}
