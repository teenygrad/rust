//===-- MLIRWrapper.cpp - C bindings for MLIR types -----------------------===//
//
// Implementation of C-compatible bindings for MLIR types.
//
//===----------------------------------------------------------------------===//

#include "MLIRWrapper.h"

#include "mlir/IR/DialectRegistry.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

extern "C" void mlirLoadTritonDialect(MlirContext ctx) {
  MLIRContext *context = unwrap(ctx);
  DialectRegistry registry;

  registry.insert<triton::TritonDialect>();
  context->appendDialectRegistry(registry);

  context->loadDialect<triton::TritonDialect>();
}

extern "C" MlirType mlirCreateTritonPointerType(MlirType pointee,
                                                int address_space) {
  auto type = unwrap(pointee);

  auto pointer_type = triton::getPointerType(type, address_space);

  return wrap(pointer_type);
}

extern "C" void mlirLoadTritonGPUDialect(MlirContext ctx) {
  MLIRContext *context = unwrap(ctx);
  DialectRegistry registry;

  registry.insert<triton::gpu::TritonGPUDialect>();
  context->appendDialectRegistry(registry);

  context->loadDialect<triton::gpu::TritonGPUDialect>();
}

// teenyc-6mv: builds a `!ttg.memdesc<...>` type for a 1-D, unswizzled
// (vec=1, perPhase=1, maxPhase=1 -- Triton's own "no swizzle" case),
// single-CTA shared-memory buffer. Hides `SwizzledSharedEncodingAttr` /
// `SharedMemorySpaceAttr` / `CGAEncodingAttr` entirely C++-side; Rust only
// sees (element_type, num_elements, mutable_memory).
extern "C" MlirType mlirCreateTritonGPUSharedMemDescType(MlirContext ctx,
                                                         MlirType elementType,
                                                         int64_t numElements,
                                                         bool mutableMemory) {
  MLIRContext *context = unwrap(ctx);
  Type elemTy = unwrap(elementType);

  auto cgaLayout = triton::gpu::CGAEncodingAttr::get1CTALayout(context, /*rank=*/1);
  llvm::SmallVector<unsigned, 1> order{0};
  auto encoding = triton::gpu::SwizzledSharedEncodingAttr::get(
      context, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1, order, cgaLayout);
  auto memSpace = triton::gpu::SharedMemorySpaceAttr::get(context);

  llvm::SmallVector<int64_t, 1> shape{numElements};
  auto memDescTy = triton::gpu::MemDescType::get(
      context, shape, elemTy, encoding, memSpace, mutableMemory, shape);

  return wrap(memDescTy);
}
