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

// teenyc-6mv / teenygrad-3w0.10: builds a `!ttg.memdesc<...>` type for an
// N-D, unswizzled (vec=1, perPhase=1, maxPhase=1 -- Triton's own "no
// swizzle" case), single-CTA shared-memory buffer. Hides
// `SwizzledSharedEncodingAttr` / `SharedMemorySpaceAttr` / `CGAEncodingAttr`
// entirely C++-side; Rust only sees (element_type, shape, rank, order,
// mutable_memory). Default `order` (null) is row-major (dim `rank-1`
// varies fastest), matching how a [BLOCK_M, BLOCK_N] tile is stored so
// `memdesc_index` slices dim 0 (rows) for the write phase.
// `memdesc_trans` of that buffer needs the swapped order (e.g. `[0, 1]`
// for 2-D) on the result type -- pass it explicitly. See
// `rustc_mlir::triton::tensor::{memdesc_index, memdesc_trans}`.
extern "C" MlirType mlirCreateTritonGPUSharedMemDescType(
    MlirContext ctx, MlirType elementType, const int64_t *shape, int64_t rank,
    const unsigned *orderIn, bool mutableMemory) {
  MLIRContext *context = unwrap(ctx);
  Type elemTy = unwrap(elementType);

  auto cgaLayout = triton::gpu::CGAEncodingAttr::get1CTALayout(context, rank);
  llvm::SmallVector<unsigned> order;
  if (orderIn) {
    order.append(orderIn, orderIn + rank);
  } else {
    for (int64_t i = 0; i < rank; ++i) {
      order.push_back(static_cast<unsigned>(rank - 1 - i));
    }
  }
  auto encoding = triton::gpu::SwizzledSharedEncodingAttr::get(
      context, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1, order, cgaLayout);
  auto memSpace = triton::gpu::SharedMemorySpaceAttr::get(context);

  llvm::SmallVector<int64_t> shapeVec(shape, shape + rank);
  auto memDescTy = triton::gpu::MemDescType::get(
      context, shapeVec, elemTy, encoding, memSpace, mutableMemory, shapeVec);

  return wrap(memDescTy);
}
