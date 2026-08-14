//===-- MLIRWrapper.h - C bindings for MLIR types ---------------*- C++ -*-===//
//
// Provides C-compatible bindings for MLIR types that can be used from Rust
// via FFI. This follows the pattern from LLVM's C API bindings.
//
// These bindings are designed to work alongside melior, providing access to
// MLIR functionality that melior doesn't expose directly.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_WRAPPER_H
#define MLIR_WRAPPER_H

#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Types.h"
#include <cstdint>
#include <mlir-c/IR.h>

using namespace mlir;

DEFINE_C_API_PTR_METHODS(MlirContext, MLIRContext)

DEFINE_C_API_METHODS(MlirType, Type)

#ifdef __cplusplus
extern "C" {
#endif

void mlirTritonLoadDialects(MlirContext context);

MlirType mlirCreateTritonPointerType(MlirType pointee, int address_space);

// teenyc-6mv: CUDA shared-memory primitives (TritonGPU dialect).

/// Loads the `TritonGPU` (`ttg`) dialect into `context`, needed to build
/// shared-memory ops (`ttg.local_alloc`/`local_store`/`local_load`) and the
/// `!ttg.memdesc<...>` type by hand.
void mlirLoadTritonGPUDialect(MlirContext context);

/// Builds a `!ttg.memdesc<...>` type for an N-D, unswizzled, single-CTA
/// shared-memory buffer of `element_type` scalars, shaped by `shape`/`rank`
/// (`rank == 1` reproduces the original 1-D behaviour). Hides
/// TritonGPU-specific encoding/memory-space attributes entirely C++-side.
///
/// `order` is the SwizzledSharedEncodingAttr dimension order (length ==
/// `rank`); a null pointer selects row-major (`[rank-1, ..., 0]`, last dim
/// varies fastest). Pass `[0, 1, ...]` for the column-major / transposed
/// view used as the result of `ttg.memdesc_trans`.
MlirType mlirCreateTritonGPUSharedMemDescType(MlirContext context,
                                              MlirType element_type,
                                              const int64_t *shape,
                                              int64_t rank,
                                              const unsigned *order,
                                              bool mutable_memory);

// `ttg.memdesc_index` / `ttg.memdesc_subslice` / `ttg.memdesc_trans` /
// `ttg.barrier` are built directly from Rust via the raw
// `OperationBuilder::new(...)` idiom (see `rustc_mlir::triton::tensor`) --
// like `local_alloc`/`local_store`/`local_load`, they need no C++-only
// encoding-attribute construction beyond this type helper.

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLIR_WRAPPER_H
