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

/// Builds a `!ttg.memdesc<...>` type for a 1-D, unswizzled, single-CTA
/// shared-memory buffer of `num_elements` scalars of `element_type`. Hides
/// TritonGPU-specific encoding/memory-space attributes entirely C++-side.
MlirType mlirCreateTritonGPUSharedMemDescType(MlirContext context,
                                              MlirType element_type,
                                              int64_t num_elements,
                                              bool mutable_memory);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLIR_WRAPPER_H
