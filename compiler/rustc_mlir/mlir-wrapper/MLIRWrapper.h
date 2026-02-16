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

#ifdef __cplusplus
extern "C" {
#endif

#include <mlir-c/IR.h>

void mlirLoadTritonDialect(MlirContext context);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLIR_WRAPPER_H
