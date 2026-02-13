//===-- TritonWrapper.h - C bindings for Triton types -----------*- C++ -*-===//
//
// Provides C-compatible bindings for Triton dialect types and attributes
// that can be used from Rust via FFI.
//
// This wraps OpenAI Triton's MLIR dialect types including:
// - Triton IR types (pointer, tensor)
// - Triton GPU types (blocked encoding, shared encoding)
// - Triton attributes (program ID, etc.)
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_WRAPPER_H
#define TRITON_WRAPPER_H

#include "MLIRWrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

/// Initialize Triton dialects and register them with the context
/// This registers: triton, triton_gpu, and related dialects
MLIRRustResult tritonRustInitDialects(MLIRContextRef ctx);

/// Register only the core Triton dialect
MLIRRustResult tritonRustRegisterTritonDialect(MLIRDialectRegistryRef registry);

/// Register the Triton GPU dialect
MLIRRustResult tritonRustRegisterTritonGPUDialect(MLIRDialectRegistryRef registry);

/// Check if Triton dialects are available (compiled with TRITON_ENABLED)
int tritonRustIsAvailable(void);

//===----------------------------------------------------------------------===//
// Triton Pointer Type
//===----------------------------------------------------------------------===//

/// Create a Triton pointer type with the given pointee type and address space
/// address_space: 0 = generic, 1 = global, 3 = shared
MLIRTypeRef tritonRustPointerTypeGet(MLIRTypeRef pointeeType, int addressSpace);

/// Check if a type is a Triton pointer type
int tritonRustTypeIsPointer(MLIRTypeRef type);

/// Get the pointee type from a Triton pointer type
MLIRTypeRef tritonRustPointerTypeGetPointeeType(MLIRTypeRef type);

/// Get the address space from a Triton pointer type
int tritonRustPointerTypeGetAddressSpace(MLIRTypeRef type);

//===----------------------------------------------------------------------===//
// Triton Tensor Type (RankedTensorType with Triton encoding)
//===----------------------------------------------------------------------===//

/// Create a Triton tensor type with shape and element type
/// encoding can be NULL for unencoded tensors
MLIRTypeRef tritonRustRankedTensorTypeGet(size_t rank, const int64_t *shape,
                                          MLIRTypeRef elementType,
                                          MLIRAttributeRef encoding);

/// Get the rank of a ranked tensor type
size_t tritonRustRankedTensorTypeGetRank(MLIRTypeRef type);

/// Get a dimension of a ranked tensor type
int64_t tritonRustRankedTensorTypeGetDimSize(MLIRTypeRef type, size_t dim);

/// Get the element type of a tensor
MLIRTypeRef tritonRustRankedTensorTypeGetElementType(MLIRTypeRef type);

/// Get the encoding attribute of a tensor (may be NULL)
MLIRAttributeRef tritonRustRankedTensorTypeGetEncoding(MLIRTypeRef type);

//===----------------------------------------------------------------------===//
// Triton GPU Blocked Encoding Attribute
//===----------------------------------------------------------------------===//

/// Create a blocked encoding attribute
/// sizePerThread, threadsPerWarp, warpsPerCTA, order are arrays of length rank
/// CTAsPerCGA and CTASplitNum are arrays of length rank (can be NULL for default)
MLIRAttributeRef tritonRustBlockedEncodingAttrGet(
    MLIRContextRef ctx,
    size_t rank,
    const unsigned *sizePerThread,
    const unsigned *threadsPerWarp,
    const unsigned *warpsPerCTA,
    const unsigned *order,
    const unsigned *CTAsPerCGA,
    const unsigned *CTASplitNum,
    const unsigned *CTAOrder);

/// Check if an attribute is a blocked encoding
int tritonRustAttrIsBlockedEncoding(MLIRAttributeRef attr);

//===----------------------------------------------------------------------===//
// Triton GPU Shared Encoding Attribute
//===----------------------------------------------------------------------===//

/// Create a shared memory encoding attribute
MLIRAttributeRef tritonRustSharedEncodingAttrGet(
    MLIRContextRef ctx,
    unsigned vec,
    unsigned perPhase,
    unsigned maxPhase,
    size_t orderLen,
    const unsigned *order,
    int hasLeadingOffset);

/// Check if an attribute is a shared encoding
int tritonRustAttrIsSharedEncoding(MLIRAttributeRef attr);

//===----------------------------------------------------------------------===//
// Triton GPU Slice Encoding Attribute
//===----------------------------------------------------------------------===//

/// Create a slice encoding attribute (for reduced dimensions)
MLIRAttributeRef tritonRustSliceEncodingAttrGet(MLIRContextRef ctx,
                                                unsigned dim,
                                                MLIRAttributeRef parent);

/// Check if an attribute is a slice encoding
int tritonRustAttrIsSliceEncoding(MLIRAttributeRef attr);

//===----------------------------------------------------------------------===//
// Triton GPU MMA (Matrix Multiply Accumulate) Encoding
//===----------------------------------------------------------------------===//

/// Create an NVIDIA MMA encoding (for tensor cores)
MLIRAttributeRef tritonRustNvidiaMmaEncodingAttrGet(
    MLIRContextRef ctx,
    unsigned versionMajor,
    unsigned versionMinor,
    size_t warpsPerCTALen,
    const unsigned *warpsPerCTA,
    size_t CTAsPerCGALen,
    const unsigned *CTAsPerCGA,
    size_t CTASplitNumLen,
    const unsigned *CTASplitNum,
    size_t CTAOrderLen,
    const unsigned *CTAOrder,
    size_t instrShapeLen,
    const unsigned *instrShape);

/// Create an AMD MMA encoding (for matrix cores)
MLIRAttributeRef tritonRustAMDMfmaEncodingAttrGet(
    MLIRContextRef ctx,
    unsigned versionMajor,
    unsigned versionMinor,
    size_t warpsPerCTALen,
    const unsigned *warpsPerCTA,
    unsigned mDim,
    unsigned nDim,
    int isTransposed,
    size_t CTAsPerCGALen,
    const unsigned *CTAsPerCGA,
    size_t CTASplitNumLen,
    const unsigned *CTASplitNum,
    size_t CTAOrderLen,
    const unsigned *CTAOrder);

//===----------------------------------------------------------------------===//
// Triton Attributes
//===----------------------------------------------------------------------===//

/// Create a program ID attribute (axis: 0, 1, or 2 for x, y, z)
MLIRAttributeRef tritonRustProgramIdAttrGet(MLIRContextRef ctx, int axis);

/// Create a load cache modifier attribute
/// Values: none=0, ca=1, cg=2, wb=3, cs=4, wt=5
MLIRAttributeRef tritonRustCacheModifierAttrGet(MLIRContextRef ctx, int modifier);

/// Create an eviction policy attribute
/// Values: normal=0, evict_first=1, evict_last=2
MLIRAttributeRef tritonRustEvictionPolicyAttrGet(MLIRContextRef ctx, int policy);

/// Create a padding option attribute
/// Values: zero=0, undef=1
MLIRAttributeRef tritonRustPaddingOptionAttrGet(MLIRContextRef ctx, int option);

/// Create a propagate NaN attribute
/// Values: none=0, all=1
MLIRAttributeRef tritonRustPropagateNanAttrGet(MLIRContextRef ctx, int option);

/// Create an atomic RMW operation attribute
/// Values: and=0, or=1, xor=2, add=3, fadd=4, max=5, min=6, umax=7, umin=8, xchg=9
MLIRAttributeRef tritonRustRMWOpAttrGet(MLIRContextRef ctx, int op);

//===----------------------------------------------------------------------===//
// Triton operation helpers
//===----------------------------------------------------------------------===//

/// Create a triton.make_range operation
/// Returns the created operation
MLIROperationRef tritonRustMakeRangeOp(MLIROpBuilderRef builder,
                                       MLIRLocationRef loc,
                                       int32_t start,
                                       int32_t end);

/// Create a triton.splat operation (broadcast scalar to tensor)
MLIROperationRef tritonRustSplatOp(MLIROpBuilderRef builder,
                                   MLIRLocationRef loc,
                                   MLIRValueRef src,
                                   MLIRTypeRef resultType);

/// Create a triton.broadcast operation
MLIROperationRef tritonRustBroadcastOp(MLIROpBuilderRef builder,
                                       MLIRLocationRef loc,
                                       MLIRValueRef src,
                                       MLIRTypeRef resultType);

/// Create a triton.expand_dims operation
MLIROperationRef tritonRustExpandDimsOp(MLIROpBuilderRef builder,
                                        MLIRLocationRef loc,
                                        MLIRValueRef src,
                                        int axis);

/// Create a triton.addptr operation (pointer arithmetic)
MLIROperationRef tritonRustAddPtrOp(MLIROpBuilderRef builder,
                                    MLIRLocationRef loc,
                                    MLIRValueRef ptr,
                                    MLIRValueRef offset);

/// Create a triton.load operation
MLIROperationRef tritonRustLoadOp(MLIROpBuilderRef builder,
                                  MLIRLocationRef loc,
                                  MLIRValueRef ptr,
                                  MLIRAttributeRef cache,
                                  MLIRAttributeRef evict,
                                  int isVolatile);

/// Create a triton.store operation
MLIROperationRef tritonRustStoreOp(MLIROpBuilderRef builder,
                                   MLIRLocationRef loc,
                                   MLIRValueRef ptr,
                                   MLIRValueRef value,
                                   MLIRAttributeRef cache);

/// Create a triton.dot operation (matrix multiply)
MLIROperationRef tritonRustDotOp(MLIROpBuilderRef builder,
                                 MLIRLocationRef loc,
                                 MLIRValueRef a,
                                 MLIRValueRef b,
                                 MLIRValueRef c,
                                 int allowTF32,
                                 int maxNumImpreciseAcc);

/// Create a triton.reduce operation
/// axis is the reduction dimension
/// The reduceOp should be added to the region
MLIROperationRef tritonRustReduceOp(MLIROpBuilderRef builder,
                                    MLIRLocationRef loc,
                                    MLIRValueRef operand,
                                    int axis);

/// Create a triton.get_program_id operation
MLIROperationRef tritonRustGetProgramIdOp(MLIROpBuilderRef builder,
                                          MLIRLocationRef loc,
                                          int axis);

/// Create a triton.get_num_programs operation
MLIROperationRef tritonRustGetNumProgramsOp(MLIROpBuilderRef builder,
                                            MLIRLocationRef loc,
                                            int axis);

/// Create a tt.func operation (Triton function)
MLIROperationRef tritonRustFuncOp(MLIROpBuilderRef builder,
                                  MLIRLocationRef loc,
                                  MLIRStringRef name,
                                  size_t numInputs,
                                  MLIRTypeRef const *inputTypes,
                                  size_t numResults,
                                  MLIRTypeRef const *resultTypes);

/// Create a tt.return operation
MLIROperationRef tritonRustReturnOp(MLIROpBuilderRef builder,
                                    MLIRLocationRef loc,
                                    size_t numValues,
                                    MLIRValueRef const *values);

//===----------------------------------------------------------------------===//
// Triton GPU operation helpers
//===----------------------------------------------------------------------===//

/// Create a triton_gpu.convert_layout operation
MLIROperationRef tritonRustConvertLayoutOp(MLIROpBuilderRef builder,
                                           MLIRLocationRef loc,
                                           MLIRValueRef src,
                                           MLIRTypeRef dstType);

/// Create a triton_gpu.alloc_tensor operation (shared memory allocation)
MLIROperationRef tritonRustAllocTensorOp(MLIROpBuilderRef builder,
                                         MLIRLocationRef loc,
                                         MLIRTypeRef tensorType);

/// Create a triton_gpu.insert_slice_async operation
MLIROperationRef tritonRustInsertSliceAsyncOp(MLIROpBuilderRef builder,
                                              MLIRLocationRef loc,
                                              MLIRValueRef src,
                                              MLIRValueRef dst,
                                              MLIRValueRef index,
                                              MLIRAttributeRef cache,
                                              MLIRAttributeRef evict,
                                              int isVolatile);

/// Create a triton_gpu.async_wait operation
MLIROperationRef tritonRustAsyncWaitOp(MLIROpBuilderRef builder,
                                       MLIRLocationRef loc,
                                       int num);

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Get Triton version string
const char *tritonRustGetVersion(void);

/// Get the number of warps for a GPU target
unsigned tritonRustGetNumWarps(MLIRStringRef target);

/// Get the number of threads per warp for a GPU target
unsigned tritonRustGetThreadsPerWarp(MLIRStringRef target);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TRITON_WRAPPER_H
