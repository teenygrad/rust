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

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Result types
//===----------------------------------------------------------------------===//

/// Result enum for operations that can fail
typedef enum {
    MLIRRustResult_Success = 0,
    MLIRRustResult_Failure = 1
} MLIRRustResult;

//===----------------------------------------------------------------------===//
// Opaque pointer types
//===----------------------------------------------------------------------===//

/// Opaque reference to an MLIR context
typedef struct MLIROpaqueContext *MLIRContextRef;

/// Opaque reference to an MLIR location
typedef struct MLIROpaqueLocation *MLIRLocationRef;

/// Opaque reference to an MLIR module
typedef struct MLIROpaqueModule *MLIRModuleRef;

/// Opaque reference to an MLIR operation
typedef struct MLIROpaqueOperation *MLIROperationRef;

/// Opaque reference to an MLIR block
typedef struct MLIROpaqueBlock *MLIRBlockRef;

/// Opaque reference to an MLIR region
typedef struct MLIROpaqueRegion *MLIRRegionRef;

/// Opaque reference to an MLIR type
typedef struct MLIROpaqueType *MLIRTypeRef;

/// Opaque reference to an MLIR attribute
typedef struct MLIROpaqueAttribute *MLIRAttributeRef;

/// Opaque reference to an MLIR value
typedef struct MLIROpaqueValue *MLIRValueRef;

/// Opaque reference to an OpBuilder
typedef struct MLIROpaqueOpBuilder *MLIROpBuilderRef;

/// Opaque reference to a dialect registry
typedef struct MLIROpaqueDialectRegistry *MLIRDialectRegistryRef;

//===----------------------------------------------------------------------===//
// String reference (non-owning)
//===----------------------------------------------------------------------===//

typedef struct {
    const char *data;
    size_t length;
} MLIRStringRef;

/// Create a string ref from a null-terminated C string
MLIRStringRef mlirStringRefCreateFromCString(const char *str);

/// Create a string ref from data and length
MLIRStringRef mlirStringRefCreate(const char *data, size_t length);

//===----------------------------------------------------------------------===//
// Context operations
//===----------------------------------------------------------------------===//

/// Create a new MLIR context
MLIRContextRef mlirRustContextCreate(void);

/// Destroy an MLIR context
void mlirRustContextDestroy(MLIRContextRef ctx);

/// Enable multi-threading for the context
void mlirRustContextEnableMultithreading(MLIRContextRef ctx, int enable);

/// Allow unregistered dialects
void mlirRustContextSetAllowUnregisteredDialects(MLIRContextRef ctx, int allow);

/// Load all available dialects
void mlirRustContextLoadAllAvailableDialects(MLIRContextRef ctx);

/// Get the number of loaded dialects
size_t mlirRustContextGetNumLoadedDialects(MLIRContextRef ctx);

//===----------------------------------------------------------------------===//
// Dialect registry operations
//===----------------------------------------------------------------------===//

/// Create a new dialect registry
MLIRDialectRegistryRef mlirRustDialectRegistryCreate(void);

/// Destroy a dialect registry
void mlirRustDialectRegistryDestroy(MLIRDialectRegistryRef registry);

/// Append a dialect registry to a context
void mlirRustContextAppendDialectRegistry(MLIRContextRef ctx,
                                          MLIRDialectRegistryRef registry);

//===----------------------------------------------------------------------===//
// Location operations
//===----------------------------------------------------------------------===//

/// Create an unknown location
MLIRLocationRef mlirRustLocationUnknownGet(MLIRContextRef ctx);

/// Create a file:line:column location
MLIRLocationRef mlirRustLocationFileLineColGet(MLIRContextRef ctx,
                                               MLIRStringRef filename,
                                               unsigned line,
                                               unsigned col);

/// Create a named location
MLIRLocationRef mlirRustLocationNameGet(MLIRContextRef ctx,
                                        MLIRStringRef name,
                                        MLIRLocationRef childLoc);

//===----------------------------------------------------------------------===//
// Module operations
//===----------------------------------------------------------------------===//

/// Create an empty module
MLIRModuleRef mlirRustModuleCreateEmpty(MLIRLocationRef loc);

/// Create a module by parsing MLIR assembly
MLIRModuleRef mlirRustModuleCreateParse(MLIRContextRef ctx,
                                        MLIRStringRef module_str);

/// Destroy a module
void mlirRustModuleDestroy(MLIRModuleRef module);

/// Get the module's body block
MLIRBlockRef mlirRustModuleGetBody(MLIRModuleRef module);

/// Get the module as an operation
MLIROperationRef mlirRustModuleGetOperation(MLIRModuleRef module);

/// Get the context from a module
MLIRContextRef mlirRustModuleGetContext(MLIRModuleRef module);

/// Print module to string (caller must free with mlirRustStringDestroy)
char *mlirRustModulePrint(MLIRModuleRef module);

/// Verify the module, returns Success if valid
MLIRRustResult mlirRustModuleVerify(MLIRModuleRef module);

//===----------------------------------------------------------------------===//
// OpBuilder operations
//===----------------------------------------------------------------------===//

/// Create an OpBuilder positioned at the start of a context
MLIROpBuilderRef mlirRustOpBuilderCreate(MLIRContextRef ctx);

/// Create an OpBuilder positioned at the end of a block
MLIROpBuilderRef mlirRustOpBuilderCreateAtBlockEnd(MLIRBlockRef block);

/// Destroy an OpBuilder
void mlirRustOpBuilderDestroy(MLIROpBuilderRef builder);

/// Set insertion point to the end of a block
void mlirRustOpBuilderSetInsertionPointToEnd(MLIROpBuilderRef builder,
                                             MLIRBlockRef block);

/// Set insertion point to the start of a block
void mlirRustOpBuilderSetInsertionPointToStart(MLIROpBuilderRef builder,
                                               MLIRBlockRef block);

/// Set insertion point before an operation
void mlirRustOpBuilderSetInsertionPoint(MLIROpBuilderRef builder,
                                        MLIROperationRef op);

/// Set insertion point after an operation
void mlirRustOpBuilderSetInsertionPointAfter(MLIROpBuilderRef builder,
                                             MLIROperationRef op);

/// Get the context from a builder
MLIRContextRef mlirRustOpBuilderGetContext(MLIROpBuilderRef builder);

/// Get the current insertion block
MLIRBlockRef mlirRustOpBuilderGetInsertionBlock(MLIROpBuilderRef builder);

//===----------------------------------------------------------------------===//
// Type operations
//===----------------------------------------------------------------------===//

/// Get the context from a type
MLIRContextRef mlirRustTypeGetContext(MLIRTypeRef type);

/// Check if type is integer type
int mlirRustTypeIsInteger(MLIRTypeRef type);

/// Check if type is float type
int mlirRustTypeIsFloat(MLIRTypeRef type);

/// Check if type is index type
int mlirRustTypeIsIndex(MLIRTypeRef type);

/// Get integer type with given width
MLIRTypeRef mlirRustIntegerTypeGet(MLIRContextRef ctx, unsigned width);

/// Get signless integer type with given width
MLIRTypeRef mlirRustIntegerTypeSignlessGet(MLIRContextRef ctx, unsigned width);

/// Get signed integer type with given width
MLIRTypeRef mlirRustIntegerTypeSignedGet(MLIRContextRef ctx, unsigned width);

/// Get unsigned integer type with given width
MLIRTypeRef mlirRustIntegerTypeUnsignedGet(MLIRContextRef ctx, unsigned width);

/// Get index type
MLIRTypeRef mlirRustIndexTypeGet(MLIRContextRef ctx);

/// Get f16 type
MLIRTypeRef mlirRustF16TypeGet(MLIRContextRef ctx);

/// Get bf16 type
MLIRTypeRef mlirRustBF16TypeGet(MLIRContextRef ctx);

/// Get f32 type
MLIRTypeRef mlirRustF32TypeGet(MLIRContextRef ctx);

/// Get f64 type
MLIRTypeRef mlirRustF64TypeGet(MLIRContextRef ctx);

/// Get none type
MLIRTypeRef mlirRustNoneTypeGet(MLIRContextRef ctx);

//===----------------------------------------------------------------------===//
// Attribute operations
//===----------------------------------------------------------------------===//

/// Get unit attribute
MLIRAttributeRef mlirRustUnitAttrGet(MLIRContextRef ctx);

/// Get bool attribute
MLIRAttributeRef mlirRustBoolAttrGet(MLIRContextRef ctx, int value);

/// Get integer attribute
MLIRAttributeRef mlirRustIntegerAttrGet(MLIRTypeRef type, int64_t value);

/// Get float attribute
MLIRAttributeRef mlirRustFloatAttrGet(MLIRTypeRef type, double value);

/// Get string attribute
MLIRAttributeRef mlirRustStringAttrGet(MLIRContextRef ctx, MLIRStringRef value);

/// Get type attribute
MLIRAttributeRef mlirRustTypeAttrGet(MLIRTypeRef type);

/// Get flat symbol ref attribute
MLIRAttributeRef mlirRustFlatSymbolRefAttrGet(MLIRContextRef ctx,
                                              MLIRStringRef symbol);

//===----------------------------------------------------------------------===//
// Operation state (for building operations)
//===----------------------------------------------------------------------===//

/// Opaque operation state for building operations
typedef struct MLIROpaqueOperationState *MLIROperationStateRef;

/// Create an operation state
MLIROperationStateRef mlirRustOperationStateCreate(MLIRStringRef name,
                                                   MLIRLocationRef loc);

/// Destroy an operation state
void mlirRustOperationStateDestroy(MLIROperationStateRef state);

/// Add results to operation state
void mlirRustOperationStateAddResults(MLIROperationStateRef state,
                                      size_t n,
                                      MLIRTypeRef const *results);

/// Add operands to operation state
void mlirRustOperationStateAddOperands(MLIROperationStateRef state,
                                       size_t n,
                                       MLIRValueRef const *operands);

/// Add attributes to operation state
void mlirRustOperationStateAddAttribute(MLIROperationStateRef state,
                                        MLIRStringRef name,
                                        MLIRAttributeRef attr);

/// Add a region to operation state
void mlirRustOperationStateAddOwnedRegions(MLIROperationStateRef state,
                                           size_t n,
                                           MLIRRegionRef const *regions);

/// Add successors to operation state
void mlirRustOperationStateAddSuccessors(MLIROperationStateRef state,
                                         size_t n,
                                         MLIRBlockRef const *successors);

//===----------------------------------------------------------------------===//
// Operation creation
//===----------------------------------------------------------------------===//

/// Create an operation from operation state
MLIROperationRef mlirRustOperationCreate(MLIROperationStateRef state);

/// Destroy an operation (if not attached to IR)
void mlirRustOperationDestroy(MLIROperationRef op);

/// Get the number of results
size_t mlirRustOperationGetNumResults(MLIROperationRef op);

/// Get a result by index
MLIRValueRef mlirRustOperationGetResult(MLIROperationRef op, size_t pos);

/// Get the number of operands
size_t mlirRustOperationGetNumOperands(MLIROperationRef op);

/// Get an operand by index
MLIRValueRef mlirRustOperationGetOperand(MLIROperationRef op, size_t pos);

/// Get the parent block
MLIRBlockRef mlirRustOperationGetBlock(MLIROperationRef op);

/// Get the number of regions
size_t mlirRustOperationGetNumRegions(MLIROperationRef op);

/// Get a region by index
MLIRRegionRef mlirRustOperationGetRegion(MLIROperationRef op, size_t pos);

/// Print operation to string (caller must free with mlirRustStringDestroy)
char *mlirRustOperationPrint(MLIROperationRef op);

//===----------------------------------------------------------------------===//
// Block operations
//===----------------------------------------------------------------------===//

/// Create a new empty block
MLIRBlockRef mlirRustBlockCreate(size_t nArgs, MLIRTypeRef const *argTypes,
                                 MLIRLocationRef const *argLocs);

/// Destroy a block (if not attached to IR)
void mlirRustBlockDestroy(MLIRBlockRef block);

/// Get the number of arguments
size_t mlirRustBlockGetNumArguments(MLIRBlockRef block);

/// Get an argument by index
MLIRValueRef mlirRustBlockGetArgument(MLIRBlockRef block, size_t pos);

/// Add an argument to a block
MLIRValueRef mlirRustBlockAddArgument(MLIRBlockRef block, MLIRTypeRef type,
                                      MLIRLocationRef loc);

/// Get the first operation in a block
MLIROperationRef mlirRustBlockGetFirstOperation(MLIRBlockRef block);

/// Get the terminator operation
MLIROperationRef mlirRustBlockGetTerminator(MLIRBlockRef block);

/// Append an operation to a block
void mlirRustBlockAppendOperation(MLIRBlockRef block, MLIROperationRef op);

/// Insert an operation before another operation in a block
void mlirRustBlockInsertOwnedOperationBefore(MLIRBlockRef block,
                                             MLIROperationRef ref,
                                             MLIROperationRef op);

//===----------------------------------------------------------------------===//
// Region operations
//===----------------------------------------------------------------------===//

/// Create a new empty region
MLIRRegionRef mlirRustRegionCreate(void);

/// Destroy a region (if not attached to IR)
void mlirRustRegionDestroy(MLIRRegionRef region);

/// Get the number of blocks in a region
size_t mlirRustRegionGetNumBlocks(MLIRRegionRef region);

/// Get the first block in a region
MLIRBlockRef mlirRustRegionGetFirstBlock(MLIRRegionRef region);

/// Append a block to a region
void mlirRustRegionAppendBlock(MLIRRegionRef region, MLIRBlockRef block);

/// Insert a block before another block
void mlirRustRegionInsertOwnedBlockBefore(MLIRRegionRef region,
                                          MLIRBlockRef ref,
                                          MLIRBlockRef block);

//===----------------------------------------------------------------------===//
// Value operations
//===----------------------------------------------------------------------===//

/// Get the type of a value
MLIRTypeRef mlirRustValueGetType(MLIRValueRef value);

/// Check if a value is a block argument
int mlirRustValueIsBlockArgument(MLIRValueRef value);

/// Check if a value is an operation result
int mlirRustValueIsOpResult(MLIRValueRef value);

/// Print value to string (caller must free with mlirRustStringDestroy)
char *mlirRustValuePrint(MLIRValueRef value);

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

/// Free a string allocated by this library
void mlirRustStringDestroy(char *str);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLIR_WRAPPER_H
