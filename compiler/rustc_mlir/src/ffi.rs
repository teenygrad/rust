//! Raw FFI bindings to the mlir-wrapper C++ library.
//!
//! This module provides unsafe FFI declarations for the C functions exported
//! by the mlir-wrapper. These are low-level bindings; prefer using the safe
//! wrappers in the `triton` and `context` modules.

use std::ffi::{c_char, c_int, c_void};
use std::os::raw::c_uint;

/// Result enum matching MLIRRustResult from C++
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MLIRRustResult {
    Success = 0,
    Failure = 1,
}

impl MLIRRustResult {
    pub fn is_success(self) -> bool {
        self == MLIRRustResult::Success
    }

    pub fn is_failure(self) -> bool {
        self == MLIRRustResult::Failure
    }
}

/// String reference (non-owning) matching MLIRStringRef from C++
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MLIRStringRef {
    pub data: *const c_char,
    pub length: usize,
}

impl MLIRStringRef {
    /// Create a string ref from a Rust string slice
    pub fn from_str(s: &str) -> Self {
        MLIRStringRef {
            data: s.as_ptr() as *const c_char,
            length: s.len(),
        }
    }

    /// Create an empty string ref
    pub fn empty() -> Self {
        MLIRStringRef {
            data: std::ptr::null(),
            length: 0,
        }
    }
}

impl From<&str> for MLIRStringRef {
    fn from(s: &str) -> Self {
        MLIRStringRef::from_str(s)
    }
}

impl From<&String> for MLIRStringRef {
    fn from(s: &String) -> Self {
        MLIRStringRef::from_str(s.as_str())
    }
}

// Opaque types from C++
// These are never instantiated in Rust; we only use pointers to them

/// Opaque MLIR context
#[repr(C)]
pub struct MLIRContext {
    _private: [u8; 0],
}

/// Opaque MLIR location
#[repr(C)]
pub struct MLIRLocation {
    _private: [u8; 0],
}

/// Opaque MLIR module
#[repr(C)]
pub struct MLIRModule {
    _private: [u8; 0],
}

/// Opaque MLIR operation
#[repr(C)]
pub struct MLIROperation {
    _private: [u8; 0],
}

/// Opaque MLIR block
#[repr(C)]
pub struct MLIRBlock {
    _private: [u8; 0],
}

/// Opaque MLIR region
#[repr(C)]
pub struct MLIRRegion {
    _private: [u8; 0],
}

/// Opaque MLIR type
#[repr(C)]
pub struct MLIRType {
    _private: [u8; 0],
}

/// Opaque MLIR attribute
#[repr(C)]
pub struct MLIRAttribute {
    _private: [u8; 0],
}

/// Opaque MLIR value
#[repr(C)]
pub struct MLIRValue {
    _private: [u8; 0],
}

/// Opaque MLIR OpBuilder
#[repr(C)]
pub struct MLIROpBuilder {
    _private: [u8; 0],
}

/// Opaque dialect registry
#[repr(C)]
pub struct MLIRDialectRegistry {
    _private: [u8; 0],
}

/// Opaque operation state
#[repr(C)]
pub struct MLIROperationState {
    _private: [u8; 0],
}

// Type aliases for pointers (matching the C typedefs)
pub type MLIRContextRef = *mut MLIRContext;
pub type MLIRLocationRef = *mut MLIRLocation;
pub type MLIRModuleRef = *mut MLIRModule;
pub type MLIROperationRef = *mut MLIROperation;
pub type MLIRBlockRef = *mut MLIRBlock;
pub type MLIRRegionRef = *mut MLIRRegion;
pub type MLIRTypeRef = *mut MLIRType;
pub type MLIRAttributeRef = *mut MLIRAttribute;
pub type MLIRValueRef = *mut MLIRValue;
pub type MLIROpBuilderRef = *mut MLIROpBuilder;
pub type MLIRDialectRegistryRef = *mut MLIRDialectRegistry;
pub type MLIROperationStateRef = *mut MLIROperationState;

#[link(name = "mlir-wrapper", kind = "static")]
extern "C" {
    //=========================================================================
    // String operations
    //=========================================================================

    pub fn mlirStringRefCreateFromCString(str: *const c_char) -> MLIRStringRef;
    pub fn mlirStringRefCreate(data: *const c_char, length: usize) -> MLIRStringRef;

    //=========================================================================
    // Context operations
    //=========================================================================

    pub fn mlirRustContextCreate() -> MLIRContextRef;
    pub fn mlirRustContextDestroy(ctx: MLIRContextRef);
    pub fn mlirRustContextEnableMultithreading(ctx: MLIRContextRef, enable: c_int);
    pub fn mlirRustContextSetAllowUnregisteredDialects(ctx: MLIRContextRef, allow: c_int);
    pub fn mlirRustContextLoadAllAvailableDialects(ctx: MLIRContextRef);
    pub fn mlirRustContextGetNumLoadedDialects(ctx: MLIRContextRef) -> usize;

    //=========================================================================
    // Dialect registry operations
    //=========================================================================

    pub fn mlirRustDialectRegistryCreate() -> MLIRDialectRegistryRef;
    pub fn mlirRustDialectRegistryDestroy(registry: MLIRDialectRegistryRef);
    pub fn mlirRustContextAppendDialectRegistry(
        ctx: MLIRContextRef,
        registry: MLIRDialectRegistryRef,
    );

    //=========================================================================
    // Location operations
    //=========================================================================

    pub fn mlirRustLocationUnknownGet(ctx: MLIRContextRef) -> MLIRLocationRef;
    pub fn mlirRustLocationFileLineColGet(
        ctx: MLIRContextRef,
        filename: MLIRStringRef,
        line: c_uint,
        col: c_uint,
    ) -> MLIRLocationRef;
    pub fn mlirRustLocationNameGet(
        ctx: MLIRContextRef,
        name: MLIRStringRef,
        child_loc: MLIRLocationRef,
    ) -> MLIRLocationRef;

    //=========================================================================
    // Module operations
    //=========================================================================

    pub fn mlirRustModuleCreateEmpty(loc: MLIRLocationRef) -> MLIRModuleRef;
    pub fn mlirRustModuleCreateParse(
        ctx: MLIRContextRef,
        module_str: MLIRStringRef,
    ) -> MLIRModuleRef;
    pub fn mlirRustModuleDestroy(module: MLIRModuleRef);
    pub fn mlirRustModuleGetBody(module: MLIRModuleRef) -> MLIRBlockRef;
    pub fn mlirRustModuleGetOperation(module: MLIRModuleRef) -> MLIROperationRef;
    pub fn mlirRustModuleGetContext(module: MLIRModuleRef) -> MLIRContextRef;
    pub fn mlirRustModulePrint(module: MLIRModuleRef) -> *mut c_char;
    pub fn mlirRustModuleVerify(module: MLIRModuleRef) -> MLIRRustResult;

    //=========================================================================
    // OpBuilder operations
    //=========================================================================

    pub fn mlirRustOpBuilderCreate(ctx: MLIRContextRef) -> MLIROpBuilderRef;
    pub fn mlirRustOpBuilderCreateAtBlockEnd(block: MLIRBlockRef) -> MLIROpBuilderRef;
    pub fn mlirRustOpBuilderDestroy(builder: MLIROpBuilderRef);
    pub fn mlirRustOpBuilderSetInsertionPointToEnd(
        builder: MLIROpBuilderRef,
        block: MLIRBlockRef,
    );
    pub fn mlirRustOpBuilderSetInsertionPointToStart(
        builder: MLIROpBuilderRef,
        block: MLIRBlockRef,
    );
    pub fn mlirRustOpBuilderSetInsertionPoint(builder: MLIROpBuilderRef, op: MLIROperationRef);
    pub fn mlirRustOpBuilderSetInsertionPointAfter(
        builder: MLIROpBuilderRef,
        op: MLIROperationRef,
    );
    pub fn mlirRustOpBuilderGetContext(builder: MLIROpBuilderRef) -> MLIRContextRef;
    pub fn mlirRustOpBuilderGetInsertionBlock(builder: MLIROpBuilderRef) -> MLIRBlockRef;

    //=========================================================================
    // Type operations
    //=========================================================================

    pub fn mlirRustTypeGetContext(ty: MLIRTypeRef) -> MLIRContextRef;
    pub fn mlirRustTypeIsInteger(ty: MLIRTypeRef) -> c_int;
    pub fn mlirRustTypeIsFloat(ty: MLIRTypeRef) -> c_int;
    pub fn mlirRustTypeIsIndex(ty: MLIRTypeRef) -> c_int;
    pub fn mlirRustIntegerTypeGet(ctx: MLIRContextRef, width: c_uint) -> MLIRTypeRef;
    pub fn mlirRustIntegerTypeSignlessGet(ctx: MLIRContextRef, width: c_uint) -> MLIRTypeRef;
    pub fn mlirRustIntegerTypeSignedGet(ctx: MLIRContextRef, width: c_uint) -> MLIRTypeRef;
    pub fn mlirRustIntegerTypeUnsignedGet(ctx: MLIRContextRef, width: c_uint) -> MLIRTypeRef;
    pub fn mlirRustIndexTypeGet(ctx: MLIRContextRef) -> MLIRTypeRef;
    pub fn mlirRustF16TypeGet(ctx: MLIRContextRef) -> MLIRTypeRef;
    pub fn mlirRustBF16TypeGet(ctx: MLIRContextRef) -> MLIRTypeRef;
    pub fn mlirRustF32TypeGet(ctx: MLIRContextRef) -> MLIRTypeRef;
    pub fn mlirRustF64TypeGet(ctx: MLIRContextRef) -> MLIRTypeRef;
    pub fn mlirRustNoneTypeGet(ctx: MLIRContextRef) -> MLIRTypeRef;

    //=========================================================================
    // Attribute operations
    //=========================================================================

    pub fn mlirRustUnitAttrGet(ctx: MLIRContextRef) -> MLIRAttributeRef;
    pub fn mlirRustBoolAttrGet(ctx: MLIRContextRef, value: c_int) -> MLIRAttributeRef;
    pub fn mlirRustIntegerAttrGet(ty: MLIRTypeRef, value: i64) -> MLIRAttributeRef;
    pub fn mlirRustFloatAttrGet(ty: MLIRTypeRef, value: f64) -> MLIRAttributeRef;
    pub fn mlirRustStringAttrGet(ctx: MLIRContextRef, value: MLIRStringRef) -> MLIRAttributeRef;
    pub fn mlirRustTypeAttrGet(ty: MLIRTypeRef) -> MLIRAttributeRef;
    pub fn mlirRustFlatSymbolRefAttrGet(
        ctx: MLIRContextRef,
        symbol: MLIRStringRef,
    ) -> MLIRAttributeRef;

    //=========================================================================
    // Operation state operations
    //=========================================================================

    pub fn mlirRustOperationStateCreate(
        name: MLIRStringRef,
        loc: MLIRLocationRef,
    ) -> MLIROperationStateRef;
    pub fn mlirRustOperationStateDestroy(state: MLIROperationStateRef);
    pub fn mlirRustOperationStateAddResults(
        state: MLIROperationStateRef,
        n: usize,
        results: *const MLIRTypeRef,
    );
    pub fn mlirRustOperationStateAddOperands(
        state: MLIROperationStateRef,
        n: usize,
        operands: *const MLIRValueRef,
    );
    pub fn mlirRustOperationStateAddAttribute(
        state: MLIROperationStateRef,
        name: MLIRStringRef,
        attr: MLIRAttributeRef,
    );
    pub fn mlirRustOperationStateAddOwnedRegions(
        state: MLIROperationStateRef,
        n: usize,
        regions: *const MLIRRegionRef,
    );
    pub fn mlirRustOperationStateAddSuccessors(
        state: MLIROperationStateRef,
        n: usize,
        successors: *const MLIRBlockRef,
    );

    //=========================================================================
    // Operation operations
    //=========================================================================

    pub fn mlirRustOperationCreate(state: MLIROperationStateRef) -> MLIROperationRef;
    pub fn mlirRustOperationDestroy(op: MLIROperationRef);
    pub fn mlirRustOperationGetNumResults(op: MLIROperationRef) -> usize;
    pub fn mlirRustOperationGetResult(op: MLIROperationRef, pos: usize) -> MLIRValueRef;
    pub fn mlirRustOperationGetNumOperands(op: MLIROperationRef) -> usize;
    pub fn mlirRustOperationGetOperand(op: MLIROperationRef, pos: usize) -> MLIRValueRef;
    pub fn mlirRustOperationGetBlock(op: MLIROperationRef) -> MLIRBlockRef;
    pub fn mlirRustOperationGetNumRegions(op: MLIROperationRef) -> usize;
    pub fn mlirRustOperationGetRegion(op: MLIROperationRef, pos: usize) -> MLIRRegionRef;
    pub fn mlirRustOperationPrint(op: MLIROperationRef) -> *mut c_char;

    //=========================================================================
    // Block operations
    //=========================================================================

    pub fn mlirRustBlockCreate(
        n_args: usize,
        arg_types: *const MLIRTypeRef,
        arg_locs: *const MLIRLocationRef,
    ) -> MLIRBlockRef;
    pub fn mlirRustBlockDestroy(block: MLIRBlockRef);
    pub fn mlirRustBlockGetNumArguments(block: MLIRBlockRef) -> usize;
    pub fn mlirRustBlockGetArgument(block: MLIRBlockRef, pos: usize) -> MLIRValueRef;
    pub fn mlirRustBlockAddArgument(
        block: MLIRBlockRef,
        ty: MLIRTypeRef,
        loc: MLIRLocationRef,
    ) -> MLIRValueRef;
    pub fn mlirRustBlockGetFirstOperation(block: MLIRBlockRef) -> MLIROperationRef;
    pub fn mlirRustBlockGetTerminator(block: MLIRBlockRef) -> MLIROperationRef;
    pub fn mlirRustBlockAppendOperation(block: MLIRBlockRef, op: MLIROperationRef);
    pub fn mlirRustBlockInsertOwnedOperationBefore(
        block: MLIRBlockRef,
        reference: MLIROperationRef,
        op: MLIROperationRef,
    );

    //=========================================================================
    // Region operations
    //=========================================================================

    pub fn mlirRustRegionCreate() -> MLIRRegionRef;
    pub fn mlirRustRegionDestroy(region: MLIRRegionRef);
    pub fn mlirRustRegionGetNumBlocks(region: MLIRRegionRef) -> usize;
    pub fn mlirRustRegionGetFirstBlock(region: MLIRRegionRef) -> MLIRBlockRef;
    pub fn mlirRustRegionAppendBlock(region: MLIRRegionRef, block: MLIRBlockRef);
    pub fn mlirRustRegionInsertOwnedBlockBefore(
        region: MLIRRegionRef,
        reference: MLIRBlockRef,
        block: MLIRBlockRef,
    );

    //=========================================================================
    // Value operations
    //=========================================================================

    pub fn mlirRustValueGetType(value: MLIRValueRef) -> MLIRTypeRef;
    pub fn mlirRustValueIsBlockArgument(value: MLIRValueRef) -> c_int;
    pub fn mlirRustValueIsOpResult(value: MLIRValueRef) -> c_int;
    pub fn mlirRustValuePrint(value: MLIRValueRef) -> *mut c_char;

    //=========================================================================
    // Memory management
    //=========================================================================

    pub fn mlirRustStringDestroy(str: *mut c_char);
}

// Triton-specific FFI declarations
#[link(name = "mlir-wrapper", kind = "static")]
extern "C" {
    //=========================================================================
    // Dialect registration
    //=========================================================================

    pub fn tritonRustInitDialects(ctx: MLIRContextRef) -> MLIRRustResult;
    pub fn tritonRustRegisterTritonDialect(registry: MLIRDialectRegistryRef) -> MLIRRustResult;
    pub fn tritonRustRegisterTritonGPUDialect(registry: MLIRDialectRegistryRef) -> MLIRRustResult;
    pub fn tritonRustIsAvailable() -> c_int;

    //=========================================================================
    // Triton Pointer Type
    //=========================================================================

    pub fn tritonRustPointerTypeGet(pointee_type: MLIRTypeRef, address_space: c_int)
        -> MLIRTypeRef;
    pub fn tritonRustTypeIsPointer(ty: MLIRTypeRef) -> c_int;
    pub fn tritonRustPointerTypeGetPointeeType(ty: MLIRTypeRef) -> MLIRTypeRef;
    pub fn tritonRustPointerTypeGetAddressSpace(ty: MLIRTypeRef) -> c_int;

    //=========================================================================
    // Triton Tensor Type
    //=========================================================================

    pub fn tritonRustRankedTensorTypeGet(
        rank: usize,
        shape: *const i64,
        element_type: MLIRTypeRef,
        encoding: MLIRAttributeRef,
    ) -> MLIRTypeRef;
    pub fn tritonRustRankedTensorTypeGetRank(ty: MLIRTypeRef) -> usize;
    pub fn tritonRustRankedTensorTypeGetDimSize(ty: MLIRTypeRef, dim: usize) -> i64;
    pub fn tritonRustRankedTensorTypeGetElementType(ty: MLIRTypeRef) -> MLIRTypeRef;
    pub fn tritonRustRankedTensorTypeGetEncoding(ty: MLIRTypeRef) -> MLIRAttributeRef;

    //=========================================================================
    // Triton GPU Encodings
    //=========================================================================

    pub fn tritonRustBlockedEncodingAttrGet(
        ctx: MLIRContextRef,
        rank: usize,
        size_per_thread: *const c_uint,
        threads_per_warp: *const c_uint,
        warps_per_cta: *const c_uint,
        order: *const c_uint,
        ctas_per_cga: *const c_uint,
        cta_split_num: *const c_uint,
        cta_order: *const c_uint,
    ) -> MLIRAttributeRef;
    pub fn tritonRustAttrIsBlockedEncoding(attr: MLIRAttributeRef) -> c_int;

    pub fn tritonRustSharedEncodingAttrGet(
        ctx: MLIRContextRef,
        vec: c_uint,
        per_phase: c_uint,
        max_phase: c_uint,
        order_len: usize,
        order: *const c_uint,
        has_leading_offset: c_int,
    ) -> MLIRAttributeRef;
    pub fn tritonRustAttrIsSharedEncoding(attr: MLIRAttributeRef) -> c_int;

    pub fn tritonRustSliceEncodingAttrGet(
        ctx: MLIRContextRef,
        dim: c_uint,
        parent: MLIRAttributeRef,
    ) -> MLIRAttributeRef;
    pub fn tritonRustAttrIsSliceEncoding(attr: MLIRAttributeRef) -> c_int;

    pub fn tritonRustNvidiaMmaEncodingAttrGet(
        ctx: MLIRContextRef,
        version_major: c_uint,
        version_minor: c_uint,
        warps_per_cta_len: usize,
        warps_per_cta: *const c_uint,
        ctas_per_cga_len: usize,
        ctas_per_cga: *const c_uint,
        cta_split_num_len: usize,
        cta_split_num: *const c_uint,
        cta_order_len: usize,
        cta_order: *const c_uint,
        instr_shape_len: usize,
        instr_shape: *const c_uint,
    ) -> MLIRAttributeRef;

    pub fn tritonRustAMDMfmaEncodingAttrGet(
        ctx: MLIRContextRef,
        version_major: c_uint,
        version_minor: c_uint,
        warps_per_cta_len: usize,
        warps_per_cta: *const c_uint,
        m_dim: c_uint,
        n_dim: c_uint,
        is_transposed: c_int,
        ctas_per_cga_len: usize,
        ctas_per_cga: *const c_uint,
        cta_split_num_len: usize,
        cta_split_num: *const c_uint,
        cta_order_len: usize,
        cta_order: *const c_uint,
    ) -> MLIRAttributeRef;

    //=========================================================================
    // Triton Attributes
    //=========================================================================

    pub fn tritonRustProgramIdAttrGet(ctx: MLIRContextRef, axis: c_int) -> MLIRAttributeRef;
    pub fn tritonRustCacheModifierAttrGet(ctx: MLIRContextRef, modifier: c_int)
        -> MLIRAttributeRef;
    pub fn tritonRustEvictionPolicyAttrGet(ctx: MLIRContextRef, policy: c_int)
        -> MLIRAttributeRef;
    pub fn tritonRustPaddingOptionAttrGet(ctx: MLIRContextRef, option: c_int) -> MLIRAttributeRef;
    pub fn tritonRustPropagateNanAttrGet(ctx: MLIRContextRef, option: c_int) -> MLIRAttributeRef;
    pub fn tritonRustRMWOpAttrGet(ctx: MLIRContextRef, op: c_int) -> MLIRAttributeRef;

    //=========================================================================
    // Triton Operation Helpers
    //=========================================================================

    pub fn tritonRustMakeRangeOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        start: i32,
        end: i32,
    ) -> MLIROperationRef;

    pub fn tritonRustSplatOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        src: MLIRValueRef,
        result_type: MLIRTypeRef,
    ) -> MLIROperationRef;

    pub fn tritonRustBroadcastOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        src: MLIRValueRef,
        result_type: MLIRTypeRef,
    ) -> MLIROperationRef;

    pub fn tritonRustExpandDimsOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        src: MLIRValueRef,
        axis: c_int,
    ) -> MLIROperationRef;

    pub fn tritonRustAddPtrOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        ptr: MLIRValueRef,
        offset: MLIRValueRef,
    ) -> MLIROperationRef;

    pub fn tritonRustLoadOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        ptr: MLIRValueRef,
        cache: MLIRAttributeRef,
        evict: MLIRAttributeRef,
        is_volatile: c_int,
    ) -> MLIROperationRef;

    pub fn tritonRustStoreOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        ptr: MLIRValueRef,
        value: MLIRValueRef,
        cache: MLIRAttributeRef,
    ) -> MLIROperationRef;

    pub fn tritonRustDotOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        a: MLIRValueRef,
        b: MLIRValueRef,
        c: MLIRValueRef,
        allow_tf32: c_int,
        max_num_imprecise_acc: c_int,
    ) -> MLIROperationRef;

    pub fn tritonRustReduceOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        operand: MLIRValueRef,
        axis: c_int,
    ) -> MLIROperationRef;

    pub fn tritonRustGetProgramIdOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        axis: c_int,
    ) -> MLIROperationRef;

    pub fn tritonRustGetNumProgramsOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        axis: c_int,
    ) -> MLIROperationRef;

    pub fn tritonRustFuncOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        name: MLIRStringRef,
        num_inputs: usize,
        input_types: *const MLIRTypeRef,
        num_results: usize,
        result_types: *const MLIRTypeRef,
    ) -> MLIROperationRef;

    pub fn tritonRustReturnOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        num_values: usize,
        values: *const MLIRValueRef,
    ) -> MLIROperationRef;

    //=========================================================================
    // Triton GPU Operation Helpers
    //=========================================================================

    pub fn tritonRustConvertLayoutOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        src: MLIRValueRef,
        dst_type: MLIRTypeRef,
    ) -> MLIROperationRef;

    pub fn tritonRustAllocTensorOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        tensor_type: MLIRTypeRef,
    ) -> MLIROperationRef;

    pub fn tritonRustInsertSliceAsyncOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        src: MLIRValueRef,
        dst: MLIRValueRef,
        index: MLIRValueRef,
        cache: MLIRAttributeRef,
        evict: MLIRAttributeRef,
        is_volatile: c_int,
    ) -> MLIROperationRef;

    pub fn tritonRustAsyncWaitOp(
        builder: MLIROpBuilderRef,
        loc: MLIRLocationRef,
        num: c_int,
    ) -> MLIROperationRef;

    //=========================================================================
    // Utility functions
    //=========================================================================

    pub fn tritonRustGetVersion() -> *const c_char;
    pub fn tritonRustGetNumWarps(target: MLIRStringRef) -> c_uint;
    pub fn tritonRustGetThreadsPerWarp(target: MLIRStringRef) -> c_uint;
}

/// Helper to convert a C string to Rust String, freeing the C string
pub unsafe fn c_str_to_string(ptr: *mut c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    let c_str = std::ffi::CStr::from_ptr(ptr);
    let result = c_str.to_string_lossy().into_owned();
    mlirRustStringDestroy(ptr);
    result
}
