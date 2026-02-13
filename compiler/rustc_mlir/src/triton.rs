//! Safe Rust wrappers for Triton dialect types and attributes.
//!
//! This module provides type-safe wrappers around the Triton FFI bindings,
//! making it easier to work with Triton types from Rust code.

use crate::ffi::{self, MLIRAttributeRef, MLIRContextRef, MLIRTypeRef, MLIRStringRef};
use crate::{Error, Result};
use std::ffi::CStr;
use std::ptr;

/// Check if Triton support is available
pub fn is_available() -> bool {
    unsafe { ffi::tritonRustIsAvailable() != 0 }
}

/// Get the Triton version string
pub fn version() -> String {
    unsafe {
        let ptr = ffi::tritonRustGetVersion();
        if ptr.is_null() {
            return String::from("unknown");
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Initialize and register Triton dialects with the context
pub fn register_dialects(ctx: MLIRContextRef) -> Result<()> {
    let result = unsafe { ffi::tritonRustInitDialects(ctx) };
    if result.is_success() {
        Ok(())
    } else {
        Err(Error::TritonNotAvailable)
    }
}

/// Address space constants for Triton pointers
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressSpace {
    /// Generic address space
    Generic = 0,
    /// Global memory
    Global = 1,
    /// Shared memory
    Shared = 3,
}

/// Cache modifier for load/store operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheModifier {
    None = 0,
    /// Cache at all levels
    CA = 1,
    /// Cache at global level
    CG = 2,
    /// Write-back
    WB = 3,
    /// Cache streaming
    CS = 4,
    /// Write-through
    WT = 5,
}

/// Eviction policy for memory operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    Normal = 0,
    EvictFirst = 1,
    EvictLast = 2,
}

/// Padding option for masked loads
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingOption {
    Zero = 0,
    Undef = 1,
}

/// NaN propagation option
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagateNan {
    None = 0,
    All = 1,
}

/// Atomic RMW operation types
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RMWOp {
    And = 0,
    Or = 1,
    Xor = 2,
    Add = 3,
    FAdd = 4,
    Max = 5,
    Min = 6,
    UMax = 7,
    UMin = 8,
    Xchg = 9,
}

/// Program ID axis
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramAxis {
    X = 0,
    Y = 1,
    Z = 2,
}

/// Triton pointer type operations
pub mod pointer {
    use super::*;

    /// Create a Triton pointer type
    pub fn create(pointee: MLIRTypeRef, address_space: AddressSpace) -> MLIRTypeRef {
        unsafe { ffi::tritonRustPointerTypeGet(pointee, address_space as i32) }
    }

    /// Check if a type is a Triton pointer
    pub fn is_pointer(ty: MLIRTypeRef) -> bool {
        unsafe { ffi::tritonRustTypeIsPointer(ty) != 0 }
    }

    /// Get the pointee type from a pointer
    pub fn pointee_type(ty: MLIRTypeRef) -> MLIRTypeRef {
        unsafe { ffi::tritonRustPointerTypeGetPointeeType(ty) }
    }

    /// Get the address space from a pointer
    pub fn address_space(ty: MLIRTypeRef) -> AddressSpace {
        let space = unsafe { ffi::tritonRustPointerTypeGetAddressSpace(ty) };
        match space {
            0 => AddressSpace::Generic,
            1 => AddressSpace::Global,
            3 => AddressSpace::Shared,
            _ => AddressSpace::Generic,
        }
    }
}

/// Triton tensor type operations
pub mod tensor {
    use super::*;

    /// Create a ranked tensor type with optional encoding
    pub fn create(
        shape: &[i64],
        element_type: MLIRTypeRef,
        encoding: Option<MLIRAttributeRef>,
    ) -> MLIRTypeRef {
        unsafe {
            ffi::tritonRustRankedTensorTypeGet(
                shape.len(),
                shape.as_ptr(),
                element_type,
                encoding.unwrap_or(ptr::null_mut()),
            )
        }
    }

    /// Get the rank of a tensor type
    pub fn rank(ty: MLIRTypeRef) -> usize {
        unsafe { ffi::tritonRustRankedTensorTypeGetRank(ty) }
    }

    /// Get a dimension size
    pub fn dim_size(ty: MLIRTypeRef, dim: usize) -> i64 {
        unsafe { ffi::tritonRustRankedTensorTypeGetDimSize(ty, dim) }
    }

    /// Get the element type
    pub fn element_type(ty: MLIRTypeRef) -> MLIRTypeRef {
        unsafe { ffi::tritonRustRankedTensorTypeGetElementType(ty) }
    }

    /// Get the encoding attribute (if any)
    pub fn encoding(ty: MLIRTypeRef) -> Option<MLIRAttributeRef> {
        let enc = unsafe { ffi::tritonRustRankedTensorTypeGetEncoding(ty) };
        if enc.is_null() {
            None
        } else {
            Some(enc)
        }
    }

    /// Get the full shape as a vector
    pub fn shape(ty: MLIRTypeRef) -> Vec<i64> {
        let r = rank(ty);
        (0..r).map(|i| dim_size(ty, i)).collect()
    }
}

/// Triton GPU encoding attributes
pub mod encoding {
    use super::*;

    /// Parameters for blocked encoding
    pub struct BlockedParams {
        pub size_per_thread: Vec<u32>,
        pub threads_per_warp: Vec<u32>,
        pub warps_per_cta: Vec<u32>,
        pub order: Vec<u32>,
        pub ctas_per_cga: Option<Vec<u32>>,
        pub cta_split_num: Option<Vec<u32>>,
        pub cta_order: Option<Vec<u32>>,
    }

    impl BlockedParams {
        /// Create blocked encoding params for a 2D tensor
        pub fn for_2d(
            size_per_thread: [u32; 2],
            threads_per_warp: [u32; 2],
            warps_per_cta: [u32; 2],
        ) -> Self {
            BlockedParams {
                size_per_thread: size_per_thread.to_vec(),
                threads_per_warp: threads_per_warp.to_vec(),
                warps_per_cta: warps_per_cta.to_vec(),
                order: vec![1, 0], // Column-major by default
                ctas_per_cga: None,
                cta_split_num: None,
                cta_order: None,
            }
        }
    }

    /// Create a blocked encoding attribute
    pub fn blocked(ctx: MLIRContextRef, params: &BlockedParams) -> MLIRAttributeRef {
        let rank = params.size_per_thread.len();
        unsafe {
            ffi::tritonRustBlockedEncodingAttrGet(
                ctx,
                rank,
                params.size_per_thread.as_ptr(),
                params.threads_per_warp.as_ptr(),
                params.warps_per_cta.as_ptr(),
                params.order.as_ptr(),
                params.ctas_per_cga.as_ref().map(|v| v.as_ptr()).unwrap_or(ptr::null()),
                params.cta_split_num.as_ref().map(|v| v.as_ptr()).unwrap_or(ptr::null()),
                params.cta_order.as_ref().map(|v| v.as_ptr()).unwrap_or(ptr::null()),
            )
        }
    }

    /// Check if an attribute is a blocked encoding
    pub fn is_blocked(attr: MLIRAttributeRef) -> bool {
        unsafe { ffi::tritonRustAttrIsBlockedEncoding(attr) != 0 }
    }

    /// Create a shared memory encoding attribute
    pub fn shared(
        ctx: MLIRContextRef,
        vec: u32,
        per_phase: u32,
        max_phase: u32,
        order: &[u32],
        has_leading_offset: bool,
    ) -> MLIRAttributeRef {
        unsafe {
            ffi::tritonRustSharedEncodingAttrGet(
                ctx,
                vec,
                per_phase,
                max_phase,
                order.len(),
                order.as_ptr(),
                has_leading_offset as i32,
            )
        }
    }

    /// Check if an attribute is a shared encoding
    pub fn is_shared(attr: MLIRAttributeRef) -> bool {
        unsafe { ffi::tritonRustAttrIsSharedEncoding(attr) != 0 }
    }

    /// Create a slice encoding attribute
    pub fn slice(ctx: MLIRContextRef, dim: u32, parent: MLIRAttributeRef) -> MLIRAttributeRef {
        unsafe { ffi::tritonRustSliceEncodingAttrGet(ctx, dim, parent) }
    }

    /// Check if an attribute is a slice encoding
    pub fn is_slice(attr: MLIRAttributeRef) -> bool {
        unsafe { ffi::tritonRustAttrIsSliceEncoding(attr) != 0 }
    }

    /// Create an NVIDIA MMA encoding for tensor cores
    pub fn nvidia_mma(
        ctx: MLIRContextRef,
        version_major: u32,
        version_minor: u32,
        warps_per_cta: &[u32],
        ctas_per_cga: &[u32],
        cta_split_num: &[u32],
        cta_order: &[u32],
        instr_shape: &[u32],
    ) -> MLIRAttributeRef {
        unsafe {
            ffi::tritonRustNvidiaMmaEncodingAttrGet(
                ctx,
                version_major,
                version_minor,
                warps_per_cta.len(),
                warps_per_cta.as_ptr(),
                ctas_per_cga.len(),
                ctas_per_cga.as_ptr(),
                cta_split_num.len(),
                cta_split_num.as_ptr(),
                cta_order.len(),
                cta_order.as_ptr(),
                instr_shape.len(),
                instr_shape.as_ptr(),
            )
        }
    }

    /// Create an AMD MFMA encoding for matrix cores
    pub fn amd_mfma(
        ctx: MLIRContextRef,
        version_major: u32,
        version_minor: u32,
        warps_per_cta: &[u32],
        m_dim: u32,
        n_dim: u32,
        is_transposed: bool,
        ctas_per_cga: &[u32],
        cta_split_num: &[u32],
        cta_order: &[u32],
    ) -> MLIRAttributeRef {
        unsafe {
            ffi::tritonRustAMDMfmaEncodingAttrGet(
                ctx,
                version_major,
                version_minor,
                warps_per_cta.len(),
                warps_per_cta.as_ptr(),
                m_dim,
                n_dim,
                is_transposed as i32,
                ctas_per_cga.len(),
                ctas_per_cga.as_ptr(),
                cta_split_num.len(),
                cta_split_num.as_ptr(),
                cta_order.len(),
                cta_order.as_ptr(),
            )
        }
    }
}

/// Triton attribute helpers
pub mod attr {
    use super::*;

    /// Create a cache modifier attribute
    pub fn cache_modifier(ctx: MLIRContextRef, modifier: CacheModifier) -> MLIRAttributeRef {
        unsafe { ffi::tritonRustCacheModifierAttrGet(ctx, modifier as i32) }
    }

    /// Create an eviction policy attribute
    pub fn eviction_policy(ctx: MLIRContextRef, policy: EvictionPolicy) -> MLIRAttributeRef {
        unsafe { ffi::tritonRustEvictionPolicyAttrGet(ctx, policy as i32) }
    }

    /// Create a padding option attribute
    pub fn padding_option(ctx: MLIRContextRef, option: PaddingOption) -> MLIRAttributeRef {
        unsafe { ffi::tritonRustPaddingOptionAttrGet(ctx, option as i32) }
    }

    /// Create a propagate NaN attribute
    pub fn propagate_nan(ctx: MLIRContextRef, option: PropagateNan) -> MLIRAttributeRef {
        unsafe { ffi::tritonRustPropagateNanAttrGet(ctx, option as i32) }
    }

    /// Create an atomic RMW operation attribute
    pub fn rmw_op(ctx: MLIRContextRef, op: RMWOp) -> MLIRAttributeRef {
        unsafe { ffi::tritonRustRMWOpAttrGet(ctx, op as i32) }
    }
}

/// GPU target information
pub mod target {
    use super::*;

    /// Get the default number of warps for a target
    pub fn num_warps(target: &str) -> u32 {
        unsafe { ffi::tritonRustGetNumWarps(MLIRStringRef::from_str(target)) }
    }

    /// Get the threads per warp for a target
    pub fn threads_per_warp(target: &str) -> u32 {
        unsafe { ffi::tritonRustGetThreadsPerWarp(MLIRStringRef::from_str(target)) }
    }
}
