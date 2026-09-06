/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::ffi::{CStr, CString};

use mlir_sys::{MlirContext, MlirModule};

use crate::ffi::{self, CompileOptions, MlirTritonCompiler};

/// Safe wrapper around the Triton compiler C API.
///
/// Use [TritonCompiler::new] to create a compiler, then [TritonCompiler::compile]
/// to run the pipeline on an MLIR module. The compiled output (e.g. IR string)
/// is available via [TritonCompiler::get_output] until the next successful
/// [TritonCompiler::compile] or until the compiler is dropped.
pub struct TritonCompiler {
    raw: MlirTritonCompiler,
}

impl TritonCompiler {
    /// Creates a new Triton compiler for the given MLIR context and target.
    ///
    /// * `context` - MLIR context (e.g. from melior).
    /// * `target`  - Target name, e.g. `"cuda"`.
    /// * `options` - Compile options; use [`CompileOptions::default_cuda`] for
    ///               sensible CUDA defaults.
    ///
    /// Returns `None` if creation failed (e.g. invalid context or target).
    pub fn new(context: MlirContext, target: &str, options: &CompileOptions) -> Option<Self> {
        let target_c = CString::new(target).ok()?;

        let raw = unsafe {
            ffi::mlirTritonCompilerCreate(context, target_c.as_ptr(), options as *const _)
        };

        if raw.ptr.is_null() {
            return None;
        }

        Some(Self { raw })
    }

    /// Runs the Triton compilation pipeline on `module`.
    ///
    /// The module is transformed in-place. On success, the compiler stores
    /// the output (e.g. textual IR) for retrieval via [TritonCompiler::get_output].
    ///
    /// Returns `true` if compilation succeeded, `false` otherwise.
    pub fn compile(&mut self, module: MlirModule) -> bool {
        unsafe { ffi::mlirTritonCompilerCompile(self.raw, module) }
    }

    /// Returns the output string from the last successful [TritonCompiler::compile].
    /// This is the assembly (e.g. PTX) output. Prefer [TritonCompiler::get_asm] for clarity.
    ///
    /// The returned slice is valid until the next successful [TritonCompiler::compile]
    /// on this compiler or until the compiler is dropped. Returns `None` if
    /// there is no output or the pointer is invalid.
    pub fn get_asm(&self) -> Option<&str> {
        ptr_to_str(self, unsafe { ffi::mlirTritonCompilerGetASM(self.raw) })
    }

    /// Returns the raw bytes of the compiled binary from the last successful
    /// [TritonCompiler::compile], e.g. RiscvBackend's linked ELF shared
    /// library. This uses the backend's explicit byte length (`getBINSize`)
    /// rather than scanning for a NUL terminator, since the underlying
    /// buffer is not guaranteed to be NUL-terminated and may contain
    /// embedded NULs. Returns `None` if there is no output.
    ///
    /// The returned slice's lifetime rules match [TritonCompiler::get_asm].
    pub fn get_bin_bytes(&self) -> Option<&[u8]> {
        let ptr = unsafe { ffi::mlirTritonCompilerGetBIN(self.raw) };
        if ptr.is_null() {
            return None;
        }
        let len = unsafe { ffi::mlirTritonCompilerGetBINSize(self.raw) };
        if len == 0 {
            return None;
        }
        // Safety: `ptr` is non-null and owned by `self` (the C++ compiler
        // handle), valid for `len` bytes per getBINSize()'s contract, and
        // the returned slice's lifetime is tied to `&self` below.
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Returns the LLIR (input MLIR) string from the last successful compile.
    pub fn get_llir(&self) -> Option<&str> {
        ptr_to_str(self, unsafe { ffi::mlirTritonCompilerGetLLIR(self.raw) })
    }

    /// Returns the TTIR (Triton IR) string from the last successful compile.
    pub fn get_ttir(&self) -> Option<&str> {
        ptr_to_str(self, unsafe { ffi::mlirTritonCompilerGetTTIR(self.raw) })
    }

    /// Returns the TTGIR (Triton GPU IR) string from the last successful compile.
    pub fn get_ttgir(&self) -> Option<&str> {
        ptr_to_str(self, unsafe { ffi::mlirTritonCompilerGetTTGIR(self.raw) })
    }

    /// Returns the LLVM IR string from the last successful compile.
    pub fn get_llvm_ir(&self) -> Option<&str> {
        ptr_to_str(self, unsafe { ffi::mlirTritonCompilerGetLLVMIR(self.raw) })
    }
}

// Tie the returned string's lifetime to the compiler so that:
// - the reference can't outlive the TritonCompiler (which owns the C++ object)
// - a &mut borrow for compile() will conflict with any live reference, preventing
//   use-after-reallocation when m_asm is replaced by the next compile() call.
fn ptr_to_str<'a>(_anchor: &'a TritonCompiler, ptr: *const std::ffi::c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr).to_str().ok() }
}

impl Drop for TritonCompiler {
    fn drop(&mut self) {
        if !self.raw.ptr.is_null() {
            unsafe {
                ffi::mlirTritonCompilerFree(self.raw);
            }
            self.raw.ptr = std::ptr::null_mut();
        }
    }
}
