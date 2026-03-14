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

use melior::Context;
use mlir_sys::{MlirContext, MlirModule};

use crate::ffi::{self, MlirTritonCompiler};

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
    /// * `target` - Target name, e.g. `"cuda"`.
    /// * `options` - Optional compiler options as a string; pass `None` for defaults.
    ///
    /// Returns `None` if creation failed (e.g. invalid context or target).
    pub fn new(context: MlirContext, target: &str, options: &str) -> Option<Self> {
        let target_c = CString::new(target).ok()?;
        let options_c = CString::new(options).ok()?;

        let raw = unsafe {
            ffi::mlirTritonCompilerCreate(context, target_c.as_ptr(), options_c.as_ptr())
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
    ///
    /// The returned slice is valid until the next successful [TritonCompiler::compile]
    /// on this compiler or until the compiler is dropped. Returns `None` if
    /// there is no output or the pointer is invalid.
    pub fn get_output(&self) -> Option<&str> {
        let ptr = unsafe { ffi::mlirTritonCompilerGetOutput(self.raw) };
        if ptr.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(ptr).to_str().ok() }
    }
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
