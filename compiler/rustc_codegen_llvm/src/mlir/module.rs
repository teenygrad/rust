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

use std::ffi::CStr;

use melior::Context;
use melior::ir::{Location, Module};
use rustc_codegen_ssa::back::write::CodegenContext;
use rustc_errors::DiagCtxtHandle;

use crate::mlir::backend::MlirCodegenBackend;

/// Represents an MLIR module during codegen
pub struct MlirModule<'c> {
    pub name: String,
    pub mlir: Module<'c>,
    pub context: Context,
}

unsafe impl<'c> Send for MlirModule<'c> {}
unsafe impl<'c> Sync for MlirModule<'c> {}

impl<'c> MlirModule<'c> {
    pub fn new(mod_name: &str) -> Self {
        let context = Context::new();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        Self { name: mod_name.to_string(), context, mlir: module }
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn parse(
        _cgcx: &CodegenContext<MlirCodegenBackend>,
        name: &CStr,
        _buffer: &[u8],
        _dcx: DiagCtxtHandle<'_>,
    ) -> Self {
        let context = Context::new();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        Self { name: name.to_string_lossy().to_string(), context, mlir: module }
    }

    pub fn set_llmod(&mut self, llmod: Module<'c>) {
        self.mlir = llmod;
    }

    pub fn llmod(&self) -> &Module<'c> {
        &self.mlir
    }
}
