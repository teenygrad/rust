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

use rustc_macros::Diagnostic;

#[derive(Diagnostic, Debug)]
pub enum MlirError {
    #[diag("mlir codegen failed: {$err}")]
    CodegenFailed { err: String },

    #[diag("mlir create operation failed: {$err}")]
    CreateOperation { err: rustc_mlir::errors::Error },

    #[diag("invalid scalar operand: {$node}")]
    InvalidScalar { node: String },

    #[diag("invalid type: {$msg}")]
    InvalidType { msg: String },

    #[diag("incomaptibale types: {$msg}")]
    IncompatibleTypes { msg: String },
}
