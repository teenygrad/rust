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

/// Error type for MLIR operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("MLIR operation failed")]
    OperationFailed,

    #[error("Triton dialects not available (compiled without TRITON_ENABLED)")]
    TritonNotAvailable,

    #[error("Invalid type: {0}")]
    InvalidType(String),

    #[error("Invalid attribute: {0}")]
    InvalidAttribute(String),

    #[error("Module verification failed")]
    VerificationFailed,

    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Result type for MLIR operations
pub type Result<T> = std::result::Result<T, Error>;
