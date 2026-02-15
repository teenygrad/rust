/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
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
