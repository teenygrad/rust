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

#ifndef RVV_DIALECT
#define RVV_DIALECT

include "mlir/IR/OpBase.td"

def RVV_Dialect : Dialect {
  let name = "rvv";
  let cppNamespace = "::mlir::rvv";

  let summary = "RISC-V Vector (RVV) extension dialect.";
  let description = [{
    Models RISC-V Vector (RVV) extension concepts that have no equivalent in
    MLIR's builtin `vector` dialect -- starting with dynamic vector-length
    negotiation (`vsetvli`), the defining feature of RVV's
    vector-length-agnostic (VLA) programming model. Ordinary vector
    arithmetic over a statically-known (possibly scalable) length is
    expressed with the builtin `vector<[N]xT>` type and the `vector`/`arith`
    dialects; this dialect only adds what those can't already express.

    Deliberately lives outside the `triton` MLIR dialects (compare
    `NVGPUDialect`, `TritonNvidiaGPUDialect`) and outside the `src/triton`
    submodule entirely: unlike an NVIDIA/AMD Triton backend, `RiscvBackend`
    (see `triton/backend/RiscvBackend.h`) does not lower through Triton's
    TTIR/TTGIR pipeline -- RISC-V's CPU/vector execution model doesn't map
    onto Triton's GPU-SIMT abstractions (warps, shared memory, etc.) the
    way that pipeline assumes. This dialect is this crate's own, consumed
    directly by RiscvBackend's (currently placeholder) LLVM-IR lowering.

    This is intentionally a minimal first cut: it exists to establish the
    dialect scaffolding and the single most-justified op (`rvv.vsetvli`),
    not to model the full RVV instruction set. Extend it as real
    MIR-to-LLVM-IR lowering for RISC-V actually needs more of it.
  }];
}

#endif // RVV_DIALECT
