// teenyc-6mv: correctly-formed hand-built shared-memory round-trip kernel.
//
// This is the fix for the SIGSEGV reproduced by naive_shared_memory.mlir. It
// reconstructs what Gluon's Python frontend emits implicitly, as a *real*
// kernel: a void `tt.func` that stages a tile through shared memory and writes
// it back through an output pointer (the "load -> shared stage -> compute
// offsets -> store back" round-trip from the issue). Kernels must be void — a
// kernel that returned a tensor would fail `tt.return` legalization during
// TritonGPU->LLVM lowering (only callee sub-functions return values).
//
// The three things a hand-built module must get right (all absent in the naive
// version, which is why it crashed):
//
//   1. The required module-level attributes (`ttg.num-warps`, `ttg.num-ctas`,
//      `ttg.threads-per-warp`, `ttg.target`). The encoding passes read these
//      (e.g. `lookupNumWarps`) and assume they are present.
//   2. A distributed (blocked) encoding on EVERY tensor — including the
//      `tt.make_range` offsets and the pointer tensor. `1 * 32 * 4 = 128`
//      matches the `128xi32` shape for num-warps=4, threads-per-warp=32.
//      A single null-encoding tensor anywhere re-triggers the crash.
//   3. `#ttg.shared_memory` + `mutable` on the (uninitialized) `local_alloc`.
//
// This module survives the full Gluon TTGIR pipeline, the shared-memory
// allocation pass, AND lowering to the LLVM dialect / PTX:
//
//     triton-opt golden_shared_memory.mlir \
//       --gluon-inline --gluon-infer-coalesced-encodings \
//       --gluon-resolve-auto-encodings --gluon-canonicalize --sccp \
//       --gluon-canonicalize --tritongpu-combine-tensor-select-and-if \
//       --allocate-shared-memory-nv --convert-scf-to-cf \
//       --convert-triton-gpu-to-llvm
//
// After `--allocate-shared-memory-nv` the module carries `ttg.shared = 512`
// (128 * 4 bytes) and each `ttg.local_alloc` gets an `allocation.offset`.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func public @golden_shared_mem(%out: !tt.ptr<i32>) {
    %c7 = arith.constant 7 : i32
    %t = tt.splat %c7 : i32 -> tensor<128xi32, #blocked>
    %sm = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    ttg.local_store %t, %sm : tensor<128xi32, #blocked> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    %r = ttg.local_load %sm : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> tensor<128xi32, #blocked>
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %outs = tt.splat %out : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %ptrs = tt.addptr %outs, %range : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %ptrs, %r : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
