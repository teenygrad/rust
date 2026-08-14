// teenyc-6mv / teenygrad-3w0.10: indexed 2-D shared-memory buffer.
//
// The row-loop transpose (the only proven idiom in this codebase) produces
// the transpose via address arithmetic on the write side -- no T::trans
// call, no 2-D tensor. mark_stage_shared is same-shape only, so it cannot
// express "write row i in one loop iteration, read column j in a later
// one, buffer persists across the kernel". This fixture is that primitive:
//
//   local_alloc [128, 128]
//     -> memdesc_index(i) -> local_store          // write a row
//     -> ttg.barrier local                        // CTA handshake
//     -> memdesc_trans {order = [1, 0]}
//     -> memdesc_index(j) -> local_load           // read a column
//
// Encoding contract (verified against Triton's verifier):
//   * 2-D parent uses swizzled_shared order = [1, 0] (row-major).
//   * 1-D slice MUST use a 1-D encoding (order = [0]); reusing the 2-D
//     parent encoding on the slice fails with "rank must be equal to or
//     one less than the shape size".
//   * trans result uses the swapped order = [0, 1]. A full alloc stays a
//     full alloc after trans, so the trans result is a legal
//     memdesc_index source.
//
//     triton-opt indexed_shared_memory.mlir \
//       --gluon-inline --gluon-infer-coalesced-encodings \
//       --gluon-resolve-auto-encodings --gluon-canonicalize --sccp \
//       --gluon-canonicalize --tritongpu-combine-tensor-select-and-if \
//       --allocate-shared-memory-nv --convert-scf-to-cf \
//       --convert-triton-gpu-to-llvm --reconcile-unrealized-casts
//
// After `--allocate-shared-memory-nv` the module carries `ttg.shared = 65536`
// (128 * 128 * 4 bytes). LLVM lowering emits st.shared + nvvm.barrier0 + a
// shared-memory read (ldmatrix from ptr<3>).
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared2d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2dT = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func public @indexed_shared_mem(%out: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c7 = arith.constant 7 : i32
    %t = tt.splat %c7 : i32 -> tensor<128xi32, #blocked>
    %sm = ttg.local_alloc : () -> !ttg.memdesc<128x128xi32, #shared2d, #smem, mutable>
    %row = ttg.memdesc_index %sm[%c0] : !ttg.memdesc<128x128xi32, #shared2d, #smem, mutable> -> !ttg.memdesc<128xi32, #shared1d, #smem, mutable>
    ttg.local_store %t, %row : tensor<128xi32, #blocked> -> !ttg.memdesc<128xi32, #shared1d, #smem, mutable>
    ttg.barrier local
    %smT = ttg.memdesc_trans %sm {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xi32, #shared2d, #smem, mutable> -> !ttg.memdesc<128x128xi32, #shared2dT, #smem, mutable>
    %col = ttg.memdesc_index %smT[%c0] : !ttg.memdesc<128x128xi32, #shared2dT, #smem, mutable> -> !ttg.memdesc<128xi32, #shared1d, #smem, mutable>
    %r = ttg.local_load %col : !ttg.memdesc<128xi32, #shared1d, #smem, mutable> -> tensor<128xi32, #blocked>
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %outs = tt.splat %out : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %ptrs = tt.addptr %outs, %range : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %ptrs, %r : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
