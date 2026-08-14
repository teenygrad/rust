// teenyc-6mv: minimal hand-built shared-memory round-trip that reproduces the
// SIGSEGV in the Gluon (`Language::GLUON`) pipeline.
//
// This mirrors the "naive" construction from the reverted
// `feat/ttg-shared-memory-primitives` exploration: a load -> shared stage ->
// load-back round-trip built directly as `ttg` ops, WITHOUT the distributed
// (blocked) encoding that Gluon's own Python frontend always attaches to every
// tensor, and WITHOUT the required module-level `ttg.*` attributes.
//
// Running any of the Gluon encoding-inference passes over this module
// segfaults, e.g.:
//
//     triton-opt naive_shared_memory.mlir --gluon-infer-coalesced-encodings
//     triton-opt naive_shared_memory.mlir --gluon-resolve-auto-encodings
//
// Root cause (see golden_shared_memory.mlir for the corrected form):
//   `src/triton/lib/Dialect/Gluon/Transforms/InferCoalescedEncodings.cpp`
//   `src/triton/lib/Dialect/Gluon/Transforms/ResolveAutoEncodings.cpp`
//   both classify tensors with
//       `tensorTy && isa<...EncodingAttr>(tensorTy.getEncoding())`
//   and `mlir::triton::gluon::inferLayout` calls that predicate on the
//   function's argument/result types. For a plain `tensor<128xi32>` the
//   encoding attribute is null, and `isa<>` on a null `Attribute` dereferences
//   null -> SIGSEGV. Gluon's frontend never emits null-encoding tensors, so the
//   predicate was written assuming `getEncoding()` is always present.
//
// To debug:
//   gdb -q -ex run --args triton-opt naive_shared_memory.mlir \
//       --gluon-infer-coalesced-encodings -o /dev/null
//   (bt shows isCoalescedEncodingTensorType <- inferLayout <- runOnOperation)
module {
  tt.func public @naive_shared_mem() -> tensor<128xi32> {
    %c7 = arith.constant 7 : i32
    %t = tt.splat %c7 : i32 -> tensor<128xi32>
    %sm = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttg.local_store %t, %sm : tensor<128xi32> -> !ttg.memdesc<128xi32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %r = ttg.local_load %sm : !ttg.memdesc<128xi32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> tensor<128xi32>
    tt.return %r : tensor<128xi32>
  }
}
