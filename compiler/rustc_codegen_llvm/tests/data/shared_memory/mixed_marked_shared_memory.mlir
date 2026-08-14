// teenyc-6mv: "mixed" kernel prototype for shared-memory staging.
//
// This is the *input* a codegen would emit when it wants to mix ordinary
// (register-layout) Triton ops with an explicit shared-memory staging step,
// WITHOUT hand-encoding every tensor and WITHOUT routing through the Gluon
// pipeline. Every tensor is plain/unencoded; the only shared-memory intent is
// the discardable `ttg.stage_shared` unit attribute on the value we want to
// stage (the loaded tensor).
//
// The processing recipe is:
//   1. --convert-triton-to-tritongpu   (assigns a #ttg.blocked encoding to
//                                        every tensor; PRESERVES the marker)
//   2. --tritongpu-stage-shared-memory  (rewrites the marked, now-encoded value
//                                        into local_alloc / local_store /
//                                        local_load using that same encoding)
//   3. --allocate-shared-memory-nv --convert-triton-gpu-to-llvm
//                                       (lowers to LLVM: st.shared + barrier +
//                                        shared-memory read)
//
// Doing the staging *after* step 1 is the whole point: the shared-memory ops
// reuse the encoding the conversion already chose, so there is no null encoding
// (which crashed the Gluon encoding-inference passes) and no unresolved
// encoded<->unencoded materialization (which failed the direct route).
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func public @mixed_shared_mem(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %ins = tt.splat %in : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %iptrs = tt.addptr %ins, %range : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %x = tt.load %iptrs {ttg.stage_shared} : tensor<128x!tt.ptr<i32>>
    %outs = tt.splat %out : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %optrs = tt.addptr %outs, %range : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    tt.store %optrs, %x : tensor<128x!tt.ptr<i32>>
    tt.return
  }
}
