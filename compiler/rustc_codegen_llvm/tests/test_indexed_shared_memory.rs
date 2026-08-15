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

//! teenygrad-3w0.10: drive an *indexed* shared-memory buffer — a
//! kernel-lifetime `[128, 128]` buffer written at row `i`, barriered,
//! transposed, and read back at column `j` — end-to-end from Rust/melior
//! through the normal `Language::TRITON` pipeline
//! ([`TritonCompiler::compile`]).
//!
//! This is the indexed-buffer counterpart to `test_mixed_shared_memory.rs`.
//! `ttg.stage_shared`/`tritongpu-stage-shared-memory` only supports
//! same-shape stage-and-readback, so it cannot express a real cross-thread
//! transpose. This test instead uses the five `tt.shared_*` marker ops
//! (`tt.shared_alloc`/`shared_store_index`/`shared_barrier`/`shared_trans`/
//! `shared_load_index`, TritonOps.td) — real ops with real tensor/index
//! operands, not a bare attribute on a borrowed op — and the
//! `tritongpu-lower-indexed-shared-memory` pass that rewrites them into the
//! real `ttg.local_alloc`/`memdesc_index`/`local_store`/`barrier`/
//! `memdesc_trans`/`local_load` sequence once `convert-triton-to-tritongpu`
//! has assigned real encodings. No hand-encoding, no Gluon pipeline.
//!
//! This test only exercises row 0 / column 0 (a single index, not a real
//! loop) — enough to prove the marker-op-to-real-op materialization works
//! through the real pipeline. `kernels/teeny-kernels`'s real
//! `transpose_2d_forward` kernel (teenygrad-3w0.10 Step 1) is what drives
//! this with a genuine per-row loop.

#![feature(rustc_private)]

use melior::ir::operation::OperationLike;
use melior::ir::r#type::IntegerType;
use melior::ir::{Block, BlockLike, Location, Operation, RegionLike, Type, Value};
use rustc_codegen_llvm::mlir::MlirModule;
use rustc_mlir::shared::arith::{Int, create_int_constant};
use rustc_mlir::triton::tensor::{
    CacheModifier, EvictionPolicy, add_ptr, load, shared_alloc, shared_barrier,
    shared_load_index, shared_store_index, shared_trans, splat, store,
};
use rustc_mlir::triton::tt::{MakeRangeOperation, ReturnOperation};
use rustc_mlir::triton::{create_func, load_triton_dialect, load_triton_gpu_dialect};

/// A kernel using the five `tt.shared_*` indexed-buffer markers must lower
/// through the ordinary TRITON pipeline: `tritongpu-lower-indexed-shared-
/// memory` turns them into a real `ttg.local_alloc` / `memdesc_index` /
/// `local_store` / `barrier` / `memdesc_trans` / `local_load` sequence
/// (visible in the TTGIR), and the kernel reaches PTX with a real
/// shared-memory buffer.
#[test]
fn indexed_shared_memory_kernel_lowers_from_rust() {
    let mut module = MlirModule::new_with_capability("indexed_shared_mem", 90);
    load_triton_dialect(&module.context);
    load_triton_gpu_dialect(&module.context);

    let location = Location::unknown(&module.context);
    let i32_ty: Type = IntegerType::new(&module.context, 32).into();

    // Everything is plain / un-encoded: this is what a front-end emits before
    // `convert-triton-to-tritongpu` assigns layouts. No `#ttg.blocked` here.
    let tensor_ty: Type =
        Type::parse(&module.context, "tensor<128xi32>").expect("tensor<128xi32> should parse");
    let ptr_tensor_ty: Type = Type::parse(&module.context, "tensor<128x!tt.ptr<i32>>")
        .expect("tensor<128x!tt.ptr<i32>> should parse");
    let ptr_ty: Type =
        Type::parse(&module.context, "!tt.ptr<i32>").expect("!tt.ptr<i32> should parse");

    // Void kernel: (input ptr, output ptr) -> ().
    let func_op = create_func(
        &module.context,
        location,
        "indexed_shared_mem",
        "public",
        &[ptr_ty, ptr_ty],
        &[],
        16,
    )
    .expect("create_func should succeed");

    let block = Block::new(&[(ptr_ty, location), (ptr_ty, location)]);
    let in_ptr: Value = block.argument(0).unwrap().into();
    let out_ptr: Value = block.argument(1).unwrap().into();

    let range_op: Operation = MakeRangeOperation::builder(&module.context, location)
        .start(melior::ir::attribute::IntegerAttribute::new(i32_ty, 0))
        .end(melior::ir::attribute::IntegerAttribute::new(i32_ty, 128))
        .result(tensor_ty)
        .build()
        .into();
    let range_val: Value = range_op.result(0).unwrap().into();
    block.append_operation(range_op);

    // tile = load(in + arange) : tensor<128xi32> -- the row we stage.
    let in_splat_op: Operation = splat(&module.context, location, in_ptr, ptr_tensor_ty)
        .expect("tt.splat of input pointer should build")
        .into();
    let in_ptrs_base: Value = in_splat_op.result(0).unwrap().into();
    block.append_operation(in_splat_op);

    let in_addptr_op: Operation =
        add_ptr(&module.context, location, in_ptrs_base, range_val, ptr_tensor_ty)
            .expect("tt.addptr (input) should build")
            .into();
    let in_ptrs: Value = in_addptr_op.result(0).unwrap().into();
    block.append_operation(in_addptr_op);

    let load_op: Operation = load(
        &module.context,
        location,
        in_ptrs,
        None,
        None,
        tensor_ty,
        CacheModifier::None,
        EvictionPolicy::Normal,
        false,
    )
    .expect("tt.load should build");
    let tile: Value = load_op.result(0).unwrap().into();
    block.append_operation(load_op);

    // idx = 0 : i32 -- both the write row and (post-transpose) read column.
    let idx_op: Operation = create_int_constant(&module.context, location, Int::I32(0))
        .expect("arith.constant i32 should build")
        .into();
    let idx: Value = idx_op.result(0).unwrap().into();
    block.append_operation(idx_op);

    // buf = tt.shared_alloc {shape = [128, 128], elem_type = i32}
    let alloc_op: Operation = shared_alloc(&module.context, location, &[128, 128], i32_ty)
        .expect("tt.shared_alloc should build")
        .into();
    let buf: Value = alloc_op.result(0).unwrap().into();
    block.append_operation(alloc_op);

    // tt.shared_store_index(buf, idx, tile) -- write row 0.
    let store_idx_op: Operation = shared_store_index(location, buf, idx, tile)
        .expect("tt.shared_store_index should build")
        .into();
    block.append_operation(store_idx_op);

    // tt.shared_barrier -- handshake before any thread reads back.
    let bar_op: Operation =
        shared_barrier(location).expect("tt.shared_barrier should build").into();
    block.append_operation(bar_op);

    // bufT = tt.shared_trans(buf)
    let trans_op: Operation = shared_trans(&module.context, location, buf)
        .expect("tt.shared_trans should build")
        .into();
    let buf_t: Value = trans_op.result(0).unwrap().into();
    block.append_operation(trans_op);

    // loaded = tt.shared_load_index(bufT, idx) -- read column 0 of the
    // original buffer (row 0 of the transposed view).
    let load_idx_op: Operation = shared_load_index(location, buf_t, idx, tensor_ty)
        .expect("tt.shared_load_index should build")
        .into();
    let loaded: Value = load_idx_op.result(0).unwrap().into();
    block.append_operation(load_idx_op);

    // store(out + arange, loaded)
    let out_splat_op: Operation = splat(&module.context, location, out_ptr, ptr_tensor_ty)
        .expect("tt.splat of output pointer should build")
        .into();
    let out_ptrs_base: Value = out_splat_op.result(0).unwrap().into();
    block.append_operation(out_splat_op);

    let out_addptr_op: Operation =
        add_ptr(&module.context, location, out_ptrs_base, range_val, ptr_tensor_ty)
            .expect("tt.addptr (output) should build")
            .into();
    let out_ptrs: Value = out_addptr_op.result(0).unwrap().into();
    block.append_operation(out_addptr_op);

    let global_store_op: Operation = store(
        &module.context,
        location,
        out_ptrs,
        loaded,
        None,
        CacheModifier::None,
        EvictionPolicy::Normal,
    )
    .expect("tt.store should build");
    block.append_operation(global_store_op);

    let ret_op = ReturnOperation::builder(&module.context, location).srcs(&[]).build();
    block.append_operation(ret_op.into());
    func_op.body().unwrap().append_block(block);
    module.mlir.body().append_operation(func_op.into());

    // Drive the ordinary TRITON pipeline (NOT compile_gluon): convert ->
    // lower-indexed-shared-memory -> ... -> PTX.
    let ok = module.compiler.compile(module.mlir.to_raw());

    // First contract: the lowering pass ran post-encoding and rewrote the
    // marker ops into the real indexed shared-memory sequence, visible in
    // the TTGIR.
    let ttgir = module.compiler.get_ttgir().unwrap_or_default();
    for needle in [
        "ttg.local_alloc",
        "ttg.memdesc_index",
        "ttg.local_store",
        "ttg.barrier",
        "ttg.memdesc_trans",
        "ttg.local_load",
    ] {
        assert!(
            ttgir.contains(needle),
            "expected `{needle}` in TTGIR after lowering the indexed shared-memory \
             kernel (compile ok={ok}), got:\n{ttgir}"
        );
    }
    assert!(
        !ttgir.contains("tt.shared_"),
        "every `tt.shared_*` marker must be consumed by the lowering pass \
         (compile ok={ok}), got:\n{ttgir}"
    );

    // Second contract: the kernel lowers all the way to PTX, and the indexed
    // round-trip materializes a real `.shared` buffer.
    let asm = module.compiler.get_asm().unwrap_or_default();
    assert!(
        asm.contains("indexed_shared_mem"),
        "expected PTX for `indexed_shared_mem` (compile ok={ok}), got:\n{asm}"
    );
    assert!(
        asm.contains(".shared"),
        "expected a `.shared` buffer in the PTX from the indexed round-trip \
         (compile ok={ok}), got:\n{asm}"
    );
}
