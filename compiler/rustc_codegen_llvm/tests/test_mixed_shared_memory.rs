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

//! teenyc-6mv: drive a *mixed* kernel — ordinary register-layout Triton ops
//! plus one shared-memory staging step — end-to-end from Rust/melior through
//! the normal `Language::TRITON` pipeline ([`TritonCompiler::compile`]).
//!
//! This is the "real fix" counterpart to `test_gluon_shared_memory.rs`. Instead
//! of hand-emitting `ttg.local_alloc`/`local_store`/`local_load` on
//! hand-encoded tensors and bypassing `convert-triton-to-tritongpu` (the Gluon
//! route, which crashed on any un-encoded tensor), this test:
//!
//!   1. builds a plain, **un-encoded** kernel (`tensor<128xi32>`, no `#ttg`
//!      layouts, no module `ttg.*` attributes) — exactly what a front-end
//!      naturally emits — that loads from an input pointer and stores to an
//!      output pointer;
//!   2. marks *only* the loaded value with
//!      [`rustc_mlir::triton::tensor::mark_stage_shared`] (`ttg.stage_shared`);
//!   3. runs the ordinary [`TritonCompiler::compile`] pipeline.
//!
//! `convert-triton-to-tritongpu` assigns every tensor a distributed
//! `#ttg.blocked` encoding and preserves the marker; the new
//! `tritongpu-stage-shared-memory` pass (wired into `CudaBackend::makeTTGIR`
//! right after the conversion) then rewrites the marked value into the three
//! shared-memory ops, reusing that encoding. The result lowers all the way to
//! PTX with a real shared-memory round-trip — no hand-encoding, no Gluon
//! pipeline, no null-encoding crash.

#![feature(rustc_private)]

use melior::ir::attribute::IntegerAttribute;
use melior::ir::operation::OperationLike;
use melior::ir::r#type::IntegerType;
use melior::ir::{Block, BlockLike, Location, Operation, RegionLike, Type, Value};
use rustc_codegen_llvm::mlir::MlirModule;
use rustc_mlir::triton::tensor::{
    CacheModifier, EvictionPolicy, add_ptr, load, mark_stage_shared, splat, store,
};
use rustc_mlir::triton::tt::{MakeRangeOperation, ReturnOperation};
use rustc_mlir::triton::{create_func, load_triton_dialect, load_triton_gpu_dialect};

/// A mixed kernel whose only shared-memory intent is a `ttg.stage_shared`
/// marker must lower through the ordinary TRITON pipeline: the staging pass
/// turns the marked value into `ttg.local_alloc`/`local_store`/`local_load`
/// (visible in the TTGIR) and the kernel reaches PTX with a `.shared` buffer.
#[test]
fn mixed_marked_kernel_stages_through_shared_memory_from_rust() {
    let mut module = MlirModule::new_with_capability("mixed_shared_mem", 90);
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
    let out_ptr_ty: Type =
        Type::parse(&module.context, "!tt.ptr<i32>").expect("!tt.ptr<i32> should parse");

    // Void kernel: (input ptr, output ptr) -> ().
    let func_op = create_func(
        &module.context,
        location,
        "mixed_shared_mem",
        "public",
        &[out_ptr_ty, out_ptr_ty],
        &[],
        16,
    )
    .expect("create_func should succeed");

    let block = Block::new(&[(out_ptr_ty, location), (out_ptr_ty, location)]);
    let in_ptr: Value = block.argument(0).unwrap().into();
    let out_ptr: Value = block.argument(1).unwrap().into();

    // arange(0, 128) for both the input and output address computations.
    let range_op: Operation = MakeRangeOperation::builder(&module.context, location)
        .start(IntegerAttribute::new(i32_ty, 0))
        .end(IntegerAttribute::new(i32_ty, 128))
        .result(tensor_ty)
        .build()
        .into();
    let range_val: Value = range_op.result(0).unwrap().into();
    block.append_operation(range_op);

    // in_ptrs = in + arange
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

    // x = load(in_ptrs); this is the value we stage through shared memory.
    let mut load_op: Operation = load(
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
    // The whole point: mark the loaded value for shared-memory staging. The
    // marker survives `convert-triton-to-tritongpu`; the staging pass consumes
    // it post-encoding.
    mark_stage_shared(&module.context, &mut load_op);
    let loaded: Value = load_op.result(0).unwrap().into();
    block.append_operation(load_op);

    // out_ptrs = out + arange
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
    // stage-shared-memory -> ... -> PTX.
    let ok = module.compiler.compile(module.mlir.to_raw());

    // First contract: the staging pass ran post-encoding and rewrote the marked
    // value into shared-memory ops, which are visible in the TTGIR.
    let ttgir = module.compiler.get_ttgir().unwrap_or_default();
    for needle in ["ttg.local_alloc", "ttg.local_store", "ttg.local_load"] {
        assert!(
            ttgir.contains(needle),
            "expected `{needle}` in TTGIR after staging the marked mixed kernel \
             (compile ok={ok}), got:\n{ttgir}"
        );
    }
    assert!(
        !ttgir.contains("stage_shared"),
        "the `ttg.stage_shared` marker must be consumed by the pass (compile ok={ok}), \
         got:\n{ttgir}"
    );

    // Second contract: the mixed kernel lowers all the way to PTX, and the
    // staged round-trip materializes a `.shared` buffer.
    let asm = module.compiler.get_asm().unwrap_or_default();
    assert!(
        asm.contains("mixed_shared_mem"),
        "expected PTX for `mixed_shared_mem` (compile ok={ok}), got:\n{asm}"
    );
    assert!(
        asm.contains(".shared"),
        "expected a `.shared` buffer in the PTX from the staged round-trip \
         (compile ok={ok}), got:\n{asm}"
    );
}
