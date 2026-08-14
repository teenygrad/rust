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

//! teenyc-6mv: drive a hand-built shared-memory round-trip through the
//! `Language::GLUON` pipeline entirely from Rust/melior.
//!
//! This is the end-to-end counterpart to the standalone `triton-opt`
//! reproduction in `test_shared_memory_pipeline.rs`. It builds a real void
//! kernel — `ttg.local_alloc` -> `local_store` -> `local_load`, then a
//! per-lane `tt.store` back through an output pointer — with the restored
//! primitives ([`rustc_mlir::triton::tensor::local_alloc`] etc.), sets the
//! module attributes Gluon's frontend sets implicitly, and lowers it via the
//! new [`rustc_mlir::triton::TritonCompiler::compile_gluon`] FFI entry (which
//! runs `CudaBackend::gluonToTTGIR`, bypassing `convert-triton-to-tritongpu`).
//!
//! Two details matter, both of which the naive hand-built module got wrong:
//!
//! * Every tensor threaded through the ops carries a distributed `#ttg.blocked`
//!   encoding. Without it the Gluon encoding-inference passes dereference a null
//!   encoding attribute and SIGSEGV (see `test_shared_memory_pipeline.rs`).
//! * The kernel is void and writes results through an output pointer. A kernel
//!   that *returned* a tensor would fail `tt.return` legalization during the
//!   TritonGPU->LLVM lowering (only callee sub-functions return values), so it
//!   would never reach PTX.

#![feature(rustc_private)]

use melior::ir::attribute::{IntegerAttribute, StringAttribute};
use melior::ir::operation::{OperationLike, OperationMutLike};
use melior::ir::r#type::IntegerType;
use melior::ir::{Block, BlockLike, Location, Operation, RegionLike, Type, Value};
use rustc_codegen_llvm::mlir::MlirModule;
use rustc_mlir::shared::arith::{Int, create_int_constant};
use rustc_mlir::triton::tensor::{
    CacheModifier, EvictionPolicy, add_ptr, local_alloc, local_load, local_store, splat, store,
};
use rustc_mlir::triton::tt::{MakeRangeOperation, ReturnOperation};
use rustc_mlir::triton::{
    create_func, load_triton_dialect, load_triton_gpu_dialect, shared_mem_desc_type,
};

/// A hand-built shared-memory round-trip lowers through the Gluon pipeline
/// without crashing, and the shared-memory ops survive into TTGIR. This proves
/// the round-trip is drivable from Rust and that the null-encoding SIGSEGV is
/// avoided by attaching a distributed encoding + the required module attrs.
#[test]
fn gluon_shared_memory_roundtrip_is_drivable_from_rust() {
    let mut module = MlirModule::new_with_capability("gluon_shared_mem", 90);
    load_triton_dialect(&module.context);
    load_triton_gpu_dialect(&module.context);

    let location = Location::unknown(&module.context);
    let i32_ty: Type = IntegerType::new(&module.context, 32).into();

    // Distributed (blocked) encoding is REQUIRED: 1 * 32 * 4 = 128 matches the
    // 128-element tile for num-warps=4, threads-per-warp=32. A plain
    // `tensor<128xi32>` here would SIGSEGV the Gluon encoding passes.
    let tensor_ty: Type = Type::parse(
        &module.context,
        "tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], \
         warpsPerCTA = [4], order = [0]}>>",
    )
    .expect("blocked-encoded tensor type should parse");
    // The output pointer tensor needs the same distributed encoding.
    let ptr_tensor_ty: Type = Type::parse(
        &module.context,
        "tensor<128x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], \
         warpsPerCTA = [4], order = [0]}>>",
    )
    .expect("blocked-encoded pointer tensor type should parse");
    let out_ptr_ty: Type =
        Type::parse(&module.context, "!tt.ptr<i32>").expect("!tt.ptr<i32> should parse");
    let mem_desc_ty = shared_mem_desc_type(&module.context, i32_ty, 128, true);

    // A real kernel is void: it takes the output pointer as an argument and
    // returns nothing.
    let func_op = create_func(
        &module.context,
        location,
        "gluon_shared_mem",
        "public",
        &[out_ptr_ty],
        &[],
        16,
    )
    .expect("create_func should succeed");

    let block = Block::new(&[(out_ptr_ty, location)]);
    let out_ptr: Value = block.argument(0).unwrap().into();

    let const_op: Operation = create_int_constant(&module.context, location, Int::I32(7))
        .expect("arith.constant should build")
        .into();
    let const_val: Value = const_op.result(0).unwrap().into();
    block.append_operation(const_op);

    let splat_op: Operation = splat(&module.context, location, const_val, tensor_ty)
        .expect("tt.splat should build")
        .into();
    let tensor_val: Value = splat_op.result(0).unwrap().into();
    block.append_operation(splat_op);

    let alloc_op: Operation = local_alloc(&module.context, location, None, None, mem_desc_ty)
        .expect("ttg.local_alloc should build")
        .into();
    let mem_desc: Value = alloc_op.result(0).unwrap().into();
    block.append_operation(alloc_op);

    let store_op: Operation =
        local_store(location, tensor_val, mem_desc).expect("ttg.local_store should build").into();
    block.append_operation(store_op);

    let load_op: Operation =
        local_load(location, mem_desc, tensor_ty).expect("ttg.local_load should build").into();
    let loaded: Value = load_op.result(0).unwrap().into();
    block.append_operation(load_op);

    // Compute per-lane output addresses: `out + arange(0, 128)`. The range must
    // also carry the distributed encoding, otherwise its null encoding
    // re-triggers the Gluon crash.
    let range_op: Operation = MakeRangeOperation::builder(&module.context, location)
        .start(IntegerAttribute::new(i32_ty, 0))
        .end(IntegerAttribute::new(i32_ty, 128))
        .result(tensor_ty)
        .build()
        .into();
    let range_val: Value = range_op.result(0).unwrap().into();
    block.append_operation(range_op);

    let out_splat_op: Operation = splat(&module.context, location, out_ptr, ptr_tensor_ty)
        .expect("tt.splat of output pointer should build")
        .into();
    let out_ptrs_base: Value = out_splat_op.result(0).unwrap().into();
    block.append_operation(out_splat_op);

    let addptr_op: Operation =
        add_ptr(&module.context, location, out_ptrs_base, range_val, ptr_tensor_ty)
            .expect("tt.addptr should build")
            .into();
    let out_ptrs: Value = addptr_op.result(0).unwrap().into();
    block.append_operation(addptr_op);

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

    // Required module attributes, exactly as Gluon's frontend sets them up front
    // (the encoding passes read `ttg.num-warps` etc. and assume they exist).
    module
        .mlir
        .as_operation_mut()
        .set_attribute("ttg.num-warps", IntegerAttribute::new(i32_ty, 4).into());
    module
        .mlir
        .as_operation_mut()
        .set_attribute("ttg.num-ctas", IntegerAttribute::new(i32_ty, 1).into());
    module
        .mlir
        .as_operation_mut()
        .set_attribute("ttg.threads-per-warp", IntegerAttribute::new(i32_ty, 32).into());
    module
        .mlir
        .as_operation_mut()
        .set_attribute("ttg.target", StringAttribute::new(&module.context, "cuda:90").into());

    let ok = module.compiler.compile_gluon(module.mlir.to_raw());

    // First contract: the Gluon pipeline consumed the hand-built shared-memory
    // ops without crashing and emitted TTGIR still containing them.
    // `applyPasses` populates the TTGIR string right after `gluonToTTGIR`.
    let ttgir = module.compiler.get_ttgir().unwrap_or_default();
    for needle in ["ttg.local_alloc", "ttg.local_store", "ttg.local_load"] {
        assert!(
            ttgir.contains(needle),
            "expected `{needle}` in Gluon TTGIR (compile_gluon ok={ok}), got:\n{ttgir}"
        );
    }

    // Second contract: because the kernel is void, the round-trip lowers all the
    // way to PTX. `makeASM` only needs the in-tree NVPTX target, so the PTX must
    // be present even if the optional `ptxas` cubin step (`makeBIN`) is
    // unavailable in this environment (in which case `ok` is false). A non-empty
    // PTX proves `tt.return` legalized — the regression a tensor-returning kernel
    // would have hit.
    let asm = module.compiler.get_asm().unwrap_or_default();
    assert!(
        asm.contains(".visible .entry gluon_shared_mem") || asm.contains("gluon_shared_mem"),
        "expected PTX for `gluon_shared_mem` (compile_gluon ok={ok}), got:\n{asm}"
    );
}
