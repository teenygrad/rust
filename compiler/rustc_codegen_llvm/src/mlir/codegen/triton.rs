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

use melior::ir::{BlockLike, Location, Operation};
use melior::utility::register_all_llvm_translations;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::ty::layout::MaybeResult;
use rustc_middle::ty::{Instance, Ty, TyCtxt, TypingEnv};
use rustc_mlir::load_all_dialects;
use rustc_mlir::triton::{create_tt_func_with_divisibility, load_triton_dialect};

use crate::mlir::MlirModule;
use crate::mlir::codegen::Codegen;
use crate::mlir::codegen::types::TypeMapper;
use crate::mlir::errors::MlirError;

pub(crate) struct TritonCodegen<'a> {
    module: &'a MlirModule<'static>,
    type_mapper: TypeMapper<'a>,
}

impl<'a> TritonCodegen<'a> {
    pub(crate) fn new(module: &'a MlirModule<'static>) -> Self {
        let context = module.context();

        load_all_dialects(context);
        register_all_llvm_translations(context);
        load_triton_dialect(context);

        Self { module, type_mapper: TypeMapper::new(context) }
    }

    fn codegen_function<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        fn_ty: Ty<'tcx>,
        instance: &Instance<'tcx>,
    ) -> Result<(), MlirError> {
        // Downcast to a FnSig
        let fn_sig = fn_ty.fn_sig(tcx);
        let fn_sig = fn_sig.skip_binder(); // Remove late-bound lifetimes

        // Extract a friendly function name, preferring unmangled if possible
        let func_name = tcx.symbol_name(*instance).name;

        // Try to demangle using the Rust symbol demangling crate if available.
        // Since in rustc we don't always bring in the rustc-demangle crate, but
        // the symbol_name should generally be readable for non-generic items.
        // Otherwise, fallback to `def_path_str` (should give a crate-relative path).
        let friendly_name = if func_name.starts_with("_R") {
            // Looks like a Rust-mangled symbol. Try to show a better name.
            tcx.def_path_str(instance.def_id())
        } else {
            func_name.to_string()
        };

        eprintln!(
            "[DEBUG] TritonCodegen: function name: {} (raw symbol: {})",
            friendly_name, func_name
        );

        // Arguments
        let arg_types: Vec<_> =
            fn_sig.inputs().iter().map(|ty| self.type_mapper.map_type(&tcx, ty)).collect();

        // Result type
        let ret_types = self.type_mapper.map_type(&tcx, &fn_sig.output()).to_result().ok();
        let ret_types = ret_types.as_slice();

        // DEBUG output: print argument and result types
        eprintln!("[DEBUG] TritonCodegen: instance function signature (argument types):");
        for (i, arg_ty) in arg_types.iter().enumerate() {
            eprintln!("    arg[{}]: {}", i, arg_ty);
        }
        eprintln!(
            "[DEBUG] TritonCodegen: instance function signature (return type): {:?}",
            ret_types
        );

        let func_op = create_tt_func_with_divisibility(
            self.module.context(),
            Location::unknown(self.module.context()),
            func_name,
            &arg_types,
            ret_types,
            16,
        );

        self.module.llmod().body().append_operation(func_op.into());

        Ok(())
    }
}

impl<'a> Codegen for TritonCodegen<'a> {
    fn codegen<'tcx>(&mut self, tcx: TyCtxt<'tcx>, item: &MonoItem<'tcx>) -> Result<(), MlirError> {
        match item {
            MonoItem::Fn(instance) => {
                let fn_ty = instance.ty(tcx, TypingEnv::fully_monomorphized());
                let is_fn_ty = matches!(
                    fn_ty.kind(),
                    rustc_middle::ty::TyKind::FnDef(..) | rustc_middle::ty::TyKind::FnPtr(_, _)
                );

                if !is_fn_ty {
                    todo!(
                        "[DEBUG] TritonCodegen: instance.ty(tcx) is not a function type: {:?}",
                        fn_ty
                    );
                }

                self.codegen_function(tcx, fn_ty, instance)
            }
            MonoItem::Static(_def_id) => {
                // TODO: Implement Triton codegen for statics
                todo!()
            }
            MonoItem::GlobalAsm(_item_id) => {
                // TODO: Implement Triton codegen for global asm
                todo!()
            }
        }
    }
}
