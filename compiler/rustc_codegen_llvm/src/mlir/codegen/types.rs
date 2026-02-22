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

use melior::Context;
use melior::ir::Type;
use melior::ir::r#type::IntegerType;
use rustc_middle::ty::{
    AdtDef, AliasTy, AliasTyKind, GenericArg, GenericArgKind, Ty, TyCtxt, TyKind, TypingEnv,
};
use rustc_mlir::triton::{create_triton_pointer, create_triton_ranked_tensor};

pub struct TypeMapper<'a> {
    context: &'a Context,
}

impl<'a> TypeMapper<'a> {
    pub fn new(context: &'a Context) -> Self {
        Self { context }
    }

    pub fn map_type<'tcx>(&self, tcx: &TyCtxt<'tcx>, ty: &Ty<'tcx>) -> Type<'a> {
        match ty.kind() {
            TyKind::Int(_bits) => todo!("Int: {:?}", _bits),
            TyKind::Uint(_uint_ty) => todo!("Uint: {:?}", _uint_ty),
            TyKind::Float(_bits) => todo!("Float: {:?}", _bits),
            TyKind::Bool => todo!(),
            TyKind::Array(_elem_ty, _len) => todo!("Array: {:?} {:?}", _elem_ty, _len),
            TyKind::Char => todo!("Char"),
            TyKind::Adt(def, args) => self.map_adt_ty(tcx, def, args.as_slice()),
            TyKind::Foreign(_id) => todo!("Foreign: {:?}", _id),
            TyKind::Str => todo!("Str"),
            TyKind::Pat(_ty, _pat) => todo!("Pat: {:?} {:?}", _ty, _pat),
            TyKind::Slice(_ty) => todo!("Slice: {:?}", _ty),
            TyKind::RawPtr(_ty, _mutability) => todo!("RawPtr: {:?} {:?}", _ty, _mutability),
            TyKind::Ref(_region, _ty, _mutability) => {
                todo!("Ref: {:?} {:?} {:?}", _region, _ty, _mutability)
            }
            TyKind::FnDef(_def, _args) => todo!("FnDef: {:?} {:?}", _def, _args),
            TyKind::FnPtr(_binder, _fn_header) => todo!("FnPtr: {:?} {:?}", _binder, _fn_header),
            TyKind::UnsafeBinder(_unsafe_binder_inner) => {
                todo!("UnsafeBinder: {:?}", _unsafe_binder_inner)
            }
            TyKind::Dynamic(_existential_predicates, _region) => {
                todo!("Dynamic: {:?} {:?}", _existential_predicates, _region)
            }
            TyKind::Closure(_def, _args) => todo!("Closure: {:?} {:?}", _def, _args),
            TyKind::CoroutineClosure(_def, _args) => {
                todo!("CoroutineClosure: {:?} {:?}", _def, _args)
            }
            TyKind::Coroutine(_def, _args) => todo!("Coroutine: {:?} {:?}", _def, _args),
            TyKind::CoroutineWitness(_def, _args) => {
                todo!("CoroutineWitness: {:?} {:?}", _def, _args)
            }
            TyKind::Never => todo!("Never"),
            TyKind::Tuple(_tys) => todo!("Tuple: {:?}", _tys),
            TyKind::Alias(alias_ty_kind, alias_ty) => {
                self.map_alias_ty(ty, tcx, alias_ty_kind, alias_ty)
            }
            TyKind::Param(_param_ty) => todo!("Param: {:?}", _param_ty),
            TyKind::Bound(bound_var_index_kind, _bound_ty) => {
                todo!("Bound: {:?} {:?}", bound_var_index_kind, _bound_ty)
            }
            TyKind::Placeholder(_placeholder_ty) => todo!("Placeholder: {:?}", _placeholder_ty),
            TyKind::Infer(_infer_ty) => todo!("Infer: {:?}", _infer_ty),
            TyKind::Error(_error_guaranteed) => todo!("Error: {:?}", _error_guaranteed),
        }
    }

    fn map_adt_ty<'tcx>(
        &self,
        tcx: &TyCtxt<'tcx>,
        def: &AdtDef,
        args: &[GenericArg<'tcx>],
    ) -> Type<'a> {
        let name = tcx.def_path_str(def.did());

        if name == "triton::llvm::triton::tensor::Tensor" {
            debug_assert_eq!(args.len(), 1, "Tensor should have 1 argument");

            let arg_ty = args[0].expect_ty();
            let arg_type = self.map_type(tcx, &arg_ty);
            return self.create_tensor_type(arg_type);
        } else if name == "triton::llvm::triton::pointer::Pointer" {
            debug_assert_eq!(args.len(), 1, "Pointer should have 1 argument");
            let arg_ty = args[0].expect_ty();
            let arg_type = self.map_type(tcx, &arg_ty);
            return self.create_pointer_type(arg_type);
        } else if name == "triton::llvm::triton::num::F32" {
            return self.create_f32_type();
        } else if name == "triton::llvm::triton::types::Bool" {
            return self.create_bool_type();
        }

        todo!("map_adt_ty: name:{:?} {:?} {:?}", name, def, args);
    }

    fn map_alias_ty<'tcx>(
        &self,
        ty: &Ty<'tcx>,
        tcx: &TyCtxt<'tcx>,
        _alias_ty_kind: &AliasTyKind,
        alias_ty: &AliasTy<'tcx>,
    ) -> Type<'a> {
        let typing_env = TypingEnv::post_analysis(*tcx, alias_ty.def_id);
        let normalized = tcx.normalize_erasing_regions(typing_env, *ty);
        self.map_type(tcx, &normalized)
    }

    fn map_projection_alias_ty<'tcx>(
        &self,
        _ty: &Ty<'tcx>,
        tcx: &TyCtxt<'tcx>,
        alias_ty: &AliasTy<'tcx>,
    ) -> Type<'a> {
        // Get the definition from the definition id of alias_ty
        let def_id = alias_ty.def_id;

        // Use the TyCtxt if available to fetch definition, otherwise just return a placeholder
        // For now, just print the def_id for debugging
        eprintln!("[DEBUG] map_projection_alias_ty: def_id={:?}", def_id);

        for ty in alias_ty.args.iter() {
            match ty.kind() {
                GenericArgKind::Lifetime(_region) => eprintln!("Lifetime: {:?}", _region),
                GenericArgKind::Type(ty) => {
                    eprintln!("Ty: {:?}", ty);
                    self.map_type(tcx, &ty);
                }
                GenericArgKind::Const(_const) => eprintln!("Const: {:?}", _const),
            }
        }

        todo!()
    }

    fn create_tensor_type(&self, arg_type: Type<'a>) -> Type<'a> {
        create_triton_ranked_tensor(&arg_type)
    }

    fn create_pointer_type(&self, arg_type: Type<'a>) -> Type<'a> {
        create_triton_pointer(&arg_type)
    }

    fn create_f32_type(&self) -> Type<'a> {
        Type::float32(self.context)
    }

    fn create_bool_type(&self) -> Type<'a> {
        // bools are 1-bit integers
        IntegerType::new(self.context, 1).into()
    }
}
