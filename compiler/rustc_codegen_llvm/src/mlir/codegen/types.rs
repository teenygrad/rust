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
use melior::ir::r#type::{IntegerType, TupleType};
use rustc_ast::{FloatTy, IntTy, UintTy};
use rustc_middle::ty::{
    AdtDef, AliasTy, AliasTyKind, GenericArg, ParamTy, Ty, TyCtxt, TyKind, TypingEnv,
};
use rustc_mlir::triton::{create_triton_pointer, create_triton_ranked_tensor};

type AdtHandler = for<'a, 'tcx> fn(&TypeMapper<'a>, &TyCtxt<'tcx>, &[GenericArg<'tcx>]) -> Type<'a>;

use std::collections::HashMap;
use std::sync::OnceLock;

static ADT_HANDLER_MAP: OnceLock<HashMap<&'static str, AdtHandler>> = OnceLock::new();

pub struct TypeMapper<'a> {
    context: &'a Context,
}

impl<'a> TypeMapper<'a> {
    pub fn new(context: &'a Context) -> Self {
        Self { context }
    }

    pub fn map_type<'tcx>(&self, tcx: &TyCtxt<'tcx>, ty: &Ty<'tcx>) -> Type<'a> {
        match ty.kind() {
            TyKind::Int(int_ty) => self.create_int_type(tcx, int_ty),
            TyKind::Uint(uint_ty) => self.create_uint_type(tcx, uint_ty),
            TyKind::Float(float_ty) => self.create_float_type(tcx, float_ty),
            TyKind::Bool => self.create_bool_type(),
            TyKind::Array(_elem_ty, _len) => todo!("Array: {:?} {:?}", _elem_ty, _len),
            TyKind::Char => todo!("Char"),
            TyKind::Adt(def, args) => self.map_adt_ty(tcx, def, args.as_slice()),
            TyKind::Foreign(_id) => todo!("Foreign: {:?}", _id),
            TyKind::Str => todo!("Str"),
            TyKind::Pat(_ty, _pat) => todo!("Pat: {:?} {:?}", _ty, _pat),
            TyKind::Slice(_ty) => todo!("Slice: {:?}", _ty),
            TyKind::RawPtr(ty, _mutability) => self.create_raw_ptr_type(tcx, ty),
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
            TyKind::Tuple(tys) => self.create_tuple_type(tcx, tys.as_slice()),
            TyKind::Alias(alias_ty_kind, alias_ty) => {
                self.map_alias_ty(ty, tcx, alias_ty_kind, alias_ty)
            }
            TyKind::Param(_param_ty) => self.create_param_type(tcx, _param_ty),
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
        println!("map_adt_ty: name:{:?} {:?} {:?}", name, def, args);

        let handler = get_adt_handler(&name);
        handler(self, tcx, args)
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

    fn create_param_type<'tcx>(&self, _tcx: &TyCtxt<'tcx>, param_ty: &ParamTy) -> Type<'a> {
        todo!("Param: {:?}", param_ty);
    }

    fn create_int_type<'tcx>(&self, _tcx: &TyCtxt<'tcx>, int_ty: &IntTy) -> Type<'a> {
        let num_bits = match int_ty {
            IntTy::Isize => unimplemented!("isize is not supported as it is device-dependent"),
            IntTy::I8 => 8,
            IntTy::I16 => 16,
            IntTy::I32 => 32,
            IntTy::I64 => 64,
            IntTy::I128 => 128,
        };

        IntegerType::new(self.context, num_bits).into()
    }

    fn create_uint_type<'tcx>(&self, _tcx: &TyCtxt<'tcx>, uint_ty: &UintTy) -> Type<'a> {
        let num_bits = match uint_ty {
            UintTy::Usize => unimplemented!("usize is not supported as it is device-dependent"),
            UintTy::U8 => 8,
            UintTy::U16 => 16,
            UintTy::U32 => 32,
            UintTy::U64 => 64,
            UintTy::U128 => 128,
        };

        // for the moment we use the signless variant of the integer type
        IntegerType::new(self.context, num_bits).into()
    }

    fn create_float_type<'tcx>(&self, _tcx: &TyCtxt<'tcx>, float_ty: &FloatTy) -> Type<'a> {
        match float_ty {
            FloatTy::F16 => Type::float16(self.context),
            FloatTy::F32 => Type::float32(self.context),
            FloatTy::F64 => Type::float64(self.context),
            FloatTy::F128 => unimplemented!("f128 is not supported"),
        }
    }

    fn create_bool_type(&self) -> Type<'a> {
        // bools are 1-bit integers
        IntegerType::new(self.context, 1).into()
    }

    fn create_tuple_type<'tcx>(&self, tcx: &TyCtxt<'tcx>, tys: &[Ty<'tcx>]) -> Type<'a> {
        let types = tys.iter().map(|ty| self.map_type(tcx, ty)).collect::<Vec<_>>();
        TupleType::new(self.context, &types).into()
    }

    fn create_raw_ptr_type<'tcx>(&self, tcx: &TyCtxt<'tcx>, ty: &Ty<'tcx>) -> Type<'a> {
        let ty = self.map_type(tcx, ty);
        create_triton_pointer(ty)
    }
}

fn get_adt_handler(adt: &str) -> AdtHandler {
    let map = ADT_HANDLER_MAP.get_or_init(|| {
        let entries: Vec<(&'static str, AdtHandler)> = vec![
            ("triton::llvm::triton::tensor::Tensor", triton_tensor_handler),
            ("triton::llvm::triton::pointer::Pointer", triton_pointer_handler),
            ("triton::llvm::triton::num::I32", triton_i32_handler),
            ("triton::llvm::triton::num::F32", triton_f32_handler),
            ("triton::llvm::triton::types::Bool", triton_bool_handler),
            ("triton::ProgramAxis", triton_program_axis_handler),
        ];
        entries.into_iter().collect()
    });

    map.get(adt).copied().unwrap_or_else(|| panic!("Handler not found: {:?}", adt))
}

pub fn triton_tensor_handler<'a, 'tcx>(
    type_mapper: &TypeMapper<'a>,
    tcx: &TyCtxt<'tcx>,
    args: &[GenericArg<'tcx>],
) -> Type<'a> {
    debug_assert_eq!(args.len(), 1, "Tensor should have 1 argument");
    let arg_ty = args[0].expect_ty();
    let arg_type = type_mapper.map_type(tcx, &arg_ty);
    create_triton_ranked_tensor(arg_type)
}

pub fn triton_pointer_handler<'a, 'tcx>(
    type_mapper: &TypeMapper<'a>,
    tcx: &TyCtxt<'tcx>,
    args: &[GenericArg<'tcx>],
) -> Type<'a> {
    debug_assert_eq!(args.len(), 1, "Pointer should have 1 argument");
    let arg_ty = args[0].expect_ty();
    let arg_type = type_mapper.map_type(tcx, &arg_ty);
    create_triton_pointer(arg_type)
}

pub fn triton_i32_handler<'a, 'tcx>(
    type_mapper: &TypeMapper<'a>,
    _tcx: &TyCtxt<'tcx>,
    _args: &[GenericArg<'tcx>],
) -> Type<'a> {
    IntegerType::new(type_mapper.context, 32).into()
}

pub fn triton_f32_handler<'a, 'tcx>(
    type_mapper: &TypeMapper<'a>,
    _tcx: &TyCtxt<'tcx>,
    _args: &[GenericArg<'tcx>],
) -> Type<'a> {
    Type::float32(type_mapper.context)
}

pub fn triton_bool_handler<'a, 'tcx>(
    type_mapper: &TypeMapper<'a>,
    _tcx: &TyCtxt<'tcx>,
    _args: &[GenericArg<'tcx>],
) -> Type<'a> {
    type_mapper.create_bool_type()
}

pub fn triton_program_axis_handler<'a, 'tcx>(
    type_mapper: &TypeMapper<'a>,
    _tcx: &TyCtxt<'tcx>,
    _args: &[GenericArg<'tcx>],
) -> Type<'a> {
    IntegerType::new(type_mapper.context, 32).into()
}
