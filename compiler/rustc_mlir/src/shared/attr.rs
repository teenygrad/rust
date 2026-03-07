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
use melior::ir::r#type::IntegerType;
use melior::ir::{Attribute, Type};
use rustc_middle::ty::{ScalarInt, Ty, TyKind};

use crate::errors::Error;

pub fn create_scalar_attr<'ctx, 'tcx>(
    context: &'ctx Context,
    ty: Ty<'tcx>,
    scalar: ScalarInt,
) -> Result<(Attribute<'ctx>, Type<'ctx>), Error> {
    // AXM: TODO: fix this with the correct integer type for unsigned integers, we need to
    // to use the signless variant of the integer type

    // Only support integer types for now
    let attr_source = match ty.kind() {
        TyKind::Int(int_ty) => match int_ty {
            rustc_ast::IntTy::I8 => {
                (format!("{} : i8", scalar.to_i8()), IntegerType::new(context, 8).into())
            }
            rustc_ast::IntTy::I16 => {
                (format!("{} : i16", scalar.to_i16()), IntegerType::new(context, 16).into())
            }
            rustc_ast::IntTy::I32 => {
                (format!("{} : i32", scalar.to_i32()), IntegerType::new(context, 32).into())
            }
            rustc_ast::IntTy::I64 => {
                (format!("{} : i64", scalar.to_i64()), IntegerType::new(context, 64).into())
            }
            rustc_ast::IntTy::I128 => {
                (format!("{} : i128", scalar.to_i128()), IntegerType::new(context, 128).into())
            }
            rustc_ast::IntTy::Isize => {
                return Err(Error::InvalidType {
                    msg: "isize is device-dependent and not supported".to_string(),
                });
            }
        },
        TyKind::Uint(uint_ty) => match uint_ty {
            rustc_ast::UintTy::U32 => {
                (format!("{} : i64", scalar.to_u32()), IntegerType::new(context, 64).into())
            }
            _ => {
                return Err(Error::InvalidType {
                    msg: format!("Unsupported unsigned type for constant: {:?}", uint_ty),
                });
            }
        },
        _ => {
            return Err(Error::InvalidType {
                msg: format!("Unsupported type for constant: {:?}", ty),
            });
        }
    };

    Ok((
        Attribute::parse(context, &attr_source.0)
            .unwrap_or_else(|| panic!("failed to parse attribute: {}", attr_source.0)),
        attr_source.1,
    ))
}
