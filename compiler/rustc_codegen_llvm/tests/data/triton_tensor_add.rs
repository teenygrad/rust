/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#![allow(non_camel_case_types)]
#![allow(internal_features)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![feature(no_core)]
#![feature(intrinsics, lang_items)]
#![feature(arbitrary_self_types)]
#![feature(const_trait_impl)]
#![feature(auto_traits)]
#![no_core]
#![no_implicit_prelude]

#[lang = "freeze"]
pub unsafe auto trait Freeze {}

#[lang = "meta_sized"]
pub unsafe auto trait MetaSized {}

#[lang = "pointee_sized"]
pub unsafe auto trait PointeeSized {}

// Required language items for no_core
#[lang = "sized"]
pub trait Sized {}

#[lang = "clone"]
pub trait Clone {
    fn clone(&self) -> Self;
}

#[lang = "copy"]
pub trait Copy: Clone {}

impl<T> Copy for *const T {}
impl<T> Copy for *mut T {}
impl<T> Clone for *const T {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Clone for *mut T {
    fn clone(&self) -> Self {
        *self
    }
}

#[lang = "legacy_receiver"]
pub trait LegacyReceiver {}

#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // This function is a shim that the compiler fills in
    unsafe { drop_in_place(to_drop) }
}

// Required language items for arithmetic operations
#[lang = "panic_const_add_overflow"]
pub fn panic_const_add_overflow() -> ! {
    loop {}
}

#[lang = "panic_const_sub_overflow"]
pub fn panic_const_sub_overflow() -> ! {
    loop {}
}

#[lang = "panic_const_mul_overflow"]
pub fn panic_const_mul_overflow() -> ! {
    loop {}
}

#[lang = "panic_const_div_overflow"]
pub fn panic_const_div_overflow() -> ! {
    loop {}
}

#[lang = "panic_const_rem_overflow"]
pub fn panic_const_rem_overflow() -> ! {
    loop {}
}

#[lang = "panic_location"]
pub struct PanicLocation {
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
}

// Also implement Copy for other primitive types that might be needed
impl Copy for i32 {}
impl Clone for i32 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for f32 {}
impl Clone for f32 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for i8 {}
impl Clone for i8 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for i16 {}
impl Clone for i16 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for i64 {}
impl Clone for i64 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for u8 {}
impl Clone for u8 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for u16 {}
impl Clone for u16 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for u32 {}
impl Clone for u32 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for u64 {}
impl Clone for u64 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for bool {}
impl Clone for bool {
    fn clone(&self) -> Self {
        *self
    }
}

pub enum Option<T> {
    None,
    Some(T),
}

use Option::*;

pub const trait Into<T>: Sized {
    /// Converts this type into the (usually inferred) input type.
    fn into(self) -> T;
}

pub const trait From<T>: Sized {
    /// Converts to this type from the input type.
    fn from(value: T) -> Self;
}

impl<T, U> Into<U> for T
where
    U: From<T>,
{
    fn into(self) -> U {
        U::from(self)
    }
}

pub mod std {
    pub mod ops {
        // Arithmetic operation lang items
        #[lang = "mul"]
        pub trait Mul<RHS = Self> {
            type Output;
            fn mul(self, rhs: RHS) -> Self::Output;
        }

        impl Mul for i32 {
            type Output = i32;
            fn mul(self, rhs: i32) -> Self::Output {
                0
            }
        }

        impl Mul for i64 {
            type Output = i64;
            fn mul(self, rhs: i64) -> Self::Output {
                0
            }
        }

        #[lang = "add"]
        pub trait Add<RHS = Self> {
            type Output;
            fn add(self, rhs: RHS) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for i32 {
            type Output = i32;
            fn add(self, rhs: i32) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for u32 {
            type Output = u32;
            fn add(self, rhs: u32) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for u64 {
            type Output = u64;
            fn add(self, rhs: u64) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for i64 {
            type Output = i64;
            fn add(self, rhs: i64) -> Self::Output {
                0
            }
        }

        #[lang = "sub"]
        pub trait Sub<RHS = Self> {
            type Output;
            fn sub(self, rhs: RHS) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Sub for i32 {
            type Output = i32;
            fn sub(self, rhs: i32) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Sub for u32 {
            type Output = u32;
            fn sub(self, rhs: u32) -> Self::Output {
                0
            }
        }

        #[lang = "div"]
        pub trait Div<RHS = Self> {
            type Output;
            fn div(self, rhs: RHS) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Div for i32 {
            type Output = i32;
            fn div(self, rhs: i32) -> Self::Output {
                0
            }
        }

        #[lang = "rem"]
        pub trait Rem<RHS = Self> {
            type Output;
            fn rem(self, rhs: RHS) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Rem for i32 {
            type Output = i32;
            fn rem(self, rhs: i32) -> Self::Output {
                0
            }
        }
    }
}
pub mod triton {
    /*
     * Copyright (c) 2025 Teenygrad. All rights reserved.
     *
     * This program is free software: you can redistribute it and/or modify it
     * under the terms of the GNU General Public License as published by the
     * Free Software Foundation, either version 3 of the License, or (at your
     * option) any later version.
     *
     * This program is distributed in the hope that it will be useful, but
     * WITHOUT ANY WARRANTY; without even the implied warranty of
     * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
     * General Public License for more details.
     *
     * You should have received a copy of the GNU General Public License
     * along with this program. If not, see <https://www.gnu.org/licenses/>.
     */
    use std::ops::{Add, Mul};

    pub use types::*;

    use self::types::{self as ty};
    pub use super::*;

    #[repr(i32)]
    pub enum ProgramAxis {
        Axis0 = 0,
        Axis1 = 1,
        Axis2 = 2,
    }

    pub trait Triton
    where
        Self::I32: Mul<u32, Output = Self::I32>,
        Self::I32Tensor: Add<Self::I32, Output = Self::I32Tensor>,
        Self::I32Tensor: Comparison<Self::I32, BoolTensor = Self::BoolTensor>,
    {
        type Bool: ty::Bool;
        type I32: ty::I32;
        type I64: ty::I64;
        type BF16: ty::BF16;

        type BoolTensor: ty::BoolTensor<Bool = Self::Bool>;
        type I32Tensor: ty::I32Tensor<I32 = Self::I32>;
        type Tensor<D: ty::Dtype>: ty::Tensor<D> + Add<Self::Tensor<D>, Output = Self::Tensor<D>>;
        type Pointer<D: ty::Dtype>: ty::Pointer<D, I32 = Self::I32, I32Tensor = Self::I32Tensor>
            + AddOffsets<Self::I32, Self::I32Tensor, Output = Self::Tensor<Self::Pointer<D>>>;

        fn program_id(axis: ProgramAxis) -> Self::I32;

        fn num_programs(axis: ProgramAxis) -> Self::I32;

        fn arange(
            start: impl Into<Self::I32>,
            end: impl Into<Self::I32>,
            step: impl Into<Self::I32>,
        ) -> Self::I32Tensor;

        fn load<D: ty::Dtype>(
            ptr: Self::Tensor<Self::Pointer<D>>,
            mask: Self::BoolTensor,
        ) -> Self::Tensor<D>;

        fn store<D: ty::Dtype>(
            dest: Self::Tensor<Self::Pointer<D>>,
            src: Self::Tensor<D>,
            mask: Self::BoolTensor,
        );
    }
    pub mod types {
        /*
         * Copyright (c) 2025 Teenygrad. All rights reserved.
         *
         * This program is free software: you can redistribute it and/or modify it
         * under the terms of the GNU General Public License as published by the
         * Free Software Foundation, either version 3 of the License, or (at your
         * option) any later version.
         *
         * This program is distributed in the hope that it will be useful, but
         * WITHOUT ANY WARRANTY; without even the implied warranty of
         * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
         * General Public License for more details.
         *
         * You should have received a copy of the GNU General Public License
         * along with this program. If not, see <https://www.gnu.org/licenses/>.
         */
        use std::ops::{Add, Mul};

        pub use super::*;

        // Dtype Type
        pub trait Dtype: Copy + Clone {}

        pub trait Num: Dtype {}

        pub trait Float: Num {}
        pub trait Int: Num {}
        pub trait Bool: Dtype + Copy {}

        // Tensor
        pub trait RankedTensor<D: Dtype>: Copy + Clone {}

        // Floating-point types
        pub trait F8E4M3FN: Float {}
        pub trait F8E4M3FNUZ: Float {}
        pub trait F8E5M2: Float {}
        pub trait F8E5M2FNUZ: Float {}

        pub trait F16: Float {}
        pub trait BF16: Float {}
        pub trait F32: Float {}
        pub trait F64: Float {}

        // Supported integer types
        pub trait I1: Int {}

        pub trait I4: Int {}
        pub trait I8: Int {}
        pub trait I16: Int {}
        pub trait I32: Int + From<u32> + From<i32> + Mul<u32> {}

        pub trait I64: Int {}

        // Int Tensor
        pub trait Tensor<D: Dtype>: RankedTensor<D> {}

        pub trait BoolTensor: Tensor<Self::Bool> {
            type Bool: Bool;
        }

        pub trait Comparison<I: Num> {
            type BoolTensor: BoolTensor;

            fn lt(self, other: I) -> Self::BoolTensor;
        }
        pub trait I32Tensor: Tensor<Self::I32> + Add<Self::I32> + Comparison<Self::I32> {
            type I32: I32;
        }

        // Offsets trait for adding tensor offsets to pointers
        pub trait AddOffsets<I: Int, T: Tensor<I>> {
            type Output;

            fn add_offsets(self, offsets: T) -> Self::Output;
        }

        // Pointer Type
        pub trait Pointer<D: Dtype>:
            Sized + Copy + Clone + Dtype + AddOffsets<Self::I32, Self::I32Tensor> + Add<Self>
        {
            type I32: I32;
            type I32Tensor: I32Tensor<I32 = Self::I32>;
        }
    }
    pub mod llvm {
        pub use super::super::*;
        /*
         * Copyright (c) 2025 Teenygrad. All rights reserved.
         *
         * This program is free software: you can redistribute it and/or modify it
         * under the terms of the GNU General Public License as published by the
         * Free Software Foundation, either version 3 of the License, or (at your
         * option) any later version.
         *
         * This program is distributed in the hope that it will be useful, but
         * WITHOUT ANY WARRANTY; without even the implied warranty of
         * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
         * General Public License for more details.
         *
         * You should have received a copy of the GNU General Public License
         * along with this program. If not, see <https://www.gnu.org/licenses/>.
         */

        pub mod triton {
            pub use super::super::super::*;
            /*
             * Copyright (c) 2025 Teenygrad. All rights reserved.
             *
             * This program is free software: you can redistribute it and/or modify it
             * under the terms of the GNU General Public License as published by the
             * Free Software Foundation, either version 3 of the License, or (at your
             * option) any later version.
             *
             * This program is distributed in the hope that it will be useful, but
             * WITHOUT ANY WARRANTY; without even the implied warranty of
             * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
             * General Public License for more details.
             *
             * You should have received a copy of the GNU General Public License
             * along with this program. If not, see <https://www.gnu.org/licenses/>.
             */
            use super::super::Triton;
            use super::super::{ProgramAxis, types as ty};

            pub struct LlvmTriton {}

            impl Triton for LlvmTriton {
                type I32 = num::I32;
                type I64 = num::I64;
                type BF16 = num::BF16;

                type Bool = types::Bool;
                type BoolTensor = tensor::BoolTensor;
                type I32Tensor = tensor::I32Tensor;
                type Tensor<D: ty::Dtype> = tensor::Tensor<D>;
                type Pointer<D: ty::Dtype> = pointer::Pointer<D>;

                #[inline(never)]
                fn program_id(_axis: ProgramAxis) -> Self::I32 {
                    // dummy implementation not used in final output
                    0.into()
                }

                #[inline(never)]
                fn num_programs(_axis: ProgramAxis) -> Self::I32 {
                    // dummy implementation not used in final output
                    0.into()
                }

                #[inline(never)]
                fn arange(
                    _start: impl Into<Self::I32>,
                    _end: impl Into<Self::I32>,
                    _step: impl Into<Self::I32>,
                ) -> Self::I32Tensor {
                    loop {}
                }

                #[inline(never)]
                fn load<D: ty::Dtype>(
                    _ptr: Self::Tensor<Self::Pointer<D>>,
                    _mask: Self::BoolTensor,
                ) -> Self::Tensor<D> {
                    // dummy implementation not used in final output
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                fn store<D: ty::Dtype>(
                    _dest: Self::Tensor<Self::Pointer<D>>,
                    _src: Self::Tensor<D>,
                    _mask: Self::BoolTensor,
                ) {
                    // nop
                }
            }
            pub mod types {
                /*
                 * Copyright (c) 2025 Teenygrad. All rights reserved.
                 *
                 * This program is free software: you can redistribute it and/or modify it
                 * under the terms of the GNU General Public License as published by the
                 * Free Software Foundation, either version 3 of the License, or (at your
                 * option) any later version.
                 *
                 * This program is distributed in the hope that it will be useful, but
                 * WITHOUT ANY WARRANTY; without even the implied warranty of
                 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
                 * General Public License for more details.
                 *
                 * You should have received a copy of the GNU General Public License
                 * along with this program. If not, see <https://www.gnu.org/licenses/>.
                 */
                use super::super::super::types as ty;
                pub use super::super::super::*;

                /*--------------------------------- Bool ---------------------------------*/

                pub struct Bool(bool);
                impl Copy for Bool {}
                impl Clone for Bool {
                    #[inline(always)]
                    fn clone(&self) -> Self {
                        *self
                    }
                }

                impl ty::Dtype for Bool {}
                impl ty::Bool for Bool {}
            }
            pub mod pointer {
                /*
                 * Copyright (c) 2025 Teenygrad. All rights reserved.
                 *
                 * This program is free software: you can redistribute it and/or modify it
                 * under the terms of the GNU General Public License as published by the
                 * Free Software Foundation, either version 3 of the License, or (at your
                 * option) any later version.
                 *
                 * This program is distributed in the hope that it will be useful, but
                 * WITHOUT ANY WARRANTY; without even the implied warranty of
                 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
                 * General Public License for more details.
                 *
                 * You should have received a copy of the GNU General Public License
                 * along with this program. If not, see <https://www.gnu.org/licenses/>.
                 */
                use std::ops::Add;

                use super::super::super::types::{self as ty};
                pub use super::super::super::*;
                use super::num::I32;
                use crate::triton::llvm::triton::tensor::{I32Tensor, Tensor};

                pub struct Pointer<D: ty::Dtype>(pub *mut D);
                impl<D: ty::Dtype> Clone for Pointer<D> {
                    fn clone(&self) -> Self {
                        *self
                    }
                }
                impl<D: ty::Dtype> Copy for Pointer<D> {}

                impl<D: ty::Dtype> ty::Dtype for Pointer<D> {}

                impl<D: ty::Dtype> ty::Pointer<D> for Pointer<D> {
                    type I32 = I32;
                    type I32Tensor = I32Tensor;
                }

                // Implement AddOffsets for I64Tensor
                impl<D: ty::Dtype> ty::AddOffsets<I32, I32Tensor> for Pointer<D> {
                    type Output = Tensor<Self>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn add_offsets(self, _offsets: I32Tensor) -> Self::Output {
                        // dummy implementation not used in final output
                        Tensor(0 as *mut Self)
                    }
                }

                impl<D: ty::Dtype> Add<Pointer<D>> for Pointer<D> {
                    type Output = Self;

                    #[inline(never)]
                    fn add(self, _other: Pointer<D>) -> Self::Output {
                        // dummy implementation not used in final output
                        self
                    }
                }
            }
            pub mod num {
                /*
                 * Copyright (c) 2025 Teenygrad. All rights reserved.
                 *
                 * This program is free software: you can redistribute it and/or modify it
                 * under the terms of the GNU General Public License as published by the
                 * Free Software Foundation, either version 3 of the License, or (at your
                 * option) any later version.
                 *
                 * This program is distributed in the hope that it will be useful, but
                 * WITHOUT ANY WARRANTY; without even the implied warranty of
                 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
                 * General Public License for more details.
                 *
                 * You should have received a copy of the GNU General Public License
                 * along with this program. If not, see <https://www.gnu.org/licenses/>.
                 */
                use std::ops::Mul;

                use super::super::super::types as ty;
                pub use super::super::super::*;

                /*--------------------------------- I1 ---------------------------------*/

                pub struct I1(pub bool);
                impl Copy for I1 {}
                impl Clone for I1 {
                    #[inline(always)]
                    fn clone(&self) -> Self {
                        *self
                    }
                }

                impl ty::Dtype for I1 {}
                impl ty::Num for I1 {}
                impl ty::Int for I1 {}
                impl ty::I1 for I1 {}

                /*--------------------------------- I32 ---------------------------------*/

                pub struct I32(pub i32);
                impl Copy for I32 {}
                impl Clone for I32 {
                    #[inline(always)]
                    fn clone(&self) -> Self {
                        *self
                    }
                }

                impl ty::Dtype for I32 {}
                impl ty::Num for I32 {}
                impl ty::Int for I32 {}
                impl ty::I32 for I32 {}

                impl Mul<u32> for I32 {
                    type Output = I32;

                    #[inline(always)]
                    fn mul(self, rhs: u32) -> Self::Output {
                        I32(self.0 * rhs as i32)
                    }
                }

                impl From<u32> for I32 {
                    #[inline(always)]
                    fn from(value: u32) -> Self {
                        Self(value as i32)
                    }
                }

                impl From<i32> for I32 {
                    #[inline(always)]
                    fn from(value: i32) -> Self {
                        Self(value)
                    }
                }

                /*--------------------------------- I64 ---------------------------------*/

                pub struct I64(pub i64);
                impl Copy for I64 {}
                impl Clone for I64 {
                    #[inline(always)]
                    fn clone(&self) -> Self {
                        *self
                    }
                }

                impl ty::Dtype for I64 {}
                impl ty::Num for I64 {}
                impl ty::Int for I64 {}
                impl ty::I64 for I64 {}

                /*--------------------------------- F32 ---------------------------------*/

                pub struct F32(pub f32);
                impl ty::Dtype for F32 {}
                impl ty::Num for F32 {}
                impl ty::Float for F32 {}
                impl ty::F32 for F32 {}

                impl Copy for F32 {}
                impl Clone for F32 {
                    #[inline(always)]
                    fn clone(&self) -> Self {
                        *self
                    }
                }

                /*--------------------------------- BF16 ---------------------------------*/

                pub struct BF16;
                impl Copy for BF16 {}
                impl Clone for BF16 {
                    #[inline(always)]
                    fn clone(&self) -> Self {
                        *self
                    }
                }

                impl ty::Dtype for BF16 {}
                impl ty::Num for BF16 {}
                impl ty::Float for BF16 {}
                impl ty::BF16 for BF16 {}
            }
            pub mod tensor {
                /*
                 * Copyright (c) 2025 Teenygrad. All rights reserved.
                 *
                 * This program is free software: you can redistribute it and/or modify it
                 * under the terms of the GNU General Public License as published by the
                 * Free Software Foundation, either version 3 of the License, or (at your
                 * option) any later version.
                 *
                 * This program is distributed in the hope that it will be useful, but
                 * WITHOUT ANY WARRANTY; without even the implied warranty of
                 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
                 * General Public License for more details.
                 *
                 * You should have received a copy of the GNU General Public License
                 * along with this program. If not, see <https://www.gnu.org/licenses/>.
                 */
                use std::ops::Add;

                use super::super::super::types::{self as ty};
                pub use super::super::super::*;
                use super::num::{I32, I64};
                use super::types::Bool;

                /*--------------------------------- Tensor ---------------------------------*/

                pub struct Tensor<D: ty::Dtype>(pub *mut D);
                impl<D: ty::Dtype> Clone for Tensor<D> {
                    fn clone(&self) -> Self {
                        *self
                    }
                }
                impl<D: ty::Dtype> Copy for Tensor<D> {}

                impl<D: ty::Dtype> ty::Tensor<D> for Tensor<D> {}
                impl<D: ty::Dtype> ty::RankedTensor<D> for Tensor<D> {}

                // Element-wise addition for tensors
                impl<D: ty::Dtype> Add<Tensor<D>> for Tensor<D> {
                    type Output = Tensor<D>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn add(self, _rhs: Tensor<D>) -> Self::Output {
                        // dummy implementation not used in final output
                        Tensor(0 as *mut D)
                    }
                }

                pub type BoolTensor = Tensor<Bool>;
                impl ty::BoolTensor for BoolTensor {
                    type Bool = Bool;
                }

                pub type I32Tensor = Tensor<I32>;

                impl ty::I32Tensor for I32Tensor {
                    type I32 = I32;
                }

                impl ty::Comparison<I32> for I32Tensor {
                    type BoolTensor = BoolTensor;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn lt(self, _other: I32) -> Self::BoolTensor {
                        // dummy implementation not used in final output
                        Tensor(0 as *mut Bool)
                    }
                }

                // Blanket implementation for any type implementing I64, including <I32 as Mul<u32>>::Output
                impl<R: ty::I32> Add<R> for I32Tensor {
                    type Output = I32Tensor;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn add(self, _rhs: R) -> Self::Output {
                        // dummy implementation not used in final output
                        Tensor(0 as *mut I32)
                    }
                }
            }
        }
    }
}
pub use triton::*;

#[no_mangle]
pub extern "C" fn entry_point(n_elements: i32) {
    use triton::llvm::triton::num::*;
    use triton::llvm::triton::pointer::Pointer;

    let x_ptr = Pointer(0 as *mut _);
    let y_ptr = Pointer(0 as *mut _);
    let output_ptr = Pointer(0 as *mut _);
    let n_elements = I32(n_elements);

    tensor_add::<triton::llvm::triton::LlvmTriton, F32, 128>(x_ptr, y_ptr, output_ptr, n_elements);
}
pub extern "C" fn tensor_add<T: Triton, D: types::Dtype, const BLOCK_SIZE: u32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    n_elements: T::I32,
) {
    let pid = T::program_id(ProgramAxis::Axis0);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE, 1) + block_start;
    let mask = offsets.lt(n_elements);
    let x = T::load(x_ptr.add_offsets(offsets), mask);
    let y = T::load(y_ptr.add_offsets(offsets), mask);
    let output = x + y;
    T::store(output_ptr.add_offsets(offsets), output, mask);
}
