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

#ifndef RVV_OPS
#define RVV_OPS

include "RVVDialect.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class RVV_Op<string mnemonic, list<Trait> traits = []> :
    Op<RVV_Dialect, mnemonic, traits>;

def RVV_VSetVLIOp : RVV_Op<"vsetvli", [Pure]> {
  let summary = "Negotiates the active vector length (RVV `vsetvli`)";
  let description = [{
    Models RVV's `vsetvli` instruction: given a requested application
    vector length (AVL), a selected element width (SEW), and register
    grouping (LMUL), returns the vector length the hardware actually
    grants for subsequent vector ops. This dynamic negotiation is RVV's
    defining vector-length-agnostic (VLA) mechanism -- a loop written
    against the returned `vl` runs correctly on any RVV implementation's
    native vector width, without recompiling per-width. It has no
    equivalent in MLIR's builtin `vector` dialect, which assumes a
    statically known (possibly scalable) length instead.

    ```mlir
    %vl = rvv.vsetvli %avl, sew = 32, lmul = 1 : index
    ```
  }];

  let arguments = (ins Index:$avl, I32Attr:$sew, I32Attr:$lmul);
  let results = (outs Index:$vl);

  let assemblyFormat =
      "$avl `,` `sew` `=` $sew `,` `lmul` `=` $lmul attr-dict `:` type($vl)";
}

def RVV_VAddOp : RVV_Op<"vadd", [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Vector-vector add over the active vector length (RVV `vadd.vv`)";
  let description = [{
    Models RVV's `vadd.vv`: an elementwise add of `lhs` and `rhs` over `vl`
    (as returned by a preceding `rvv.vsetvli`) active elements; elements at
    or beyond `vl` are left unspecified. Unlike `arith.addf`/`arith.addi`
    over MLIR's builtin `vector` type, the active-element count here is a
    dynamic SSA value rather than part of the operand type, reflecting
    RVV's VLA model -- the same operand types can run with a different
    `vl` on each iteration of a stripmined loop.

    ```mlir
    %r = rvv.vadd %a, %b, %vl : vector<[4]xf32>
    ```
  }];

  let arguments = (ins AnyVectorOfAnyRank:$lhs, AnyVectorOfAnyRank:$rhs, Index:$vl);
  let results = (outs AnyVectorOfAnyRank:$result);

  let assemblyFormat = "$lhs `,` $rhs `,` $vl attr-dict `:` type($result)";
}

#endif // RVV_OPS
