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

//! teenyc-6mv: shared-memory (`ttg.local_alloc`/`local_store`/`local_load`)
//! Gluon-pipeline reproduction / regression test.
//!
//! # What this test is for
//!
//! The reverted `feat/ttg-shared-memory-primitives` exploration reported that a
//! hand-built shared-memory round-trip *segfaults* when driven through the
//! `Language::GLUON` pipeline (`CudaBackend::gluonToTTGIR`), with no debugger
//! attached to diagnose it. This test makes that crash a first-class,
//! reproducible, debuggable artifact instead of a one-off note.
//!
//! It drives the already-built standalone `triton-opt` tool (from the vendored
//! Triton build under `target/build/triton-build/build/bin`) over two committed
//! MLIR fixtures, so it needs **no** rebuild of the C++ `mlir-wrapper` and gives
//! a tight edit/run/`gdb` loop:
//!
//!   * `tests/data/shared_memory/naive_shared_memory.mlir` — the naive
//!     construction (plain, un-encoded tensors; no module `ttg.*` attributes).
//!     Running a Gluon encoding pass over it **crashes**.
//!   * `tests/data/shared_memory/golden_shared_memory.mlir` — the corrected
//!     construction (distributed blocked encoding on every tensor; required
//!     module attributes present). It survives the whole pipeline.
//!
//! # Root cause (found via `gdb` on the naive fixture)
//!
//! `mlir::triton::gluon::inferLayout` (`InferLayoutUtils.cpp`) calls a
//! per-`Type` predicate on the function's argument/result types. Both predicates
//!
//! ```text
//! // InferCoalescedEncodings.cpp (pre-fix)
//! bool isCoalescedEncodingTensorType(Type ty) {
//!   auto tensorTy = dyn_cast<RankedTensorType>(ty);
//!   return tensorTy && isa<gluon::CoalescedEncodingAttr>(tensorTy.getEncoding());
//! }
//! // ResolveAutoEncodings.cpp: same shape with AutoEncodingAttr
//! ```
//!
//! called `isa<...>(tensorTy.getEncoding())`. For a plain `tensor<128xi32>` the
//! encoding is a **null `Attribute`**, and `isa<>` on a null attribute
//! dereferences null -> SIGSEGV. Gluon's own Python frontend never emits
//! null-encoding tensors (it attaches a distributed layout to everything and
//! sets `ttg.num-warps` / `ttg.num-ctas` / `ttg.threads-per-warp` / `ttg.target`
//! on the module up front), so this contract was implicit and undocumented.
//!
//! # The fix (teenyc-6mv)
//!
//! Two complementary changes in the vendored Triton (`src/triton`):
//!
//!   * **Fix #1 — null guards.** Both predicates above now also check
//!     `tensorTy.getEncoding()` before the `isa<>`, so an unencoded tensor is
//!     treated as "not a coalesced/auto encoding" instead of crashing. The
//!     naive fixture that used to SIGSEGV now passes cleanly (see
//!     `naive_gluon_shared_memory_no_longer_segfaults`).
//!   * **Fix #2 — TTGIR-level staging.** A new `tritongpu-stage-shared-memory`
//!     pass lets a "mixed" kernel (plain Triton ops + a shared-memory staging
//!     step) route through the ordinary `convert-triton-to-tritongpu` pipeline:
//!     the value to stage is marked with a discardable `ttg.stage_shared`
//!     attribute, the conversion assigns it a real `#ttg.blocked` encoding, and
//!     the new pass then rewrites it into `local_alloc` / `local_store` /
//!     `local_load` reusing that encoding (see
//!     `mixed_marked_shared_memory_stages_through_shared_memory`).
//!
//! If `triton-opt` cannot be located (a checkout without the vendored Triton
//! build), the tests print a skip notice and pass, rather than failing.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Locate the standalone `triton-opt` built alongside the MLIR wrapper.
///
/// Honors a `TRITON_OPT` override, otherwise walks up from this crate looking
/// for `target/build/triton-build/build/bin/triton-opt`. Returns `None` when it
/// can't be found so callers can skip gracefully.
fn find_triton_opt() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("TRITON_OPT") {
        let path = PathBuf::from(explicit);
        if path.is_file() {
            return Some(path);
        }
    }

    const REL: &str = "target/build/triton-build/build/bin/triton-opt";
    let mut dir: Option<&Path> = Some(Path::new(env!("CARGO_MANIFEST_DIR")));
    while let Some(d) = dir {
        let candidate = d.join(REL);
        if candidate.is_file() {
            return Some(candidate);
        }
        dir = d.parent();
    }
    None
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/shared_memory").join(name)
}

/// The Gluon TTGIR pipeline, mirroring `CudaBackend::gluonToTTGIR` +
/// `make_llir`'s `allocate-shared-memory-nv`. This is the sequence that a
/// hand-built `Language::GLUON` module actually goes through.
const GLUON_PIPELINE: &[&str] = &[
    "--gluon-inline",
    "--gluon-infer-coalesced-encodings",
    "--gluon-resolve-auto-encodings",
    "--gluon-canonicalize",
    "--sccp",
    "--gluon-canonicalize",
    "--tritongpu-combine-tensor-select-and-if",
    "--allocate-shared-memory-nv",
];

fn run_triton_opt(opt: &Path, input: &Path, passes: &[&str]) -> Output {
    Command::new(opt)
        .arg(input)
        .args(passes)
        .args(["-o", "/dev/null"])
        .output()
        .expect("failed to spawn triton-opt")
}

/// Fix #1 regression: the naive, un-encoded hand-built module used to SIGSEGV
/// when a single Gluon encoding-inference pass ran over it (null-encoding deref
/// in `isCoalescedEncodingTensorType`). With the null guards in place it must
/// now run to completion WITHOUT any crash signal — the pass simply treats an
/// unencoded tensor as "not a coalesced encoding" and leaves it alone.
///
/// If this ever regresses to a crash again, this assertion fails loudly at the
/// `signal.is_none()` check.
#[test]
fn naive_gluon_shared_memory_no_longer_segfaults() {
    let Some(opt) = find_triton_opt() else {
        eprintln!("skipping naive_gluon_shared_memory_no_longer_segfaults: triton-opt not found");
        return;
    };

    let output = run_triton_opt(
        &opt,
        &fixture("naive_shared_memory.mlir"),
        &["--gluon-infer-coalesced-encodings"],
    );

    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        let signal = output.status.signal();
        assert!(
            signal.is_none(),
            "the null-encoding guard (teenyc-6mv fix #1) must keep the naive \
             Gluon shared-memory module from crashing, but triton-opt was killed \
             by signal {:?} (11 = SIGSEGV, the original bug).\nstderr:\n{}",
            signal,
            String::from_utf8_lossy(&output.stderr),
        );
        assert!(
            output.status.success(),
            "expected the guarded pass to exit cleanly on the naive fixture, got \
             {:?}.\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    #[cfg(not(unix))]
    {
        assert!(
            output.status.success(),
            "expected the guarded pass to exit cleanly on the naive fixture"
        );
    }
}

/// The corrected module (distributed encodings + required module attributes)
/// must survive the entire Gluon pipeline through shared-memory allocation, and
/// the three shared-memory ops plus the `ttg.shared` allocation annotation must
/// be present in the result. This pins the construction contract a hand-built
/// shared-memory lowering has to satisfy.
#[test]
fn golden_gluon_shared_memory_survives_pipeline() {
    let Some(opt) = find_triton_opt() else {
        eprintln!("skipping golden_gluon_shared_memory_survives_pipeline: triton-opt not found");
        return;
    };

    // Emit real IR (not /dev/null) so we can assert on the lowered form.
    let output = Command::new(&opt)
        .arg(fixture("golden_shared_memory.mlir"))
        .args(GLUON_PIPELINE)
        .output()
        .expect("failed to spawn triton-opt");

    assert!(
        output.status.success(),
        "expected the golden Gluon shared-memory module to lower cleanly, but \
         triton-opt failed ({:?}).\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );

    let ir = String::from_utf8_lossy(&output.stdout);
    for needle in ["ttg.local_alloc", "ttg.local_store", "ttg.local_load", "ttg.shared ="] {
        assert!(
            ir.contains(needle),
            "expected `{needle}` in the lowered Gluon shared-memory IR, got:\n{ir}"
        );
    }
}

/// Fix #2 prototype: a "mixed" kernel (plain register-layout Triton ops plus a
/// value marked `ttg.stage_shared`) must route through the ordinary
/// `convert-triton-to-tritongpu` pipeline and then get its marked value staged
/// through shared memory by the new `tritongpu-stage-shared-memory` pass —
/// reusing the `#ttg.blocked` encoding the conversion assigned, so no null
/// encoding and no unresolved encoded<->unencoded materialization.
///
/// This asserts the intermediate TTGIR (the three shared-memory ops appear and
/// the marker is consumed) AND that the staged module lowers all the way to LLVM
/// with a real shared-memory write + barrier and no leftover casts.
#[test]
fn mixed_marked_shared_memory_stages_through_shared_memory() {
    let Some(opt) = find_triton_opt() else {
        eprintln!(
            "skipping mixed_marked_shared_memory_stages_through_shared_memory: triton-opt not found"
        );
        return;
    };

    let convert = "--convert-triton-to-tritongpu=target=cuda:90 num-warps=4 \
                   threads-per-warp=32 num-ctas=1";

    // Stage 1: convert (encode) + stage-shared-memory -> TTGIR with the ops.
    let ttgir_out = Command::new(&opt)
        .arg(fixture("mixed_marked_shared_memory.mlir"))
        .args([convert, "--tritongpu-stage-shared-memory"])
        .output()
        .expect("failed to spawn triton-opt");
    assert!(
        ttgir_out.status.success(),
        "convert + stage-shared-memory must succeed on the mixed kernel, got \
         {:?}.\nstderr:\n{}",
        ttgir_out.status,
        String::from_utf8_lossy(&ttgir_out.stderr),
    );
    let ttgir = String::from_utf8_lossy(&ttgir_out.stdout);
    for needle in ["ttg.local_alloc", "ttg.local_store", "ttg.local_load"] {
        assert!(
            ttgir.contains(needle),
            "expected `{needle}` after staging the marked mixed kernel, got:\n{ttgir}"
        );
    }
    assert!(
        !ttgir.contains("stage_shared"),
        "the `ttg.stage_shared` marker must be consumed by the pass, got:\n{ttgir}"
    );

    // Stage 2: lower the staged TTGIR all the way to LLVM and confirm a real
    // shared-memory write + sync survive, with no unresolved casts left.
    let llvm_out = Command::new(&opt)
        .arg(fixture("mixed_marked_shared_memory.mlir"))
        .args([
            convert,
            "--tritongpu-stage-shared-memory",
            "--allocate-shared-memory-nv",
            "--convert-triton-gpu-to-llvm",
            "--reconcile-unrealized-casts",
        ])
        .output()
        .expect("failed to spawn triton-opt");
    assert!(
        llvm_out.status.success(),
        "the staged mixed kernel must lower to LLVM cleanly, got {:?}.\nstderr:\n{}",
        llvm_out.status,
        String::from_utf8_lossy(&llvm_out.stderr),
    );
    let llvm = String::from_utf8_lossy(&llvm_out.stdout);
    assert!(
        llvm.contains("st.shared") || llvm.contains("ptr<3>"),
        "expected a shared-memory write (st.shared / addrspace(3)) in the lowered \
         mixed kernel, got:\n{llvm}"
    );
    assert!(
        llvm.contains("barrier"),
        "expected a barrier synchronizing the shared-memory round-trip, got:\n{llvm}"
    );
    assert!(
        !llvm.contains("unrealized_conversion_cast"),
        "expected no leftover unrealized_conversion_cast after reconciliation, got:\n{llvm}"
    );
}
