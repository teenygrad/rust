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
//! // InferCoalescedEncodings.cpp
//! bool isCoalescedEncodingTensorType(Type ty) {
//!   auto tensorTy = dyn_cast<RankedTensorType>(ty);
//!   return tensorTy && isa<gluon::CoalescedEncodingAttr>(tensorTy.getEncoding());
//! }
//! // ResolveAutoEncodings.cpp: same shape with AutoEncodingAttr
//! ```
//!
//! call `isa<...>(tensorTy.getEncoding())`. For a plain `tensor<128xi32>` the
//! encoding is a **null `Attribute`**, and `isa<>` on a null attribute
//! dereferences null -> SIGSEGV. Gluon's own Python frontend never emits
//! null-encoding tensors (it attaches a distributed layout to everything and
//! sets `ttg.num-warps` / `ttg.num-ctas` / `ttg.threads-per-warp` / `ttg.target`
//! on the module up front), so this contract was implicit and undocumented. The
//! golden fixture reconstructs it by hand.
//!
//! # To debug the crash manually
//!
//! ```text
//! TRITON_OPT=target/build/triton-build/build/bin/triton-opt
//! gdb -q -ex run --args "$TRITON_OPT" \
//!     compiler/rustc_codegen_llvm/tests/data/shared_memory/naive_shared_memory.mlir \
//!     --gluon-infer-coalesced-encodings -o /dev/null
//! # bt: isCoalescedEncodingTensorType <- inferLayout <- runOnOperation
//! ```
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

/// The naive, un-encoded hand-built module must reproduce the crash: driving a
/// single Gluon encoding-inference pass over it terminates the process with a
/// signal (SIGSEGV in practice). This pins the bug so that a future fix flips
/// this assertion loudly instead of silently changing behavior.
#[test]
fn naive_gluon_shared_memory_segfaults() {
    let Some(opt) = find_triton_opt() else {
        eprintln!("skipping naive_gluon_shared_memory_segfaults: triton-opt not found");
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
            signal.is_some(),
            "expected the naive Gluon shared-memory module to CRASH \
             (null-encoding deref in isCoalescedEncodingTensorType), but \
             triton-opt exited normally with {:?}.\nIf the underlying \
             null-encoding guard was fixed upstream, update this test to assert \
             success instead.\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr),
        );
        // SIGSEGV (11) is the observed signature; accept any crash signal but
        // surface the actual one to make regressions in the failure *mode*
        // visible.
        assert_eq!(
            signal,
            Some(11),
            "expected SIGSEGV (11) from the null-encoding deref, got signal {:?}",
            signal,
        );
    }

    #[cfg(not(unix))]
    {
        assert!(
            !output.status.success(),
            "expected the naive Gluon shared-memory module to fail hard"
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
