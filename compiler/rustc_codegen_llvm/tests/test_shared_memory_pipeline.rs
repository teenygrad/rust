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
//! TTGIR-staging pipeline test.
//!
//! # What this test is for
//!
//! A "mixed" kernel (plain register-layout Triton ops plus a value marked
//! `ttg.stage_shared`) must route through the ordinary
//! `convert-triton-to-tritongpu` pipeline and then get its marked value staged
//! through shared memory by the `tritongpu-stage-shared-memory` pass — reusing
//! the `#ttg.blocked` encoding the conversion assigned, so there's no null
//! encoding and no unresolved encoded<->unencoded materialization.
//!
//! It drives the already-built standalone `triton-opt` tool (from the vendored
//! Triton build under `target/build/triton-build/build/bin`) over a committed
//! MLIR fixture, so it needs **no** rebuild of the C++ `mlir-wrapper` and gives
//! a tight edit/run/`gdb` loop.
//!
//! If `triton-opt` cannot be located (a checkout without the vendored Triton
//! build), the test prints a skip notice and passes, rather than failing.

use std::path::{Path, PathBuf};
use std::process::Command;

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

/// A "mixed" kernel (plain register-layout Triton ops plus a
/// value marked `ttg.stage_shared`) must route through the ordinary
/// `convert-triton-to-tritongpu` pipeline and then get its marked value staged
/// through shared memory by the `tritongpu-stage-shared-memory` pass —
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
