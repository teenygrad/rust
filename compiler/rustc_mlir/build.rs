use std::env;
use std::path::PathBuf;

use cargo_metadata::MetadataCommand;
use rustc_helper::llvm::Llvm;
use rustc_helper::triton::Triton;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Get project directory and target directory using cargo_metadata
    // Use rustc_llvm directory since that's where llvm.toml and triton.toml are located
    let metadata = MetadataCommand::new().exec().unwrap();
    let llvm_package = metadata.packages.iter().find(|p| p.name.as_str() == "rustc_llvm").unwrap();
    let project_dir: PathBuf = llvm_package.manifest_path.parent().unwrap().into();
    let target_dir: PathBuf = metadata.target_directory.into();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let wrapper_dir = manifest_dir.join("mlir-wrapper");

    // Get LLVM and Triton configs using helper functions
    let llvm = Llvm::new(&project_dir, &target_dir);
    let triton = Triton::new(&project_dir, &target_dir);

    // Use LLVM install_dir for cmake config paths
    let llvm_dir = &llvm.install_dir;
    let mlir_dir = llvm_dir.join("lib/cmake/mlir");

    // Configure cmake build
    let mut config = cmake::Config::new(&wrapper_dir);

    config
        .generator("Ninja")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("LLVM_DIR", llvm_dir.join("lib/cmake/llvm"))
        .define("MLIR_DIR", &mlir_dir)
        .define("TRITON_SOURCE_DIR", &triton.source_dir())
        .define("TRITON_BUILD_DIR", &triton.build_dir());

    let dst = config.build();

    // Link the built library
    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=static=mlir-wrapper");

    // Link MLIR libraries
    let mlir_lib_dir = llvm.build_dir.join("lib");
    println!("cargo:rustc-link-search=native={}", mlir_lib_dir.display());

    // Core MLIR libraries needed
    let mlir_libs = [
        "MLIRIR",
        "MLIRSupport",
        "MLIRParser",
        "MLIRPass",
        "MLIRTransforms",
        "MLIRAnalysis",
        "MLIRDialect",
    ];

    for lib in &mlir_libs {
        println!("cargo:rustc-link-lib={}", lib);
    }

    // Link LLVM support libraries
    println!("cargo:rustc-link-lib=LLVMSupport");
    println!("cargo:rustc-link-lib=LLVMCore");

    // Link Triton libraries
    println!("cargo:rustc-link-search=native={}", triton.link_dir().display());
    for lib in triton.link_libs() {
        println!("cargo:rustc-link-lib={}", lib);
    }
    // Additional Triton libraries
    println!("cargo:rustc-link-lib=TritonIR");
    println!("cargo:rustc-link-lib=TritonGPUIR");

    // Link C++ standard library
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Rerun if wrapper sources change
    println!("cargo:rerun-if-changed=mlir-wrapper/");
    // Note: llvm.toml and triton.toml are in rustc_llvm directory, tracked by rustc_llvm's build.rs

    // Generate bindings header path for use in lib.rs
    println!("cargo:include={}", wrapper_dir.display());
    println!("cargo:out_dir={}", out_dir.display());
}
