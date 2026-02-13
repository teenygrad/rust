//
// Copyright (c) 2025, The TeenyGrad Contributors
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

use std::path::{Path, PathBuf};

use cmake::Config;
use serde::Deserialize;

use crate::llvm::Llvm;
use crate::{create_dir, read_toml};

#[derive(Debug, Deserialize)]
pub struct TritonConfig {
    pub build_type: String,
    pub backends: String,
    pub python_module: String,
    pub proton: String,
}

#[derive(Debug)]
pub struct Triton {
    pub source_dir: PathBuf,
    pub out_dir: PathBuf,
    pub install_dir: PathBuf,
}

impl Triton {
    pub fn new(project_dir: &Path, target_dir: &Path) -> Self {
        let source_dir = project_dir.join("../../src/triton");
        let out_dir = target_dir.join("build/triton-build");
        let install_dir = target_dir.join("install");

        Self { source_dir, out_dir, install_dir }
    }

    pub fn source_dir(&self) -> PathBuf {
        self.source_dir.clone()
    }

    pub fn build_dir(&self) -> PathBuf {
        self.out_dir.clone()
    }

    pub fn include_dirs(&self) -> Vec<PathBuf> {
        vec![
            self.source_dir.join("include"),
            self.source_dir.join("third_party"),
            self.out_dir.join("build/include"),
            self.out_dir.join("build/third_party"),
        ]
    }

    pub fn link_dir(&self) -> PathBuf {
        self.out_dir.join("build")
    }

    pub fn link_libs(&self) -> Vec<String> {
        vec!["triton".to_string()]
    }
}

pub fn build_triton(project_dir: &Path, target_dir: &Path, llvm: &Llvm) -> Triton {
    let config: TritonConfig = read_toml(&project_dir.join("triton.toml"));
    let triton = Triton::new(project_dir, target_dir);

    create_dir(&triton.out_dir);
    create_dir(&triton.install_dir);

    let llvm_build_dir = llvm.out_dir.join("build");

    Config::new(&triton.source_dir)
        .generator("Ninja")
        .env("LLVM_BUILD_DIR", llvm_build_dir.clone())
        .env("LLVM_INCLUDE_DIRS", llvm_build_dir.join("include"))
        .env("LLVM_LIBRARY_DIR", llvm_build_dir.join("lib"))
        .define("LLD_DIR", llvm_build_dir.join("lib/cmake/lld"))
        .define("LLVM_SYSPATH", llvm_build_dir.clone())
        .define("TRITON_BUILD_PYTHON_MODULE", config.python_module)
        .define("TRITON_BUILD_PROTON", config.proton)
        .define("TRITON_CODEGEN_BACKENDS", config.backends)
        .define("TRITON_WHEEL_DIR", Path::new("/tmp"))
        .define("CMAKE_BUILD_TYPE", config.build_type)
        .define("CMAKE_INSTALL_PREFIX", &triton.install_dir)
        .define("CMAKE_INCLUDE_PATH", triton.source_dir.join("third_party"))
        .out_dir(&triton.out_dir)
        .build();

    triton
}
