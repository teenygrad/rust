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
use crate::triton::Triton;
use crate::{create_dir, read_toml};

#[derive(Debug, Deserialize)]
pub struct WrapperConfig {
    pub build_type: String,
}

#[derive(Debug)]
pub struct Wrapper {
    pub source_dir: PathBuf,
    pub build_dir: PathBuf,
    pub install_dir: PathBuf,
}

impl Wrapper {
    pub fn new(source_dir: PathBuf, build_dir: PathBuf, install_dir: PathBuf) -> Self {
        Self { source_dir, build_dir, install_dir }
    }

    pub fn source_dir(&self) -> PathBuf {
        self.source_dir.clone()
    }

    pub fn build_dir(&self) -> PathBuf {
        self.build_dir.clone()
    }

    pub fn include_dirs(&self) -> Vec<PathBuf> {
        vec![
            self.source_dir.join("include"),
            self.source_dir.join("third_party"),
            self.build_dir.join("build/include"),
            self.build_dir.join("build/third_party"),
        ]
    }

    pub fn link_dir(&self) -> PathBuf {
        self.build_dir.join("build")
    }

    pub fn link_libs(&self) -> Vec<String> {
        vec!["triton".to_string()]
    }
}

pub fn build_wrapper(
    project_dir: &Path,
    target_dir: &Path,
    llvm: &Llvm,
    triton: &Triton,
) -> Wrapper {
    let config: WrapperConfig = read_toml(&project_dir.join("llvm-wrapper.toml"));
    let source_dir = project_dir.join("llvm-wrapper");
    let out_dir = target_dir.join("build/llvm-wrapper-build");
    let install_dir = target_dir.join("install");

    create_dir(&out_dir);
    create_dir(&install_dir);

    let llvm_build_dir = llvm.out_dir.join("build");
    let triton_build_dir = triton.out_dir.join("build");

    Config::new(&source_dir)
        .generator("Ninja")
        .env("LLVM_BUILD_DIR", llvm_build_dir.clone())
        .env("LLVM_INCLUDE_DIRS", llvm_build_dir.join("include"))
        .env("LLVM_LIBRARY_DIR", llvm_build_dir.join("lib"))
        .env("TRITON_SOURCE_DIR", &triton.source_dir)
        .env("TRITON_BUILD_DIR", triton_build_dir)
        .define("LLD_DIR", llvm_build_dir.join("lib/cmake/lld"))
        .define("LLVM_SYSPATH", llvm_build_dir.clone())
        .define("CMAKE_BUILD_TYPE", config.build_type)
        .define("CMAKE_INSTALL_PREFIX", &install_dir)
        .out_dir(&out_dir)
        .build();

    Wrapper::new(source_dir, out_dir, install_dir)
}
