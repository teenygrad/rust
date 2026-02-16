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

use crate::{create_dir, read_toml};

#[derive(Debug, Deserialize)]
pub struct LlvmConfig {
    pub build_type: String,
    pub enabled_projects: String,
    pub enabled_targets: String,
}

#[derive(Debug)]
pub struct Llvm {
    pub root_dir: PathBuf,
    pub source_dir: PathBuf,
    pub out_dir: PathBuf,
    pub llvm_config: PathBuf,
    pub install_dir: PathBuf,
}

impl Llvm {
    pub fn new(root_dir: &Path, target_dir: &Path) -> Self {
        let source_dir = root_dir.join("src/llvm-project/llvm");
        let out_dir = target_dir.join("build/llvm-build");
        let install_dir = target_dir.join("install");

        Llvm {
            root_dir: root_dir.to_path_buf(),
            source_dir,
            out_dir,
            llvm_config: install_dir.join("bin/llvm-config"),
            install_dir,
        }
    }
}

pub fn build_llvm(root_dir: &Path, project_dir: &Path, target_dir: &Path) -> Llvm {
    let llvm = Llvm::new(root_dir, target_dir);
    let config: LlvmConfig = read_toml(&project_dir.join("llvm.toml"));

    create_dir(&llvm.out_dir);
    create_dir(&llvm.install_dir);

    Config::new(&llvm.source_dir)
        .generator("Ninja")
        .define("LLVM_ENABLE_PROJECTS", config.enabled_projects)
        .define("LLVM_TARGETS_TO_BUILD", config.enabled_targets)
        .define("CMAKE_BUILD_TYPE", config.build_type)
        .define("CMAKE_INSTALL_PREFIX", &llvm.install_dir)
        .define("LLVM_INSTALL_UTILS", "ON")
        .out_dir(&llvm.out_dir)
        .build();

    llvm
}
