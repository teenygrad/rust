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
    pub enable_projects: String,
}

#[derive(Debug)]
pub struct Llvm {
    pub build_dir: PathBuf,
    pub install_dir: PathBuf,
    pub llvm_config: PathBuf,
}

pub fn build_llvm(project_dir: &Path, target_dir: &Path) -> Llvm {
    let config: LlvmConfig = read_toml(&project_dir.join("llvm.toml"));
    let source_dir = project_dir.join("../../src/llvm-project/llvm");
    let out_dir = target_dir.join("build/llvm-build");
    let install_dir = target_dir.join("install");

    create_dir(&out_dir);
    create_dir(&install_dir);

    Config::new(&source_dir)
        .generator("Ninja")
        .define("LLVM_ENABLE_PROJECTS", config.enable_projects)
        .define("CMAKE_BUILD_TYPE", config.build_type)
        .define("CMAKE_INSTALL_PREFIX", &install_dir)
        .out_dir(&out_dir)
        .build();

    Llvm {
        build_dir: out_dir.join("build"),
        llvm_config: install_dir.join("bin/llvm-config"),
        install_dir,
    }
}
