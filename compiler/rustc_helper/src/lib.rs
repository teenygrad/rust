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

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub mod llvm;
pub mod llvm_wrapper;
pub mod triton;

pub(crate) fn read_toml<T: for<'de> Deserialize<'de>>(toml_path: &PathBuf) -> T {
    let contents = fs::read_to_string(toml_path)
        .unwrap_or_else(|_| panic!("Failed to read {}", toml_path.display()));
    toml::from_str::<T>(&contents)
        .unwrap_or_else(|_| panic!("Failed to parse {}", toml_path.display()))
}

pub(crate) fn create_dir(dir: &Path) {
    fs::create_dir_all(dir).unwrap_or_else(|_| panic!("Failed to create {}", dir.display()));
}
