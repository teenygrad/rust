//! Test harness for MLIR codegen backend.
//!
//! This module provides utilities and documentation for testing the MLIR codegen backend,
//! particularly useful for JIT compilation scenarios.
//!
//! ## Usage
//!
//! The test harness can be used via rustc command line:
//!
//! ```bash
//! # Enable MLIR backend with verbose logging
//! RUST_LOG=info rustc --codegen-backend=mlir -Z frontend=triton your_file.rs
//!
//! # Or use x.py for testing within the repo
//! ./x.py test tests/codegen/your_test.rs --stage 1 --keep-stage 1
//! ```
//!
//! ## Test Patterns
//!
//! The `test_patterns` module contains example Rust code snippets that exercise
//! different MIR constructs, useful for testing the visitor.

use std::path::PathBuf;

/// Configuration for the test harness.
#[derive(Debug, Clone)]
pub struct TestConfig {
    /// Enable verbose MIR logging
    pub verbose_mir: bool,
    /// Enable summary-only mode (no detailed MIR)
    pub summary_only: bool,
    /// Log to file instead of stdout
    pub log_file: Option<PathBuf>,
    /// Filter functions by name pattern
    pub function_filter: Option<String>,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            verbose_mir: true,
            summary_only: false,
            log_file: None,
            function_filter: None,
        }
    }
}

impl TestConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn verbose(mut self) -> Self {
        self.verbose_mir = true;
        self.summary_only = false;
        self
    }

    pub fn summary_only(mut self) -> Self {
        self.summary_only = true;
        self.verbose_mir = false;
        self
    }

    pub fn with_log_file(mut self, path: PathBuf) -> Self {
        self.log_file = Some(path);
        self
    }

    pub fn filter_function(mut self, pattern: &str) -> Self {
        self.function_filter = Some(pattern.to_string());
        self
    }
}

/// Example test patterns for common MIR structures.
///
/// These are example Rust code snippets that exercise different
/// MIR constructs. Useful for testing the visitor.
pub mod test_patterns {
    /// Simple arithmetic - tests basic assignments and binary ops
    pub const ARITHMETIC: &str = r#"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn multiply_add(a: i32, b: i32, c: i32) -> i32 {
    a * b + c
}
"#;

    /// Control flow - tests branches, switches, loops
    pub const CONTROL_FLOW: &str = r#"
pub fn if_else(x: i32) -> i32 {
    if x > 0 {
        x + 1
    } else {
        x - 1
    }
}

pub fn loop_sum(n: i32) -> i32 {
    let mut sum = 0;
    let mut i = 0;
    while i < n {
        sum += i;
        i += 1;
    }
    sum
}

pub fn match_arm(x: Option<i32>) -> i32 {
    match x {
        Some(v) => v,
        None => 0,
    }
}
"#;

    /// References and borrows - tests Ref, RawPtr, Deref
    pub const REFERENCES: &str = r#"
pub fn borrow(x: &i32) -> i32 {
    *x
}

pub fn mutable_borrow(x: &mut i32) {
    *x += 1;
}

pub fn raw_pointer(x: *const i32) -> i32 {
    unsafe { *x }
}
"#;

    /// Aggregates - tests struct, tuple, array construction
    pub const AGGREGATES: &str = r#"
pub struct Point {
    x: i32,
    y: i32,
}

pub fn make_point(x: i32, y: i32) -> Point {
    Point { x, y }
}

pub fn make_tuple(a: i32, b: i32) -> (i32, i32) {
    (a, b)
}

pub fn make_array() -> [i32; 3] {
    [1, 2, 3]
}
"#;

    /// Function calls - tests Call, TailCall terminators
    pub const FUNCTION_CALLS: &str = r#"
fn helper(x: i32) -> i32 {
    x * 2
}

pub fn caller(x: i32) -> i32 {
    helper(x) + 1
}

pub fn recursive_fib(n: i32) -> i32 {
    if n <= 1 {
        n
    } else {
        recursive_fib(n - 1) + recursive_fib(n - 2)
    }
}
"#;

    /// Closures - tests closure capture and invocation
    pub const CLOSURES: &str = r#"
pub fn with_closure(x: i32) -> i32 {
    let add_one = |y: i32| y + 1;
    add_one(x)
}

pub fn capturing_closure(x: i32) -> i32 {
    let multiplier = 2;
    let multiply = |y: i32| y * multiplier;
    multiply(x)
}
"#;

    /// Generics - tests generic instantiation
    pub const GENERICS: &str = r#"
pub fn identity<T>(x: T) -> T {
    x
}

pub fn pair<T, U>(a: T, b: U) -> (T, U) {
    (a, b)
}

pub fn use_generics() -> i32 {
    let x = identity(42i32);
    let (a, _b) = pair(1i32, 2i64);
    x + a
}
"#;

    /// Enums and discriminants - tests enum handling
    pub const ENUMS: &str = r#"
pub enum Color {
    Red,
    Green,
    Blue,
    Rgb(u8, u8, u8),
}

pub fn color_value(c: Color) -> u8 {
    match c {
        Color::Red => 0,
        Color::Green => 1,
        Color::Blue => 2,
        Color::Rgb(r, _, _) => r,
    }
}
"#;
}

/// Utility to run the test harness from command line or script.
///
/// Example usage in a test script:
/// ```bash
/// # Set up tracing
/// export RUST_LOG=info
///
/// # Compile with MLIR backend
/// rustc --codegen-backend=mlir -Z frontend=triton test.rs 2>&1 | tee mir_output.log
/// ```
pub fn print_usage() {
    println!(
        r#"
MLIR Codegen Backend Test Harness
==================================

This test harness helps test the MLIR codegen backend by logging MIR structures.

Command Line Usage:
-------------------
1. Set RUST_LOG=info for verbose output
2. Use --codegen-backend=mlir to select the MLIR backend
3. Use -Z frontend=triton to specify the Triton frontend

Example:
  RUST_LOG=info rustc --codegen-backend=mlir -Z frontend=triton test.rs

Test Patterns Available:
------------------------
- ARITHMETIC: Basic math operations
- CONTROL_FLOW: if/else, loops, match
- REFERENCES: borrows, raw pointers
- AGGREGATES: structs, tuples, arrays
- FUNCTION_CALLS: calls, recursion
- CLOSURES: closure capture
- GENERICS: generic instantiation
- ENUMS: enum discriminants

For JIT/API usage, integrate directly with the rustc_interface crate.
"#
    );
}
