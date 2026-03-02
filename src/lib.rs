//! `facttool` library crate.
//!
//! Exposes all modules so they can be used by integration tests and by other
//! crates embedding `facttool` as a library dependency.

pub mod cli;
pub mod config;
pub mod factcheck;
pub mod llm;
pub mod reader;
pub mod schema;
pub mod search;
