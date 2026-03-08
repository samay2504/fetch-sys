//! `fetchsys` library crate.
//!
//! Exposes all modules for integration tests and downstream agent crates.
//!
//! ## Module tree
//! ```text
//! fetchsys
//!  ├── cli          — CLI argument definitions (clap)
//!  ├── config       — unified config loading (TOML + env + CLI)
//!  ├── schema       — agent JSON contract
//!  ├── search       — 6-tier search waterfall
//!  ├── reader       — URL-to-Markdown reader (Jina / Readability / anytomd / Firecrawl / raw HTTP)
//!  ├── llm          — LLM provider waterfall
//!  ├── factcheck    — claim extraction + cross-referencing
//!  │
//!  │   ── performance crawler engine ──
//!  ├── crawler      — concurrent JoinSet scheduler + login integration
//!  ├── fetch        — tiered fetch (static + CAPTCHA bypass + scroll)
//!  ├── human        — UA rotation, delays, domain politeness
//!  ├── captcha      — CAPTCHA/bot detection + bypass strategy planner
//!  ├── extract      — page classification + readability extraction
//!  ├── ranking      — multi-factor result scoring
//!  ├── storage      — in-memory TTL page cache
//!  ├── login        — mock credential vault + login-wall automation
//!  └── scroll       — infinite-scroll content accumulation
//! ```

pub mod cli;
pub mod config;
pub mod factcheck;
pub mod llm;
pub mod reader;
pub mod schema;
pub mod search;

// Performance crawler engine modules
pub mod captcha;
pub mod crawler;
pub mod extract;
pub mod fetch;
pub mod human;
pub mod login;
pub mod ranking;
pub mod scroll;
pub mod storage;
