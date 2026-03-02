//! Agent-mode JSON contract types.
//!
//! All public types must be stable; treat this as an API surface.
//! Any breaking change here requires a major version bump.
//!
//! The JSON schema produced by [`AgentResponse`] is tested in
//! `tests/schema_tests.rs`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sub-types
// ---------------------------------------------------------------------------

/// The synthesised answer over all gathered sources.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Answer {
    /// Final answer text (may be markdown for human display).
    pub text: String,
    /// Model confidence: 0.0 (no confidence) → 1.0 (fully confident).
    /// Derived from cross-source agreement and LLM logprob if available.
    pub confidence: f64,
}

/// A single web source retrieved and read.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Source {
    /// Canonical URL of the source.
    pub url: String,
    /// Page title as returned by the reader.
    pub title: String,
    /// Short excerpt / snippet from search results.
    pub snippet: String,
    /// 1-based rank in the search result list.
    pub rank: usize,
    /// Raw LLM-friendly markdown returned by the reader adapter (may be empty
    /// if reader failed for this URL).
    #[serde(default)]
    pub reader_raw: String,
}

/// Claim-level fact-check result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactCheck {
    /// The specific factual claim being evaluated.
    pub claim: String,
    /// Overall status across sources.
    pub status: FactCheckStatus,
    /// URLs of sources that provided evidence for this status.
    pub evidence_urls: Vec<String>,
}

/// Possible outcomes of a fact-check.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FactCheckStatus {
    /// Multiple independent sources confirm the claim.
    Confirmed,
    /// At least one source directly contradicts the claim.
    Contradicted,
    /// Insufficient or conflicting evidence to make a determination.
    Inconclusive,
}

impl std::fmt::Display for FactCheckStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactCheckStatus::Confirmed => write!(f, "confirmed"),
            FactCheckStatus::Contradicted => write!(f, "contradicted"),
            FactCheckStatus::Inconclusive => write!(f, "inconclusive"),
        }
    }
}

/// Request/response provenance metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Metadata {
    /// ISO-8601 UTC timestamp of when the response was generated.
    pub timestamp: DateTime<Utc>,
    /// Names of providers that were actually used (not just attempted).
    pub providers_used: Vec<String>,
    /// End-to-end latency from CLI invocation to output ready.
    pub latency_ms: u64,
}

// ---------------------------------------------------------------------------
// Root response type
// ---------------------------------------------------------------------------

/// The complete agent-mode JSON response.
///
/// Serialises to / deserialises from the strict schema documented in the PRD.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentResponse {
    /// The original query string.
    pub query: String,
    /// Synthesised answer over all sources.
    pub answer: Answer,
    /// Ordered list of sources (rank ascending).
    pub sources: Vec<Source>,
    /// Per-claim fact-check results.
    pub fact_checks: Vec<FactCheck>,
    /// Provenance and request metadata.
    pub metadata: Metadata,
}

impl AgentResponse {
    /// Construct a new response with the given query; fill other fields with defaults.
    pub fn new(query: String, providers_used: Vec<String>, latency_ms: u64) -> Self {
        Self {
            query,
            answer: Answer {
                text: String::new(),
                confidence: 0.0,
            },
            sources: Vec::new(),
            fact_checks: Vec::new(),
            metadata: Metadata {
                timestamp: Utc::now(),
                providers_used,
                latency_ms,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Validation helpers (used by tests)
// ---------------------------------------------------------------------------

/// Assert that a JSON value matches the expected schema shape.
/// Returns `Ok(())` on success, `Err(description)` on failure.
pub fn validate_schema(value: &serde_json::Value) -> Result<(), String> {
    let required_top = ["query", "answer", "sources", "fact_checks", "metadata"];
    for key in &required_top {
        if value.get(key).is_none() {
            return Err(format!("Missing top-level key: {key}"));
        }
    }

    let answer = &value["answer"];
    if answer.get("text").is_none() {
        return Err("answer.text missing".into());
    }
    if answer.get("confidence").is_none() {
        return Err("answer.confidence missing".into());
    }

    let meta = &value["metadata"];
    for key in &["timestamp", "providers_used", "latency_ms"] {
        if meta.get(key).is_none() {
            return Err(format!("metadata.{key} missing"));
        }
    }

    if let Some(sources) = value["sources"].as_array() {
        for (i, src) in sources.iter().enumerate() {
            for key in &["url", "title", "snippet", "rank"] {
                if src.get(key).is_none() {
                    return Err(format!("sources[{i}].{key} missing"));
                }
            }
        }
    } else {
        return Err("sources must be an array".into());
    }

    if let Some(fcs) = value["fact_checks"].as_array() {
        for (i, fc) in fcs.iter().enumerate() {
            for key in &["claim", "status", "evidence_urls"] {
                if fc.get(key).is_none() {
                    return Err(format!("fact_checks[{i}].{key} missing"));
                }
            }
        }
    } else {
        return Err("fact_checks must be an array".into());
    }

    Ok(())
}
