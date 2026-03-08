//! Multi-factor result ranking engine.
//!
//! Scores crawled pages on four axes:
//!
//! | Axis | Weight | Notes |
//! |------|--------|-------|
//! | Domain reputation | 0.30 | TLD trust + known-quality domain patterns |
//! | Content length | 0.25 | Longer substantive pages score higher (up to a cap) |
//! | Keyword density | 0.35 | Query term coverage in the extracted text |
//! | Freshness | 0.10 | ISO-8601 date extracted from URL path or content |
//!
//! All axes are normalised to [0, 1] before weighting.

use chrono::{Datelike, Utc};
use once_cell::sync::Lazy;
use regex::Regex;
use tracing::debug;

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

const W_DOMAIN: f64 = 0.30;
const W_CONTENT_LEN: f64 = 0.25;
const W_KEYWORD: f64 = 0.35;
const W_FRESHNESS: f64 = 0.10;

/// Content length above which the length bonus is fully saturated (~5 000 words).
const CONTENT_LEN_CAP: usize = 30_000; // characters

/// Compiled regex for ISO date extraction from URLs / text.
static DATE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(\d{4})[/\-_](\d{1,2})[/\-_](\d{1,2})").unwrap()
});

// ---------------------------------------------------------------------------
// Domain reputation table
// ---------------------------------------------------------------------------

/// Manually-curated high-reputation domain suffixes / patterns.
/// Score contribution: 0.0 (unknown) → 1.0 (academic / major reference).
const HIGH_REP_DOMAINS: &[(&str, f64)] = &[
    // Academic / research
    (".edu", 1.0),
    (".gov", 1.0),
    ("arxiv.org", 0.95),
    ("pubmed.ncbi.nlm.nih.gov", 0.95),
    ("scholar.google", 0.90),
    ("semanticscholar.org", 0.90),
    // Major reference
    ("wikipedia.org", 0.88),
    ("britannica.com", 0.85),
    // Tech documentation
    ("docs.rs", 0.85),
    ("doc.rust-lang.org", 0.90),
    ("developer.mozilla.org", 0.88),
    ("docs.python.org", 0.88),
    // Quality tech media
    ("stackoverflow.com", 0.80),
    ("github.com", 0.75),
    ("arxiv-sanity.com", 0.70),
    ("towardsdatascience.com", 0.65),
    ("medium.com", 0.55),
    ("dev.to", 0.60),
    ("hackernews.com", 0.62),
    ("news.ycombinator.com", 0.62),
    // Generic neutral
    (".org", 0.60),
    (".net", 0.50),
    (".com", 0.45),
    (".io", 0.50),
];

/// Look up a domain reputation score in [0, 1].
fn domain_score(url: &str) -> f64 {
    let url_lower = url.to_lowercase();
    for (pattern, score) in HIGH_REP_DOMAINS {
        if url_lower.contains(pattern) {
            return *score;
        }
    }
    0.35 // unknown domain baseline
}

// ---------------------------------------------------------------------------
// Keyword density score
// ---------------------------------------------------------------------------

/// Compute fraction of unique query terms present in the text.
fn keyword_score(text: &str, query: &str) -> f64 {
    let text_lower = text.to_lowercase();
    let terms: Vec<&str> = query
        .split_whitespace()
        .filter(|t| t.len() > 2) // skip stop words by length proxy
        .collect();

    if terms.is_empty() {
        return 0.5;
    }

    let hits = terms.iter().filter(|&&t| text_lower.contains(t)).count();
    hits as f64 / terms.len() as f64
}

// ---------------------------------------------------------------------------
// Content-length score
// ---------------------------------------------------------------------------

fn content_len_score(content: &str) -> f64 {
    let len = content.len().min(CONTENT_LEN_CAP);
    len as f64 / CONTENT_LEN_CAP as f64
}

// ---------------------------------------------------------------------------
// Freshness score
// ---------------------------------------------------------------------------

/// Extract a year from the URL or content and compute a decay score.
/// Pages from the current year score 1.0; score decays ~0.15 per year.
fn freshness_score(url: &str, content: &str) -> f64 {
    let current_year = Utc::now().year() as i64;

    let candidate = DATE_RE
        .captures(url)
        .or_else(|| DATE_RE.captures(content))
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<i64>().ok());

    match candidate {
        Some(year) if year >= 2000 && year <= current_year => {
            let age = (current_year - year) as f64;
            (1.0 - age * 0.12).max(0.05)
        }
        Some(_) => 0.05,
        None => 0.40, // no date found — assume middling freshness
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Composite score for a single result. Returns a value in [0, 1].
pub fn score_result(url: &str, content: &str, query: &str) -> f64 {
    let d = domain_score(url);
    let l = content_len_score(content);
    let k = keyword_score(content, query);
    let f = freshness_score(url, content);

    let score = W_DOMAIN * d + W_CONTENT_LEN * l + W_KEYWORD * k + W_FRESHNESS * f;
    debug!(url, score, domain = d, length = l, keyword = k, freshness = f, "Result scored");
    score
}

/// Rank a list of `(url, content)` pairs for a query.
/// Returns indices sorted descending by composite score.
pub fn rank_results<'a>(results: &'a [(String, String)], query: &str) -> Vec<(usize, f64)> {
    let mut scored: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .map(|(i, (url, content))| (i, score_result(url, content, query)))
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}
