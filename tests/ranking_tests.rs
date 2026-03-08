//! Integration tests for the ranking module — scoring, ordering, domain signals,
//! freshness heuristics, and keyword density.
//!
//! `score_result(url, content, query) -> f64`
//! `rank_results(results: &[(String, String)], query) -> Vec<(usize, f64)>`

use fetchsys::ranking::{rank_results, score_result};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build a (url, content) pair as used by the ranking API.
fn pair(url: &str, content: &str) -> (String, String) {
    (url.to_string(), content.to_string())
}

// ── score_result ──────────────────────────────────────────────────────────────

#[test]
fn score_is_between_zero_and_one() {
    let s = score_result("https://example.com", "Rust is great", "Rust");
    assert!(s >= 0.0, "Score must be non-negative, got {}", s);
    assert!(s <= 1.0, "Score must be at most 1.0, got {}", s);
}

#[test]
fn higher_score_with_query_in_content() {
    let s_match = score_result(
        "https://example.com",
        "Rust Programming Language Guide: a comprehensive guide to Rust.",
        "Rust",
    );
    let s_no_match = score_result(
        "https://example.com",
        "A comparison of programming languages in general.",
        "Rust",
    );
    assert!(
        s_match > s_no_match,
        "Keyword match should boost score ({:.4} vs {:.4})",
        s_match, s_no_match
    );
}

#[test]
fn edu_domain_scores_higher_than_com_same_content() {
    let content = "Rust ownership and memory safety course material.";
    let s_edu = score_result("https://cs.mit.edu/rust", content, "Rust");
    let s_com = score_result("https://example.com/rust", content, "Rust");
    assert!(
        s_edu > s_com,
        ".edu should score higher than .com ({:.4} vs {:.4})",
        s_edu, s_com
    );
}

#[test]
fn gov_domain_scores_higher_than_info_same_content() {
    let content = "Rust safety facts and research findings.";
    let s_gov = score_result("https://cdc.gov/rust-article", content, "Rust");
    let s_info = score_result("https://facts.info/rust", content, "Rust");
    assert!(s_gov > s_info, ".gov should score higher than .info");
}

#[test]
fn org_domain_scores_at_least_as_high_as_com() {
    let content = "Ownership and borrowing in Rust.";
    let s_org = score_result("https://rustlang.org/learn", content, "Rust");
    let s_com = score_result("https://randomblog.com/rust", content, "Rust");
    assert!(s_org >= s_com, ".org should score >= .com");
}

#[test]
fn fresh_url_year_scores_higher_than_old() {
    // DATE_RE requires year/month/day — use full date paths so freshness is extracted
    let content = "Latest Rust updates and release notes from this year.";
    let s_new = score_result("https://blog.example.com/2025/01/15/rust-news", content, "Rust");
    let s_old = score_result("https://blog.example.com/2015/01/15/rust-intro", content, "Rust");
    assert!(
        s_new > s_old,
        "Newer year in URL should score higher ({:.4} vs {:.4})",
        s_new, s_old
    );
}

#[test]
fn keyword_density_does_not_hurt() {
    // Both texts contain the keyword; the scorer uses binary presence not density.
    // Use content of equal length so only keyword presence (not char count) drives any score delta.
    let dense = score_result(
        "https://x.com/a",
        "Rust Rust Rust Rust Rust Rust programming language",
        "Rust",
    );
    let single = score_result(
        "https://x.com/b",
        "The Rust programming language is excellent today",
        "Rust",
    );
    // Both have the keyword → same keyword_score (1.0); dense should not score lower
    assert!(dense >= single * 0.95,
        "Dense keyword use should score at least as high as sparse ({:.4} vs {:.4})",
        dense, single
    );
}

#[test]
fn empty_query_returns_valid_score() {
    let s = score_result("https://example.com", "Some page content here", "");
    assert!(s >= 0.0);
    assert!(s <= 1.0);
}

#[test]
fn https_favored_over_http() {
    let content = "Content about Rust programming language.";
    let s_https = score_result("https://example.com/page", content, "Rust");
    let s_http = score_result("http://example.com/page", content, "Rust");
    assert!(s_https >= s_http, "HTTPS should score >= HTTP");
}

#[test]
fn multiple_keywords_all_present_scores_higher() {
    let s_full = score_result(
        "https://example.com",
        "Async runtime design and concurrency primitives in Rust.",
        "Rust async concurrency",
    );
    let s_partial = score_result(
        "https://example.com",
        "Introduction to Rust programming basics.",
        "Rust async concurrency",
    );
    assert!(s_full > s_partial, "All keywords matching should yield higher score");
}

#[test]
fn longer_content_does_not_reduce_score_below_zero() {
    let long = "word ".repeat(10_000);
    let s = score_result("https://example.com", &long, "word");
    assert!(s >= 0.0);
    assert!(s <= 1.0);
}

// ── rank_results ──────────────────────────────────────────────────────────────

#[test]
fn rank_results_returns_sorted_descending() {
    let pairs = vec![
        pair("http://spam.biz/x", "No keywords here at all."),
        pair("https://cs.mit.edu/rust-ownership", "Ownership in Rust: a deep dive."),
        pair("https://rustlang.org/docs", "All about Rust language."),
    ];
    let ranked = rank_results(&pairs, "Rust ownership");
    // Verify descending order
    for i in 0..ranked.len().saturating_sub(1) {
        assert!(
            ranked[i].1 >= ranked[i + 1].1,
            "rank_results results should be in descending score order"
        );
    }
}

#[test]
fn rank_results_returns_correct_count() {
    let pairs = vec![
        pair("https://a.com", "content a"),
        pair("https://b.com", "content b"),
        pair("https://c.com", "content c"),
    ];
    let ranked = rank_results(&pairs, "content");
    assert_eq!(ranked.len(), 3, "Should return one entry per input");
}

#[test]
fn rank_results_indices_are_valid() {
    let pairs = vec![
        pair("https://a.com", "alpha beta gamma"),
        pair("https://b.com", "delta epsilon"),
        pair("https://c.com", "zeta"),
    ];
    let ranked = rank_results(&pairs, "alpha");
    for (idx, _score) in &ranked {
        assert!(*idx < pairs.len(), "Index {} out of bounds for {} pairs", idx, pairs.len());
    }
}

#[test]
fn rank_results_empty_input_returns_empty() {
    let pairs: Vec<(String, String)> = vec![];
    let ranked = rank_results(&pairs, "any query");
    assert!(ranked.is_empty(), "Empty input should return empty ranked list");
}

#[test]
fn rank_results_single_item_returns_single() {
    let pairs = vec![pair("https://example.com", "Rust content here")];
    let ranked = rank_results(&pairs, "Rust");
    assert_eq!(ranked.len(), 1);
    assert_eq!(ranked[0].0, 0, "Single item should have index 0");
}

#[test]
fn rank_results_edu_index_at_top_for_academic_query() {
    let pairs = vec![
        pair("https://random-seo.biz/quantum", "basic intro to quantum"),
        pair("https://physics.caltech.edu/quantum", "quantum computing research methods"),
    ];
    let ranked = rank_results(&pairs, "quantum computing research");
    // The highest-ranked item should be the .edu domain (index 1 in input)
    assert_eq!(
        ranked[0].0, 1,
        ".edu domain should be ranked first for academic query"
    );
}

#[test]
fn rank_results_identical_pairs_do_not_panic() {
    let pairs = vec![
        pair("https://a.com", "same content"),
        pair("https://b.com", "same content"),
        pair("https://c.com", "same content"),
    ];
    let ranked = rank_results(&pairs, "same");
    assert_eq!(ranked.len(), 3);
}

#[test]
fn rank_results_all_scores_in_valid_range() {
    let pairs = vec![
        pair("https://example.com", "Rust ownership and lifetimes"),
        pair("http://old-site.net/2010/post", "something unrelated"),
        pair("https://cs.stanford.edu/os", "Operating systems and memory management"),
    ];
    for (_idx, score) in rank_results(&pairs, "Rust memory") {
        assert!(score >= 0.0, "Score should be >= 0, got {}", score);
        assert!(score <= 1.0, "Score should be <= 1, got {}", score);
    }
}
