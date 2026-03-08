//! Fact-checker / aggregator.
//!
//! Takes the top-N read documents, extracts key factual claims (via LLM or
//! heuristics), cross-references them across sources, and computes a
//! per-claim status and an overall confidence score.

use tracing::{debug, info};

use crate::{
    config::FactCheckConfig,
    llm::LlmProvider,
    reader::ReadResult,
    schema::{FactCheck, FactCheckStatus},
};

// ---------------------------------------------------------------------------
// Intermediate types (internal to this module)
// ---------------------------------------------------------------------------

/// A claim extracted from one or more sources with per-source support info.
#[derive(Debug, Clone)]
pub struct AnnotatedClaim {
    pub text: String,
    /// URLs that confirm / support the claim.
    pub supporting_urls: Vec<String>,
    /// URLs that contradict the claim.
    pub contradicting_urls: Vec<String>,
    /// Total sources examined.
    pub total_sources: usize,
}

impl AnnotatedClaim {
    /// Derive a [`FactCheckStatus`] based on threshold parameters.
    pub fn status(&self, cfg: &FactCheckConfig) -> FactCheckStatus {
        let total = self.total_sources as f64;
        if total == 0.0 {
            return FactCheckStatus::Inconclusive;
        }
        let confirm_frac = self.supporting_urls.len() as f64 / total;
        let contradict_frac = self.contradicting_urls.len() as f64 / total;

        if contradict_frac >= cfg.contradict_threshold && contradict_frac > confirm_frac {
            FactCheckStatus::Contradicted
        } else if confirm_frac >= cfg.confirm_threshold {
            FactCheckStatus::Confirmed
        } else {
            FactCheckStatus::Inconclusive
        }
    }

    /// Evidence URLs relevant to the resolved status.
    pub fn evidence_urls(&self, status: &FactCheckStatus) -> Vec<String> {
        match status {
            FactCheckStatus::Contradicted => self.contradicting_urls.clone(),
            FactCheckStatus::Confirmed => self.supporting_urls.clone(),
            FactCheckStatus::Inconclusive => {
                let mut urls = self.supporting_urls.clone();
                urls.extend(self.contradicting_urls.iter().cloned());
                urls
            }
        }
    }

    pub fn to_fact_check(&self, cfg: &FactCheckConfig) -> FactCheck {
        let status = self.status(cfg);
        let evidence_urls = self.evidence_urls(&status);
        FactCheck {
            claim: self.text.clone(),
            status,
            evidence_urls,
        }
    }
}

// ---------------------------------------------------------------------------
// Claim extraction
// ---------------------------------------------------------------------------

/// Aggregate result from the fact-check pipeline.
#[derive(Debug)]
pub struct FactCheckOutput {
    pub answer: String,
    pub confidence: f64,
    pub fact_checks: Vec<AnnotatedClaim>,
}

/// Extract factual claims via LLM given the aggregated context.
async fn extract_claims_via_llm(
    query: &str,
    docs: &[ReadResult],
    provider: &dyn LlmProvider,
) -> anyhow::Result<Vec<String>> {
    use crate::llm::{build_context, Message};

    let context = build_context(docs, 800);

    let messages = vec![
        Message::system(
            "You extract factual claims from source text. Rules:\n\
             1. Output ONLY a JSON array of 3-5 short factual claim strings.\n\
             2. Each claim must be a single, self-contained factual sentence (max 25 words).\n\
             3. No markdown fences, no commentary, no explanation.\n\
             4. Each claim must be directly about the query topic, not metadata like dates or page warnings.\n\
             5. Example output: [\"SHAP is based on Shapley values from game theory\", \"SHAP was introduced by Lundberg and Lee in 2017\"]",
        ),
        Message::user(format!(
            "Query: {query}\n\nSources:\n{context}\n\nJSON array of 3-5 factual claims:"
        )),
    ];

    let raw = provider.complete(messages).await?;
    debug!(raw_len = raw.len(), raw_preview = %raw.chars().take(300).collect::<String>(), "LLM claim extraction response");

    // Parse the JSON array from the response (LLM might wrap in markdown fences,
    // or the response might have been truncated by max_tokens)
    let json_str = extract_json_array(&raw).unwrap_or_else(|| raw.trim().to_owned());
    let claims: Vec<String> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                error = %e,
                json_str = %json_str.chars().take(200).collect::<String>(),
                "LLM claim JSON parse failed; falling back to heuristic"
            );
            return Ok(Vec::new());
        }
    };

    // Filter out any claim that looks like metadata rather than a factual statement
    let filtered: Vec<String> = claims
        .into_iter()
        .filter(|c| !is_metadata_line(c))
        .collect();

    if filtered.is_empty() {
        tracing::warn!("LLM returned zero usable claims after filtering; falling back to heuristic");
    } else {
        info!(count = filtered.len(), "LLM extracted claims");
    }

    Ok(filtered)
}

/// Heuristic fallback when LLM is disabled or claim extraction fails:
/// derive factual claims from document content by extracting declarative
/// sentences that look like factual statements.
///
/// Filters out:
/// - Metadata lines (timestamps, published dates, CAPTCHA warnings)
/// - Headings (lines starting with `#` or `=`)
/// - Very short or navigation-like lines
/// - Lines that are questions or imperatives
fn extract_claims_heuristic(query: &str, docs: &[ReadResult]) -> Vec<String> {
    let query_lower = query.to_lowercase();
    // Extract 2-3 content-bearing keywords from the query
    let query_keywords: Vec<&str> = query_lower
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .collect();

    let mut claims: Vec<String> = Vec::new();

    for doc in docs.iter().take(5) {
        for line in doc.content.lines() {
            if claims.len() >= 5 {
                break;
            }
            let trimmed = line.trim();

            // Skip empty, too-short, or too-long lines
            if trimmed.len() < 40 || trimmed.len() > 300 {
                continue;
            }

            // Skip metadata and noise
            if is_metadata_line(trimmed) {
                continue;
            }

            // Must contain at least one query keyword to be relevant
            let lower = trimmed.to_lowercase();
            if !query_keywords.iter().any(|kw| lower.contains(kw)) {
                continue;
            }

            let claim: String = trimmed.chars().take(200).collect();
            // Deduplicate by checking overlap with already-collected claims
            if !claims.iter().any(|existing| {
                let ex_lower = existing.to_lowercase();
                // >60% word overlap → duplicate
                let words: Vec<&str> = lower.split_whitespace().collect();
                let overlap = words.iter().filter(|w| ex_lower.contains(**w)).count();
                overlap as f64 / words.len().max(1) as f64 > 0.6
            }) {
                claims.push(claim);
            }
        }
    }

    // If we couldn't find anything, use a minimal generic claim
    if claims.is_empty() {
        claims.push(format!(
            "Multiple web sources were found containing information about: {}",
            query
        ));
    }

    claims
}

/// Returns `true` if the line looks like metadata / boilerplate rather than
/// a factual claim.  Used by both LLM and heuristic claim paths.
fn is_metadata_line(line: &str) -> bool {
    let l = line.to_lowercase();
    let l = l.trim();

    // Timestamps / published dates
    if l.starts_with("published")
        || l.starts_with("date:")
        || l.starts_with("last updated")
        || l.starts_with("modified")
        || l.starts_with("posted on")
    {
        return true;
    }

    // Regex-free date pattern: contains yyyy-mm-dd or common date headers
    if l.contains("published time:") || l.contains("publish date") {
        return true;
    }

    // Page warnings / CAPTCHA / cookie notices
    if l.contains("captcha")
        || l.contains("cookie")
        || l.contains("javascript is required")
        || l.contains("enable javascript")
        || l.contains("please verify")
        || l.contains("access denied")
        || l.contains("403 forbidden")
    {
        return true;
    }

    // Navigation / UI chrome
    if l.starts_with("skip to")
        || l.starts_with("jump to")
        || l.starts_with("table of contents")
        || l.starts_with("menu")
        || l.starts_with("search")
        || l.starts_with("sign in")
        || l.starts_with("log in")
    {
        return true;
    }

    // Headings / separators
    if l.starts_with('#') || l.starts_with('=') || l.starts_with("---") {
        return true;
    }

    // Questions are not claims
    if l.ends_with('?') {
        return true;
    }

    // Lines that are mostly non-alpha (URLs, code, paths)
    let alpha_count = l.chars().filter(|c| c.is_alphabetic()).count();
    if l.len() > 10 && (alpha_count as f64 / l.len() as f64) < 0.5 {
        return true;
    }

    false
}

/// Cross-reference a claim against all documents using keyword coverage +
/// proximity-windowed negation.
///
/// Design:
/// - A document is *relevant* if >50% of the claim's content-words appear in it.
/// - A relevant document is *contradicting* only if a **strong negation phrase**
///   (e.g. "is false", "is a myth", "debunked") appears within
///   `NEGATION_WINDOW` bytes of a keyword hit.
/// - Checking the whole document for single words like "not" or "wrong" causes
///   every technical article (which routinely discusses problems and anti-patterns)
///   to be falsely marked contradicting — this fixes that.
///
/// Replace with embedding-similarity reranking when a vector store is available.
fn cross_reference(claim: &str, docs: &[ReadResult], docs_lower: &[String]) -> AnnotatedClaim {
    // Meaningful content words only (skip stop-words under 4 chars)
    let claim_lower = claim.to_lowercase();
    let keywords: Vec<&str> = claim_lower
        .split_whitespace()
        .filter(|w| w.len() > 4)
        .collect();

    // Strong contradiction phrases — multi-word to minimise false positives.
    // Single words like "not" / "wrong" are too common in technical writing.
    const NEGATION_PHRASES: &[&str] = &[
        "is false", "is incorrect", "is wrong", "is a myth", "is untrue",
        "not true", "not correct", "not accurate", "not the case",
        "never true", "debunked", "disproven", "contrary to", "contradicts",
    ];
    // Search window (bytes) on each side of a keyword match
    const NEGATION_WINDOW: usize = 120;

    /// Snap a byte index to the nearest valid UTF-8 char boundary.
    /// `dir` == `true` → search forward; `false` → search backward.
    #[inline]
    fn snap_to_char_boundary(s: &str, idx: usize, forward: bool) -> usize {
        let len = s.len();
        if idx >= len { return len; }
        if s.is_char_boundary(idx) { return idx; }
        if forward {
            (idx..=len).find(|&i| s.is_char_boundary(i)).unwrap_or(len)
        } else {
            (0..idx).rev().find(|&i| s.is_char_boundary(i)).unwrap_or(0)
        }
    }

    let mut supporting = Vec::new();
    let mut contradicting = Vec::new();

    for (doc, content_lower) in docs.iter().zip(docs_lower.iter()) {

        // Always consider docments relevant when no meaningful keywords exist
        if keywords.is_empty() {
            supporting.push(doc.url.clone());
            continue;
        }

        let hits = keywords.iter().filter(|k| content_lower.contains(**k)).count();

        // Skip documents that do not cover at least half the claim keywords
        if hits as f64 / keywords.len() as f64 <= 0.5 {
            continue;
        }

        // Proximity check: scan each keyword occurrence and look for a strong
        // negation phrase within NEGATION_WINDOW bytes on either side.
        // All slice boundaries are snapped to valid UTF-8 char boundaries to
        // prevent panics on multi-byte characters (e.g. ≠, →, é).
        let has_proximate_negation = keywords.iter().any(|kw| {
            let mut search_from = 0usize;
            let sf = snap_to_char_boundary(&content_lower, search_from, true);
            let _ = sf; // ensure search_from itself is safe below
            while search_from < content_lower.len() {
                let safe_from = snap_to_char_boundary(&content_lower, search_from, true);
                let haystack = &content_lower[safe_from..];
                let rel = match haystack.find(kw) {
                    Some(r) => r,
                    None => break,
                };
                let abs = safe_from + rel;
                let win_start = snap_to_char_boundary(
                    &content_lower,
                    abs.saturating_sub(NEGATION_WINDOW),
                    false,
                );
                let win_end = snap_to_char_boundary(
                    &content_lower,
                    (abs + kw.len() + NEGATION_WINDOW).min(content_lower.len()),
                    true,
                );
                let window = &content_lower[win_start..win_end];
                if NEGATION_PHRASES.iter().any(|neg| window.contains(neg)) {
                    return true;
                }
                search_from = abs + 1;
            }
            false
        });

        if has_proximate_negation {
            contradicting.push(doc.url.clone());
        } else {
            supporting.push(doc.url.clone());
        }
    }

    AnnotatedClaim {
        text: claim.to_owned(),
        supporting_urls: supporting,
        contradicting_urls: contradicting,
        total_sources: docs.len(),
    }
}

// ---------------------------------------------------------------------------
// Synthesis
// ---------------------------------------------------------------------------

/// Generate the final synthesised answer text.
async fn synthesise(
    query: &str,
    docs: &[ReadResult],
    provider: &dyn LlmProvider,
    llm_enabled: bool,
) -> String {
    if !llm_enabled || docs.is_empty() {
        // Fallback: aggregate snippets
        let snippets: Vec<String> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| format!("[{}] {}: {}", i + 1, d.title, d.snippet(300)))
            .collect();
        return format!(
            "## Search results for: {query}\n\n{}",
            snippets.join("\n\n")
        );
    }

    use crate::llm::{build_context, synthesis_prompt};
    let context = build_context(docs, 1200);
    let msgs = synthesis_prompt(query, &context);

    match provider.complete(msgs).await {
        Ok(text) => text,
        Err(e) => {
            tracing::warn!(error = %e, "LLM synthesis failed; falling back to snippets");
            docs.iter()
                .enumerate()
                .map(|(i, d)| format!("[{}] {} — {}", i + 1, d.url, d.snippet(200)))
                .collect::<Vec<_>>()
                .join("\n\n")
        }
    }
}

/// Compute overall confidence from the fact-check results.
///
/// Algorithm (production-grade, avoids the 0%-for-everything failure):
///
/// 1. `decided`  = confirmed + contradicted  (claims with a verdict)
/// 2. `precision` = confirmed / decided       (how much evidence agrees)
/// 3. `coverage`  = decided  / total          (how many claims have evidence)
/// 4. score = precision × 0.7 + coverage × 0.3
///
/// Key properties:
/// - Inconclusive claims are *neutral*: they don't dilute precision.
/// - If ALL claims are inconclusive (no verdict either way), we return a
///   moderate 0.5 — sources were found and read, contradictions absent.
/// - Contradictions still penalise: a claim counted as contradicted is
///   excluded from the numerator of precision.
fn compute_confidence(claims: &[AnnotatedClaim], cfg: &FactCheckConfig) -> f64 {
    if claims.is_empty() {
        return 0.0;
    }

    let confirmed = claims
        .iter()
        .filter(|c| matches!(c.status(cfg), FactCheckStatus::Confirmed))
        .count();
    let contradicted = claims
        .iter()
        .filter(|c| matches!(c.status(cfg), FactCheckStatus::Contradicted))
        .count();
    let decided = confirmed + contradicted;

    if decided == 0 {
        // All claims inconclusive — moderate baseline: evidence was read but
        // the heuristic found no strong verdicts in either direction.
        return 0.5;
    }

    let precision = confirmed as f64 / decided as f64;
    let coverage  = decided  as f64 / claims.len() as f64;
    (precision * 0.7 + coverage * 0.3).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the full fact-check pipeline over the collected documents.
///
/// - Extracts claims (via LLM or heuristics)
/// - Cross-references each claim across all docs
/// - Synthesises a final answer
/// - Computes an overall confidence score
pub async fn check(
    docs: &[ReadResult],
    query: &str,
    provider: &dyn LlmProvider,
    llm_enabled: bool,
    cfg: &FactCheckConfig,
) -> FactCheckOutput {
    info!(doc_count = docs.len(), "Starting fact-check pipeline");

    // 1+2. Synthesise answer and extract claims concurrently
    let (answer, claims_text) = if llm_enabled {
        let synth_fut = synthesise(query, docs, provider, llm_enabled);
        let claims_fut = async {
            let llm_claims = extract_claims_via_llm(query, docs, provider)
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "LLM claim extraction failed; using heuristic");
                    Vec::new()
                });
            if llm_claims.is_empty() {
                tracing::info!("Using heuristic claim extraction (LLM produced no claims)");
                extract_claims_heuristic(query, docs)
            } else {
                llm_claims
            }
        };
        tokio::join!(synth_fut, claims_fut)
    } else {
        let answer = synthesise(query, docs, provider, llm_enabled).await;
        let claims_text = extract_claims_heuristic(query, docs);
        (answer, claims_text)
    };

    // 3. Cross-reference each claim (pre-compute lowercased docs once)
    let docs_lower: Vec<String> = docs.iter().map(|d| d.content.to_lowercase()).collect();
    let annotated: Vec<AnnotatedClaim> = claims_text
        .iter()
        .map(|c| cross_reference(c, docs, &docs_lower))
        .collect();

    // 4. Compute confidence
    let confidence = compute_confidence(&annotated, cfg);
    info!(confidence, "Fact-check complete");

    FactCheckOutput {
        answer,
        confidence,
        fact_checks: annotated,
    }
}

/// Helper to pull a JSON array out of text that may include markdown fences
/// or may have been truncated by max_tokens.
///
/// Handles:
/// - Clean: `["a", "b"]`
/// - Markdown-fenced: `` ```json\n["a", "b"]\n``` ``
/// - Truncated: `["a", "b", "c` → repairs to `["a", "b"]` (best-effort)
/// - Unescaped inner quotes: `["The paper "X" was..."]` → escapes inner `"`
fn extract_json_array(text: &str) -> Option<String> {
    let start = text.find('[')?;

    // Try the full bracket match first (ideal case)
    if let Some(end) = text.rfind(']') {
        if end > start {
            let candidate = &text[start..=end];
            // Quick validation: if it parses, great
            if serde_json::from_str::<Vec<String>>(candidate).is_ok() {
                return Some(candidate.to_owned());
            }
            // Try sanitising unescaped inner quotes before giving up on full match
            if let Some(fixed) = sanitize_json_inner_quotes(candidate) {
                return Some(fixed);
            }
        }
    }

    // Truncated response: find the last complete string element.
    // Walk backward from the end to find the last `"` that closes a string,
    // then find the corresponding `,` or `[` before it to cut cleanly.
    let after_bracket = &text[start..];

    // Find all positions of `",` or `"\n` which mark end-of-element boundaries
    let mut last_clean_end = None;
    let mut i = 0;
    let bytes = after_bracket.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'"' {
            // Check if this quote is followed by `,` or `]` or whitespace+`]`
            let mut j = i + 1;
            // Skip whitespace
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\n' || bytes[j] == b'\r' || bytes[j] == b'\t') {
                j += 1;
            }
            if j < bytes.len() && (bytes[j] == b',' || bytes[j] == b']') {
                last_clean_end = Some(i);
            }
        }
        i += 1;
    }

    if let Some(end_quote) = last_clean_end {
        // Build a repaired array: take everything up to and including the last
        // cleanly-terminated string, then close the array
        let partial = &after_bracket[..=end_quote];
        let repaired = format!("{}]", partial.trim_end_matches(',').trim());
        if serde_json::from_str::<Vec<String>>(&repaired).is_ok() {
            return Some(repaired);
        }
        // Try sanitising the repaired string too
        if let Some(fixed) = sanitize_json_inner_quotes(&repaired) {
            return Some(fixed);
        }
    }

    None
}

/// Fix unescaped inner quotes in a JSON array of strings.
///
/// LLMs sometimes produce output like:
///   `["The paper "Attention Is All You Need" introduced..."]`
/// where the inner quotes are not escaped. This function walks the
/// byte stream and escapes any `"` that appears mid-string (i.e. not
/// at an element boundary like `["`, `",` or `"]`).
fn sanitize_json_inner_quotes(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    if len < 2 || bytes[0] != b'[' {
        return None;
    }

    let mut out = Vec::with_capacity(len + 64);
    out.push(b'[');
    let mut i = 1;

    // Skip leading whitespace inside the array
    while i < len && bytes[i].is_ascii_whitespace() {
        out.push(bytes[i]);
        i += 1;
    }

    while i < len {
        // End of array
        if bytes[i] == b']' {
            out.push(b']');
            break;
        }
        // Expect opening quote of a string element
        if bytes[i] != b'"' {
            // Skip unexpected chars (whitespace, commas already handled)
            out.push(bytes[i]);
            i += 1;
            continue;
        }

        // ── Process one string element ──
        out.push(b'"'); // opening quote
        i += 1;

        while i < len {
            // Already-escaped char — pass through
            if bytes[i] == b'\\' && i + 1 < len {
                out.push(bytes[i]);
                out.push(bytes[i + 1]);
                i += 2;
                continue;
            }
            if bytes[i] == b'"' {
                // Is this the *closing* quote?  Look ahead past whitespace
                // for `,` or `]` which indicate element boundary.
                let mut j = i + 1;
                while j < len && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j >= len || bytes[j] == b',' || bytes[j] == b']' {
                    // Closing quote
                    out.push(b'"');
                    i = j;
                    break;
                }
                // Inner quote — escape it
                out.push(b'\\');
                out.push(b'"');
                i += 1;
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }

        // Consume comma + whitespace between elements
        while i < len && bytes[i].is_ascii_whitespace() {
            out.push(bytes[i]);
            i += 1;
        }
        if i < len && bytes[i] == b',' {
            out.push(b',');
            i += 1;
        }
        while i < len && bytes[i].is_ascii_whitespace() {
            out.push(bytes[i]);
            i += 1;
        }
    }

    let repaired = String::from_utf8(out).ok()?;
    if serde_json::from_str::<Vec<String>>(&repaired).is_ok() {
        Some(repaired)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FactCheckConfig;

    fn cfg() -> FactCheckConfig {
        FactCheckConfig {
            confirm_threshold: 0.6,
            contradict_threshold: 0.4,
        }
    }

    fn claim(sup: usize, con: usize, total: usize) -> AnnotatedClaim {
        AnnotatedClaim {
            text: "test claim".into(),
            supporting_urls: (0..sup).map(|i| format!("https://s{i}.com")).collect(),
            contradicting_urls: (0..con).map(|i| format!("https://c{i}.com")).collect(),
            total_sources: total,
        }
    }

    #[test]
    fn confirmed_when_majority_support() {
        let c = claim(3, 0, 4);
        assert_eq!(c.status(&cfg()), FactCheckStatus::Confirmed);
    }

    #[test]
    fn contradicted_when_majority_contradict() {
        let c = claim(1, 3, 4);
        assert_eq!(c.status(&cfg()), FactCheckStatus::Contradicted);
    }

    #[test]
    fn inconclusive_when_split() {
        let c = claim(1, 1, 4);
        assert_eq!(c.status(&cfg()), FactCheckStatus::Inconclusive);
    }

    #[test]
    fn inconclusive_on_zero_sources() {
        let c = claim(0, 0, 0);
        assert_eq!(c.status(&cfg()), FactCheckStatus::Inconclusive);
    }

    #[test]
    fn confidence_zero_when_no_claims() {
        let conf = compute_confidence(&[], &cfg());
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn confidence_high_when_all_confirmed() {
        let claims = vec![claim(3, 0, 4), claim(3, 0, 4)];
        let conf = compute_confidence(&claims, &cfg());
        assert!(conf > 0.5);
    }

    #[test]
    fn extract_json_array_from_markdown() {
        let raw = "Here are the claims:\n```json\n[\"claim A\", \"claim B\"]\n```";
        let json = extract_json_array(raw).unwrap();
        let parsed: Vec<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, vec!["claim A", "claim B"]);
    }

    #[test]
    fn extract_json_array_truncated_response() {
        // Simulates LLM output truncated by max_tokens mid-string
        let raw = "```json\n[\"SHAP is based on Shapley values\", \"SHAP was introduced in 2017\", \"SHAP uses game theor";
        let json = extract_json_array(raw).unwrap();
        let parsed: Vec<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], "SHAP is based on Shapley values");
    }

    #[test]
    fn extract_json_array_clean_input() {
        let raw = r#"["claim 1", "claim 2", "claim 3"]"#;
        let json = extract_json_array(raw).unwrap();
        let parsed: Vec<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 3);
    }

    #[test]
    fn extract_json_array_with_unescaped_inner_quotes() {
        // LLM produces: ["The paper "Attention Is All You Need" introduced transformers"]
        let raw = r#"["The paper "Attention Is All You Need" introduced transformers"]"#;
        let json = extract_json_array(raw).unwrap();
        let parsed: Vec<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
        assert!(parsed[0].contains("Attention Is All You Need"));
    }

    #[test]
    fn extract_json_array_with_multiple_unescaped_quotes() {
        let raw = r#"["The paper "X" introduced "Y" concept", "Another claim here"]"#;
        let json = extract_json_array(raw).unwrap();
        let parsed: Vec<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 2);
        assert!(parsed[0].contains("\"X\""));
        assert_eq!(parsed[1], "Another claim here");
    }

    #[test]
    fn sanitize_json_inner_quotes_already_valid() {
        let input = r#"["valid", "already"]"#;
        // Should return None because the input already parses cleanly
        assert!(sanitize_json_inner_quotes(input).is_none() ||
                sanitize_json_inner_quotes(input) == Some(input.to_owned()));
    }

    #[test]
    fn is_metadata_filters_timestamps() {
        assert!(is_metadata_line("Published Time: 2024-01-03 12:44:19"));
        assert!(is_metadata_line("Last updated: March 2026"));
        assert!(is_metadata_line("Warning: This page maybe requiring CAPTCHA, please make sure you are authorized."));
    }

    #[test]
    fn is_metadata_allows_factual_claims() {
        assert!(!is_metadata_line("SHAP is based on Shapley values from cooperative game theory"));
        assert!(!is_metadata_line("Machine learning models can be interpreted using feature importance"));
    }

    #[test]
    fn heuristic_excludes_metadata() {
        let docs = vec![
            crate::reader::ReadResult {
                url: "https://example.com".into(),
                title: "Test".into(),
                content: "Published Time: 2024-01-03 12:44:19\n\
                          Warning: This page requires CAPTCHA verification\n\
                          # SHAP Guide\n\
                          SHAP is a method for explaining machine learning model predictions using Shapley values."
                    .into(),
                adapter: "test".into(),
            },
        ];
        let claims = extract_claims_heuristic("what is shap", &docs);
        assert!(!claims.iter().any(|c| c.contains("Published")));
        assert!(!claims.iter().any(|c| c.contains("CAPTCHA")));
        assert!(claims.iter().any(|c| c.contains("SHAP") || c.contains("shap")));
    }
}
