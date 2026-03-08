//! Multi-tier search engine client.
//!
//! Implements the waterfall / fallback strategy:
//!   Tier 1 (free/self-hosted) → Tier 2 (free-tier cloud) → Tier 3 (paid)
//!
//! Adding a new provider: implement [`SearchProvider`] and register it in
//! [`ProviderRegistry::build`].

use anyhow::{bail, Context};
use async_trait::async_trait;
use once_cell::sync::Lazy;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, sync::Arc, time::Duration};
use tokio::time::sleep;
use tracing::{debug, info, warn};
use url::Url;

use crate::config::SearchConfig;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// A single search result returned by any provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Canonical URL.
    pub url: String,
    /// Page title.
    pub title: String,
    /// Short excerpt / snippet.
    pub snippet: String,
    /// 1-based rank within this result set.
    pub rank: usize,
    /// Human-readable provider name (for provenance tracking).
    pub provider: String,
    /// Simple relevance quality score in [0.0, 1.0].
    pub quality: f64,
}

impl SearchResult {
    /// Attempt to return a normalised host for deduplication.
    pub fn host(&self) -> Option<String> {
        Url::parse(&self.url).ok().and_then(|u| {
            u.host_str().map(|h| h.trim_start_matches("www.").to_lowercase())
        })
    }
}

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait SearchProvider: Send + Sync {
    /// Provider identifier (used for provenance metadata).
    fn name(&self) -> &'static str;

    /// Submit a search query and return raw results.
    async fn search(&self, query: &str, max: usize) -> anyhow::Result<Vec<SearchResult>>;
}

// ---------------------------------------------------------------------------
// Tier 1 — SearXNG
// ---------------------------------------------------------------------------

pub struct SearXNGProvider {
    client: reqwest::Client,
    base_url: String,
}

impl SearXNGProvider {
    pub fn new(base_url: &str, timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent("fetchsys/0.1 (+https://github.com/your-org/fetchsys)")
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_owned(),
        }
    }
}

#[derive(Deserialize)]
struct SearXNGResponse {
    results: Vec<SearXNGResult>,
}

#[derive(Deserialize)]
struct SearXNGResult {
    url: String,
    title: String,
    content: Option<String>,
    score: Option<f64>,
}

#[async_trait]
impl SearchProvider for SearXNGProvider {
    fn name(&self) -> &'static str {
        "searxng"
    }

    async fn search(&self, query: &str, max: usize) -> anyhow::Result<Vec<SearchResult>> {
        let url = format!("{}/search", self.base_url);
        debug!(%url, "SearXNG search");

        let resp = self
            .client
            .get(&url)
            .query(&[
                ("q", query),
                ("format", "json"),
                ("language", "en"),
                ("pageno", "1"),
            ])
            .send()
            .await
            .context("SearXNG request failed")?
            .error_for_status()
            .context("SearXNG returned an error status")?;

        let body: SearXNGResponse = resp.json().await.context("SearXNG JSON parse failed")?;

        let results = body
            .results
            .into_iter()
            .take(max)
            .enumerate()
            .map(|(i, r)| SearchResult {
                url: r.url,
                title: r.title,
                snippet: r.content.unwrap_or_default(),
                rank: i + 1,
                provider: "searxng".into(),
                quality: r.score.unwrap_or(0.5).min(1.0).max(0.0),
            })
            .collect();

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tier 2 — DuckDuckGo (HTML scrape, no API key)
// ---------------------------------------------------------------------------

pub struct DuckDuckGoProvider {
    client: reqwest::Client,
}

impl DuckDuckGoProvider {
    pub fn new(timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { client }
    }

    /// DDG HTML results wrap destination URLs in `/l/?uddg=<encoded-url>` redirects.
    fn decode_ddg_href(href: &str) -> String {
        Url::parse(&format!("https://duckduckgo.com{href}"))
            .ok()
            .and_then(|u| u.query_pairs().find(|(k, _)| k == "uddg").map(|(_, v)| v.into_owned()))
            .unwrap_or_else(|| href.to_owned())
    }
}

#[async_trait]
impl SearchProvider for DuckDuckGoProvider {
    fn name(&self) -> &'static str {
        "duckduckgo"
    }

    async fn search(&self, query: &str, max: usize) -> anyhow::Result<Vec<SearchResult>> {
        use scraper::{Html, Selector};
        debug!("DuckDuckGo search");

        let html = self
            .client
            .get("https://html.duckduckgo.com/html/")
            .query(&[("q", query), ("kl", "en-us")])
            .header("Accept-Language", "en-US,en;q=0.9")
            .send()
            .await
            .context("DuckDuckGo request failed")?
            .error_for_status()
            .context("DuckDuckGo returned an error status")?
            .text()
            .await
            .context("DuckDuckGo response body read failed")?;

        static DDG_RESULT_SEL: Lazy<Selector> = Lazy::new(|| Selector::parse("div.result").unwrap());
        static DDG_TITLE_SEL: Lazy<Selector>  = Lazy::new(|| Selector::parse("a.result__a").unwrap());
        static DDG_SNIP_SEL: Lazy<Selector>   = Lazy::new(|| Selector::parse("a.result__snippet").unwrap());

        let document = Html::parse_document(&html);

        let results = document
            .select(&DDG_RESULT_SEL)
            .take(max)
            .enumerate()
            .filter_map(|(i, div)| {
                let a = div.select(&DDG_TITLE_SEL).next()?;
                let title = a.text().collect::<String>().trim().to_owned();
                let href  = a.value().attr("href").unwrap_or("").to_owned();
                if title.is_empty() || href.is_empty() {
                    return None;
                }
                let url     = Self::decode_ddg_href(&href);
                let snippet = div.select(&DDG_SNIP_SEL).next()
                    .map(|s| s.text().collect::<String>().trim().to_owned())
                    .unwrap_or_default();
                Some(SearchResult {
                    url, title, snippet,
                    rank: i + 1,
                    provider: "duckduckgo".into(),
                    quality: 0.6,
                })
            })
            .collect();

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tier 3 — Brave Search API
// ---------------------------------------------------------------------------

pub struct BraveProvider {
    client: reqwest::Client,
    api_key: String,
}

impl BraveProvider {
    pub fn new(api_key: String, timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent("fetchsys/0.1")
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { client, api_key }
    }
}

#[derive(Deserialize)]
struct BraveResponse {
    web: Option<BraveWeb>,
}

#[derive(Deserialize)]
struct BraveWeb {
    results: Vec<BraveResult>,
}

#[derive(Deserialize)]
struct BraveResult {
    url: String,
    title: String,
    description: Option<String>,
}

#[async_trait]
impl SearchProvider for BraveProvider {
    fn name(&self) -> &'static str {
        "brave"
    }

    async fn search(&self, query: &str, max: usize) -> anyhow::Result<Vec<SearchResult>> {
        debug!("Brave search");
        let resp = self
            .client
            .get("https://api.search.brave.com/res/v1/web/search")
            .header("Accept", "application/json")
            .header("Accept-Encoding", "gzip")
            .header("X-Subscription-Token", &self.api_key)
            .query(&[("q", query), ("count", &max.to_string())])
            .send()
            .await
            .context("Brave request failed")?
            .error_for_status()
            .context("Brave returned an error status")?;

        let body: BraveResponse = resp.json().await.context("Brave JSON parse failed")?;

        let results = body
            .web
            .map(|w| w.results)
            .unwrap_or_default()
            .into_iter()
            .take(max)
            .enumerate()
            .map(|(i, r)| SearchResult {
                url: r.url,
                title: r.title,
                snippet: r.description.unwrap_or_default(),
                rank: i + 1,
                provider: "brave".into(),
                quality: 0.7, // Brave doesn't expose a score; use a fixed baseline
            })
            .collect();

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tier 4 — Bing (HTML scrape, no API key, emergency fallback)
// ---------------------------------------------------------------------------

pub struct BingProvider {
    client: reqwest::Client,
}

/// Bing wraps outbound links in a tracking redirect:
/// `https://www.bing.com/ck/a?!&&p=...&u=a1<base64url>&ntb=1`
/// This function extracts and base64url-decodes the real URL from the `u` param.
fn decode_bing_url(href: &str) -> String {
    if !href.contains("bing.com/ck/") && !href.starts_with("/ck/") {
        return href.to_owned();
    }
    let base = if href.starts_with('/') {
        format!("https://www.bing.com{href}")
    } else {
        href.to_owned()
    };
    let Ok(parsed) = Url::parse(&base) else {
        return href.to_owned();
    };
    let Some(u_param) = parsed
        .query_pairs()
        .find(|(k, _)| k == "u")
        .map(|(_, v)| v.into_owned())
    else {
        return href.to_owned();
    };
    // Strip the leading "a1" prefix Bing prepends
    let encoded = u_param.strip_prefix("a1").unwrap_or(&u_param);
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .unwrap_or_else(|| href.to_owned())
}

impl BingProvider {
    pub fn new(timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { client }
    }
}

#[async_trait]
impl SearchProvider for BingProvider {
    fn name(&self) -> &'static str {
        "bing"
    }

    async fn search(&self, query: &str, max: usize) -> anyhow::Result<Vec<SearchResult>> {
        use scraper::{Html, Selector};
        debug!("Bing search");

        let html = self
            .client
            .get("https://www.bing.com/search")
            .query(&[("q", query), ("setlang", "en"), ("mkt", "en-US")])
            .header("Accept-Language", "en-US,en;q=0.9")
            .send()
            .await
            .context("Bing request failed")?
            .error_for_status()
            .context("Bing returned an error status")?
            .text()
            .await
            .context("Bing response body read failed")?;

        let document = Html::parse_document(&html);
        static BING_ITEM_SEL: Lazy<Selector>  = Lazy::new(|| Selector::parse("li.b_algo").unwrap());
        static BING_TITLE_SEL: Lazy<Selector> = Lazy::new(|| Selector::parse("h2 > a").unwrap());
        static BING_SNIP_SEL: Lazy<Selector>  = Lazy::new(|| Selector::parse("p").unwrap());

        let results = document
            .select(&BING_ITEM_SEL)
            .take(max)
            .enumerate()
            .filter_map(|(i, li)| {
                let a = li.select(&BING_TITLE_SEL).next()?;
                let title = a.text().collect::<String>().trim().to_owned();
                let raw_url = a.value().attr("href").unwrap_or("");
                let url = decode_bing_url(raw_url);
                if title.is_empty() || url.is_empty() {
                    return None;
                }
                let snippet = li.select(&BING_SNIP_SEL).next()
                    .map(|p| p.text().collect::<String>().trim().to_owned())
                    .unwrap_or_default();
                Some(SearchResult {
                    url, title, snippet,
                    rank: i + 1,
                    provider: "bing".into(),
                    quality: 0.55,
                })
            })
            .collect();

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tier 5 — Google (HTML scrape, no API key, emergency fallback)
// ---------------------------------------------------------------------------

pub struct GoogleProvider {
    client: reqwest::Client,
}

impl GoogleProvider {
    pub fn new(timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { client }
    }
}

/// Decode Google redirect URLs: /url?q=<real-url>&sa=...
fn decode_google_url(href: &str) -> String {
    // Google wraps results in /url?q=<encoded_url>
    if href.starts_with("/url?") || href.contains("google.com/url?") {
        let full = if href.starts_with('/') {
            format!("https://www.google.com{href}")
        } else {
            href.to_owned()
        };
        if let Ok(parsed) = Url::parse(&full) {
            if let Some(q) = parsed.query_pairs().find(|(k, _)| k == "q").map(|(_, v)| v.into_owned()) {
                return q;
            }
        }
    }
    href.to_owned()
}

#[async_trait]
impl SearchProvider for GoogleProvider {
    fn name(&self) -> &'static str {
        "google"
    }

    async fn search(&self, query: &str, max: usize) -> anyhow::Result<Vec<SearchResult>> {
        use scraper::{Html, Selector};
        debug!("Google search");

        let html = self
            .client
            .get("https://www.google.com/search")
            .query(&[("q", query), ("hl", "en"), ("num", &max.to_string())])
            .header("Accept-Language", "en-US,en;q=0.9")
            .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
            .send()
            .await
            .context("Google request failed")?
            .error_for_status()
            .context("Google returned an error status")?
            .text()
            .await
            .context("Google response body read failed")?;

        let document = Html::parse_document(&html);

        // Google organic result selectors (multiple patterns for robustness)
        static GOOGLE_RESULT_SEL: Lazy<Selector> = Lazy::new(|| Selector::parse("div.g").unwrap());
        static GOOGLE_TITLE_SEL: Lazy<Selector> = Lazy::new(|| Selector::parse("h3").unwrap());
        static GOOGLE_LINK_SEL: Lazy<Selector> = Lazy::new(|| Selector::parse("a").unwrap());
        static GOOGLE_SNIP_SEL: Lazy<Selector> = Lazy::new(|| {
            // Google uses various class names; try common ones
            Selector::parse("div.VwiC3b, span.aCOpRe, div[data-sncf], div.IsZvec").unwrap()
        });

        let results: Vec<SearchResult> = document
            .select(&GOOGLE_RESULT_SEL)
            .take(max)
            .enumerate()
            .filter_map(|(i, div)| {
                let a = div.select(&GOOGLE_LINK_SEL).next()?;
                let raw_href = a.value().attr("href").unwrap_or("");
                let url = decode_google_url(raw_href);

                // Skip internal Google links, images, etc.
                if url.starts_with('/') || url.starts_with('#') || url.contains("google.com/") {
                    return None;
                }
                // Validate it's a real URL
                if Url::parse(&url).is_err() {
                    return None;
                }

                let title = div.select(&GOOGLE_TITLE_SEL).next()
                    .map(|h3| h3.text().collect::<String>().trim().to_owned())
                    .unwrap_or_default();

                if title.is_empty() {
                    return None;
                }

                let snippet = div.select(&GOOGLE_SNIP_SEL).next()
                    .map(|s| s.text().collect::<String>().trim().to_owned())
                    .unwrap_or_default();

                Some(SearchResult {
                    url, title, snippet,
                    rank: i + 1,
                    provider: "google".into(),
                    quality: 0.65,
                })
            })
            .collect();

        if results.is_empty() {
            bail!("Google returned no parseable results (possible CAPTCHA or layout change)");
        }

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tier 6 — Serper.dev
// ---------------------------------------------------------------------------

pub struct SerperProvider {
    client: reqwest::Client,
    api_key: String,
}

impl SerperProvider {
    pub fn new(api_key: String, timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { client, api_key }
    }
}

#[derive(Deserialize)]
struct SerperResponse {
    organic: Vec<SerperResult>,
}

#[derive(Deserialize)]
struct SerperResult {
    link: String,
    title: String,
    snippet: Option<String>,
    position: Option<usize>,
}

#[async_trait]
impl SearchProvider for SerperProvider {
    fn name(&self) -> &'static str {
        "serper"
    }

    async fn search(&self, query: &str, max: usize) -> anyhow::Result<Vec<SearchResult>> {
        debug!("Serper search");
        let body = serde_json::json!({ "q": query, "num": max });
        let resp = self
            .client
            .post("https://google.serper.dev/search")
            .header("X-API-KEY", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Serper request failed")?
            .error_for_status()
            .context("Serper returned an error status")?;

        let parsed: SerperResponse = resp.json().await.context("Serper JSON parse failed")?;

        let results = parsed
            .organic
            .into_iter()
            .take(max)
            .enumerate()
            .map(|(i, r)| SearchResult {
                rank: r.position.unwrap_or(i + 1),
                url: r.link,
                title: r.title,
                snippet: r.snippet.unwrap_or_default(),
                provider: "serper".into(),
                quality: 0.8,
            })
            .collect();

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Provider registry
// ---------------------------------------------------------------------------

pub struct ProviderRegistry {
    /// Ordered list of providers (Tier 1 first).
    providers: Vec<Arc<dyn SearchProvider>>,
}

impl ProviderRegistry {
    pub fn build(cfg: &SearchConfig) -> Self {
        let timeout = Duration::from_secs(cfg.timeout_secs);
        let mut providers: Vec<Arc<dyn SearchProvider>> = Vec::new();

        for name in &cfg.providers {
            match name.to_lowercase().as_str() {
                "searxng" => {
                    // SearXNG is local (Docker) — use a short timeout so we fail
                    // fast when the container is not running instead of blocking
                    // the full provider timeout × retries.
                    let searxng_timeout = Duration::from_secs(timeout.as_secs().min(3));
                    providers.push(Arc::new(SearXNGProvider::new(&cfg.searxng_url, searxng_timeout)));
                }
                "duckduckgo" => {
                    providers.push(Arc::new(DuckDuckGoProvider::new(timeout)));
                }
                "brave" => {
                    match cfg.brave_api_key.as_deref().filter(|k| !k.is_empty()) {
                        Some(key) => providers.push(Arc::new(BraveProvider::new(key.to_owned(), timeout))),
                        None => warn!("Brave provider requested but BRAVE_API_KEY not set; skipping"),
                    }
                }
                "bing" => {
                    providers.push(Arc::new(BingProvider::new(timeout)));
                }
                "google" => {
                    providers.push(Arc::new(GoogleProvider::new(timeout)));
                }
                "serper" => {
                    match cfg.serper_api_key.as_deref().filter(|k| !k.is_empty()) {
                        Some(key) => providers.push(Arc::new(SerperProvider::new(key.to_owned(), timeout))),
                        None => warn!("Serper provider requested but SERPER_API_KEY not set; skipping"),
                    }
                }
                other => {
                    warn!(%other, "Unknown search provider name; skipping");
                }
            }
        }

        Self { providers }
    }

    pub fn providers(&self) -> &[Arc<dyn SearchProvider>] {
        &self.providers
    }
}

// ---------------------------------------------------------------------------
// Retry helper (exponential backoff + jitter)
// ---------------------------------------------------------------------------

async fn with_retry<F, Fut, T>(
    label: &str,
    retries: u32,
    mut f: F,
) -> anyhow::Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<T>>,
{
    let mut attempt = 0u32;
    loop {
        match f().await {
            Ok(v) => return Ok(v),
            Err(e) if attempt < retries => {
                attempt += 1;
                let base = 2u64.pow(attempt);
                let jitter: u64 = rand::thread_rng().gen_range(0..500);
                let delay = Duration::from_millis(base * 300 + jitter);
                warn!(%label, attempt, ?delay, error = %e, "Transient error; retrying");
                sleep(delay).await;
            }
            Err(e) => return Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Deduplication & scoring
// ---------------------------------------------------------------------------

/// Remove duplicate URLs (keep first occurrence by rank) and re-rank.
fn deduplicate(mut results: Vec<SearchResult>) -> Vec<SearchResult> {
    let mut seen_urls = HashSet::new();
    results.retain(|r| seen_urls.insert(r.url.clone()));
    results
        .into_iter()
        .enumerate()
        .map(|(i, mut r)| {
            r.rank = i + 1;
            r
        })
        .collect()
}

/// Compute a naive quality score for a result set.
fn result_set_quality(results: &[SearchResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let avg: f64 = results.iter().map(|r| r.quality).sum::<f64>() / results.len() as f64;
    avg
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Perform a multi-tier search with automatic fallback and a hard deadline.
///
/// - Attempts each provider in order (as configured by `providers` list).
/// - Accepts a tier's results if the quality score meets or exceeds `min_quality_score`.
/// - Falls back to next provider on failure or low quality.
/// - Returns the best available result set, deduplicated and re-ranked.
/// - A hard deadline of `3 × timeout_secs` caps total wall-clock time.
pub async fn multi_tier_search(
    query: &str,
    cfg: &SearchConfig,
) -> anyhow::Result<Vec<SearchResult>> {
    let registry = ProviderRegistry::build(cfg);
    let providers = registry.providers();

    if providers.is_empty() {
        bail!("No search providers configured. Enable searxng/duckduckgo/bing (no key), or set BRAVE_API_KEY / SERPER_API_KEY.");
    }

    // Hard deadline: 3× individual provider timeout caps overall search wall-clock
    let deadline = Duration::from_secs(cfg.timeout_secs.saturating_mul(3).max(15));

    let search_future = async {
        let mut best: Vec<SearchResult> = Vec::new();

        for provider in providers {
            let pname = provider.name();
            info!(%pname, "Attempting search provider");

            let result = with_retry(pname, cfg.retries, || {
                let p = provider.clone();
                let q = query.to_owned();
                let max = cfg.max_results;
                async move { p.search(&q, max).await }
            })
            .await;

            match result {
                Err(e) => {
                    warn!(%pname, error = %e, "Provider failed; falling back");
                    continue;
                }
                Ok(results) if results.is_empty() => {
                    warn!(%pname, "Provider returned no results; falling back");
                    continue;
                }
                Ok(results) => {
                    let q = result_set_quality(&results);
                    info!(%pname, quality = q, count = results.len(), "Provider returned results");

                    if best.is_empty() || q > result_set_quality(&best) {
                        best = results;
                    }

                    if q >= cfg.min_quality_score {
                        // Acceptable quality — stop cascading
                        break;
                    }
                    warn!(%pname, quality = q, min = cfg.min_quality_score, "Quality below threshold; trying next provider");
                }
            }
        }

        if best.is_empty() {
            bail!("All search providers exhausted with no usable results for query: {query}");
        }

        Ok(deduplicate(best))
    };

    match tokio::time::timeout(deadline, search_future).await {
        Ok(result) => result,
        Err(_) => {
            warn!(deadline_secs = deadline.as_secs(), "Search deadline exceeded");
            bail!("Search timed out after {}s", deadline.as_secs());
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(url: &str, quality: f64, rank: usize) -> SearchResult {
        SearchResult {
            url: url.into(),
            title: "Test".into(),
            snippet: "snippet".into(),
            rank,
            provider: "test".into(),
            quality,
        }
    }

    #[test]
    fn deduplication_removes_duplicates_and_reranks() {
        let results = vec![
            make_result("https://example.com/a", 0.9, 1),
            make_result("https://example.com/b", 0.8, 2),
            make_result("https://example.com/a", 0.7, 3), // duplicate
            make_result("https://example.com/c", 0.6, 4),
        ];
        let deduped = deduplicate(results);
        assert_eq!(deduped.len(), 3);
        // Ranks should be re-assigned 1,2,3
        assert_eq!(deduped[0].rank, 1);
        assert_eq!(deduped[1].rank, 2);
        assert_eq!(deduped[2].rank, 3);
    }

    #[test]
    fn quality_score_empty_returns_zero() {
        assert_eq!(result_set_quality(&[]), 0.0);
    }

    #[test]
    fn quality_score_averages_correctly() {
        let results = vec![
            make_result("https://a.com", 0.8, 1),
            make_result("https://b.com", 0.4, 2),
        ];
        let q = result_set_quality(&results);
        assert!((q - 0.6).abs() < 1e-9);
    }

    #[test]
    fn provider_registry_skips_missing_keys() {
        let cfg = SearchConfig {
            providers: vec!["brave".into(), "serper".into()],
            brave_api_key: None,
            serper_api_key: None,
            ..Default::default()
        };
        let registry = ProviderRegistry::build(&cfg);
        assert!(registry.providers().is_empty());
    }
}
