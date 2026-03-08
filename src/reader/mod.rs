//! Reader / Scraper adapter layer.
//!
//! Converts URLs to LLM-friendly markdown text using a **5-tier waterfall**:
//!
//!   Tier 1 → Jina Reader (r.jina.ai) — external API, highest-quality Markdown
//!             *(skipped automatically when no API key is set)*
//!   Tier 2 → Readability + htmd — Rust-native article extraction + HTML→Markdown
//!             *(free, local, no external service — replicates Jina locally)*
//!   Tier 3 → anytomd — pure-Rust document converter, broader format support
//!   Tier 4 → Firecrawl API (firecrawl.dev) — managed cloud reader
//!             *(skipped automatically when no API key is set)*
//!   Tier 5 → Raw HTTP fetch + HTML scraper — plain-text last resort
//!
//! ## Design rationale
//! - **Open-source first**: Tiers 2, 3, 5 are fully local with zero external deps
//! - **Paid optional**: Tiers 1, 4 are API-based (free tiers available)
//! - **Cross-platform**: Works on Windows, Linux, macOS; no native C deps
//! - **Production-grade**: Circuit-breaker on APIs, retries with jitter, deadline caps
//!
//! Adding a new reader adapter: implement [`ReaderAdapter`] and register it in
//! [`read_url`]'s priority list.

use async_trait::async_trait;
use once_cell::sync::Lazy;
use rand::Rng;
use scraper::{Html, Selector};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, info, warn};

use crate::config::ReaderConfig;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Content read from a single URL.
#[derive(Debug, Clone)]
pub struct ReadResult {
    /// Source URL.
    pub url: String,
    /// Page title (may be empty if extraction failed).
    pub title: String,
    /// Full LLM-friendly markdown text.
    pub content: String,
    /// Human-readable name of the adapter that produced this result.
    pub adapter: String,
}

impl ReadResult {
    /// Return a truncated snippet suitable for display / agent output.
    pub fn snippet(&self, max_chars: usize) -> String {
        let s = self.content.trim();
        if s.chars().count() <= max_chars {
            s.to_owned()
        } else {
            // Collect exactly max_chars chars, then break at a word boundary
            let truncated: String = s.chars().take(max_chars).collect();
            match truncated.rfind(|c: char| c.is_whitespace()) {
                Some(pos) => format!("{}…", &truncated[..pos]),
                None => format!("{truncated}…"),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reader adapter trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ReaderAdapter: Send + Sync {
    fn name(&self) -> &'static str;

    /// Fetch and convert `url` to a [`ReadResult`].
    async fn read(&self, url: &str) -> anyhow::Result<ReadResult>;
}

// ---------------------------------------------------------------------------
// Tier 1 — Jina Reader (r.jina.ai)
// ---------------------------------------------------------------------------

pub struct JinaReader {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    max_bytes: usize,
}

impl JinaReader {
    pub fn new(base_url: &str, api_key: Option<String>, timeout: Duration, max_bytes: usize) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent("fetchsys/0.1")
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_owned(),
            api_key,
            max_bytes,
        }
    }
}

#[async_trait]
impl ReaderAdapter for JinaReader {
    fn name(&self) -> &'static str {
        "jina_reader"
    }

    async fn read(&self, url: &str) -> anyhow::Result<ReadResult> {
        let reader_url = format!("{}/{}", self.base_url, url);
        debug!(%reader_url, "Jina Reader fetch");

        let mut req = self
            .client
            .get(&reader_url)
            .header("Accept", "text/plain, text/markdown")
            .header("X-Return-Format", "markdown");

        // Send API key if available (grants higher rate limits)
        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req
            .send()
            .await?
            .error_for_status()?;

        let text = resp.text().await?;

        // Truncate if over budget (UTF-8 safe: snap to char boundary)
        let content = if text.len() > self.max_bytes {
            let mut end = self.max_bytes;
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            text[..end].to_owned()
        } else {
            text
        };

        // Jina prepends a "Title:" line in its markdown output
        let title = content
            .lines()
            .find(|l| l.starts_with("Title:"))
            .map(|l| l.trim_start_matches("Title:").trim().to_owned())
            .unwrap_or_default();

        Ok(ReadResult {
            url: url.to_owned(),
            title,
            content,
            adapter: "jina_reader".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tier 2 — Readability + htmd (Rust-native article extraction → Markdown)
// ---------------------------------------------------------------------------
// Replicates what Jina Reader does but runs 100% locally:
// 1. Fetch raw HTML via reqwest (browser-like UA)
// 2. Extract main article content via Mozilla Readability algorithm
// 3. Convert cleaned HTML → Markdown via htmd (turndown.js-inspired)
//
// Zero API cost • zero rate limits • cross-platform • production-grade

/// User-agent pool for direct HTTP fetches — rotated to reduce bot detection.
const UA_POOL: &[&str] = &[
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
];

/// Pick a random user agent from the pool.
fn random_ua() -> &'static str {
    UA_POOL[rand::thread_rng().gen_range(0..UA_POOL.len())]
}

/// Build a reqwest client with browser-like headers for direct fetches.
fn build_browser_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .user_agent(random_ua())
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

pub struct ReadabilityReader {
    client: reqwest::Client,
    max_bytes: usize,
}

impl ReadabilityReader {
    pub fn new(timeout: Duration, max_bytes: usize) -> Self {
        Self {
            client: build_browser_client(timeout),
            max_bytes,
        }
    }
}

#[async_trait]
impl ReaderAdapter for ReadabilityReader {
    fn name(&self) -> &'static str {
        "readability"
    }

    async fn read(&self, url: &str) -> anyhow::Result<ReadResult> {
        use anyhow::Context;
        debug!(%url, "Readability+htmd fetch");

        let bytes = self
            .client
            .get(url)
            .header("Accept", "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8")
            .header("Accept-Language", "en-US,en;q=0.9")
            .header("Accept-Encoding", "gzip, deflate, br")
            .header("DNT", "1")
            .header("Sec-Fetch-Dest", "document")
            .header("Sec-Fetch-Mode", "navigate")
            .header("Sec-Fetch-Site", "none")
            .header("Sec-Fetch-User", "?1")
            .send()
            .await
            .context("Readability HTTP request failed")?
            .error_for_status()
            .context("Readability HTTP returned error status")?
            .bytes()
            .await
            .context("Readability body read failed")?;

        // Clamp to budget
        let data = if bytes.len() > self.max_bytes {
            &bytes[..self.max_bytes]
        } else {
            &bytes[..]
        };
        let html_str = String::from_utf8_lossy(data).into_owned();
        let url_owned = url.to_owned();

        // Run CPU-bound extraction off the async executor
        let result = tokio::task::spawn_blocking(move || {
            readability_extract(&html_str, &url_owned)
        })
        .await
        .context("Readability task panicked")?
        .context("Readability extraction failed")?;

        if result.content.trim().is_empty() {
            anyhow::bail!("Readability produced empty content");
        }

        Ok(result)
    }
}

/// Run Mozilla Readability extraction + htmd Markdown conversion.
fn readability_extract(html: &str, url: &str) -> anyhow::Result<ReadResult> {
    use readability::extractor;

    // readability::extractor::extract expects a Read impl
    let mut cursor = std::io::Cursor::new(html.as_bytes());
    let product = extractor::extract(&mut cursor, &url::Url::parse(url).unwrap_or_else(|_| {
        url::Url::parse("https://example.com").unwrap()
    }))
    .map_err(|e| anyhow::anyhow!("Readability extract failed: {e}"))?;

    // Convert cleaned article HTML → Markdown
    let markdown = htmd::convert(&product.content)
        .unwrap_or_else(|_| product.text.clone());

    let content = if markdown.trim().is_empty() {
        // Fallback to plain text if markdown conversion yields nothing
        product.text
    } else {
        markdown
    };

    Ok(ReadResult {
        url: url.to_owned(),
        title: product.title,
        content,
        adapter: "readability".into(),
    })
}

// ---------------------------------------------------------------------------
// Tier 3 — anytomd (local HTML → Markdown, no external service required)
// ---------------------------------------------------------------------------

/// Fetches raw HTML from the URL and converts it to clean Markdown using
/// the [`anytomd`] crate — a pure-Rust document converter with no external
/// runtime dependency. Unlike `JinaReader` this never calls an external API;
/// unlike `RawHttpReader` it produces properly-structured Markdown instead of
/// concatenated plain text.
pub struct AnyToMdReader {
    client: reqwest::Client,
    max_bytes: usize,
}

impl AnyToMdReader {
    pub fn new(timeout: Duration, max_bytes: usize) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent(
                "Mozilla/5.0 (compatible; fetchsys/0.1; +https://github.com/samay2504/fetchsys)",
            )
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { client, max_bytes }
    }
}

#[async_trait]
impl ReaderAdapter for AnyToMdReader {
    fn name(&self) -> &'static str {
        "anytomd"
    }

    async fn read(&self, url: &str) -> anyhow::Result<ReadResult> {
        use anyhow::Context;
        debug!(%url, "anytomd fetch");

        let bytes = self
            .client
            .get(url)
            .header("Accept", "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8")
            .send()
            .await
            .context("anytomd HTTP request failed")?
            .error_for_status()
            .context("anytomd HTTP returned error status")?
            .bytes()
            .await
            .context("anytomd body read failed")?;

        // Clamp to budget before handing off to the converter
        let data: Vec<u8> = if bytes.len() > self.max_bytes {
            bytes[..self.max_bytes].to_vec()
        } else {
            bytes.to_vec()
        };

        // `convert_bytes` is synchronous (CPU-bound); run off the async executor
        let result = tokio::task::spawn_blocking(move || {
            anytomd::convert_bytes(&data, "html", &anytomd::ConversionOptions::default())
        })
        .await
        .context("anytomd task panicked")?
        .context("anytomd conversion failed")?;

        if result.markdown.trim().is_empty() {
            anyhow::bail!("anytomd produced empty markdown");
        }

        Ok(ReadResult {
            url: url.to_owned(),
            title: result.title.unwrap_or_default(),
            content: result.markdown,
            adapter: "anytomd".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tier 3 — Raw HTTP fetch + scraper fallback
// ---------------------------------------------------------------------------

pub struct RawHttpReader {
    client: reqwest::Client,
    max_bytes: usize,
}

impl RawHttpReader {
    pub fn new(timeout: Duration, max_bytes: usize) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent(
                "Mozilla/5.0 (compatible; fetchsys/0.1; +https://github.com/samay2504/fetchsys)",
            )
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { client, max_bytes }
    }
}

fn extract_text_from_html(html: &str) -> (String, String) {
    static TITLE_SEL: Lazy<Selector> = Lazy::new(|| Selector::parse("title").unwrap());

    let document = Html::parse_document(html);

    // Extract title
    let title = document
        .select(&TITLE_SEL)
        .next()
        .map(|e| e.text().collect::<Vec<_>>().join("").trim().to_owned())
        .unwrap_or_default();

    // Build text from common content selectors (fallback to body)
    let content_selectors = ["article", "main", "[role='main']", "section"];
    let found_sel = content_selectors
        .iter()
        .find_map(|s| {
            let sel = Selector::parse(s).ok()?;
            document.select(&sel).next().map(|_| sel)
        });
    let body_sel = Selector::parse("body").unwrap();
    let text_selector = found_sel.as_ref().unwrap_or(&body_sel);

    let text = document
        .select(text_selector)
        .flat_map(|el| el.text())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    (title, text)
}

#[async_trait]
impl ReaderAdapter for RawHttpReader {
    fn name(&self) -> &'static str {
        "raw_http"
    }

    async fn read(&self, url: &str) -> anyhow::Result<ReadResult> {
        debug!(%url, "Raw HTTP fetch");
        let resp = self
            .client
            .get(url)
            .send()
            .await?
            .error_for_status()?;

        let bytes = resp.bytes().await?;
        let bytes = if bytes.len() > self.max_bytes {
            &bytes[..self.max_bytes]
        } else {
            &bytes
        };
        let html = String::from_utf8_lossy(bytes);
        let (title, content) = extract_text_from_html(&html);

        Ok(ReadResult {
            url: url.to_owned(),
            title,
            content,
            adapter: "raw_http".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tier 4 (optional) — Firecrawl API (managed cloud reader, open-source)
// ---------------------------------------------------------------------------
// Firecrawl (firecrawl.dev) is open-source (AGPL-3.0) and can be self-hosted.
// When `FIRECRAWL_API_KEY` is set, this tier provides high-quality LLM-ready
// markdown via the Firecrawl API. When not set, this tier is skipped.

pub struct FirecrawlReader {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl FirecrawlReader {
    pub fn new(api_key: String, base_url: &str, timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            api_key,
            base_url: base_url.trim_end_matches('/').to_owned(),
        }
    }
}

#[derive(serde::Deserialize)]
struct FirecrawlResponse {
    #[allow(dead_code)]
    success: Option<bool>,
    data: Option<FirecrawlData>,
}

#[derive(serde::Deserialize)]
struct FirecrawlData {
    markdown: Option<String>,
    metadata: Option<FirecrawlMetadata>,
}

#[derive(serde::Deserialize)]
struct FirecrawlMetadata {
    title: Option<String>,
}

#[async_trait]
impl ReaderAdapter for FirecrawlReader {
    fn name(&self) -> &'static str {
        "firecrawl"
    }

    async fn read(&self, url: &str) -> anyhow::Result<ReadResult> {
        use anyhow::Context;
        debug!(%url, "Firecrawl API fetch");

        let body = serde_json::json!({
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": true,
        });

        let resp: FirecrawlResponse = self
            .client
            .post(format!("{}/v1/scrape", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Firecrawl HTTP request failed")?
            .error_for_status()
            .context("Firecrawl HTTP returned error status")?
            .json()
            .await
            .context("Firecrawl response parse failed")?;

        let data = resp.data.context("Firecrawl returned no data")?;
        let content = data.markdown.unwrap_or_default();

        if content.trim().is_empty() {
            anyhow::bail!("Firecrawl returned empty markdown");
        }

        let title = data.metadata
            .and_then(|m| m.title)
            .unwrap_or_default();

        Ok(ReadResult {
            url: url.to_owned(),
            title,
            content,
            adapter: "firecrawl".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// API circuit breaker (shared between Jina and Firecrawl)
// ---------------------------------------------------------------------------
// After N consecutive failures an API tier is disabled for the rest of the run.
// This avoids wasting latency budget on APIs that are down or rate-limited.

/// Global circuit breaker for Jina Reader API.
static JINA_CIRCUIT_OPEN: AtomicBool = AtomicBool::new(false);
/// Global circuit breaker for Firecrawl API.
static FIRECRAWL_CIRCUIT_OPEN: AtomicBool = AtomicBool::new(false);

/// Number of consecutive failures before tripping the circuit.
const API_CIRCUIT_THRESHOLD: u32 = 2;

use std::sync::atomic::AtomicU32;
/// Jina consecutive failure counter.
static JINA_FAILURES: AtomicU32 = AtomicU32::new(0);
/// Firecrawl consecutive failure counter.
static FIRECRAWL_FAILURES: AtomicU32 = AtomicU32::new(0);

fn record_api_success(failures: &AtomicU32, circuit: &AtomicBool) {
    failures.store(0, Ordering::Relaxed);
    circuit.store(false, Ordering::Relaxed);
}

fn record_api_failure(failures: &AtomicU32, circuit: &AtomicBool, label: &str) {
    let prev = failures.fetch_add(1, Ordering::Relaxed);
    if prev + 1 >= API_CIRCUIT_THRESHOLD {
        warn!(%label, "API circuit breaker tripped — skipping for remaining URLs");
        circuit.store(true, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Retry helper
// ---------------------------------------------------------------------------

async fn with_retry<F, Fut>(label: &str, retries: u32, mut f: F) -> anyhow::Result<ReadResult>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<ReadResult>>,
{
    let mut attempt = 0u32;
    loop {
        match f().await {
            Ok(v) => return Ok(v),
            Err(e) if attempt < retries => {
                attempt += 1;
                let base = 2u64.pow(attempt);
                let jitter: u64 = rand::thread_rng().gen_range(0..300);
                let delay = Duration::from_millis(base * 200 + jitter);
                warn!(%label, attempt, ?delay, error = %e, "Reader transient error; retrying");
                sleep(delay).await;
            }
            Err(e) => return Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// URL normalisation for scraping
// ---------------------------------------------------------------------------

/// Rewrite URLs for better scraping compatibility **before** entering the
/// adapter waterfall.
///
/// Current rewrites:
/// - `www.reddit.com` / `reddit.com` → `old.reddit.com`
///   (static HTML, avoids 403 bot-protection on the React SPA)
fn normalize_scrape_url(url: &str) -> String {
    if let Ok(mut parsed) = url::Url::parse(url) {
        if let Some(host) = parsed.host_str().map(str::to_lowercase) {
            if host == "www.reddit.com" || host == "reddit.com" {
                let _ = parsed.set_host(Some("old.reddit.com"));
                debug!(original = %url, rewritten = %parsed, "Reddit URL rewritten to old.reddit.com");
                return parsed.to_string();
            }
        }
    }
    url.to_owned()
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Read a single URL through the **5-tier adapter waterfall**.
///
/// Tier 1 → Jina Reader  *(skipped if no API key or circuit open)*
/// Tier 2 → Readability + htmd  *(Rust-native, free, local)*
/// Tier 3 → anytomd  *(local HTML→Markdown converter)*
/// Tier 4 → Firecrawl API  *(skipped if no API key or circuit open)*
/// Tier 5 → Raw HTTP + scraper  *(plain-text last resort)*
pub async fn read_url(url: &str, cfg: &ReaderConfig) -> anyhow::Result<ReadResult> {
    // Normalise URLs for better scraping compatibility before entering the waterfall.
    let url = normalize_scrape_url(url);
    let url = url.as_str();

    let timeout = Duration::from_secs(cfg.timeout_secs);
    let max_bytes = cfg.max_content_bytes;
    let retries = cfg.retries;

    // Tier 1: Jina Reader (only if API key is set and circuit is healthy)
    let jina_enabled = cfg.jina_api_key.is_some()
        && !JINA_CIRCUIT_OPEN.load(Ordering::Relaxed);

    if jina_enabled {
        let jina = JinaReader::new(&cfg.jina_reader_base, cfg.jina_api_key.clone(), timeout, max_bytes);
        let result = with_retry("jina_reader", retries, || {
            let j = &jina;
            let u = url.to_owned();
            async move { j.read(&u).await }
        })
        .await;

        match result {
            Ok(r) if !r.content.is_empty() => {
                record_api_success(&JINA_FAILURES, &JINA_CIRCUIT_OPEN);
                return Ok(r);
            }
            Ok(_) => {
                record_api_failure(&JINA_FAILURES, &JINA_CIRCUIT_OPEN, "jina_reader");
                warn!(%url, "Jina returned empty; trying readability");
            }
            Err(e) => {
                record_api_failure(&JINA_FAILURES, &JINA_CIRCUIT_OPEN, "jina_reader");
                warn!(%url, error = %e, "Jina failed; trying readability");
            }
        }
    } else if cfg.jina_api_key.is_none() {
        debug!(%url, "Jina skipped (no API key)");
    } else {
        debug!(%url, "Jina skipped (circuit open)");
    }

    // Tier 2: Readability + htmd (Rust-native, free, local)
    let rdbl = ReadabilityReader::new(timeout, max_bytes);
    let result = with_retry("readability", retries, || {
        let r = &rdbl;
        let u = url.to_owned();
        async move { r.read(&u).await }
    })
    .await;

    match result {
        Ok(r) if !r.content.is_empty() => return Ok(r),
        Ok(_) => warn!(%url, "Readability returned empty; trying anytomd"),
        Err(e) => warn!(%url, error = %e, "Readability failed; trying anytomd"),
    }

    // Tier 3: anytomd — local HTML-to-Markdown (no external service)
    let atm = AnyToMdReader::new(timeout, max_bytes);
    let result = with_retry("anytomd", retries, || {
        let r = &atm;
        let u = url.to_owned();
        async move { r.read(&u).await }
    })
    .await;

    match result {
        Ok(r) if !r.content.is_empty() => return Ok(r),
        Ok(_) => warn!(%url, "anytomd returned empty; trying next tier"),
        Err(e) => warn!(%url, error = %e, "anytomd failed; trying next tier"),
    }

    // Tier 4: Firecrawl API (only if API key is set and circuit is healthy)
    let firecrawl_key = std::env::var("FIRECRAWL_API_KEY").ok()
        .or_else(|| cfg.firecrawl_api_key.clone());
    let firecrawl_enabled = firecrawl_key.is_some()
        && !FIRECRAWL_CIRCUIT_OPEN.load(Ordering::Relaxed);

    if firecrawl_enabled {
        let base = std::env::var("FIRECRAWL_BASE_URL")
            .unwrap_or_else(|_| cfg.firecrawl_base_url.clone()
                .unwrap_or_else(|| "https://api.firecrawl.dev".into()));
        let fc = FirecrawlReader::new(firecrawl_key.unwrap(), &base, timeout);
        let result = with_retry("firecrawl", retries, || {
            let r = &fc;
            let u = url.to_owned();
            async move { r.read(&u).await }
        })
        .await;

        match result {
            Ok(r) if !r.content.is_empty() => {
                record_api_success(&FIRECRAWL_FAILURES, &FIRECRAWL_CIRCUIT_OPEN);
                return Ok(r);
            }
            Ok(_) => {
                record_api_failure(&FIRECRAWL_FAILURES, &FIRECRAWL_CIRCUIT_OPEN, "firecrawl");
                warn!(%url, "Firecrawl returned empty; falling back to raw HTTP");
            }
            Err(e) => {
                record_api_failure(&FIRECRAWL_FAILURES, &FIRECRAWL_CIRCUIT_OPEN, "firecrawl");
                warn!(%url, error = %e, "Firecrawl failed; falling back to raw HTTP");
            }
        }
    }

    // Tier 5: Raw HTTP + scraper (last resort)
    let raw = RawHttpReader::new(timeout, max_bytes);
    with_retry("raw_http", retries, || {
        let r = &raw;
        let u = url.to_owned();
        async move { r.read(&u).await }
    })
    .await
}

/// Read the top-N URLs from search results using a two-phase hybrid strategy:
///
/// **Phase 1 — CrawlScheduler (primary, fast):**
///   All URLs are fetched concurrently with the `crawler::CrawlScheduler`
///   (tiered: static reqwest → headless browser for JS pages). Results are
///   relevance-ranked before being returned.
///
/// **Phase 2 — Reader enrichment (fallback for thin pages):**
///   Any pages whose crawled content is thin (< 100 words) are re-processed
///   through the 5-tier reader waterfall (Jina → Readability → anytomd →
///   Firecrawl → raw HTTP) to get higher-quality Markdown. This avoids wasting
///   API quota on pages the crawler handled well.
///
/// A hard `reader_deadline_secs` deadline caps the total phase-2 wait time.
pub async fn read_top(
    results: &[crate::search::SearchResult],
    reader_cfg: &ReaderConfig,
    crawler_cfg: &crate::config::CrawlerConfig,
    query: &str,
) -> Vec<ReadResult> {
    use std::collections::HashMap;
    use tokio::sync::mpsc;
    use crate::crawler::CrawlScheduler;

    // --- Phase 1: concurrent crawl -------------------------------------------
    let urls: Vec<String> = results.iter().map(|r| r.url.clone()).collect();
    let scheduler = CrawlScheduler::new(crawler_cfg.clone());
    let crawled = scheduler.crawl_urls(urls, query).await;

    let mut results_map: HashMap<String, ReadResult> = crawled
        .into_iter()
        .map(|p| {
            let adapter = p.adapter.clone();
            let rr = ReadResult {
                url: p.url.clone(),
                title: p.title,
                content: p.content,
                adapter,
            };
            (p.url, rr)
        })
        .collect();

    info!(
        fetched = results_map.len(),
        total = results.len(),
        "CrawlScheduler phase complete"
    );

    // --- Phase 2: Jina enrichment for thin pages ------------------------------
    let thin_urls: Vec<String> = results
        .iter()
        .filter(|r| {
            results_map
                .get(&r.url)
                .map(|rr| rr.content.split_whitespace().count() < 100)
                .unwrap_or(true)
        })
        .map(|r| r.url.clone())
        .collect();

    if !thin_urls.is_empty() {
        info!(count = thin_urls.len(), "Reader enrichment pass for thin pages");
        let deadline = Duration::from_secs(reader_cfg.reader_deadline_secs);
        let (tx, mut rx) = mpsc::channel::<ReadResult>(thin_urls.len().max(1));

        for url in thin_urls {
            let cfg = reader_cfg.clone();
            let tx = tx.clone();
            tokio::spawn(async move {
                match read_url(&url, &cfg).await {
                    Ok(doc) if !doc.content.is_empty() => {
                        let _ = tx.send(doc).await;
                    }
                    Ok(_) => warn!(%url, "Reader enrichment returned empty content"),
                    Err(e) => warn!(%url, error = %e, "Reader enrichment failed"),
                }
            });
        }
        drop(tx);

        let _ = tokio::time::timeout(deadline, async {
            while let Some(doc) = rx.recv().await {
                results_map.insert(doc.url.clone(), doc);
            }
        })
        .await;
    }

    let final_count = results_map.len();
    info!(final_count, "Reader complete");

    // Return in original search-rank order (preserves recall ordering)
    results
        .iter()
        .filter_map(|r| results_map.remove(&r.url))
        .collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_truncates_at_word_boundary() {
        let r = ReadResult {
            url: "https://example.com".into(),
            title: "Test".into(),
            content: "Hello world this is a long text string for testing".into(),
            adapter: "test".into(),
        };
        let s = r.snippet(20);
        // Compare char counts — '…' is 3 UTF-8 bytes but 1 char
        assert!(s.chars().count() <= 21, "snippet too long: {:?} ({} chars)", s, s.chars().count());
        assert!(s.ends_with('…'));
    }

    #[test]
    fn snippet_returns_full_when_short() {
        let r = ReadResult {
            url: "https://example.com".into(),
            title: "Test".into(),
            content: "Short".into(),
            adapter: "test".into(),
        };
        assert_eq!(r.snippet(100), "Short");
    }

    #[test]
    fn html_extraction_title_and_body() {
        let html = r#"<html><head><title>Test Page</title></head>
            <body><main><p>Main content here.</p></main></body></html>"#;
        let (title, text) = extract_text_from_html(html);
        assert_eq!(title, "Test Page");
        assert!(text.contains("Main content here."));
    }

    #[test]
    fn normalize_reddit_url() {
        assert_eq!(
            normalize_scrape_url("https://www.reddit.com/r/rust/comments/abc"),
            "https://old.reddit.com/r/rust/comments/abc"
        );
        assert_eq!(
            normalize_scrape_url("https://reddit.com/r/learnprogramming"),
            "https://old.reddit.com/r/learnprogramming"
        );
    }

    #[test]
    fn normalize_non_reddit_url_unchanged() {
        let url = "https://example.com/page";
        assert_eq!(normalize_scrape_url(url), url);
    }
}
