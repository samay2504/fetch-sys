//! Reader / Scraper adapter layer.
//!
//! Converts URLs to LLM-friendly markdown text using a tiered approach:
//!   Tier 1 → Jina Reader (r.jina.ai) — external API, highest-quality Markdown
//!   Tier 2 → anytomd — pure-Rust local HTML→Markdown, no external service
//!   Tier 3 → Raw HTTP fetch + HTML scraper — plain-text last resort
//!
//! Adding a new reader adapter: implement [`ReaderAdapter`] and add it to
//! [`read_url`]'s priority list.

use async_trait::async_trait;
use rand::Rng;
use scraper::{Html, Selector};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};

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
            .expect("Failed to build Jina HTTP client");
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

        // Truncate if over budget
        let content = if text.len() > self.max_bytes {
            text[..self.max_bytes].to_owned()
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
// Tier 2 — anytomd (local HTML → Markdown, no external service required)
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
                "Mozilla/5.0 (compatible; fetchsys/0.1; +https://github.com/your-org/fetchsys)",
            )
            .build()
            .expect("Failed to build AnyToMd HTTP client");
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
                "Mozilla/5.0 (compatible; fetchsys/0.1; +https://github.com/your-org/fetchsys)",
            )
            .build()
            .expect("Failed to build raw HTTP client");
        Self { client, max_bytes }
    }
}

fn extract_text_from_html(html: &str) -> (String, String) {
    let document = Html::parse_document(html);

    // Extract title
    let title_sel = Selector::parse("title").unwrap();
    let title = document
        .select(&title_sel)
        .next()
        .map(|e| e.text().collect::<Vec<_>>().join("").trim().to_owned())
        .unwrap_or_default();

    // Build text from common content selectors (fallback to body)
    let content_selectors = ["article", "main", "[role='main']", "section", "body"];
    let text_selector = content_selectors
        .iter()
        .find_map(|s| {
            let sel = Selector::parse(s).ok()?;
            document.select(&sel).next().map(|_| sel)
        })
        .unwrap_or_else(|| Selector::parse("body").unwrap());

    let text = document
        .select(&text_selector)
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
// Public entry points
// ---------------------------------------------------------------------------

/// Read a single URL through the adapter waterfall.
///
/// Tier 1 → Jina Reader (external API, highest quality markdown)
/// Tier 2 → anytomd (pure-Rust local HTML→Markdown, no external service)
/// Tier 3 → Raw HTTP + scraper (plain-text fallback of last resort)
pub async fn read_url(url: &str, cfg: &ReaderConfig) -> anyhow::Result<ReadResult> {
    let timeout = Duration::from_secs(cfg.timeout_secs);
    let max_bytes = cfg.max_content_bytes;
    let retries = cfg.retries;

    // Tier 1: Jina Reader
    let jina = JinaReader::new(&cfg.jina_reader_base, cfg.jina_api_key.clone(), timeout, max_bytes);
    let result = with_retry("jina_reader", retries, || {
        let j = &jina;
        let u = url.to_owned();
        async move { j.read(&u).await }
    })
    .await;

    match result {
        Ok(r) if !r.content.is_empty() => return Ok(r),
        Ok(_) => warn!(%url, "Jina returned empty content; falling back to anytomd"),
        Err(e) => warn!(%url, error = %e, "Jina failed; falling back to anytomd"),
    }

    // Tier 2: anytomd — local HTML-to-Markdown (no external service)
    let atm = AnyToMdReader::new(timeout, max_bytes);
    let result = with_retry("anytomd", retries, || {
        let r = &atm;
        let u = url.to_owned();
        async move { r.read(&u).await }
    })
    .await;

    match result {
        Ok(r) if !r.content.is_empty() => return Ok(r),
        Ok(_) => warn!(%url, "anytomd returned empty content; falling back to raw HTTP"),
        Err(e) => warn!(%url, error = %e, "anytomd failed; falling back to raw HTTP"),
    }

    // Tier 3: Raw HTTP + scraper (last resort)
    let raw = RawHttpReader::new(timeout, max_bytes);
    with_retry("raw_http", retries, || {
        let r = &raw;
        let u = url.to_owned();
        async move { r.read(&u).await }
    })
    .await
}

/// Read the top-N URLs from search results in parallel and return collected
/// [`ReadResult`]s. URLs that fail all adapters are skipped with a warning.
///
/// A hard deadline (`reader_deadline_secs`) prevents a single slow URL from
/// stalling the entire pipeline. Results collected before the deadline are
/// returned; in-flight tasks are dropped.
pub async fn read_top(results: &[crate::search::SearchResult], cfg: &ReaderConfig) -> Vec<ReadResult> {
    use tokio::sync::mpsc;

    let deadline = Duration::from_secs(cfg.reader_deadline_secs);
    let (tx, mut rx) = mpsc::channel::<(usize, ReadResult)>(results.len().max(1));

    for (i, r) in results.iter().enumerate() {
        let url = r.url.clone();
        let cfg = cfg.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            match read_url(&url, &cfg).await {
                Ok(doc) => { let _ = tx.send((i, doc)).await; }
                Err(e) => warn!(rank = i + 1, error = %e, "Failed to read URL; skipping"),
            }
        });
    }
    // Drop our handle so rx closes when all senders are done
    drop(tx);

    let mut collected: Vec<(usize, ReadResult)> = Vec::new();

    // Collect results until the deadline or until all tasks complete
    let _ = tokio::time::timeout(deadline, async {
        while let Some(item) = rx.recv().await {
            collected.push(item);
        }
    })
    .await;

    if collected.len() < results.len() {
        warn!(
            collected = collected.len(),
            total = results.len(),
            deadline_secs = cfg.reader_deadline_secs,
            "Reader deadline reached; proceeding with partial results"
        );
    }

    // Return in original rank order
    collected.sort_by_key(|(i, _)| *i);
    collected.into_iter().map(|(_, doc)| doc).collect()
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
}
