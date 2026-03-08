//! Tiered fetch strategy: static HTML → headless browser (optional feature).
//!
//! # Tier 1 — Static fetch (always active)
//! Uses `reqwest` with human-like headers. Fast, zero extra processes.
//! Suitable for most static / SSR pages.
//!
//! # Tier 2 — Headless browser (requires `--features headless`)
//! Drives Chrome via DevTools Protocol using `chromiumoxide`.
//! Activated automatically when page classification returns `JsHeavy`.
//! Falls back to static result when Chrome is not available.
//!
//! # Decision flow
//! ```text
//!  tiered_fetch(url)
//!    → static_fetch        → if PageType::JsHeavy? → browser_fetch
//!    → classify_page           ↓ failed / not enabled
//!    → return FetchRaw     ← static result                  
//! ```

use std::time::Duration;

use anyhow::{bail, Context, Result};
use scraper::{Html, Selector};
use tracing::{debug, info, warn};

use crate::captcha::{self, BypassStrategy, CaptchaKind, MAX_CAPTCHA_RETRIES};
use crate::extract::{classify_page, PageType};
use crate::human::HumanBehavior;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Raw result of fetching a single URL.
#[derive(Debug, Clone)]
pub struct FetchRaw {
    /// Final URL after any redirects.
    pub final_url: String,
    /// Page title extracted from `<title>`.
    pub title: String,
    /// Raw HTML body.
    pub raw_html: String,
    /// HTTP status code.
    pub status: u16,
    /// How many bytes were downloaded.
    pub bytes_downloaded: usize,
    /// Whether a headless browser was used.
    pub used_browser: bool,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Fetch `url` using the most appropriate strategy for the content type.
///
/// 1. Always attempts static fetch first (fast path).
/// 2. If the page is classified as `JsHeavy` AND the `headless` feature is
///    compiled in, retries with a headless Chrome instance.
/// 3. Returns the best result obtained, or propagates the error if all tiers fail.
pub async fn tiered_fetch(
    url: &str,
    user_agent: &str,
    timeout_secs: u64,
    max_bytes: usize,
) -> Result<FetchRaw> {
    debug!(url, "Tier 1: static fetch");
    let static_result = static_fetch(url, user_agent, timeout_secs, max_bytes).await;

    match static_result {
        Ok(raw) => {
            // Check for CAPTCHA before classification
            let captcha = captcha::detect(&raw.raw_html);
            if captcha.is_blocked() {
                warn!(url, captcha = captcha.label(), "CAPTCHA detected — skipping");
                bail!("CAPTCHA detected ({})", captcha.label());
            }

            let page_type = classify_page(&raw.raw_html);

            if page_type == PageType::JsHeavy {
                info!(url, "Tier 2: JS-heavy page detected, attempting browser fetch");
                match browser_fetch(url, timeout_secs).await {
                    Ok(browser_raw) => {
                        info!(url, "Browser fetch succeeded");
                        return Ok(browser_raw);
                    }
                    Err(e) => {
                        warn!(url, error = %e, "Browser fetch failed — using static result");
                    }
                }
            }

            Ok(raw)
        }
        Err(e) => {
            warn!(url, error = %e, "Static fetch failed");
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 1 — Static fetch (reqwest)
// ---------------------------------------------------------------------------

/// Fetch a URL using `reqwest` with human-like browser headers.
/// Returns raw HTML up to `max_bytes`.
pub async fn static_fetch(
    url: &str,
    user_agent: &str,
    timeout_secs: u64,
    max_bytes: usize,
) -> Result<FetchRaw> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .user_agent(user_agent)
        .redirect(reqwest::redirect::Policy::limited(8))
        .build()
        .context("Failed to build fetch HTTP client")?;

    let resp: reqwest::Response = client
        .get(url)
        .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        .header("Accept-Language", "en-US,en;q=0.9")
        .header("Accept-Encoding", "gzip, deflate, br")
        .header("DNT", "1")
        .header("Upgrade-Insecure-Requests", "1")
        .header("Sec-Fetch-Dest", "document")
        .header("Sec-Fetch-Mode", "navigate")
        .header("Sec-Fetch-Site", "none")
        .header("Sec-Fetch-User", "?1")
        .send()
        .await
        .context("Static fetch request failed")?;

    let final_url = resp.url().to_string();
    let status = resp.status().as_u16();

    if !resp.status().is_success() {
        bail!("HTTP {} from {}", status, url);
    }

    let raw_bytes = resp.bytes().await.context("Failed to read response body")?;
    let bytes_downloaded = raw_bytes.len();

    let clamped = if raw_bytes.len() > max_bytes {
        &raw_bytes[..max_bytes]
    } else {
        &raw_bytes[..]
    };

    let raw_html = String::from_utf8_lossy(clamped).into_owned();
    let title = extract_title(&raw_html);

    debug!(url, status, bytes_downloaded, "Static fetch complete");

    Ok(FetchRaw {
        final_url,
        title,
        raw_html,
        status,
        bytes_downloaded,
        used_browser: false,
    })
}

// ---------------------------------------------------------------------------
// Tier 2 — Headless browser
// ---------------------------------------------------------------------------

/// Attempt to render a JS-heavy page using a headless Chrome instance.
///
/// This function is a **no-op stub** when the `headless` Cargo feature is
/// not enabled. Enable it with:
///
/// ```sh
/// cargo build --features headless
/// ```
///
/// When enabled, requires Google Chrome or Chromium to be installed.
/// The binary path is read from `CHROME_PATH` env var
/// (default: searches `$PATH` for `google-chrome`, `chromium`, `chrome`).
#[cfg(not(feature = "headless"))]
pub async fn browser_fetch(_url: &str, _timeout_secs: u64) -> Result<FetchRaw> {
    bail!("Headless browser support not compiled in (build with --features headless)")
}

#[cfg(feature = "headless")]
pub async fn browser_fetch(url: &str, timeout_secs: u64) -> Result<FetchRaw> {
    use chromiumoxide::browser::{Browser, BrowserConfig};
    use futures::StreamExt;

    let chrome_path = std::env::var("CHROME_PATH").ok();

    let mut config_builder = BrowserConfig::builder()
        .with_head(false) // headless
        .arg("--no-sandbox")
        .arg("--disable-gpu")
        .arg("--disable-dev-shm-usage")
        .arg("--disable-blink-features=AutomationControlled") // anti-detection
        .arg("--window-size=1920,1080");

    if let Some(path) = chrome_path {
        config_builder = config_builder.chrome_executable(path);
    }

    let config = config_builder.build().context("Failed to build Chrome config")?;
    let (mut browser, mut handler) = Browser::launch(config)
        .await
        .context("Failed to launch Chrome")?;

    // Drive the browser event loop
    let _handler_task = tokio::spawn(async move {
        while handler.next().await.is_some() {}
    });

    let page = browser
        .new_page(url)
        .await
        .context("Failed to open page in Chrome")?;

    // Wait for load + small grace period for JS hydration
    tokio::time::sleep(Duration::from_millis(2_000)).await;

    let html = page
        .content()
        .await
        .context("Failed to get page content from Chrome")?;

    let title = page
        .title()
        .await
        .unwrap_or(None)
        .unwrap_or_default();

    browser.close().await.ok();

    let bytes_downloaded = html.len();

    Ok(FetchRaw {
        final_url: url.to_owned(),
        title,
        raw_html: html,
        status: 200,
        bytes_downloaded,
        used_browser: true,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the `<title>` text from raw HTML.
pub fn extract_title(html: &str) -> String {
    let doc = Html::parse_document(html);
    let sel = Selector::parse("title").unwrap();
    doc.select(&sel)
        .next()
        .map(|el| el.text().collect::<String>().trim().to_owned())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Resilient fetch — CAPTCHA bypass + UA rotation retry loop
// ---------------------------------------------------------------------------

/// Options that control the resilient fetch pipeline.
#[derive(Debug, Clone)]
pub struct FetchOptions {
    /// How many times to retry after a CAPTCHA or transient error.
    pub max_retries: u32,
    /// Whether to attempt the headless browser tier on CAPTCHA.
    pub allow_headless: bool,
    /// Whether to enable infinite-scroll accumulation on JS-heavy pages.
    pub enable_scroll: bool,
    /// Maximum scroll steps (passed through to `scroll::ScrollConfig`).
    pub max_scroll_steps: u32,
}

impl Default for FetchOptions {
    fn default() -> Self {
        Self {
            max_retries: MAX_CAPTCHA_RETRIES,
            allow_headless: cfg!(feature = "headless"),
            enable_scroll: false,
            max_scroll_steps: 8,
        }
    }
}

/// Fetch a URL with automatic CAPTCHA detection, bypass-strategy selection,
/// UA rotation, and optional infinite-scroll accumulation.
///
/// # Retry loop
/// ```text
///  attempt 0 → tiered_fetch (static or browser)
///    ├─ CAPTCHA detected → plan_bypass() → execute strategy → attempt 1
///    ├─ CAPTCHA detected again → plan_bypass(attempt=1) → attempt 2
///    └─ MAX_CAPTCHA_RETRIES exceeded → bail
/// ```
///
/// Successful results are returned immediately without further retries.
pub async fn tiered_fetch_resilient(
    url: &str,
    behavior: &HumanBehavior,
    timeout_secs: u64,
    max_bytes: usize,
    opts: &FetchOptions,
) -> Result<FetchRaw> {
    let mut last_err: Option<anyhow::Error> = None;

    for attempt in 0..=opts.max_retries {
        let ua = behavior.user_agent();

        debug!(url, attempt, ua, "Resilient fetch attempt");

        match tiered_fetch(url, ua, timeout_secs, max_bytes).await {
            Ok(raw) => {
                // Handle scroll accumulation for JS-heavy pages
                if opts.enable_scroll && raw.used_browser {
                    return enrich_with_scroll(raw, url, opts).await;
                }
                return Ok(raw);
            }
            Err(e) => {
                let msg = e.to_string();

                // Quick probe: is the page a CAPTCHA block?
                let kind = if msg.contains("CAPTCHA") {
                    // Re-check with the kind from the error message
                    if msg.contains("cloudflare") {
                        CaptchaKind::Cloudflare
                    } else if msg.contains("recaptcha") {
                        CaptchaKind::Recaptcha
                    } else if msg.contains("hcaptcha") {
                        CaptchaKind::HCaptcha
                    } else if msg.contains("bot_manager") {
                        CaptchaKind::BotManager
                    } else {
                        CaptchaKind::Generic
                    }
                } else {
                    // Not a CAPTCHA error — propagate immediately (network issue etc.)
                    return Err(e);
                };

                let strategy = captcha::plan_bypass(&kind, attempt, opts.allow_headless);
                warn!(url, attempt, bypass = strategy.label(), captcha = kind.label(), "CAPTCHA bypass");

                match strategy {
                    BypassStrategy::Skip => {
                        bail!("CAPTCHA bypass exhausted for {} ({})", url, kind.label());
                    }
                    BypassStrategy::RotateUserAgent => {
                        // Just loop — a new UA is picked on the next iteration
                        last_err = Some(e);
                    }
                    BypassStrategy::WaitAndRetry { delay_ms } => {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        last_err = Some(e);
                    }
                    BypassStrategy::UseHeadlessBrowser => {
                        // Force browser tier regardless of page classification
                        match browser_fetch(url, timeout_secs).await {
                            Ok(raw) => return Ok(raw),
                            Err(be) => {
                                warn!(url, error = %be, "Headless bypass failed");
                                last_err = Some(be);
                            }
                        }
                    }
                }
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("Resilient fetch failed after retries")))
}

/// Enrich a `FetchRaw` by running the scroll engine and returning the
/// accumulated HTML as the new `raw_html`.
async fn enrich_with_scroll(mut raw: FetchRaw, url: &str, opts: &FetchOptions) -> Result<FetchRaw> {
    use crate::scroll::{accumulate_scroll_content, ScrollConfig};

    let scroll_cfg = ScrollConfig {
        max_steps: opts.max_scroll_steps,
        ..Default::default()
    };

    match accumulate_scroll_content(url, &raw.raw_html, &scroll_cfg).await {
        Ok(result) => {
            info!(url, steps = result.steps_taken, words = result.word_count, "Scroll enrichment complete");
            raw.raw_html = result.accumulated_html;
            raw.bytes_downloaded = raw.raw_html.len();
        }
        Err(e) => {
            warn!(url, error = %e, "Scroll enrichment failed — using static HTML");
        }
    }

    Ok(raw)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_title_basic() {
        let html = "<html><head><title>  Hello World  </title></head></html>";
        assert_eq!(extract_title(html), "Hello World");
    }

    #[test]
    fn extract_title_missing() {
        let html = "<html><body><p>No title</p></body></html>";
        assert_eq!(extract_title(html), "");
    }

    #[test]
    fn extract_title_nested_whitespace() {
        let html = "<html><head><title>\n  Rust Programming\n  </title></head></html>";
        assert_eq!(extract_title(html), "Rust Programming");
    }

    #[test]
    fn fetch_options_defaults() {
        let opts = FetchOptions::default();
        assert!(opts.max_retries > 0);
        assert!(!opts.enable_scroll); // off by default
    }

    #[test]
    fn fetch_raw_fields_set_correctly() {
        let raw = FetchRaw {
            final_url: "https://example.com".into(),
            title: "Ex".into(),
            raw_html: "<p>body</p>".into(),
            status: 200,
            bytes_downloaded: 11,
            used_browser: false,
        };
        assert!(!raw.used_browser);
        assert_eq!(raw.status, 200);
        assert_eq!(raw.bytes_downloaded, 11);
    }
}
