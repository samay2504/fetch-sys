//! Infinite-scroll and lazy-load content accumulation.
//!
//! Many modern sites defer content until the user scrolls — intersection
//! observers, virtualized lists, or `window.onscroll` triggers.  This module
//! provides:
//!
//! ```text
//!  accumulate_scroll_content(url, initial_html, config)
//!      │
//!      ├─ [no browser feature]  returns initial_html unchanged (zero steps)
//!      ├─ [headless feature]    drives Chrome via chromiumoxide
//!      └─ [selenium feature]    drives Chrome/Firefox via Selenium WebDriver
//! ```
//!
//! ### Content-growth heuristic
//! After each scroll step the word-count of the full page HTML is sampled.
//! If growth drops below `ScrollConfig::growth_threshold` (default 3 %) for
//! two consecutive steps, scrolling stops and we consider the page exhausted.
//!
//! ### Deduplication
//! Content accumulated across steps is passed through [`extract_content`]
//! which performs DOM-density scoring, so duplicate nav/footer text is
//! naturally filtered before returning to the caller.

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use tracing::debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Full control over how the scroll engine behaves.
#[derive(Debug, Clone)]
pub struct ScrollConfig {
    /// Maximum scroll steps (each step = one `scrollBy` call + wait).
    /// Hard cap preventing runaway loops on extremely long pages.
    pub max_steps: u32,
    /// Milliseconds to wait after each scroll for lazy content to hydrate.
    /// Larger values → slower but better JS-rendered content capture.
    pub pause_ms: u64,
    /// Stop early when word-count growth per step falls below this fraction.
    /// `0.03` means: stop if < 3 % new words since the last step.
    pub growth_threshold: f64,
    /// Hard character limit for accumulated HTML. Prevents OOM on infinite-scroll feeds.
    pub max_chars: usize,
    /// Consecutive stale steps before early termination (growth < threshold).
    pub stale_steps_limit: u32,
}

impl Default for ScrollConfig {
    fn default() -> Self {
        Self {
            max_steps: 12,
            pause_ms: 900,
            growth_threshold: 0.03,
            max_chars: 500_000,
            stale_steps_limit: 2,
        }
    }
}

impl ScrollConfig {
    /// Quick preset for fast scraping (fewer waits, fewer steps).
    pub fn fast() -> Self {
        Self { max_steps: 5, pause_ms: 400, ..Default::default() }
    }

    /// Preset for thorough extraction of feed-style pages.
    pub fn thorough() -> Self {
        Self { max_steps: 25, pause_ms: 1_200, stale_steps_limit: 3, ..Default::default() }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Outcome of a scroll-extraction session.
#[derive(Debug, Clone)]
pub struct ScrollResult {
    /// Number of scroll steps actually taken.
    pub steps_taken: u32,
    /// Accumulated HTML after all scroll steps.
    pub accumulated_html: String,
    /// `true` if the page appeared to reach its bottom before `max_steps`.
    pub reached_bottom: bool,
    /// Approximate word count of the final HTML.
    pub word_count: usize,
}

impl ScrollResult {
    /// Build a trivial result wrapping `initial_html` (no scrolling occurred).
    pub fn passthrough(html: &str) -> Self {
        let wc = word_count(html);
        Self {
            steps_taken: 0,
            accumulated_html: html.to_owned(),
            reached_bottom: true,
            word_count: wc,
        }
    }
}

// ---------------------------------------------------------------------------
// JavaScript snippets used in browser control
// ---------------------------------------------------------------------------

/// Scroll down by exactly one viewport height (smooth for human-like behaviour).
pub const JS_SCROLL_DOWN: &str =
    "window.scrollBy({top: window.innerHeight, left: 0, behavior: 'smooth'}); \
     return document.documentElement.scrollHeight;";

/// Return the current scroll position as a fraction of total scrollable height.
pub const JS_SCROLL_FRACTION: &str =
    "var h = document.documentElement.scrollHeight - window.innerHeight; \
     return h > 0 ? window.scrollY / h : 1.0;";

/// Retrieve the full outer HTML of the document.
pub const JS_GET_HTML: &str = "return document.documentElement.outerHTML;";

/// Check whether the viewport is within 300 px of the page bottom.
pub const JS_NEAR_BOTTOM: &str =
    "return (window.scrollY + window.innerHeight) >= \
     (document.documentElement.scrollHeight - 300);";

/// Trigger lazy-load observers by simulating a resize event.
pub const JS_TRIGGER_LAZY: &str =
    "window.dispatchEvent(new Event('scroll')); \
     window.dispatchEvent(new Event('resize'));";

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Accumulate page content by simulating infinite scroll.
///
/// - When **no browser feature** is compiled: returns `initial_html` unchanged.
/// - When **`headless`** is enabled: uses `chromiumoxide` to drive Chrome.
/// - When only **`selenium`** is enabled: uses `thirtyfour` via WebDriver.
///
/// The caller is responsible for calling [`crate::extract::extract_content`]
/// on the returned HTML to get clean, deduplicated Markdown.
pub async fn accumulate_scroll_content(
    url: &str,
    initial_html: &str,
    _config: &ScrollConfig,
) -> Result<ScrollResult> {
    let _ = url; // suppress unused-variable warnings when no feature is active

    #[cfg(feature = "headless")]
    return scroll_headless(url, initial_html, _config).await;

    #[cfg(all(not(feature = "headless"), feature = "selenium"))]
    return scroll_selenium(url, initial_html, _config).await;

    // No browser available — passthrough
    #[cfg(not(any(feature = "headless", feature = "selenium")))]
    {
        debug!(url, "Scroll: no browser feature — returning initial HTML");
        Ok(ScrollResult::passthrough(initial_html))
    }
}

// ---------------------------------------------------------------------------
// Utility functions (always compiled so tests work without optional features)
// ---------------------------------------------------------------------------

/// Strip HTML tags and count whitespace-separated tokens.
///
/// Fast approximation; not a full text parser.
pub fn word_count(html: &str) -> usize {
    static TAGS: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]+>").unwrap());
    let stripped = TAGS.replace_all(html, " ");
    stripped.split_whitespace().count()
}

/// Detect whether content growth between two samples justifies continuing.
///
/// Returns `true` when next scroll is worthwhile.
pub fn should_continue_scrolling(prev: usize, current: usize, threshold: f64) -> bool {
    if prev == 0 {
        return true;
    }
    let growth = (current as f64 - prev as f64) / prev as f64;
    growth >= threshold
}

/// Estimate the percentage of the page that has been scrolled.
///
/// `scroll_y` and `doc_height` come from JavaScript evaluations.
pub fn scroll_progress_pct(scroll_y: f64, inner_height: f64, doc_height: f64) -> f64 {
    if doc_height <= inner_height {
        return 100.0;
    }
    ((scroll_y + inner_height) / doc_height * 100.0).min(100.0)
}

// ---------------------------------------------------------------------------
// headless (chromiumoxide) implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "headless")]
async fn scroll_headless(
    url: &str,
    _initial_html: &str,
    config: &ScrollConfig,
) -> Result<ScrollResult> {
    use std::time::Duration;
    use chromiumoxide::browser::{Browser, BrowserConfig};
    use futures::StreamExt;
    use anyhow::Context;
    use tracing::{info, warn};

    let chrome_path = std::env::var("CHROME_PATH").ok();

    let mut builder = BrowserConfig::builder()
        .with_head(false)
        .arg("--no-sandbox")
        .arg("--disable-gpu")
        .arg("--disable-dev-shm-usage")
        .arg("--disable-blink-features=AutomationControlled")
        .arg("--window-size=1920,1080");

    if let Some(path) = chrome_path {
        builder = builder.chrome_executable(path);
    }

    let cfg = builder.build().context("Chrome BrowserConfig build failed")?;
    let (mut browser, mut handler) = Browser::launch(cfg).await.context("Chrome launch failed")?;
    let _h = tokio::spawn(async move { while handler.next().await.is_some() {} });

    let page = browser.new_page(url).await.context("Open page failed")?;
    tokio::time::sleep(Duration::from_secs(2)).await;

    let mut prev_words = 0usize;
    let mut stale = 0u32;
    let mut steps_taken = 0u32;
    let mut reached_bottom = false;
    let mut accumulated_html = String::new();

    for step in 0..config.max_steps {
        // Trigger lazy-load observers
        page.evaluate(JS_TRIGGER_LAZY).await.ok();

        // Capture current HTML
        let html = page.content().await.unwrap_or_default();
        let wc = word_count(&html);
        debug!(url, step, words = wc, "Headless scroll step");

        accumulated_html = html;

        // Growth check
        if !should_continue_scrolling(prev_words, wc, config.growth_threshold) {
            stale += 1;
            if stale >= config.stale_steps_limit {
                info!(url, step, "Scroll stopped — content plateau");
                reached_bottom = true;
                break;
            }
        } else {
            stale = 0;
        }
        prev_words = wc;
        steps_taken = step + 1;

        if accumulated_html.len() >= config.max_chars {
            warn!(url, "Scroll max_chars reached");
            break;
        }

        page.evaluate(JS_SCROLL_DOWN).await.ok();
        tokio::time::sleep(Duration::from_millis(config.pause_ms)).await;

        // Near-bottom check
        let near_bottom = page
            .evaluate(JS_NEAR_BOTTOM)
            .await
            .ok()
            .and_then(|v| v.value().and_then(|j| j.as_bool()))
            .unwrap_or(false);

        if near_bottom {
            reached_bottom = true;
            // One final HTML capture
            accumulated_html = page.content().await.unwrap_or(accumulated_html);
            break;
        }
    }

    browser.close().await.ok();
    let wc = word_count(&accumulated_html);
    info!(url, steps_taken, reached_bottom, words = wc, "Headless scroll complete");
    Ok(ScrollResult { steps_taken, accumulated_html, reached_bottom, word_count: wc })
}

// ---------------------------------------------------------------------------
// selenium (thirtyfour) implementation
// ---------------------------------------------------------------------------

#[cfg(all(not(feature = "headless"), feature = "selenium"))]
async fn scroll_selenium(
    url: &str,
    _initial_html: &str,
    config: &ScrollConfig,
) -> Result<ScrollResult> {
    use std::time::Duration;
    use thirtyfour::prelude::*;
    use anyhow::Context;
    use tracing::{info, warn};

    let webdriver_url =
        std::env::var("WEBDRIVER_URL").unwrap_or_else(|_| "http://localhost:4444".into());

    let caps = DesiredCapabilities::chrome();
    let driver = WebDriver::new(&webdriver_url, caps)
        .await
        .context("Cannot connect to Selenium WebDriver")?;

    driver.goto(url).await.context("Navigate failed")?;
    tokio::time::sleep(Duration::from_secs(2)).await;

    let mut prev_words = 0usize;
    let mut stale = 0u32;
    let mut steps_taken = 0u32;
    let mut reached_bottom = false;
    let mut accumulated_html = String::new();

    for step in 0..config.max_steps {
        // Trigger lazy loaders
        driver
            .execute("window.dispatchEvent(new Event('scroll'));", vec![])
            .await
            .ok();

        accumulated_html = driver.source().await.unwrap_or_default();
        let wc = word_count(&accumulated_html);
        debug!(url, step, words = wc, "Selenium scroll step");

        if !should_continue_scrolling(prev_words, wc, config.growth_threshold) {
            stale += 1;
            if stale >= config.stale_steps_limit {
                reached_bottom = true;
                break;
            }
        } else {
            stale = 0;
        }
        prev_words = wc;
        steps_taken = step + 1;

        if accumulated_html.len() >= config.max_chars {
            break;
        }

        driver
            .execute("window.scrollBy(0, window.innerHeight);", vec![])
            .await
            .ok();
        tokio::time::sleep(Duration::from_millis(config.pause_ms)).await;

        // Near-bottom check
        let at_bottom_str: String = driver
            .execute(
                "return String((window.scrollY + window.innerHeight) \
                 >= document.documentElement.scrollHeight - 300);",
                vec![],
            )
            .await
            .ok()
            .and_then(|v| v.convert().ok())
            .unwrap_or_default();

        if at_bottom_str == "true" {
            reached_bottom = true;
            accumulated_html = driver.source().await.unwrap_or(accumulated_html);
            break;
        }
    }

    driver.quit().await.ok();
    let wc = word_count(&accumulated_html);
    info!(url, steps_taken, reached_bottom, words = wc, "Selenium scroll complete");
    Ok(ScrollResult { steps_taken, accumulated_html, reached_bottom, word_count: wc })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── word_count ──────────────────────────────────────────────────────────

    #[test]
    fn word_count_strips_tags() {
        let html = "<p>Hello <b>world</b></p>";
        assert_eq!(word_count(html), 2);
    }

    #[test]
    fn word_count_empty() {
        assert_eq!(word_count(""), 0);
    }

    #[test]
    fn word_count_plain_text() {
        let html = "one two three";
        assert_eq!(word_count(html), 3);
    }

    #[test]
    fn word_count_ignores_tag_attributes() {
        let html = r#"<div class="container"><span id="title">Rust is fast</span></div>"#;
        // "Rust", "is", "fast" → 3
        assert_eq!(word_count(html), 3);
    }

    // ── should_continue_scrolling ───────────────────────────────────────────

    #[test]
    fn should_continue_when_prev_zero() {
        assert!(should_continue_scrolling(0, 100, 0.03));
    }

    #[test]
    fn should_stop_when_no_growth() {
        // 100 → 101: only 1 % growth (below 3 % threshold)
        assert!(!should_continue_scrolling(100, 101, 0.03));
    }

    #[test]
    fn should_continue_with_significant_growth() {
        // 100 → 150: 50 % growth
        assert!(should_continue_scrolling(100, 150, 0.03));
    }

    #[test]
    fn should_stop_at_exactly_threshold() {
        // threshold 0.10: 100 → 110 is exactly 10 % → should continue (growth >= threshold)
        assert!(should_continue_scrolling(100, 110, 0.10));
        // 100 → 109: 9 % < 10 % → stop
        assert!(!should_continue_scrolling(100, 109, 0.10));
    }

    // ── scroll_progress_pct ─────────────────────────────────────────────────

    #[test]
    fn progress_pct_at_top() {
        // scroll_y=0, inner_height=768, doc_height=3000
        let pct = scroll_progress_pct(0.0, 768.0, 3000.0);
        assert!((pct - 25.6).abs() < 0.1);
    }

    #[test]
    fn progress_pct_at_bottom() {
        // scroll_y = doc_height - inner_height
        let pct = scroll_progress_pct(2232.0, 768.0, 3000.0);
        assert!((pct - 100.0).abs() < 0.1);
    }

    #[test]
    fn progress_pct_page_fits_viewport() {
        // doc_height <= inner_height → 100 %
        assert_eq!(scroll_progress_pct(0.0, 1000.0, 800.0), 100.0);
    }

    // ── ScrollResult passthrough ────────────────────────────────────────────

    #[test]
    fn passthrough_preserves_html() {
        let html = "<p>Test content</p>";
        let r = ScrollResult::passthrough(html);
        assert_eq!(r.accumulated_html, html);
        assert_eq!(r.steps_taken, 0);
        assert!(r.reached_bottom);
        assert_eq!(r.word_count, 2);
    }

    // ── ScrollConfig presets ────────────────────────────────────────────────

    #[test]
    fn fast_preset_fewer_steps() {
        let fast = ScrollConfig::fast();
        let default = ScrollConfig::default();
        assert!(fast.max_steps < default.max_steps);
        assert!(fast.pause_ms < default.pause_ms);
    }

    #[test]
    fn thorough_preset_more_steps() {
        let thorough = ScrollConfig::thorough();
        let default = ScrollConfig::default();
        assert!(thorough.max_steps > default.max_steps);
    }

    // ── JS constants non-empty ──────────────────────────────────────────────

    #[test]
    fn js_constants_non_empty() {
        assert!(!JS_SCROLL_DOWN.is_empty());
        assert!(!JS_NEAR_BOTTOM.is_empty());
        assert!(!JS_GET_HTML.is_empty());
        assert!(!JS_TRIGGER_LAZY.is_empty());
    }

    // ── accumulate_scroll_content (no-browser stub) ─────────────────────────

    #[tokio::test]
    #[cfg(not(any(feature = "headless", feature = "selenium")))]
    async fn accumulate_returns_initial_html_without_browser() {
        let html = "<p>Static page content</p>";
        let result =
            accumulate_scroll_content("https://example.com", html, &ScrollConfig::default())
                .await
                .unwrap();
        assert_eq!(result.accumulated_html, html);
        assert_eq!(result.steps_taken, 0);
        assert!(result.reached_bottom);
    }
}
