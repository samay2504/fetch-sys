//! Concurrent crawl scheduler with `tokio::task::JoinSet`-based worker pool.
//!
//! Replaces the linear per-URL reader loop with a bounded-concurrency engine:
//!
//! ```text
//!  CrawlScheduler::crawl_urls(urls)
//!      │
//!      ├─► Semaphore (max_concurrent permits)
//!      │        │
//!      │   ┌────▼──────────────────────────────────────────┐
//!      │   │  Worker (tokio::spawn)                        │
//!      │   │  1. Cache lookup → hit? return                │
//!      │   │  2. HumanBehavior::domain_delay()             │
//!      │   │  3. tiered_fetch_resilient()                  │
//!      │   │      ├─ static_fetch (reqwest)                │
//!      │   │      ├─ CAPTCHA? → plan_bypass → retry        │
//!      │   │      └─ browser_fetch (headless / selenium)   │
//!      │   │  4. classify_page()                           │
//!      │   │      └─ LoginRequired? → perform_static_login │
//!      │   │  5. extract_content()                         │
//!      │   │  6. score_result()                            │
//!      │   │  7. cache.insert()                            │
//!      │   └──────────────────────────────────────────────-┘
//!      │
//!      └─► JoinSet::join_next() → Vec<CrawledPage>
//! ```
//!
//! # Concurrency
//! The semaphore limits the number of in-flight HTTP requests so we don't
//! overwhelm targets or exhaust file descriptors. Default: 8 concurrent.

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{Mutex, Semaphore};
use tokio::task::JoinSet;
use tracing::{debug, info, warn};

use crate::config::CrawlerConfig;
use crate::extract::{classify_page, extract_content, PageType};
use crate::fetch::{tiered_fetch_resilient, FetchOptions};
use crate::human::HumanBehavior;
use crate::login::{perform_static_login, CredentialVault};
use crate::ranking::score_result;
use crate::storage::{CachedPage, PageCache};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// A page ready for the LLM fact-check pipeline.
#[derive(Debug, Clone)]
pub struct CrawledPage {
    /// Final URL (after redirects).
    pub url: String,
    /// Page title.
    pub title: String,
    /// Clean Markdown content extracted from the page.
    pub content: String,
    /// How the page was classified.
    pub page_type: PageType,
    /// Composite relevance score [0, 1].
    pub score: f64,
    /// Fetch latency in milliseconds.
    pub fetch_ms: u64,
    /// Which fetch adapter was used.
    pub adapter: String,
}

// ---------------------------------------------------------------------------
// CrawlScheduler
// ---------------------------------------------------------------------------

/// Orchestrates concurrent page fetching with caching, politeness,
/// intelligent content extraction, CAPTCHA bypass, and login-wall handling.
pub struct CrawlScheduler {
    config: CrawlerConfig,
    cache: Arc<PageCache>,
    behavior: Arc<HumanBehavior>,
    /// Shared credential vault for login-wall automation.
    vault: Arc<Mutex<CredentialVault>>,
}

impl CrawlScheduler {
    pub fn new(config: CrawlerConfig) -> Self {
        let behavior = Arc::new(
            HumanBehavior::new()
                .with_delays(config.domain_gap_ms, config.jitter_max_ms),
        );
        Self {
            cache: Arc::new(PageCache::new(config.cache_ttl_secs)),
            behavior,
            vault: Arc::new(Mutex::new(CredentialVault::new())),
            config,
        }
    }

    /// Attach a pre-populated credential vault (loaded from file or env).
    pub fn with_vault(mut self, vault: CredentialVault) -> Self {
        self.vault = Arc::new(Mutex::new(vault));
        self
    }

    /// Crawl all URLs concurrently.
    ///
    /// - Respects `config.max_concurrent` (Semaphore).
    /// - Returns all successfully crawled pages sorted by relevance score.
    /// - Failed URLs are logged as warnings and omitted from results.
    pub async fn crawl_urls(&self, urls: Vec<String>, query: &str) -> Vec<CrawledPage> {
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent));
        let mut join_set: JoinSet<Option<CrawledPage>> = JoinSet::new();

        for url_str in urls {
            let sem = Arc::clone(&semaphore);
            let cache = Arc::clone(&self.cache);
            let behavior = Arc::clone(&self.behavior);
            let vault = Arc::clone(&self.vault);
            let timeout = self.config.fetch_timeout_secs;
            let max_bytes = self.config.max_content_bytes;
            let retries = self.config.retries;
            let query_owned = query.to_owned();

            join_set.spawn(async move {
                // Hard per-URL deadline: fetch_timeout + 30s buffer for login/extraction
                let per_url_deadline = std::time::Duration::from_secs(timeout + 30);
                match tokio::time::timeout(per_url_deadline, async {
                // Acquire concurrency permit
                let _permit = match sem.acquire().await {
                    Ok(p) => p,
                    Err(_) => {
                        warn!(url = %url_str, "Semaphore closed — skipping");
                        return None;
                    }
                };

                // Cache hit
                if let Some(cached) = cache.get(&url_str) {
                    debug!(url = %url_str, "Serving from cache");
                    let score = score_result(&cached.url, &cached.markdown, &query_owned);
                    return Some(CrawledPage {
                        url: cached.url,
                        title: cached.title,
                        content: cached.markdown,
                        page_type: PageType::Static, // cached — type already resolved
                        score,
                        fetch_ms: 0,
                        adapter: cached.adapter,
                    });
                }

                // Per-domain politeness delay
                behavior.domain_delay(&url_str).await;

                let fetch_opts = FetchOptions {
                    max_retries: retries,
                    allow_headless: cfg!(feature = "headless"),
                    enable_scroll: false,
                    max_scroll_steps: 8,
                };
                let t0 = Instant::now();

                let raw = match tiered_fetch_resilient(
                    &url_str,
                    &behavior,
                    timeout,
                    max_bytes,
                    &fetch_opts,
                ).await {
                    Ok(r) => r,
                    Err(e) => {
                        warn!(url = %url_str, error = %e, "Resilient fetch failed");
                        return None;
                    }
                };

                let fetch_ms = t0.elapsed().as_millis() as u64;
                let page_type = classify_page(&raw.raw_html);

                // Login-wall handling — re-fetch with session cookies on success
                let final_html;
                if page_type == PageType::LoginRequired {
                    info!(url = %url_str, "Login wall detected — attempting static login");
                    let domain = crate::login::domain_from_url(&url_str);
                    let cred = vault.lock().await.get_or_generate(&domain);
                    match perform_static_login(&url_str, &cred, behavior.user_agent(), timeout).await {
                        Ok((outcome, session)) if outcome.is_success() && !session.cookies.is_empty() => {
                            info!(url = %url_str, cookies = session.cookies.len(), "Login succeeded — re-fetching with session");
                            // Re-fetch the page with authenticated cookies
                            let client = reqwest::Client::builder()
                                .timeout(std::time::Duration::from_secs(timeout))
                                .user_agent(behavior.user_agent())
                                .build()
                                .unwrap_or_else(|_| reqwest::Client::new());
                            match client
                                .get(&url_str)
                                .header("Cookie", session.cookie_header())
                                .send()
                                .await
                            {
                                Ok(resp) => match resp.text().await {
                                    Ok(body) => {
                                        info!(url = %url_str, body_len = body.len(), "Re-fetch with cookies succeeded");
                                        final_html = body;
                                    }
                                    Err(e) => {
                                        warn!(url = %url_str, error = %e, "Re-fetch body read failed; using original");
                                        final_html = raw.raw_html.clone();
                                    }
                                },
                                Err(e) => {
                                    warn!(url = %url_str, error = %e, "Re-fetch with cookies failed; using original");
                                    final_html = raw.raw_html.clone();
                                }
                            }
                        }
                        Ok((outcome, _)) => {
                            warn!(url = %url_str, outcome = ?outcome, "Login failed");
                            final_html = raw.raw_html.clone();
                        }
                        Err(e) => {
                            warn!(url = %url_str, error = %e, "Login error");
                            final_html = raw.raw_html.clone();
                        }
                    }
                } else {
                    final_html = raw.raw_html.clone();
                }

                let content = extract_content(&final_html, &url_str);
                let score = score_result(&url_str, &content, &query_owned);
                let adapter = if raw.used_browser { "browser".to_owned() } else { "static".to_owned() };

                info!(
                    url = %url_str,
                    fetch_ms,
                    page_type = ?page_type,
                    score,
                    words = content.split_whitespace().count(),
                    adapter = %adapter,
                    "Crawled"
                );

                cache.insert(
                    url_str.clone(),
                    CachedPage {
                        url: url_str.clone(),
                        title: raw.title.clone(),
                        markdown: content.clone(),
                        adapter: adapter.clone(),
                    },
                );

                Some(CrawledPage {
                    url: url_str,
                    title: raw.title,
                    content,
                    page_type,
                    score,
                    fetch_ms,
                    adapter,
                })
                }).await {
                    Ok(result) => result,
                    Err(_) => {
                        warn!("Per-URL deadline exceeded");
                        None
                    }
                }
            });
        }

        // Collect results
        let mut pages = Vec::new();
        while let Some(outcome) = join_set.join_next().await {
            match outcome {
                Ok(Some(page)) => pages.push(page),
                Ok(None) => {}
                Err(e) => warn!(error = %e, "Worker task panicked"),
            }
        }

        // Sort by relevance score (descending)
        pages.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        pages
    }

    /// Evict expired entries from the page cache.
    pub fn evict_cache(&self) {
        self.cache.evict_expired();
        debug!(remaining = self.cache.len(), "Cache eviction complete");
    }
}

// ---------------------------------------------------------------------------
// Convenience: convert CrawledPage → reader::ReadResult
// ---------------------------------------------------------------------------

impl From<CrawledPage> for crate::reader::ReadResult {
    fn from(p: CrawledPage) -> Self {
        crate::reader::ReadResult {
            url: p.url,
            title: p.title,
            content: p.content,
            adapter: p.adapter,
        }
    }
}
