//! Human-like crawling behaviour: user-agent rotation, random delays, and
//! per-domain politeness (minimum gap between requests to the same host).
//!
//! Anti-detection philosophy:
//!  - Rotate through a large pool of real browser UAs so no single fingerprint.
//!  - Randomise inter-request delays to avoid timing-pattern detection.
//!  - Respect per-domain cooldowns to avoid triggering rate-limit responses.
//!  - Forward realistic browser Accept / Accept-Language / Accept-Encoding headers.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rand::Rng;
use tracing::debug;
use url::Url;

// ---------------------------------------------------------------------------
// User-agent pool — real desktop browser strings (Chrome, Firefox, Safari, Edge)
// ---------------------------------------------------------------------------

const USER_AGENT_POOL: &[&str] = &[
    // Chrome 122 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    // Chrome 122 — macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    // Firefox 123 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    // Firefox 123 — Linux
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    // Safari 17 — macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    // Edge 122 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    // Chrome 121 — Android
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.178 Mobile Safari/537.36",
    // Safari 17 — iPhone
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Mobile/15E148 Safari/604.1",
    // Chrome 120 — Windows (older, common fallback)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    // Firefox 120 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    // Chrome 119 — macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    // Brave (Chrome-based)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Brave/122",
];

// ---------------------------------------------------------------------------
// HumanBehavior
// ---------------------------------------------------------------------------

/// Controls timing and fingerprint randomisation for the crawler.
pub struct HumanBehavior {
    /// Minimum delay between requests to the SAME domain (ms).
    min_domain_gap_ms: u64,
    /// Jitter range added on top of the domain gap (ms).
    jitter_max_ms: u64,
    /// Per-domain last-access tracker (domain → last_request_time).
    domain_last: Mutex<HashMap<String, Instant>>,
}

impl HumanBehavior {
    pub fn new() -> Self {
        Self {
            min_domain_gap_ms: 500,
            jitter_max_ms: 1_500,
            domain_last: Mutex::new(HashMap::new()),
        }
    }

    /// Customise delay ranges (milliseconds).
    pub fn with_delays(mut self, min_domain_gap_ms: u64, jitter_max_ms: u64) -> Self {
        self.min_domain_gap_ms = min_domain_gap_ms;
        self.jitter_max_ms = jitter_max_ms;
        self
    }

    /// Pick a random user-agent string from the pool.
    pub fn user_agent(&self) -> &'static str {
        let idx = rand::thread_rng().gen_range(0..USER_AGENT_POOL.len());
        USER_AGENT_POOL[idx]
    }

    /// Sleep for a random jitter delay (no domain tracking).
    pub async fn delay(&self) {
        let ms = rand::thread_rng().gen_range(200..=self.jitter_max_ms);
        debug!(delay_ms = ms, "Human-like request delay");
        tokio::time::sleep(Duration::from_millis(ms)).await;
    }

    /// Enforce a minimum gap between requests to the same domain, plus jitter.
    /// Should be called just before making the HTTP request.
    pub async fn domain_delay(&self, url: &str) {
        let domain = extract_domain(url);
        let now = Instant::now();

        let sleep_ms = {
            let mut map = self.domain_last.lock().unwrap();
            let sleep = if let Some(last) = map.get(&domain) {
                let elapsed = now.duration_since(*last).as_millis() as u64;
                if elapsed < self.min_domain_gap_ms {
                    self.min_domain_gap_ms - elapsed
                } else {
                    0
                }
            } else {
                0
            };
            map.insert(domain.clone(), Instant::now());
            sleep
        };

        // Add random jitter on top
        let jitter = rand::thread_rng().gen_range(100..=self.jitter_max_ms);
        let total_ms = sleep_ms + jitter;
        if total_ms > 0 {
            debug!(domain, delay_ms = total_ms, "Domain politeness delay");
            tokio::time::sleep(Duration::from_millis(total_ms)).await;
        }
    }

    /// Return a set of realistic browser-like headers as a Vec<(key, value)>.
    pub fn browser_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"),
            ("Accept-Language", "en-US,en;q=0.9"),
            ("Accept-Encoding", "gzip, deflate, br"),
            ("DNT", "1"),
            ("Upgrade-Insecure-Requests", "1"),
            ("Sec-Fetch-Dest", "document"),
            ("Sec-Fetch-Mode", "navigate"),
            ("Sec-Fetch-Site", "none"),
            ("Sec-Fetch-User", "?1"),
            ("Cache-Control", "max-age=0"),
        ]
    }
}

impl Default for HumanBehavior {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the eTLD+1 host from a URL string for domain-level tracking.
fn extract_domain(url: &str) -> String {
    Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_owned()))
        .unwrap_or_else(|| url.to_owned())
}
