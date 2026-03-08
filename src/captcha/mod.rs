//! CAPTCHA and bot-challenge detection heuristics.
//!
//! Uses DOM pattern matching to classify pages that require human verification.
//! When a CAPTCHA is detected the crawler skips the page rather than hanging.
//!
//! Detection signals (roughly in precision order):
//!  - Cloudflare challenge page markers (`cf-challenge`, `cf_clearance`)
//!  - Google reCAPTCHA (`g-recaptcha`, `recaptcha`)
//!  - hCaptcha (`h-captcha`, `hcaptcha`)
//!  - Generic challenge forms (`challenge-form`, `cf-browser-verification`)
//!  - Datadome / PerimeterX / Akamai bot-manager markers
//!  - Distil Networks / Imperva
//!  - Page title heuristics ("Just a moment", "Access denied", "Verify you are human")

use once_cell::sync::Lazy;
use regex::Regex;

// ---------------------------------------------------------------------------
// Detection type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptchaKind {
    /// No CAPTCHA detected — page is safe to parse.
    None,
    /// Cloudflare challenge (JS interstitial or Turnstile).
    Cloudflare,
    /// Google reCAPTCHA v2 / v3 / Enterprise.
    Recaptcha,
    /// hCaptcha.
    HCaptcha,
    /// Datadome, PerimeterX, Akamai Bot Manager, or similar.
    BotManager,
    /// Generic challenge page (challenge-form, verify-human, etc.).
    Generic,
}

impl CaptchaKind {
    pub fn is_blocked(&self) -> bool {
        !matches!(self, CaptchaKind::None)
    }

    pub fn label(&self) -> &'static str {
        match self {
            CaptchaKind::None => "none",
            CaptchaKind::Cloudflare => "cloudflare",
            CaptchaKind::Recaptcha => "recaptcha",
            CaptchaKind::HCaptcha => "hcaptcha",
            CaptchaKind::BotManager => "bot_manager",
            CaptchaKind::Generic => "generic",
        }
    }
}

// ---------------------------------------------------------------------------
// Compiled patterns
// ---------------------------------------------------------------------------

static CF_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)(cf-challenge|cf_clearance|__cf_bm|cfduid|cf-ray|cloudflare-nginx|just a moment\.{3}|checking your browser)"#).unwrap()
});

static RECAPTCHA_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)(g-recaptcha|recaptcha\.net|google\.com/recaptcha|grecaptcha)"#).unwrap()
});

static HCAPTCHA_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)(h-captcha|hcaptcha\.com|data-sitekey)"#).unwrap()
});

static BOT_MANAGER_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)(datadome|perimeterx|px\.js|_pxAppId|akamai-bot|botd\.js|imperva|distil|incapsula|reese84)"#).unwrap()
});

static GENERIC_CHALLENGE_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)(challenge-form|verify[- ]you|verify.*human|browser.*verification|access denied|403 forbidden|please wait|ddos protection)"#).unwrap()
});

static TITLE_BLOCKED_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)<title[^>]*>\s*(just a moment|access denied|403 forbidden|rate limited|attention required|ddos protection|are you human|security check)"#).unwrap()
});

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Inspect raw HTML and return the detected CAPTCHA/challenge kind.
/// Runs all patterns and returns the highest-priority match.
pub fn detect(html: &str) -> CaptchaKind {
    // Cloudflare first — most common, very distinctive
    if CF_PATTERN.is_match(html) || TITLE_BLOCKED_PATTERN.is_match(html) {
        // Disambiguate CF vs generic using CF-specific strings
        if html.contains("cf-challenge") || html.contains("cf_clearance") || html.contains("__cf_bm") {
            return CaptchaKind::Cloudflare;
        }
    }

    if RECAPTCHA_PATTERN.is_match(html) {
        return CaptchaKind::Recaptcha;
    }

    if HCAPTCHA_PATTERN.is_match(html) {
        return CaptchaKind::HCaptcha;
    }

    if BOT_MANAGER_PATTERN.is_match(html) {
        return CaptchaKind::BotManager;
    }

    if GENERIC_CHALLENGE_PATTERN.is_match(html) || TITLE_BLOCKED_PATTERN.is_match(html) {
        return CaptchaKind::Generic;
    }

    CaptchaKind::None
}

/// Quick check: is this HTML likely a CAPTCHA/challenge page?
pub fn is_blocked(html: &str) -> bool {
    detect(html).is_blocked()
}

// ---------------------------------------------------------------------------
// Bypass strategy
// ---------------------------------------------------------------------------

/// The recommended strategy to attempt when a CAPTCHA is encountered.
///
/// Strategies are ordered from cheapest (no process spawn) to most expensive.
/// The crawler selects the right tier based on the `CaptchaKind` severity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BypassStrategy {
    /// Rotate the user-agent string and retry immediately.
    /// Effective against heuristic UA-based blocks.
    RotateUserAgent,
    /// Wait `delay_ms` milliseconds then retry with a different UA.
    /// Effective against token-bucket / rate-limit challenges.
    WaitAndRetry { delay_ms: u64 },
    /// Fetch through a headless browser to execute the JS challenge.
    /// Required for Cloudflare JS Interstitial, reCAPTCHA Invisible v3.
    UseHeadlessBrowser,
    /// Skip the URL entirely — challenge cannot be solved without external service.
    Skip,
}

impl BypassStrategy {
    /// Human-readable label for logging.
    pub fn label(&self) -> &'static str {
        match self {
            BypassStrategy::RotateUserAgent => "rotate_ua",
            BypassStrategy::WaitAndRetry { .. } => "wait_retry",
            BypassStrategy::UseHeadlessBrowser => "headless",
            BypassStrategy::Skip => "skip",
        }
    }
}

/// Choose the cheapest viable bypass strategy for the given `CaptchaKind`.
///
/// Callers can enforce a maximum allowed cost (e.g. disable headless) by
/// checking the returned variant before acting.
///
/// | Kind | Strategy |
/// |------|---------|
/// | None | — (not called in practice) |
/// | Cloudflare | HeadlessBrowser (JS challenge), else Skip |
/// | Recaptcha | Skip (requires ML solver or paid service) |
/// | HCaptcha | Skip (same reason) |
/// | BotManager | WaitAndRetry → RotateUserAgent |
/// | Generic | WaitAndRetry |
pub fn plan_bypass(kind: &CaptchaKind, attempt: u32, headless_available: bool) -> BypassStrategy {
    match kind {
        CaptchaKind::None => BypassStrategy::RotateUserAgent,

        CaptchaKind::Cloudflare => {
            if attempt == 0 {
                // First attempt: try with a fresh UA + delay (sometimes works)
                BypassStrategy::WaitAndRetry { delay_ms: 3_000 }
            } else if headless_available {
                // Second attempt: real browser to solve JS challenge
                BypassStrategy::UseHeadlessBrowser
            } else {
                BypassStrategy::Skip
            }
        }

        // reCAPTCHA / hCaptcha need a vision model — skip unless we add an OCR solver
        CaptchaKind::Recaptcha | CaptchaKind::HCaptcha => BypassStrategy::Skip,

        CaptchaKind::BotManager => {
            if attempt == 0 {
                BypassStrategy::WaitAndRetry { delay_ms: 2_000 }
            } else if attempt == 1 {
                BypassStrategy::RotateUserAgent
            } else {
                BypassStrategy::Skip
            }
        }

        CaptchaKind::Generic => {
            if attempt == 0 {
                BypassStrategy::WaitAndRetry { delay_ms: 1_500 }
            } else {
                BypassStrategy::RotateUserAgent
            }
        }
    }
}

/// Maximum useful retry attempts before giving up on a blocked URL.
pub const MAX_CAPTCHA_RETRIES: u32 = 3;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Detection ────────────────────────────────────────────────────────────

    #[test]
    fn detects_cloudflare_by_cf_challenge() {
        let html = r#"<div class="cf-challenge">Checking...</div>"#;
        assert_eq!(detect(html), CaptchaKind::Cloudflare);
    }

    #[test]
    fn detects_cloudflare_by_clearance_cookie() {
        let html = r#"<script>document.cookie = "cf_clearance=abc";</script>"#;
        assert_eq!(detect(html), CaptchaKind::Cloudflare);
    }

    #[test]
    fn detects_recaptcha_by_script() {
        let html = r#"<script src="https://www.google.com/recaptcha/api.js"></script>"#;
        assert_eq!(detect(html), CaptchaKind::Recaptcha);
    }

    #[test]
    fn detects_recaptcha_by_grecaptcha() {
        let html = "<div> grecaptcha.execute('key') </div>";
        assert_eq!(detect(html), CaptchaKind::Recaptcha);
    }

    #[test]
    fn detects_hcaptcha() {
        let html = r#"<div class="h-captcha" data-sitekey="xxx"></div>"#;
        assert_eq!(detect(html), CaptchaKind::HCaptcha);
    }

    #[test]
    fn detects_datadome() {
        let html = r#"<script src="https://ct.datadome.co/loader.js"></script>"#;
        assert_eq!(detect(html), CaptchaKind::BotManager);
    }

    #[test]
    fn detects_generic_challenge() {
        let html = "<p>Please verify you are human before continuing.</p>";
        assert_eq!(detect(html), CaptchaKind::Generic);
    }

    #[test]
    fn detects_access_denied_title() {
        let html = "<html><head><title>Access Denied</title></head></html>";
        assert_eq!(detect(html), CaptchaKind::Generic);
    }

    #[test]
    fn clean_page_returns_none() {
        let html = "<html><body><p>Normal article content here.</p></body></html>";
        assert_eq!(detect(html), CaptchaKind::None);
    }

    #[test]
    fn is_blocked_true_for_captcha() {
        let html = r#"<div class="h-captcha"></div>"#;
        assert!(is_blocked(html));
    }

    #[test]
    fn is_blocked_false_for_clean() {
        assert!(!is_blocked("<p>Hello world</p>"));
    }

    // ── CaptchaKind helpers ──────────────────────────────────────────────────

    #[test]
    fn none_is_not_blocked() {
        assert!(!CaptchaKind::None.is_blocked());
    }

    #[test]
    fn all_non_none_are_blocked() {
        for kind in [
            CaptchaKind::Cloudflare,
            CaptchaKind::Recaptcha,
            CaptchaKind::HCaptcha,
            CaptchaKind::BotManager,
            CaptchaKind::Generic,
        ] {
            assert!(kind.is_blocked(), "{:?} should be blocked", kind);
        }
    }

    #[test]
    fn labels_are_non_empty() {
        for kind in [
            CaptchaKind::None,
            CaptchaKind::Cloudflare,
            CaptchaKind::Recaptcha,
            CaptchaKind::HCaptcha,
            CaptchaKind::BotManager,
            CaptchaKind::Generic,
        ] {
            assert!(!kind.label().is_empty());
        }
    }

    // ── Bypass strategy ──────────────────────────────────────────────────────

    #[test]
    fn cloudflare_first_attempt_waits() {
        let s = plan_bypass(&CaptchaKind::Cloudflare, 0, false);
        assert!(matches!(s, BypassStrategy::WaitAndRetry { .. }));
    }

    #[test]
    fn cloudflare_second_attempt_uses_headless_if_available() {
        let s = plan_bypass(&CaptchaKind::Cloudflare, 1, true);
        assert_eq!(s, BypassStrategy::UseHeadlessBrowser);
    }

    #[test]
    fn cloudflare_second_attempt_skips_without_headless() {
        let s = plan_bypass(&CaptchaKind::Cloudflare, 1, false);
        assert_eq!(s, BypassStrategy::Skip);
    }

    #[test]
    fn recaptcha_always_skips() {
        for attempt in 0..4 {
            let s = plan_bypass(&CaptchaKind::Recaptcha, attempt, true);
            assert_eq!(s, BypassStrategy::Skip);
        }
    }

    #[test]
    fn hcaptcha_always_skips() {
        assert_eq!(plan_bypass(&CaptchaKind::HCaptcha, 0, true), BypassStrategy::Skip);
    }

    #[test]
    fn bot_manager_rotates_ua_on_second_attempt() {
        let s = plan_bypass(&CaptchaKind::BotManager, 1, false);
        assert_eq!(s, BypassStrategy::RotateUserAgent);
    }

    #[test]
    fn generic_rotates_on_retry() {
        let s = plan_bypass(&CaptchaKind::Generic, 1, false);
        assert_eq!(s, BypassStrategy::RotateUserAgent);
    }

    #[test]
    fn bypass_labels_non_empty() {
        let strategies = [
            BypassStrategy::RotateUserAgent,
            BypassStrategy::WaitAndRetry { delay_ms: 1000 },
            BypassStrategy::UseHeadlessBrowser,
            BypassStrategy::Skip,
        ];
        for s in &strategies {
            assert!(!s.label().is_empty());
        }
    }
}
