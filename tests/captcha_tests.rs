//! Integration tests for captcha detection and bypass strategy planning.
//!
//! These tests cover real-world HTML patterns from common bot-protection vendors.

use fetchsys::captcha::{
    detect, is_blocked, plan_bypass, BypassStrategy, CaptchaKind, MAX_CAPTCHA_RETRIES,
};

// ── Detection accuracy ───────────────────────────────────────────────────────

#[test]
fn clean_page_is_not_blocked() {
    let html = r#"<!DOCTYPE html>
    <html><head><title>My Blog</title></head>
    <body><article><p>This is a normal article about Rust programming.</p></article></body>
    </html>"#;
    assert_eq!(detect(html), CaptchaKind::None);
    assert!(!is_blocked(html));
}

#[test]
fn cloudflare_ray_id_header_hint() {
    // Simulates a page that echoes the CF-Ray header
    let html = r#"<html><body>
        <p>cf-ray: 8abc123456789001-IAD</p>
        <p>cloudflare-nginx</p>
    </body></html>"#;
    // Contains "cloudflare-nginx" but not the CF challenge markers directly
    // depends on regex — pattern includes it
    let kind = detect(html);
    assert!(kind == CaptchaKind::Cloudflare || kind == CaptchaKind::None,
        "Expected Cloudflare or None, got {:?}", kind);
}

#[test]
fn cloudflare_js_challenge_page() {
    let html = r#"<html><head><title>Just a moment...</title></head>
    <body>
        <div id="cf-challenge">Checking your browser before accessing...</div>
        <script>var s,t,o,p,b,r,e,a,k,i,n,g,f;</script>
    </body></html>"#;
    assert_eq!(detect(html), CaptchaKind::Cloudflare);
    assert!(is_blocked(html));
}

#[test]
fn cloudflare_clearance_cookie_marker() {
    let html = r#"<script>document.cookie = '__cf_bm=abc; SameSite=None';</script>"#;
    assert_eq!(detect(html), CaptchaKind::Cloudflare);
}

#[test]
fn recaptcha_v2_widget() {
    let html = r#"<div class="g-recaptcha" data-sitekey="6LdABC12ABCDEFghi"></div>
    <script src="https://www.google.com/recaptcha/api.js"></script>"#;
    assert_eq!(detect(html), CaptchaKind::Recaptcha);
}

#[test]
fn recaptcha_v3_inline_call() {
    let html = r#"<script>
        grecaptcha.ready(function() {
            grecaptcha.execute('6LdXXXX', {action: 'login'})
        });
    </script>"#;
    assert_eq!(detect(html), CaptchaKind::Recaptcha);
}

#[test]
fn hcaptcha_widget() {
    let html = r#"<div class="h-captcha" data-sitekey="10000000-ffff-ffff"></div>
    <script src="https://hcaptcha.com/1/api.js"></script>"#;
    assert_eq!(detect(html), CaptchaKind::HCaptcha);
}

#[test]
fn datadome_bot_manager() {
    let html = r#"<script src="https://ct.datadome.co/tags.js" async></script>"#;
    assert_eq!(detect(html), CaptchaKind::BotManager);
}

#[test]
fn perimeterx_bot_manager() {
    let html = r#"<script>window._pxAppId = 'PX123ABCDE';</script>"#;
    assert_eq!(detect(html), CaptchaKind::BotManager);
}

#[test]
fn imperva_incapsula() {
    let html = r#"<!-- Incapsula session id: 123abc456def -->"#;
    assert_eq!(detect(html), CaptchaKind::BotManager);
}

#[test]
fn generic_ddos_protection_page() {
    let html = r#"<html><body><p>DDoS protection by our security provider</p></body></html>"#;
    assert_eq!(detect(html), CaptchaKind::Generic);
}

#[test]
fn generic_access_denied_title() {
    let html = r#"<html><head><title>Access Denied</title></head><body></body></html>"#;
    assert_eq!(detect(html), CaptchaKind::Generic);
}

#[test]
fn generic_verify_human_text() {
    let html = r#"<p>Please verify you are human to continue.</p>"#;
    assert_eq!(detect(html), CaptchaKind::Generic);
}

#[test]
fn rate_limited_title() {
    let html = r#"<html><head><title>Rate Limited</title></head></html>"#;
    assert_eq!(detect(html), CaptchaKind::Generic);
}

// ── Priority ordering ────────────────────────────────────────────────────────

#[test]
fn recaptcha_wins_over_generic_when_both_present() {
    // Page has both recaptcha and generic challenge text
    let html = r#"<script src="https://www.google.com/recaptcha/api.js"></script>
    <p>Please verify you are human</p>"#;
    // Recaptcha should be detected first (higher priority)
    assert_eq!(detect(html), CaptchaKind::Recaptcha);
}

// ── CaptchaKind methods ──────────────────────────────────────────────────────

#[test]
fn is_blocked_returns_false_for_none() {
    assert!(!CaptchaKind::None.is_blocked());
}

#[test]
fn all_non_none_kinds_are_blocked() {
    let kinds = [
        CaptchaKind::Cloudflare,
        CaptchaKind::Recaptcha,
        CaptchaKind::HCaptcha,
        CaptchaKind::BotManager,
        CaptchaKind::Generic,
    ];
    for k in &kinds {
        assert!(k.is_blocked(), "{:?} should be blocked", k);
    }
}

#[test]
fn all_kinds_have_non_empty_labels() {
    let kinds = [
        CaptchaKind::None,
        CaptchaKind::Cloudflare,
        CaptchaKind::Recaptcha,
        CaptchaKind::HCaptcha,
        CaptchaKind::BotManager,
        CaptchaKind::Generic,
    ];
    for k in &kinds {
        assert!(!k.label().is_empty(), "{:?} label is empty", k);
    }
}

// ── Bypass strategy ──────────────────────────────────────────────────────────

#[test]
fn cloudflare_strategy_progression() {
    // attempt 0 without headless → wait
    assert!(matches!(plan_bypass(&CaptchaKind::Cloudflare, 0, false),
        BypassStrategy::WaitAndRetry { .. }));
    // attempt 0 with headless → still wait first
    assert!(matches!(plan_bypass(&CaptchaKind::Cloudflare, 0, true),
        BypassStrategy::WaitAndRetry { .. }));
    // attempt 1 with headless → browser
    assert_eq!(plan_bypass(&CaptchaKind::Cloudflare, 1, true), BypassStrategy::UseHeadlessBrowser);
    // attempt 1 without headless → skip
    assert_eq!(plan_bypass(&CaptchaKind::Cloudflare, 1, false), BypassStrategy::Skip);
}

#[test]
fn recaptcha_always_skips_regardless_of_attempt() {
    for attempt in 0..MAX_CAPTCHA_RETRIES + 1 {
        let s = plan_bypass(&CaptchaKind::Recaptcha, attempt, true);
        assert_eq!(s, BypassStrategy::Skip,
            "Recaptcha attempt {} should Skip", attempt);
    }
}

#[test]
fn hcaptcha_always_skips() {
    assert_eq!(plan_bypass(&CaptchaKind::HCaptcha, 0, true), BypassStrategy::Skip);
    assert_eq!(plan_bypass(&CaptchaKind::HCaptcha, 1, true), BypassStrategy::Skip);
}

#[test]
fn bot_manager_three_stage_bypass() {
    // 0 → wait
    assert!(matches!(plan_bypass(&CaptchaKind::BotManager, 0, false),
        BypassStrategy::WaitAndRetry { .. }));
    // 1 → rotate UA
    assert_eq!(plan_bypass(&CaptchaKind::BotManager, 1, false), BypassStrategy::RotateUserAgent);
    // 2 → skip
    assert_eq!(plan_bypass(&CaptchaKind::BotManager, 2, false), BypassStrategy::Skip);
}

#[test]
fn generic_two_stage_bypass() {
    assert!(matches!(plan_bypass(&CaptchaKind::Generic, 0, false),
        BypassStrategy::WaitAndRetry { .. }));
    assert_eq!(plan_bypass(&CaptchaKind::Generic, 1, false), BypassStrategy::RotateUserAgent);
}

#[test]
fn bypass_wait_delay_is_positive() {
    match plan_bypass(&CaptchaKind::Cloudflare, 0, false) {
        BypassStrategy::WaitAndRetry { delay_ms } => assert!(delay_ms > 0),
        other => panic!("Expected WaitAndRetry, got {:?}", other),
    }
}

#[test]
fn bypass_labels_unique_and_non_empty() {
    let strategies = [
        BypassStrategy::RotateUserAgent,
        BypassStrategy::WaitAndRetry { delay_ms: 1000 },
        BypassStrategy::UseHeadlessBrowser,
        BypassStrategy::Skip,
    ];
    let labels: Vec<_> = strategies.iter().map(|s| s.label()).collect();
    for label in &labels {
        assert!(!label.is_empty());
    }
    // All labels should be distinct
    let unique: std::collections::HashSet<_> = labels.iter().collect();
    assert_eq!(unique.len(), labels.len(), "Labels should be unique");
}

#[test]
fn max_captcha_retries_is_reasonable() {
    assert!(MAX_CAPTCHA_RETRIES >= 2);
    assert!(MAX_CAPTCHA_RETRIES <= 10);
}
