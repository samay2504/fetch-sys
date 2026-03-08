//! Integration tests for the human behaviour module — user-agent rotation,
//! async delay bounds, header completeness, and domain politeness.

use fetchsys::human::HumanBehavior;

// ── user_agent ─────────────────────────────────────────────────────────────

#[test]
fn user_agent_is_non_empty() {
    let ua = HumanBehavior::new().user_agent();
    assert!(!ua.is_empty(), "user_agent() must return a non-empty string");
}

#[test]
fn user_agent_looks_like_real_browser() {
    let ua = HumanBehavior::new().user_agent();
    assert!(
        ua.contains("Mozilla"),
        "UA should contain 'Mozilla', got: {:?}",
        ua
    );
}

#[test]
fn user_agent_contains_known_browser_token() {
    let ua = HumanBehavior::new().user_agent();
    let has_browser = ua.contains("Chrome")
        || ua.contains("Firefox")
        || ua.contains("Safari")
        || ua.contains("Edg");
    assert!(has_browser, "UA should name a real browser, got: {:?}", ua);
}

#[test]
fn user_agent_multiple_calls_all_valid() {
    let b = HumanBehavior::new();
    for i in 0..30 {
        let ua = b.user_agent();
        assert!(!ua.is_empty(), "Call {} returned empty UA", i);
        assert!(ua.contains("Mozilla"), "Call {} returned invalid UA: {:?}", i, ua);
    }
}

#[test]
fn user_agent_pool_has_variety() {
    // After enough calls we should see at least 2 distinct UAs
    let b = HumanBehavior::new();
    let uas: std::collections::HashSet<&'static str> = (0..50).map(|_| b.user_agent()).collect();
    assert!(uas.len() >= 2, "UA pool should contain multiple distinct agents, got 1");
}

// ── browser_headers ────────────────────────────────────────────────────────

#[test]
fn browser_headers_includes_accept() {
    let headers = HumanBehavior::new().browser_headers();
    let has_accept = headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("Accept"));
    assert!(has_accept, "browser_headers() must include Accept header");
}

#[test]
fn browser_headers_includes_accept_language() {
    let headers = HumanBehavior::new().browser_headers();
    let found = headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("Accept-Language"));
    assert!(found, "browser_headers() must include Accept-Language");
}

#[test]
fn browser_headers_includes_accept_encoding() {
    let headers = HumanBehavior::new().browser_headers();
    let found = headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("Accept-Encoding"));
    assert!(found, "browser_headers() must include Accept-Encoding");
}

#[test]
fn browser_headers_has_at_least_three_entries() {
    let headers = HumanBehavior::new().browser_headers();
    assert!(
        headers.len() >= 3,
        "Expected at least 3 browser headers, got {}",
        headers.len()
    );
}

#[test]
fn browser_headers_all_keys_and_values_nonempty() {
    for (key, value) in HumanBehavior::new().browser_headers() {
        assert!(!key.is_empty(), "Header key must not be empty");
        assert!(!value.is_empty(), "Header value for '{}' must not be empty", key);
    }
}

#[test]
fn browser_headers_accept_encoding_includes_gzip() {
    let headers = HumanBehavior::new().browser_headers();
    let enc = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("Accept-Encoding"))
        .map(|(_, v)| *v)
        .unwrap_or("");
    assert!(enc.contains("gzip"), "Accept-Encoding should include gzip, got: {:?}", enc);
}

// ── delay ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn delay_completes_without_panic() {
    // jitter_max_ms must be >= 200 (delay() uses gen_range(200..=jitter_max_ms))
    let b = HumanBehavior::new().with_delays(0, 200);
    b.delay().await;
}

#[tokio::test]
async fn delay_called_twice_works() {
    let b = HumanBehavior::new().with_delays(0, 200);
    b.delay().await;
    b.delay().await;
}

// ── domain_delay ───────────────────────────────────────────────────────────

#[tokio::test]
async fn domain_delay_first_visit_completes() {
    // jitter_max_ms must be >= 100 (domain_delay uses gen_range(100..=jitter_max_ms))
    let b = HumanBehavior::new().with_delays(0, 100);
    let start = std::time::Instant::now();
    b.domain_delay("https://example.com/page").await;
    assert!(
        start.elapsed().as_secs() < 5,
        "First domain_delay should not take > 5 s"
    );
}

#[tokio::test]
async fn domain_delay_different_domains_independent() {
    let b = HumanBehavior::new().with_delays(0, 100);
    b.domain_delay("https://site-a.com/").await;
    b.domain_delay("https://site-b.com/").await;
}

#[tokio::test]
async fn domain_delay_invalid_url_does_not_panic() {
    let b = HumanBehavior::new().with_delays(0, 100);
    b.domain_delay("not-a-valid-url").await;
}

// ── with_delays builder ────────────────────────────────────────────────────

#[test]
fn with_delays_returns_human_behavior() {
    let _b: HumanBehavior = HumanBehavior::new().with_delays(100, 500);
}

#[test]
fn default_same_as_new() {
    // Default is implemented — must not panic
    let b: HumanBehavior = Default::default();
    assert!(!b.user_agent().is_empty());
}

#[test]
fn with_delays_zero_range_accepted() {
    // Builder itself accepts any values — validity is only checked when delay() is called
    let _b = HumanBehavior::new().with_delays(0, 200);
}

// ── header composition sanity ──────────────────────────────────────────────

#[test]
fn headers_can_be_converted_to_hashmap() {
    let headers = HumanBehavior::new().browser_headers();
    let map: std::collections::HashMap<&str, &str> = headers.iter().copied().collect();
    assert!(map.contains_key("Accept"));
    assert!(map.contains_key("Accept-Language"));
    assert!(map.contains_key("Accept-Encoding"));
}
