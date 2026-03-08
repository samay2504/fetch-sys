//! Integration tests for the scroll module — word counting, growth heuristics,
//! scroll progress, ScrollConfig presets, and the no-browser passthrough path.

use fetchsys::scroll::{
    accumulate_scroll_content, scroll_progress_pct, should_continue_scrolling, word_count,
    ScrollConfig, ScrollResult, JS_GET_HTML, JS_NEAR_BOTTOM, JS_SCROLL_DOWN, JS_TRIGGER_LAZY,
};

// ── word_count ───────────────────────────────────────────────────────────────

#[test]
fn word_count_plain_text() {
    assert_eq!(word_count("hello world foo"), 3);
}

#[test]
fn word_count_strips_html_tags() {
    assert_eq!(word_count("<p>one <b>two</b> three</p>"), 3);
}

#[test]
fn word_count_empty_string_is_zero() {
    assert_eq!(word_count(""), 0);
}

#[test]
fn word_count_whitespace_only_is_zero() {
    assert_eq!(word_count("   \t\n  "), 0);
}

#[test]
fn word_count_nested_tags() {
    let html = r#"<div class="content"><p>The <em>quick</em> brown fox</p></div>"#;
    assert_eq!(word_count(html), 4);
}

#[test]
fn word_count_self_closing_tags() {
    let html = r#"<p>Hello<br/>World</p>"#;
    assert_eq!(word_count(html), 2);
}

#[test]
fn word_count_longer_article() {
    let html = r#"<article>
        <h1>Rust Memory Safety</h1>
        <p>Rust ensures memory safety through its ownership system.</p>
        <p>No garbage collector is needed because lifetimes are tracked at compile time.</p>
    </article>"#;
    let wc = word_count(html);
    assert!(wc >= 15, "Expected at least 15 words, got {}", wc);
}

// ── should_continue_scrolling ────────────────────────────────────────────────

#[test]
fn continue_when_prev_is_zero() {
    // First measurement — always continue
    assert!(should_continue_scrolling(0, 500, 0.03));
    assert!(should_continue_scrolling(0, 0, 0.03));
}

#[test]
fn continue_with_strong_growth() {
    // 200 → 400 = 100% growth → continue
    assert!(should_continue_scrolling(200, 400, 0.03));
}

#[test]
fn stop_with_no_growth() {
    // Same count → 0% growth < 3% threshold → stop
    assert!(!should_continue_scrolling(1000, 1000, 0.03));
}

#[test]
fn stop_with_tiny_growth() {
    // 1000 → 1010 = 1% growth < 3% threshold → stop
    assert!(!should_continue_scrolling(1000, 1010, 0.03));
}

#[test]
fn exact_threshold_boundary() {
    // 1000 → 1030 = exactly 3% → should continue (>= threshold)
    assert!(should_continue_scrolling(1000, 1030, 0.03));
    // 1000 → 1029 = 2.9% → should stop (< threshold)
    assert!(!should_continue_scrolling(1000, 1029, 0.03));
}

#[test]
fn custom_high_threshold() {
    // 20% threshold: 100 → 115 = 15% → stop
    assert!(!should_continue_scrolling(100, 115, 0.20));
    // 100 → 125 = 25% → continue
    assert!(should_continue_scrolling(100, 125, 0.20));
}

// ── scroll_progress_pct ──────────────────────────────────────────────────────

#[test]
fn progress_at_top_of_page() {
    // scroll_y = 0, window = 768, page = 3000 → (0+768)/3000 ≈ 25.6%
    let pct = scroll_progress_pct(0.0, 768.0, 3_000.0);
    assert!((pct - 25.6).abs() < 0.5, "Expected ~25.6%, got {:.1}%", pct);
}

#[test]
fn progress_at_bottom_of_page() {
    // scroll_y = page - window = 2232, window = 768, page = 3000 → 100%
    let pct = scroll_progress_pct(2232.0, 768.0, 3_000.0);
    assert!((pct - 100.0).abs() < 0.1, "Expected 100%, got {:.1}%", pct);
}

#[test]
fn progress_page_fits_in_viewport() {
    // page <= window → 100%
    assert_eq!(scroll_progress_pct(0.0, 1920.0, 800.0), 100.0);
    assert_eq!(scroll_progress_pct(0.0, 768.0, 768.0), 100.0);
}

#[test]
fn progress_mid_page() {
    // scroll_y + window = 1500, page = 3000 → 50%
    let pct = scroll_progress_pct(732.0, 768.0, 3_000.0);
    assert!((pct - 50.0).abs() < 0.5, "Expected ~50%, got {:.1}%", pct);
}

#[test]
fn progress_cannot_exceed_100() {
    // Over-scroll scenario
    let pct = scroll_progress_pct(5000.0, 768.0, 3_000.0);
    assert!(pct <= 100.0, "Progress should be capped at 100%");
}

// ── ScrollResult::passthrough ─────────────────────────────────────────────────

#[test]
fn passthrough_returns_initial_html_unchanged() {
    let html = "<p>Article text here for testing purposes</p>";
    let r = ScrollResult::passthrough(html);
    assert_eq!(r.accumulated_html, html);
    assert_eq!(r.steps_taken, 0);
    assert!(r.reached_bottom, "Passthrough should mark page as exhausted");
    assert!(r.word_count >= 5, "Word count should be 5+");
}

#[test]
fn passthrough_empty_html() {
    let r = ScrollResult::passthrough("");
    assert_eq!(r.steps_taken, 0);
    assert_eq!(r.word_count, 0);
    assert!(r.reached_bottom);
}

// ── ScrollConfig presets ──────────────────────────────────────────────────────

#[test]
fn default_config_has_sensible_values() {
    let cfg = ScrollConfig::default();
    assert!(cfg.max_steps > 0);
    assert!(cfg.pause_ms > 0);
    assert!(cfg.growth_threshold > 0.0 && cfg.growth_threshold < 1.0);
    assert!(cfg.max_chars > 10_000);
    assert!(cfg.stale_steps_limit >= 1);
}

#[test]
fn fast_preset_lower_than_default() {
    let fast = ScrollConfig::fast();
    let def = ScrollConfig::default();
    assert!(fast.max_steps <= def.max_steps);
    assert!(fast.pause_ms <= def.pause_ms);
}

#[test]
fn thorough_preset_higher_than_default() {
    let thorough = ScrollConfig::thorough();
    let def = ScrollConfig::default();
    assert!(thorough.max_steps >= def.max_steps);
}

#[test]
fn stale_steps_limit_at_least_one() {
    for cfg in [ScrollConfig::default(), ScrollConfig::fast(), ScrollConfig::thorough()] {
        assert!(cfg.stale_steps_limit >= 1,
            "stale_steps_limit must be >= 1");
    }
}

// ── JS constants ─────────────────────────────────────────────────────────────

#[test]
fn js_scroll_down_contains_scroll_call() {
    assert!(JS_SCROLL_DOWN.contains("scrollBy"));
}

#[test]
fn js_near_bottom_returns_boolean() {
    assert!(JS_NEAR_BOTTOM.contains("return"));
}

#[test]
fn js_get_html_returns_outer_html() {
    assert!(JS_GET_HTML.contains("outerHTML"));
}

#[test]
fn js_trigger_lazy_dispatches_events() {
    assert!(JS_TRIGGER_LAZY.contains("dispatchEvent"));
}

// ── No-browser passthrough (always runs without feature flags) ────────────────

#[tokio::test]
#[cfg(not(any(feature = "headless", feature = "selenium")))]
async fn accumulate_returns_initial_without_browser() {
    let html = "<article><p>Content paragraph one that is long enough.</p>\
                <p>Content paragraph two with more words.</p></article>";
    let result = accumulate_scroll_content(
        "https://example.com/test",
        html,
        &ScrollConfig::default(),
    )
    .await
    .expect("accumulate_scroll_content should not fail in no-browser mode");

    assert_eq!(result.accumulated_html, html, "Should return initial HTML unchanged");
    assert_eq!(result.steps_taken, 0, "No scroll steps without browser");
    assert!(result.reached_bottom, "Should mark page as exhausted");
}

#[tokio::test]
#[cfg(not(any(feature = "headless", feature = "selenium")))]
async fn accumulate_with_fast_config_still_passthrough() {
    let html = "<p>short</p>";
    let r = accumulate_scroll_content("https://x.com", html, &ScrollConfig::fast())
        .await
        .unwrap();
    assert_eq!(r.accumulated_html, html);
}
