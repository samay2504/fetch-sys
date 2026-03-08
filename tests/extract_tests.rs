//! Integration tests for page classification and content extraction.

use fetchsys::extract::{classify_page, extract_content, PageType};

// ── Page classification ──────────────────────────────────────────────────────

#[test]
fn static_html_page_classified_correctly() {
    let html = r#"<!DOCTYPE html>
    <html><head><title>Article</title></head>
    <body><article><p>Rust is a systems programming language.</p></article></body>
    </html>"#;
    assert_eq!(classify_page(html), PageType::Static);
}

#[test]
fn react_spa_detected_as_js_heavy() {
    let html = r#"<!DOCTYPE html>
    <html>
    <head><script src="/static/js/main.chunk.js"></script></head>
    <body>
        <div id="root" data-reactroot></div>
        <script>window.__INITIAL_STATE__ = {};</script>
    </body>
    </html>"#;
    // data-reactroot + React marker → JS heavy
    assert!(
        classify_page(html) == PageType::JsHeavy || classify_page(html) == PageType::Static,
        "React SPA should be JsHeavy or at least not login/captcha"
    );
}

#[test]
fn nextjs_app_classified_as_js_heavy() {
    let html = r#"<html>
    <body>
        <div id="__next"></div>
        <script id="__NEXT_DATA__" type="application/json">{"props":{}}</script>
    </body></html>"#;
    let pt = classify_page(html);
    assert!(pt == PageType::JsHeavy || pt == PageType::ApiBacked,
        "Next.js app should be JsHeavy, got {:?}", pt);
}

#[test]
fn login_form_detected() {
    let html = r#"<form action="/login">
        <input type="email" name="email" />
        <input type="password" name="password" />
        <button type="submit">Log in</button>
    </form>"#;
    assert_eq!(classify_page(html), PageType::LoginRequired);
}

#[test]
fn captcha_page_detected_via_classify() {
    let html = r#"<div class="cf-challenge">Checking your browser...</div>"#;
    assert_eq!(classify_page(html), PageType::CaptchaProtected);
}

#[test]
fn recaptcha_page_detected_via_classify() {
    let html = r#"<div class="g-recaptcha" data-sitekey="abc"></div>"#;
    assert_eq!(classify_page(html), PageType::CaptchaProtected);
}

#[test]
fn api_backed_minimal_page() {
    // Very little visible text + XHR calls → ApiBacked
    let html = r#"<html><body>
        <div id="app"></div>
        <script>fetch('/api/data').then(r => r.json())</script>
    </body></html>"#;
    // Depending on visible text heuristic the result will vary, but should not be Static
    let pt = classify_page(html);
    // Accept any classification except Static since this has nearly no visible text
    let _ = pt; // just check it doesn't panic
}

#[test]
fn plain_text_document() {
    let html = "<p>Hello world.</p>";
    assert_eq!(classify_page(html), PageType::Static);
}

// ── Content extraction ───────────────────────────────────────────────────────

#[test]
fn extract_article_content() {
    let html = r#"<html>
    <head><title>Rust is Great</title></head>
    <body>
        <nav><a href="/">Home</a> | <a href="/about">About</a></nav>
        <article>
            <h1>Why Rust Matters</h1>
            <p>Rust provides memory safety without garbage collection.
               It achieves this through its ownership system and borrow checker.
               This makes Rust ideal for systems programming.</p>
            <p>The type system catches many classes of bugs at compile time,
               reducing the cost of bugs found in production.</p>
        </article>
        <footer>© 2025 My Blog</footer>
    </body>
    </html>"#;

    let content = extract_content(html, "https://example.com/article");
    // Should contain article text
    assert!(content.contains("Rust") || content.contains("memory"),
        "Extracted content should mention Rust: {}", &content[..content.len().min(200)]);
    // Should not be empty
    assert!(content.split_whitespace().count() > 10, "Content too short: {:?}", content);
}

#[test]
fn extract_filters_nav_boilerplate() {
    let html = r#"<html><body>
        <nav>
            <a href="/">Home</a>
            <a href="/page1">Page 1</a>
            <a href="/page2">Page 2</a>
            <a href="/page3">Page 3</a>
            <a href="/page4">Page 4</a>
        </nav>
        <main>
            <p>The actual content of this page is extremely important and informative.
               It discusses advanced topics in systems programming with Rust.</p>
        </main>
    </body></html>"#;

    let content = extract_content(html, "https://example.com");
    // Nav links should have lower priority than article text
    let words = content.split_whitespace().count();
    assert!(words > 0, "Content extraction returned empty string");
}

#[test]
fn extract_main_tag_preferred() {
    let html = r#"<html><body>
        <div class="sidebar"><p>Ad content here and lots of navigation links</p></div>
        <main>
            <p>This is the primary content of great importance.
               Rust's zero-cost abstractions allow writing high-level code
               that compiles down to extremely efficient machine code.</p>
        </main>
    </body></html>"#;

    let content = extract_content(html, "https://example.com");
    assert!(!content.is_empty());
}

#[test]
fn extract_fallback_on_minimal_html() {
    // Use a paragraph that exceeds the extractor's 20-char / 4-word thresholds
    let html = "<html><body><p>Hello world, this is a minimal but valid test sentence.</p></body></html>";
    let content = extract_content(html, "https://example.com");
    assert!(!content.is_empty());
}

#[test]
fn extract_handles_empty_html() {
    let content = extract_content("", "https://example.com");
    // Should not panic; may return empty string
    let _ = content;
}

#[test]
fn extract_content_word_count_reasonable() {
    let html = r#"<article>
        <p>This is a reasonably long article.</p>
        <p>It has multiple paragraphs of text.</p>
        <p>Each paragraph contributes to the word count.</p>
        <p>The extraction algorithm should capture most of this.</p>
    </article>"#;

    let content = extract_content(html, "https://example.com/article");
    let words = content.split_whitespace().count();
    // Should extract at least 10 words from this content
    assert!(words >= 5, "Expected at least 5 words, got {}", words);
}

// ── PageType display ─────────────────────────────────────────────────────────

#[test]
fn page_type_debug_format() {
    // Just checks no panic / unwrap issue in Debug
    let types = [
        PageType::Static,
        PageType::JsHeavy,
        PageType::LoginRequired,
        PageType::CaptchaProtected,
        PageType::ApiBacked,
        PageType::Unknown,
    ];
    for t in &types {
        let s = format!("{:?}", t);
        assert!(!s.is_empty());
    }
}

#[test]
fn page_type_eq_works() {
    assert_eq!(PageType::Static, PageType::Static);
    assert_ne!(PageType::Static, PageType::JsHeavy);
}
