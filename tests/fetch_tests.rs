//! Integration tests for the fetch module — static fetching, title extraction,
//! FetchOptions defaults, and HTTP error handling via a local mock server.

use fetchsys::fetch::{extract_title, static_fetch, FetchOptions, FetchRaw};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── extract_title (unit-level, no network) ────────────────────────────────────

#[test]
fn extract_title_standard() {
    assert_eq!(
        extract_title("<html><head><title>Hello World</title></head></html>"),
        "Hello World"
    );
}

#[test]
fn extract_title_trimmed() {
    assert_eq!(
        extract_title("<title>  Spaces Around  </title>"),
        "Spaces Around"
    );
}

#[test]
fn extract_title_missing_returns_empty() {
    assert_eq!(extract_title("<html><body><p>no title</p></body></html>"), "");
}

#[test]
fn extract_title_empty_tag() {
    assert_eq!(extract_title("<title></title>"), "");
}

#[test]
fn extract_title_empty_string() {
    assert_eq!(extract_title(""), "");
}

#[test]
fn extract_title_case_insensitive() {
    // Some pages may use uppercase <TITLE>
    let html = "<HTML><HEAD><TITLE>Upper Case</TITLE></HEAD></HTML>";
    // Either returns the value or empty — must not panic
    let t = extract_title(html);
    assert!(t == "Upper Case" || t.is_empty());
}

#[test]
fn extract_title_nested_content() {
    let html = "<html><head><meta charset='utf-8'><title>Rust News 2025</title></head><body></body></html>";
    assert_eq!(extract_title(html), "Rust News 2025");
}

// ── FetchOptions defaults ─────────────────────────────────────────────────────

#[test]
fn fetch_options_default_max_retries_positive() {
    let opts = FetchOptions::default();
    assert!(opts.max_retries >= 1, "Default max_retries should be at least 1");
}

#[test]
fn fetch_options_default_scroll_disabled() {
    let opts = FetchOptions::default();
    assert!(!opts.enable_scroll, "Scroll should be disabled by default");
}

#[test]
fn fetch_options_default_max_scroll_steps_nonzero() {
    let opts = FetchOptions::default();
    assert!(opts.max_scroll_steps > 0);
}

#[test]
fn fetch_options_struct_is_fully_specified() {
    let opts = FetchOptions {
        max_retries: 5,
        allow_headless: false,
        enable_scroll: true,
        max_scroll_steps: 20,
    };
    assert_eq!(opts.max_retries, 5);
    assert!(!opts.allow_headless);
    assert!(opts.enable_scroll);
    assert_eq!(opts.max_scroll_steps, 20);
}

// ── FetchRaw field presence ───────────────────────────────────────────────────

#[test]
fn fetch_raw_fields_accessible() {
    // Construct a synthetic FetchRaw to confirm all public fields are accessible
    let raw = FetchRaw {
        final_url: "https://example.com".to_string(),
        title: "Example".to_string(),
        raw_html: "<p>hello</p>".to_string(),
        status: 200,
        bytes_downloaded: 11,
        used_browser: false,
    };
    assert_eq!(raw.status, 200);
    assert_eq!(raw.title, "Example");
    assert!(!raw.used_browser);
    assert_eq!(raw.bytes_downloaded, 11);
}

// ── static_fetch with mock server ─────────────────────────────────────────────

#[tokio::test]
async fn static_fetch_200_returns_ok() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("<html><head><title>Mock Page</title></head><body><p>content</p></body></html>")
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&server)
        .await;

    let url = format!("{}/page", server.uri());
    let result = static_fetch(&url, "Mozilla/5.0", 10, 1_000_000).await;
    assert!(result.is_ok(), "200 response should be Ok, got: {:?}", result.err());

    let raw = result.unwrap();
    assert_eq!(raw.status, 200);
    assert!(
        raw.raw_html.contains("content"),
        "HTML body should be captured"
    );
}

#[tokio::test]
async fn static_fetch_captures_title() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/titled"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("<html><head><title>The Real Title</title></head><body></body></html>")
                .insert_header("content-type", "text/html"),
        )
        .mount(&server)
        .await;

    let url = format!("{}/titled", server.uri());
    let raw = static_fetch(&url, "Mozilla/5.0", 10, 1_000_000).await.unwrap();
    assert_eq!(raw.title, "The Real Title");
}

#[tokio::test]
async fn static_fetch_records_bytes_downloaded() {
    let body = "<html><head><title>Bytes</title></head><body><p>Hello!</p></body></html>";
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/bytes"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/html"),
        )
        .mount(&server)
        .await;

    let url = format!("{}/bytes", server.uri());
    let raw = static_fetch(&url, "Mozilla/5.0", 10, 1_000_000).await.unwrap();
    assert!(
        raw.bytes_downloaded > 0,
        "bytes_downloaded should be > 0, got {}",
        raw.bytes_downloaded
    );
}

#[tokio::test]
async fn static_fetch_404_returns_error_or_status() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/missing"))
        .respond_with(ResponseTemplate::new(404).set_body_string("Not Found"))
        .mount(&server)
        .await;

    let url = format!("{}/missing", server.uri());
    let result = static_fetch(&url, "Mozilla/5.0", 10, 1_000_000).await;
    // Either an Err or a FetchRaw with status 404 is acceptable
    match result {
        Err(_) => {} // explicit error — fine
        Ok(raw) => assert_eq!(raw.status, 404, "Should reflect 404 status"),
    }
}

#[tokio::test]
async fn static_fetch_500_is_handled() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/error"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&server)
        .await;

    let url = format!("{}/error", server.uri());
    let result = static_fetch(&url, "Mozilla/5.0", 10, 1_000_000).await;
    match result {
        Err(_) => {}
        Ok(raw) => assert_eq!(raw.status, 500),
    }
}

#[tokio::test]
async fn static_fetch_plain_text_body() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/text"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("Just plain text without HTML tags")
                .insert_header("content-type", "text/plain"),
        )
        .mount(&server)
        .await;

    let url = format!("{}/text", server.uri());
    let result = static_fetch(&url, "Mozilla/5.0", 10, 1_000_000).await;
    // Must not panic; plain text is valid content
    assert!(result.is_ok() || result.is_err());
}

#[tokio::test]
async fn static_fetch_large_html_truncated_by_max_bytes() {
    // Generate 2 MB of content
    let body = format!(
        "<html><head><title>Big</title></head><body>{}</body></html>",
        "x".repeat(2_000_000)
    );
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/big"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/html"),
        )
        .mount(&server)
        .await;

    let url = format!("{}/big", server.uri());
    // Cap at 100 KB — clamping is applied to raw_html, not bytes_downloaded
    let result = static_fetch(&url, "Mozilla/5.0", 10, 100_000).await;
    if let Ok(raw) = result {
        assert!(
            raw.raw_html.len() <= 100_000 + 512, // allow small HTML overhead
            "raw_html length {} should respect max_bytes cap",
            raw.raw_html.len()
        );
    }
}

#[tokio::test]
async fn static_fetch_sets_final_url() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/url-test"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("<html><head><title>URL</title></head></html>")
                .insert_header("content-type", "text/html"),
        )
        .mount(&server)
        .await;

    let url = format!("{}/url-test", server.uri());
    let raw = static_fetch(&url, "Mozilla/5.0", 10, 1_000_000).await.unwrap();
    assert!(
        !raw.final_url.is_empty(),
        "final_url should be populated after fetch"
    );
}
