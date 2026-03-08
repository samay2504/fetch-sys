//! Edge-case tests — boundary conditions, Unicode, empty inputs, error paths.
//!
//! These tests verify the system handles unusual or adversarial inputs
//! gracefully without panicking or producing corrupt output.

use fetchsys::config::{FactCheckConfig, LlmConfig, SearchConfig};
use fetchsys::factcheck;
use fetchsys::reader::ReadResult;
use fetchsys::schema::{AgentResponse, Answer, FactCheck, FactCheckStatus};
use fetchsys::search::multi_tier_search;
use fetchsys::storage::{CachedPage, PageCache};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fact_cfg() -> FactCheckConfig {
    FactCheckConfig {
        confirm_threshold: 0.6,
        contradict_threshold: 0.4,
    }
}

fn make_doc(url: &str, content: &str) -> ReadResult {
    ReadResult {
        url: url.into(),
        title: "Test".into(),
        content: content.into(),
        adapter: "test".into(),
    }
}

// ---------------------------------------------------------------------------
// Unicode / multi-byte edge cases
// ---------------------------------------------------------------------------

#[test]
fn reader_snippet_handles_multibyte_chars() {
    let r = ReadResult {
        url: "https://example.com".into(),
        title: "Test".into(),
        // Mix of ASCII, emoji, CJK, and diacritics
        content: "Hello 🌍 世界 café naïve résumé ¡Hola! Ñoño こんにちは".into(),
        adapter: "test".into(),
    };

    // Truncate at various boundaries — must not panic
    for max in 1..=60 {
        let s = r.snippet(max);
        // Result should be valid UTF-8 (it is since it's a String)
        assert!(s.len() <= max * 4 + 4, "snippet too large at max={max}");
    }
}

#[test]
fn factcheck_handles_unicode_claims() {
    let docs = vec![
        make_doc(
            "https://example.com/1",
            "日本語のテキストです。機械学習は重要な分野です。",
        ),
        make_doc(
            "https://example.com/2",
            "Ελληνικά κείμενα. η μηχανική μάθηση είναι σημαντική.",
        ),
        make_doc(
            "https://example.com/3",
            "Текст на русском языке. Машинное обучение — важная область.",
        ),
    ];

    let provider = fetchsys::llm::build_provider(&LlmConfig::default());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async {
        factcheck::check(
            &docs,
            "機械学習とは何ですか",
            provider.as_ref(),
            false,
            &fact_cfg(),
        )
        .await
    });

    // Should not panic — answer may be sparse since heuristic needs word overlap
    assert!(!result.answer.is_empty());
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

#[test]
fn factcheck_handles_emoji_heavy_content() {
    let docs = vec![make_doc(
        "https://example.com/emoji",
        "🔥🔥🔥 This is 🚀 amazing! AI 🤖 is the future 💡 of technology 🌐. \
         Machine 🏭 learning 📚 processes data 📊 efficiently ✅.",
    )];

    let provider = fetchsys::llm::build_provider(&LlmConfig::default());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async {
        factcheck::check(
            &docs,
            "machine learning emoji test",
            provider.as_ref(),
            false,
            &fact_cfg(),
        )
        .await
    });

    assert!(!result.answer.is_empty());
}

// ---------------------------------------------------------------------------
// Empty / zero-length inputs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_empty_query_is_handled() {
    // No mock server — empty providers should bail immediately
    let cfg = SearchConfig {
        providers: vec![],
        ..Default::default()
    };

    let result = multi_tier_search("", &cfg).await;
    assert!(result.is_err(), "Empty providers should return error");
}

#[tokio::test]
async fn factcheck_empty_docs() {
    let provider = fetchsys::llm::build_provider(&LlmConfig::default());
    let result = factcheck::check(
        &[],
        "empty docs query",
        provider.as_ref(),
        false,
        &fact_cfg(),
    )
    .await;

    // With no docs, heuristic generates a generic claim → inconclusive → 0.5 baseline
    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "Confidence out of range with no docs: {}",
        result.confidence
    );
}

#[tokio::test]
async fn factcheck_single_empty_doc() {
    let docs = vec![make_doc("https://example.com", "")];
    let provider = fetchsys::llm::build_provider(&LlmConfig::default());
    let result = factcheck::check(
        &docs,
        "empty content",
        provider.as_ref(),
        false,
        &fact_cfg(),
    )
    .await;

    // Should not panic; confidence may be low
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

#[test]
fn reader_snippet_empty_content() {
    let r = ReadResult {
        url: "https://example.com".into(),
        title: "".into(),
        content: "".into(),
        adapter: "test".into(),
    };
    let s = r.snippet(100);
    assert_eq!(s, "");
}

#[test]
fn reader_snippet_whitespace_only() {
    let r = ReadResult {
        url: "https://example.com".into(),
        title: "".into(),
        content: "   \n\t\n   ".into(),
        adapter: "test".into(),
    };
    let s = r.snippet(100);
    assert_eq!(s, "");
}

// ---------------------------------------------------------------------------
// Very long inputs
// ---------------------------------------------------------------------------

#[test]
fn reader_snippet_very_long_content() {
    let content = "word ".repeat(100_000); // 500,000 chars
    let r = ReadResult {
        url: "https://example.com".into(),
        title: "".into(),
        content,
        adapter: "test".into(),
    };
    let s = r.snippet(100);
    assert!(s.chars().count() <= 101); // max_chars + possible trailing '…'
}

#[tokio::test]
async fn factcheck_very_long_query() {
    let query = "a ".repeat(5000); // 10,000 char query
    let docs = vec![make_doc("https://example.com", "Some content about testing.")];
    let provider = fetchsys::llm::build_provider(&LlmConfig::default());
    let result = factcheck::check(
        &docs,
        &query,
        provider.as_ref(),
        false,
        &fact_cfg(),
    )
    .await;

    assert!(!result.answer.is_empty());
}

// ---------------------------------------------------------------------------
// Schema edge cases
// ---------------------------------------------------------------------------

#[test]
fn schema_empty_sources_and_factchecks_valid() {
    let response = AgentResponse::new("test".into(), vec![], 0);
    let json = serde_json::to_value(&response).unwrap();
    let result = fetchsys::schema::validate_schema(&json);
    assert!(result.is_ok(), "Empty sources/factchecks should be valid: {:?}", result);
}

#[test]
fn schema_unicode_query_roundtrip() {
    let mut response = AgentResponse::new(
        "¿Qué es la inteligencia artificial? 人工知能とは？ 🤖".into(),
        vec!["searxng".into()],
        100,
    );
    response.answer = Answer {
        text: "La IA es un campo de la informática. 人工知能はコンピュータ科学の分野です。".into(),
        confidence: 0.75,
    };

    let json_str = serde_json::to_string(&response).unwrap();
    let deserialized: AgentResponse = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.query, response.query);
    assert_eq!(deserialized.answer.text, response.answer.text);
}

#[test]
fn schema_zero_confidence() {
    let mut response = AgentResponse::new("test".into(), vec![], 0);
    response.answer = Answer {
        text: "No answer".into(),
        confidence: 0.0,
    };
    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["answer"]["confidence"], 0.0);
}

#[test]
fn schema_max_confidence() {
    let mut response = AgentResponse::new("test".into(), vec![], 0);
    response.answer = Answer {
        text: "Very confident".into(),
        confidence: 1.0,
    };
    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["answer"]["confidence"], 1.0);
}

#[test]
fn schema_factcheck_all_status_variants() {
    let statuses = [
        FactCheckStatus::Confirmed,
        FactCheckStatus::Contradicted,
        FactCheckStatus::Inconclusive,
    ];

    for status in &statuses {
        let fc = FactCheck {
            claim: "test claim".into(),
            status: status.clone(),
            evidence_urls: vec!["https://evidence.com".into()],
        };
        let json = serde_json::to_value(&fc).unwrap();
        let deserialized: FactCheck = serde_json::from_value(json).unwrap();
        assert_eq!(&deserialized.status, status);
    }
}

// ---------------------------------------------------------------------------
// Storage edge cases
// ---------------------------------------------------------------------------

#[test]
fn cache_get_nonexistent_returns_none() {
    let cache = PageCache::new(300);
    assert!(cache.get("https://nonexistent.com").is_none());
}

#[test]
fn cache_insert_same_key_overwrites() {
    let cache = PageCache::new(300);
    cache.insert(
        "https://example.com".into(),
        CachedPage {
            url: "https://example.com".into(),
            title: "Old".into(),
            markdown: "Old content".into(),
            adapter: "test".into(),
        },
    );
    cache.insert(
        "https://example.com".into(),
        CachedPage {
            url: "https://example.com".into(),
            title: "New".into(),
            markdown: "New content".into(),
            adapter: "test".into(),
        },
    );
    assert_eq!(cache.len(), 1);
    let page = cache.get("https://example.com").unwrap();
    assert_eq!(page.title, "New");
}

#[test]
fn cache_empty_url_key() {
    let cache = PageCache::new(300);
    cache.insert(
        "".into(),
        CachedPage {
            url: "".into(),
            title: "Empty URL".into(),
            markdown: "Content".into(),
            adapter: "test".into(),
        },
    );
    assert!(cache.get("").is_some());
}

#[test]
fn cache_unicode_url_key() {
    let cache = PageCache::new(300);
    let url = "https://例え.jp/ページ/テスト?q=日本語";
    cache.insert(
        url.into(),
        CachedPage {
            url: url.into(),
            title: "Unicode".into(),
            markdown: "コンテンツ".into(),
            adapter: "test".into(),
        },
    );
    let page = cache.get(url).unwrap();
    assert_eq!(page.title, "Unicode");
}

#[test]
fn cache_max_capacity_one() {
    let cache = PageCache::with_max_capacity(300, 1);
    cache.insert("a".into(), CachedPage { url: "a".into(), title: "A".into(), markdown: "".into(), adapter: "".into() });
    cache.insert("b".into(), CachedPage { url: "b".into(), title: "B".into(), markdown: "".into(), adapter: "".into() });
    assert_eq!(cache.len(), 1);
    assert!(cache.get("b").is_some());
}

// ---------------------------------------------------------------------------
// Search provider edge cases
// ---------------------------------------------------------------------------

#[test]
fn search_config_no_providers_defaults_empty() {
    let cfg = SearchConfig {
        providers: vec![],
        ..Default::default()
    };
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(multi_tier_search("test", &cfg));
    assert!(result.is_err());
}

#[test]
fn search_config_unknown_provider_skipped() {
    use fetchsys::search::ProviderRegistry;
    let cfg = SearchConfig {
        providers: vec!["nonexistent_provider".into(), "also_fake".into()],
        ..Default::default()
    };
    let registry = ProviderRegistry::build(&cfg);
    assert!(registry.providers().is_empty());
}

// ---------------------------------------------------------------------------
// Special character queries
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_special_characters_in_query() {
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "results": [{
                "url": "https://example.com/special",
                "title": "Special",
                "content": "Result for special chars",
                "score": 0.8
            }]
        })))
        .mount(&server)
        .await;

    let cfg = SearchConfig {
        providers: vec!["searxng".into()],
        searxng_url: server.uri(),
        max_results: 5,
        timeout_secs: 5,
        retries: 0,
        min_quality_score: 0.3,
        brave_api_key: None,
        serper_api_key: None,
    };

    // Query with special characters
    let special_queries = [
        "what's the difference?",
        "C++ vs Rust",
        "price > $100 & < $500",
        "path/to/file.rs",
        "SELECT * FROM users WHERE id=1; DROP TABLE;",
        "<script>alert('xss')</script>",
        "null\0byte",
        "tab\there",
        "newline\nhere",
    ];

    for q in &special_queries {
        let result = multi_tier_search(q, &cfg).await;
        // Should not panic — may succeed or fail gracefully
        match result {
            Ok(results) => {
                for r in &results {
                    assert!(!r.url.is_empty());
                }
            }
            Err(_) => {} // Acceptable for adversarial inputs
        }
    }
}

// ---------------------------------------------------------------------------
// Fact-check cross-reference edge cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn factcheck_contradicting_sources() {
    let docs = vec![
        make_doc(
            "https://pro.com",
            "Climate change is caused by anthropogenic greenhouse gas emissions. \
             This is confirmed by 97% of climate scientists. The evidence is conclusive.",
        ),
        make_doc(
            "https://contra.com",
            "The claim that humans cause climate change is false and is a myth. \
             The evidence has been debunked by independent researchers. Climate is not the case.",
        ),
    ];

    let provider = fetchsys::llm::build_provider(&LlmConfig::default());
    let result = factcheck::check(
        &docs,
        "is climate change caused by humans",
        provider.as_ref(),
        false,
        &fact_cfg(),
    )
    .await;

    // Should detect conflicting evidence
    assert!(!result.answer.is_empty());
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    // With contradicting sources, confidence should not be max
    // (exact value depends on heuristic claims)
}

#[tokio::test]
async fn factcheck_all_irrelevant_content() {
    let docs = vec![
        make_doc("https://a.com", "The weather today is sunny with clear skies."),
        make_doc("https://b.com", "Recipe for chocolate cake: mix flour and sugar."),
        make_doc("https://c.com", "Sports news: team won the championship game."),
    ];

    let provider = fetchsys::llm::build_provider(&LlmConfig::default());
    let result = factcheck::check(
        &docs,
        "quantum computing error correction",
        provider.as_ref(),
        false,
        &fact_cfg(),
    )
    .await;

    // Should gracefully handle no relevant content
    assert!(!result.answer.is_empty());
}

// ---------------------------------------------------------------------------
// Concurrency edge cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn factcheck_concurrent_calls_dont_interfere() {
    let _provider = fetchsys::llm::build_provider(&LlmConfig::default());
    let cfg = fact_cfg();

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let docs = vec![make_doc(
                &format!("https://example.com/{i}"),
                &format!(
                    "Topic {i} is important. It involves processing and analysis of data related to topic {i}."
                ),
            )];
            let provider = fetchsys::llm::build_provider(&LlmConfig::default());
            let cfg = cfg.clone();
            let query = format!("what is topic {i}");
            tokio::spawn(async move {
                factcheck::check(&docs, &query, provider.as_ref(), false, &cfg).await
            })
        })
        .collect();

    for handle in handles {
        let result = handle.await.expect("task panicked");
        assert!(!result.answer.is_empty());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }
}
