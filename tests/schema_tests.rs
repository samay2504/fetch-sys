//! Schema contract tests — ensure AgentResponse serialises to / deserialises from
//! the strict JSON shape documented in the PRD.

use fetchsys::schema::{
    validate_schema, AgentResponse, Answer, FactCheck, FactCheckStatus, Metadata, Source,
};

fn sample_response() -> AgentResponse {
    AgentResponse {
        query: "who is ada lovelace".into(),
        answer: Answer {
            text: "Ada Lovelace was a 19th-century mathematician.".into(),
            confidence: 0.85,
        },
        sources: vec![Source {
            url: "https://en.wikipedia.org/wiki/Ada_Lovelace".into(),
            title: "Ada Lovelace - Wikipedia".into(),
            snippet: "Ada Lovelace is considered the first computer programmer.".into(),
            rank: 1,
            reader_raw: "Full page content…".into(),
        }],
        fact_checks: vec![FactCheck {
            claim: "Ada Lovelace was born in 1815".into(),
            status: FactCheckStatus::Confirmed,
            evidence_urls: vec!["https://en.wikipedia.org/wiki/Ada_Lovelace".into()],
        }],
        metadata: Metadata {
            timestamp: chrono::Utc::now(),
            providers_used: vec!["searxng".into(), "r.jina.ai".into()],
            latency_ms: 320,
        },
    }
}

#[test]
fn agent_response_serialises_and_validates() {
    let resp = sample_response();
    let json_str = serde_json::to_string(&resp).expect("serialisation failed");
    let value: serde_json::Value =
        serde_json::from_str(&json_str).expect("deserialisation failed");

    validate_schema(&value).expect("schema validation failed");
}

#[test]
fn agent_response_roundtrip() {
    let original = sample_response();
    let json_str = serde_json::to_string(&original).unwrap();
    let recovered: AgentResponse = serde_json::from_str(&json_str).unwrap();

    assert_eq!(original.query, recovered.query);
    assert!((original.answer.confidence - recovered.answer.confidence).abs() < 1e-9);
    assert_eq!(original.sources.len(), recovered.sources.len());
    assert_eq!(original.fact_checks.len(), recovered.fact_checks.len());
    assert_eq!(original.metadata.latency_ms, recovered.metadata.latency_ms);
}

#[test]
fn validate_schema_fails_on_missing_key() {
    let mut value = serde_json::json!({
        "query": "test",
        "answer": {"text": "x", "confidence": 0.5},
        "sources": [],
        "fact_checks": [],
        // metadata intentionally missing
    });
    assert!(validate_schema(&value).is_err());

    // Fix it
    value["metadata"] = serde_json::json!({
        "timestamp": "2026-03-02T00:00:00Z",
        "providers_used": [],
        "latency_ms": 0,
    });
    assert!(validate_schema(&value).is_ok());
}

#[test]
fn validate_schema_fails_on_missing_source_field() {
    let value = serde_json::json!({
        "query": "test",
        "answer": {"text": "x", "confidence": 0.5},
        "sources": [{"url": "https://x.com", "title": "X", "snippet": "s"}], // rank missing
        "fact_checks": [],
        "metadata": {
            "timestamp": "2026-03-02T00:00:00Z",
            "providers_used": [],
            "latency_ms": 0,
        }
    });
    assert!(validate_schema(&value).is_err());
}

#[test]
fn fact_check_status_roundtrip() {
    for status in [
        FactCheckStatus::Confirmed,
        FactCheckStatus::Contradicted,
        FactCheckStatus::Inconclusive,
    ] {
        let json = serde_json::to_string(&status).unwrap();
        let back: FactCheckStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, back);
    }
}
