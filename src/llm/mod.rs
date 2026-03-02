//! LLM provider abstraction layer.
//!
//! Provider waterfall (highest priority first):
//!   google_gemini → groq → openrouter → ollama → openai_compat → local → fallback
//!
//! Each provider is wrapped by a lightweight circuit breaker: after
//! `circuit_breaker_threshold` consecutive failures the provider is cooled
//! down for `circuit_breaker_cooldown_secs` before being retried.
//!
//! Adding a new provider: implement [`LlmProvider`] and register it in
//! [`build_provider_chain`].

use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use anyhow::Context;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::config::LlmConfig;

// ---------------------------------------------------------------------------
// Message / response types
// ---------------------------------------------------------------------------

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
}

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &'static str;

    /// Complete a conversation and return the full assistant response.
    async fn complete(&self, messages: Vec<Message>) -> anyhow::Result<String>;
}

// ---------------------------------------------------------------------------
// Circuit breaker (per-provider)
// ---------------------------------------------------------------------------

/// Lightweight per-provider circuit breaker.
///
/// Opens after `threshold` consecutive failures; resets after `cooldown`.
#[derive(Clone)]
struct CircuitBreaker {
    failures: Arc<AtomicU32>,
    threshold: u32,
    cooldown: Duration,
    opened_at: Arc<Mutex<Option<Instant>>>,
}

impl CircuitBreaker {
    fn new(threshold: u32, cooldown_secs: u64) -> Self {
        Self {
            failures: Arc::new(AtomicU32::new(0)),
            threshold,
            cooldown: Duration::from_secs(cooldown_secs),
            opened_at: Arc::new(Mutex::new(None)),
        }
    }

    /// Returns `true` if the circuit is currently open (provider unavailable).
    fn is_open(&self) -> bool {
        if let Ok(guard) = self.opened_at.lock() {
            if let Some(t) = *guard {
                return t.elapsed() < self.cooldown;
            }
        }
        false
    }

    /// Record a success — reset failure counter and close circuit.
    fn record_success(&self) {
        self.failures.store(0, Ordering::Relaxed);
        if let Ok(mut guard) = self.opened_at.lock() {
            *guard = None;
        }
    }

    /// Record a failure — open circuit if threshold exceeded.
    fn record_failure(&self) {
        let prev = self.failures.fetch_add(1, Ordering::Relaxed);
        if prev + 1 >= self.threshold {
            if let Ok(mut guard) = self.opened_at.lock() {
                if guard.is_none() {
                    *guard = Some(Instant::now());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics (lightweight, lock-free counters)
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct ProviderMetrics {
    pub invocations: u64,
    pub failures: u64,
    pub fallbacks: u64,
}

// ---------------------------------------------------------------------------
// OpenAI-compatible HTTP provider
// ---------------------------------------------------------------------------
// Covers: OpenAI, Groq, OpenRouter, Ollama, Azure OpenAI, LM Studio, etc.
// All of these expose the same `/chat/completions` endpoint format.

struct OpenAiCompatProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    /// Extra headers sent with every request (e.g. `HTTP-Referer` for OpenRouter).
    extra_headers: Vec<(String, String)>,
    model: String,
    max_tokens: u32,
    temperature: f32,
    label: &'static str,
}

impl OpenAiCompatProvider {
    fn new(
        label: &'static str,
        base_url: impl Into<String>,
        api_key: Option<String>,
        model: impl Into<String>,
        max_tokens: u32,
        temperature: f32,
        timeout_secs: u64,
        extra_headers: Vec<(String, String)>,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .user_agent("fetchsys/0.1")
            .build()
            .expect("Failed to build HTTP client");
        Self {
            client,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key,
            extra_headers,
            model: model.into(),
            max_tokens,
            temperature,
            label,
        }
    }
}

#[derive(Deserialize)]
struct OaiResponse {
    choices: Vec<OaiChoice>,
}
#[derive(Deserialize)]
struct OaiChoice {
    message: OaiMessage,
}
#[derive(Deserialize)]
struct OaiMessage {
    content: String,
}

#[async_trait]
impl LlmProvider for OpenAiCompatProvider {
    fn name(&self) -> &'static str {
        self.label
    }

    async fn complete(&self, messages: Vec<Message>) -> anyhow::Result<String> {
        let url = format!("{}/chat/completions", self.base_url);
        debug!(%url, model = %self.model, provider = %self.label, "LLM request");

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": false,
        });

        let mut req = self.client.post(&url).json(&body);
        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }
        for (k, v) in &self.extra_headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let resp = req.send().await.context("LLM HTTP request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            let category = http_error_category(status.as_u16());
            warn!(
                provider = %self.label,
                model = %self.model,
                status = status.as_u16(),
                %category,
                body = %body_text.chars().take(300).collect::<String>(),
                "LLM endpoint error"
            );
            anyhow::bail!(
                "{} HTTP {} ({}) — {}",
                self.label, status.as_u16(), category,
                body_text.chars().take(120).collect::<String>()
            );
        }

        let parsed: OaiResponse = resp
            .json()
            .await
            .context("Failed to parse LLM JSON response")?;

        parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .context("LLM response contained no choices")
    }
}

// ---------------------------------------------------------------------------
// Google Gemini provider (native REST — not OpenAI-compat)
// ---------------------------------------------------------------------------

struct GeminiProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    max_tokens: u32,
    temperature: f32,
}

impl GeminiProvider {
    fn new(api_key: String, model: String, max_tokens: u32, temperature: f32, timeout_secs: u64) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .user_agent("fetchsys/0.1")
            .build()
            .expect("Failed to build Gemini HTTP client");
        Self { client, api_key, model, max_tokens, temperature }
    }

    /// Split messages into an optional `systemInstruction` text and the
    /// conversation `contents` array for the Gemini REST API.
    ///
    /// Gemini's `contents` must alternate user/model roles. System messages
    /// (role == "system") are collected into the top-level `systemInstruction`
    /// field; they must NOT appear in `contents`.
    fn build_gemini_body(messages: &[Message], max_tokens: u32, temperature: f32) -> serde_json::Value {
        let system_text: String = messages
            .iter()
            .filter(|m| m.role == "system")
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let contents: Vec<serde_json::Value> = messages
            .iter()
            .filter(|m| m.role != "system")
            .map(|m| {
                let role = if m.role == "assistant" { "model" } else { "user" };
                serde_json::json!({
                    "role": role,
                    "parts": [{ "text": m.content }]
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            }
        });

        if !system_text.is_empty() {
            body["systemInstruction"] = serde_json::json!({
                "parts": [{ "text": system_text }]
            });
        }

        body
    }
}

#[async_trait]
impl LlmProvider for GeminiProvider {
    fn name(&self) -> &'static str {
        "google_gemini"
    }

    async fn complete(&self, messages: Vec<Message>) -> anyhow::Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );
        debug!(model = %self.model, "Gemini request");

        let body = Self::build_gemini_body(&messages, self.max_tokens, self.temperature);

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Gemini HTTP request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            let category = http_error_category(status.as_u16());
            warn!(
                model = %self.model,
                status = status.as_u16(),
                %category,
                body = %body_text.chars().take(300).collect::<String>(),
                "Gemini API error"
            );
            anyhow::bail!(
                "Gemini HTTP {} ({}) — {}",
                status.as_u16(), category,
                body_text.chars().take(120).collect::<String>()
            );
        }

        #[derive(Deserialize)]
        struct GeminiResp { candidates: Vec<GeminiCandidate> }
        #[derive(Deserialize)]
        struct GeminiCandidate { content: GeminiContent }
        #[derive(Deserialize)]
        struct GeminiContent { parts: Vec<GeminiPart> }
        #[derive(Deserialize)]
        struct GeminiPart { text: String }

        let parsed: GeminiResp = resp.json().await.context("Failed to parse Gemini response")?;
        parsed
            .candidates
            .into_iter()
            .next()
            .and_then(|c| c.content.parts.into_iter().next())
            .map(|p| p.text)
            .context("Gemini response contained no candidates")
    }
}

// ---------------------------------------------------------------------------
// Local inference stub (candle / llm crate integration point)
// ---------------------------------------------------------------------------

/// Placeholder local provider.
/// Replace `complete` body with `candle` or `llm` crate inference.
pub struct LocalLlmProvider {
    pub model_path: String,
}

#[async_trait]
impl LlmProvider for LocalLlmProvider {
    fn name(&self) -> &'static str {
        "local"
    }

    async fn complete(&self, messages: Vec<Message>) -> anyhow::Result<String> {
        warn!(model = %self.model_path, "Local LLM is a stub — integrate `candle` or `llm` crate here.");
        let last = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("unknown");
        Ok(format!(
            "[LOCAL-STUB] Query: «{last}». \
             Replace with real inference via `candle` or `llm` crate."
        ))
    }
}

// ---------------------------------------------------------------------------
// Fallback echo provider (last resort — always succeeds)
// ---------------------------------------------------------------------------

struct FallbackProvider;

#[async_trait]
impl LlmProvider for FallbackProvider {
    fn name(&self) -> &'static str {
        "fallback"
    }

    async fn complete(&self, messages: Vec<Message>) -> anyhow::Result<String> {
        let q = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");
        warn!("All LLM providers exhausted — using echo fallback");
        Ok(format!(
            "[FALLBACK] No LLM provider available. Query was: «{q}». \
             Configure GOOGLE_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, or LLM_API_KEY."
        ))
    }
}

// ---------------------------------------------------------------------------
// Fallback chain provider (ties everything together)
// ---------------------------------------------------------------------------

/// Wraps an ordered list of providers. On each `complete` call:
///  1. Skip providers whose circuit is open (cooling down).
///  2. Try; on success reset that circuit and return.
///  3. On failure record it and try the next provider.
///  4. The last entry is always `FallbackProvider` which never fails.
pub struct FallbackChainProvider {
    entries: Vec<(Box<dyn LlmProvider>, CircuitBreaker)>,
}

impl FallbackChainProvider {
    fn new(entries: Vec<(Box<dyn LlmProvider>, CircuitBreaker)>) -> Self {
        Self { entries }
    }

    /// Name of the provider that would currently be attempted first.
    pub fn active_provider_name(&self) -> &'static str {
        self.entries
            .iter()
            .find(|(_, cb)| !cb.is_open())
            .map(|(p, _)| p.name())
            .unwrap_or("none")
    }
}

#[async_trait]
impl LlmProvider for FallbackChainProvider {
    fn name(&self) -> &'static str {
        "chain"
    }

    async fn complete(&self, messages: Vec<Message>) -> anyhow::Result<String> {
        let mut last_err = anyhow::anyhow!("No providers available");

        for (provider, cb) in &self.entries {
            if cb.is_open() {
                warn!(provider = %provider.name(), "Circuit open; skipping");
                continue;
            }

            match provider.complete(messages.clone()).await {
                Ok(text) => {
                    cb.record_success();
                    info!(provider = %provider.name(), "LLM provider succeeded");
                    return Ok(text);
                }
                Err(e) => {
                    cb.record_failure();
                    warn!(provider = %provider.name(), error = %e, "Provider failed; trying next");
                    last_err = e;
                }
            }
        }

        Err(last_err)
    }
}

// ---------------------------------------------------------------------------
// HTTP error categorisation (mirrors llm_wrapper.py _should_trigger_fallback)
// ---------------------------------------------------------------------------

/// Map an HTTP status code to a human-readable category for structured logging.
/// Mirrors the logic in `llm_wrapper.py::_should_trigger_fallback`:
///   401/403 → auth   429 → rate_limit   404 → model_not_found
///   400 → bad_request   5xx → server_error
fn http_error_category(status: u16) -> &'static str {
    match status {
        401 | 403 => "auth",
        402         => "payment_required",
        429         => "rate_limit",
        404         => "model_not_found",
        400         => "bad_request",
        500..=599   => "server_error",
        _           => "client_error",
    }
}

// ---------------------------------------------------------------------------
// Factory — build_provider_chain
// ---------------------------------------------------------------------------

/// Build the provider chain from config + environment variables.
///
/// Provider preference order (from `cfg.provider_preference`):
///   `google_gemini` | `groq` | `openrouter` | `ollama` | `openai` | `local` | `fallback`
///
/// Any provider whose required key/URL is absent is silently skipped.
/// The `fallback` echo provider is always appended last.
pub fn build_provider(cfg: &LlmConfig) -> Box<dyn LlmProvider> {
    build_provider_chain(cfg)
}

/// Preferred public constructor — returns `Box<dyn LlmProvider>` backed by a
/// `FallbackChainProvider`.
pub fn build_provider_chain(cfg: &LlmConfig) -> Box<dyn LlmProvider> {
    let cb = || CircuitBreaker::new(cfg.circuit_breaker_threshold, cfg.circuit_breaker_cooldown_secs);

    let mut entries: Vec<(Box<dyn LlmProvider>, CircuitBreaker)> = Vec::new();

    for name in &cfg.provider_preference {
        match name.as_str() {
            "google_gemini" => {
                if let Some(key) = cfg.google_api_key.clone().or_else(|| std::env::var("GOOGLE_API_KEY").ok()) {
                    let model = std::env::var("GEMINI_MODEL")
                        .unwrap_or_else(|_| cfg.google_model.clone());
                    info!(%model, "Registering Google Gemini provider");
                    entries.push((
                        Box::new(GeminiProvider::new(key, model, cfg.max_tokens, cfg.temperature, cfg.timeout_secs)),
                        cb(),
                    ));
                }
            }
            "groq" => {
                if let Some(key) = cfg.groq_api_key.clone().or_else(|| std::env::var("GROQ_API_KEY").ok()) {
                    // Try multiple Groq models in order; the chain will handle failures
                    for model in &cfg.groq_models {
                        let label: &'static str = Box::leak(
                            format!("groq/{model}").into_boxed_str()
                        );
                        info!(%model, "Registering Groq provider");
                        entries.push((
                            Box::new(OpenAiCompatProvider::new(
                                label,
                                "https://api.groq.com/openai/v1",
                                Some(key.clone()),
                                model.clone(),
                                cfg.max_tokens,
                                cfg.temperature,
                                cfg.timeout_secs,
                                vec![],
                            )),
                            cb(),
                        ));
                    }
                }
            }
            "openrouter" => {
                if let Some(key) = cfg.openrouter_api_key.clone().or_else(|| std::env::var("OPENROUTER_API_KEY").ok()) {
                    let model = std::env::var("OPENROUTER_MODEL")
                        .unwrap_or_else(|_| cfg.openrouter_model.clone());
                    info!(%model, "Registering OpenRouter provider");
                    entries.push((
                        Box::new(OpenAiCompatProvider::new(
                            "openrouter",
                            "https://openrouter.ai/api/v1",
                            Some(key),
                            model,
                            cfg.max_tokens,
                            cfg.temperature,
                            cfg.timeout_secs,
                            vec![
                                ("HTTP-Referer".into(), "https://github.com/your-org/fetchsys".into()),
                                ("X-Title".into(), "fetchsys".into()),
                            ],
                        )),
                        cb(),
                    ));
                }
            }
            "ollama" => {
                // Ollama's OpenAI-compat endpoint — no key required
                let base = std::env::var("OLLAMA_URL")
                    .unwrap_or_else(|_| cfg.ollama_url.clone());
                let model = std::env::var("OLLAMA_MODEL")
                    .unwrap_or_else(|_| cfg.ollama_model.clone());
                info!(%base, %model, "Registering Ollama provider");
                entries.push((
                    Box::new(OpenAiCompatProvider::new(
                        "ollama",
                        format!("{base}/v1"),
                        None,
                        model,
                        cfg.max_tokens,
                        cfg.temperature,
                        cfg.timeout_secs,
                        vec![],
                    )),
                    cb(),
                ));
            }
            "openai" | "openai_compat" => {
                if let Some(key) = cfg.api_key.clone().or_else(|| std::env::var("LLM_API_KEY").ok()) {
                    let base = std::env::var("LLM_BASE_URL")
                        .unwrap_or_else(|_| cfg.base_url.clone());
                    let model = std::env::var("LLM_MODEL")
                        .unwrap_or_else(|_| cfg.model.clone());
                    info!(%base, %model, "Registering OpenAI-compat provider");
                    entries.push((
                        Box::new(OpenAiCompatProvider::new(
                            "openai_compat",
                            base,
                            Some(key),
                            model,
                            cfg.max_tokens,
                            cfg.temperature,
                            cfg.timeout_secs,
                            vec![],
                        )),
                        cb(),
                    ));
                }
            }
            "local" => {
                let path = std::env::var("LOCAL_MODEL_PATH")
                    .unwrap_or_else(|_| cfg.model.clone());
                info!(%path, "Registering local LLM stub");
                entries.push((Box::new(LocalLlmProvider { model_path: path }), cb()));
            }
            other => {
                // "fallback" is always auto-appended; other unknown names are skipped.
                if other != "fallback" {
                    warn!(%other, "Unknown LLM provider name; skipping");
                }
            }
        }
    }

    // Fallback echo provider is always last
    entries.push((Box::new(FallbackProvider), CircuitBreaker::new(u32::MAX, 0)));

    if entries.len() == 1 {
        warn!("No real LLM providers configured — only fallback echo will be used");
    }

    Box::new(FallbackChainProvider::new(entries))
}

// ---------------------------------------------------------------------------
// System prompts / helpers
// ---------------------------------------------------------------------------

/// Build the synthesis prompt for fact-grounded answers.
pub fn synthesis_prompt(query: &str, context: &str) -> Vec<Message> {
    vec![
        Message::system(
            "You are a precise research assistant. Given a user query and web source excerpts, \
             synthesise a concise, accurate answer in Markdown. \
             Cite sources inline using [N] notation. \
             If sources conflict, note the disagreement explicitly. \
             Never invent facts not present in the sources.",
        ),
        Message::user(format!(
            "Query: {query}\n\n--- Sources ---\n{context}\n\n--- End Sources ---\n\n\
             Provide a concise answer with citations:"
        )),
    ]
}

/// Build a context string from read documents for inclusion in a prompt.
pub fn build_context(docs: &[crate::reader::ReadResult], max_chars_per_doc: usize) -> String {
    docs.iter()
        .enumerate()
        .map(|(i, doc)| {
            let snippet = doc.snippet(max_chars_per_doc);
            format!("[{}] URL: {}\nTitle: {}\n{}", i + 1, doc.url, doc.title, snippet)
        })
        .collect::<Vec<_>>()
        .join("\n\n---\n\n")
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesis_prompt_includes_query() {
        let msgs = synthesis_prompt("Who is Ada Lovelace?", "content here");
        let user_msg = msgs.iter().find(|m| m.role == "user").unwrap();
        assert!(user_msg.content.contains("Who is Ada Lovelace?"));
    }

    #[test]
    fn build_context_includes_all_docs() {
        let docs = vec![
            crate::reader::ReadResult {
                url: "https://a.com".into(),
                title: "A".into(),
                content: "Content A".into(),
                adapter: "test".into(),
            },
            crate::reader::ReadResult {
                url: "https://b.com".into(),
                title: "B".into(),
                content: "Content B".into(),
                adapter: "test".into(),
            },
        ];
        let ctx = build_context(&docs, 500);
        assert!(ctx.contains("[1]"));
        assert!(ctx.contains("[2]"));
        assert!(ctx.contains("https://a.com"));
        assert!(ctx.contains("https://b.com"));
    }

    #[tokio::test]
    async fn local_stub_returns_string() {
        let provider = LocalLlmProvider { model_path: "dummy".into() };
        let resp = provider.complete(vec![Message::user("Hello")]).await.unwrap();
        assert!(resp.contains("LOCAL-STUB"));
    }

    #[tokio::test]
    async fn fallback_chain_uses_fallback_when_all_fail() {
        // Build a chain with only the fallback
        let entries: Vec<(Box<dyn LlmProvider>, CircuitBreaker)> = vec![
            (Box::new(FallbackProvider), CircuitBreaker::new(u32::MAX, 0)),
        ];
        let chain = FallbackChainProvider::new(entries);
        let resp = chain.complete(vec![Message::user("test")]).await.unwrap();
        assert!(resp.contains("FALLBACK"));
    }

    #[test]
    fn circuit_breaker_opens_after_threshold() {
        let cb = CircuitBreaker::new(3, 60);
        assert!(!cb.is_open());
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open()); // threshold not yet reached
        cb.record_failure();
        assert!(cb.is_open()); // opened
        cb.record_success();
        assert!(!cb.is_open()); // closed again
    }

    #[test]
    fn gemini_message_conversion_maps_roles() {
        let messages = vec![
            Message::system("Be helpful"),
            Message::user("Hello"),
            Message { role: "assistant".into(), content: "Hi!".into() },
        ];
        let body = GeminiProvider::build_gemini_body(&messages, 512, 0.5);
        // System message extracted into systemInstruction
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Be helpful");
        // contents only has user + model turns
        let arr = body["contents"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["role"], "user");
        assert_eq!(arr[1]["role"], "model");
    }
}
