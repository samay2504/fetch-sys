//! Application configuration.
//!
//! Reads from (in ascending priority):
//!   1. `config/default.toml` (committed, no secrets)
//!   2. `config/local.toml`   (gitignored, local overrides)
//!   3. Environment variables  prefixed with `FETCHSYS_` (or provider-specific names)
//!   4. CLI flags override env in the `Opts` struct
//!
//! The `Config` struct is the single source of truth passed through the app.

use anyhow::Context;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sub-structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Ordered list of provider names to attempt (tier ordering).
    pub providers: Vec<String>,
    /// Minimum quality score (0.0–1.0) to accept results from a tier.
    pub min_quality_score: f64,
    /// Maximum results to return per search.
    pub max_results: usize,
    /// SearXNG base URL.
    pub searxng_url: String,
    /// Brave Search API key.
    pub brave_api_key: Option<String>,
    /// Serper.dev API key (Tier 3).
    pub serper_api_key: Option<String>,
    /// HTTP request timeout in seconds.
    pub timeout_secs: u64,
    /// Number of retries on transient failures.
    pub retries: u32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            providers: vec![
                "searxng".into(),
                "duckduckgo".into(),
                "brave".into(),
                "bing".into(),
                "serper".into(),
            ],
            min_quality_score: 0.3,
            max_results: 10,
            searxng_url: "http://localhost:8888".into(),
            brave_api_key: None,
            serper_api_key: None,
            timeout_secs: 10,
            retries: 2,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ReaderConfig {
    /// Base URL for Jina reader (Tier 1).
    pub jina_reader_base: String,
    /// Base URL for Jina search (alternative).
    pub jina_search_base: String,
    /// Jina API key (optional — free tier available, key grants higher limits).
    pub jina_api_key: Option<String>,
    /// Timeout for reader HTTP requests.
    pub timeout_secs: u64,
    /// Hard deadline (seconds) for the entire reader stage (all URLs combined).
    /// Prevents a single slow URL from blocking the whole pipeline.
    pub reader_deadline_secs: u64,
    /// Max content length in bytes to accept per page.
    pub max_content_bytes: usize,
    /// Retries on transient failures.
    pub retries: u32,
}

impl Default for ReaderConfig {
    fn default() -> Self {
        Self {
            jina_reader_base: "https://r.jina.ai".into(),
            jina_search_base: "https://s.jina.ai".into(),
            jina_api_key: None,
            timeout_secs: 10,
            reader_deadline_secs: 25,
            max_content_bytes: 200_000,
            retries: 1,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct LlmConfig {
    /// Whether LLM synthesis is enabled at all.
    pub enabled: bool,

    /// Ordered provider preference list. Each name maps to a concrete provider.
    /// Valid values: google_gemini, groq, openrouter, ollama, openai, local, fallback
    pub provider_preference: Vec<String>,

    // ── OpenAI-compatible (openai / openai_compat) ──
    /// OpenAI-compatible API base URL.
    pub base_url: String,
    /// API key for the OpenAI-compat endpoint.
    pub api_key: Option<String>,
    /// Model name for the OpenAI-compat endpoint.
    pub model: String,

    // ── Google Gemini ──
    pub google_api_key: Option<String>,
    /// Gemini model name (e.g. gemini-2.5-flash, gemini-1.5-pro).
    pub google_model: String,

    // ── Groq ──
    pub groq_api_key: Option<String>,
    /// Ordered list of Groq models to try (each is a separate chain entry).
    pub groq_models: Vec<String>,

    // ── OpenRouter ──
    pub openrouter_api_key: Option<String>,
    /// OpenRouter model slug.
    pub openrouter_model: String,

    // ── Ollama ──
    /// Ollama server base URL (no trailing slash, no /v1).
    pub ollama_url: String,
    /// Ollama model name.
    pub ollama_model: String,

    // ── Shared inference knobs ──
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// HTTP timeout in seconds (applies to all providers).
    pub timeout_secs: u64,

    // ── Circuit breaker ──
    /// Consecutive failures before a provider's circuit opens.
    pub circuit_breaker_threshold: u32,
    /// Seconds a provider stays in cooldown after the circuit opens.
    pub circuit_breaker_cooldown_secs: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider_preference: vec![
                "google_gemini".into(),
                "groq".into(),
                "openrouter".into(),
                "ollama".into(),
                "openai".into(),
                "local".into(),
                "fallback".into(),
            ],
            base_url: "https://api.openai.com/v1".into(),
            api_key: None,
            model: "gpt-4o-mini".into(),
            google_api_key: None,
            google_model: "gemini-2.5-flash".into(),
            groq_api_key: None,
            groq_models: vec![
                "llama-3.3-70b-versatile".into(),
                "llama-3.1-8b-instant".into(),
                "mixtral-8x7b-32768".into(),
            ],
            openrouter_api_key: None,
            openrouter_model: "openai/gpt-4o-mini".into(),
            ollama_url: "http://localhost:11434".into(),
            ollama_model: "llama3.2".into(),
            max_tokens: 1024,
            temperature: 0.2,
            timeout_secs: 60,
            circuit_breaker_threshold: 3,
            circuit_breaker_cooldown_secs: 60,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct FactCheckConfig {
    /// Minimum fraction of sources that must agree for "confirmed" status (0.0–1.0).
    pub confirm_threshold: f64,
    /// Minimum fraction for "contradicted" status.
    pub contradict_threshold: f64,
}

impl Default for FactCheckConfig {
    fn default() -> Self {
        Self {
            confirm_threshold: 0.6,
            contradict_threshold: 0.4,
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default)]
pub struct Config {
    pub search: SearchConfig,
    pub reader: ReaderConfig,
    pub llm: LlmConfig,
    pub factcheck: FactCheckConfig,
    /// Number of top search results to read (can be overridden by CLI --top-n).
    pub top_n: usize,
}

impl Config {
    /// Load config from files + environment.
    ///
    /// Files are optional; if they don't exist the defaults are used.
    pub fn load(config_path: Option<&str>) -> anyhow::Result<Self> {
        // Load optional .env (silently ignored if missing)
        let _ = dotenvy::dotenv();

        let mut builder = config::Config::builder();

        // Built-in defaults
        let default_path = "config/default.toml";
        if std::path::Path::new(default_path).exists() {
            builder = builder.add_source(config::File::with_name(default_path));
        }

        // Local overrides
        let local_path = "config/local.toml";
        if std::path::Path::new(local_path).exists() {
            builder = builder.add_source(config::File::with_name(local_path));
        }

        // Explicit CLI-supplied path
        if let Some(path) = config_path {
            builder = builder.add_source(config::File::with_name(path));
        }

        // Environment variable overrides (FETCHSYS__ prefix with double-underscore for nesting)
        builder = builder.add_source(
            config::Environment::with_prefix("FETCHSYS")
                .separator("__")
                .ignore_empty(true),
        );

        // Provider-specific env vars mapped into nested config keys
        let env_mappings: &[(&str, &str)] = &[
            ("SEARXNG_URL",              "search.searxng_url"),
            ("BRAVE_API_KEY",            "search.brave_api_key"),
            ("SERPER_API_KEY",           "search.serper_api_key"),
            ("LLM_BASE_URL",             "llm.base_url"),
            ("LLM_API_KEY",              "llm.api_key"),
            ("LLM_MODEL",                "llm.model"),
            ("GOOGLE_API_KEY",           "llm.google_api_key"),
            ("GEMINI_MODEL",             "llm.google_model"),
            ("GROQ_API_KEY",             "llm.groq_api_key"),
            ("OPENROUTER_API_KEY",       "llm.openrouter_api_key"),
            ("OPENROUTER_MODEL",         "llm.openrouter_model"),
            ("OLLAMA_URL",               "llm.ollama_url"),
            ("OLLAMA_MODEL",             "llm.ollama_model"),
            ("JINA_API_KEY",             "reader.jina_api_key"),
        ];
        for (env_var, config_key) in env_mappings {
            if let Ok(val) = std::env::var(env_var) {
                builder = builder.set_override(*config_key, val)?;
            }
        }

        let cfg = builder
            .build()
            .context("Failed to build configuration")?
            .try_deserialize::<Config>()
            .context("Failed to deserialize configuration")?;

        Ok(cfg)
    }

    /// Apply CLI flag overrides on top of the loaded config.
    pub fn apply_cli_overrides(&mut self, opts: &crate::cli::Opts) {
        if let Some(ref providers) = opts.providers {
            self.search.providers.clone_from(providers);
        }
        if let Some(ref url) = opts.searxng_url {
            self.search.searxng_url.clone_from(url);
        }
        if let Some(ref key) = opts.brave_api_key {
            self.search.brave_api_key = Some(key.clone());
        }
        if let Some(ref key) = opts.serper_api_key {
            self.search.serper_api_key = Some(key.clone());
        }
        if let Some(ref url) = opts.llm_base_url {
            self.llm.base_url.clone_from(url);
        }
        if let Some(ref key) = opts.llm_api_key {
            self.llm.api_key = Some(key.clone());
        }
        if let Some(ref model) = opts.llm_model {
            self.llm.model.clone_from(model);
        }
        if opts.no_llm {
            self.llm.enabled = false;
        }
        self.search.min_quality_score = opts.min_quality;
        self.top_n = opts.top_n;
    }
}
