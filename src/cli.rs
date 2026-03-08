//! CLI argument definitions via `clap` derive.
//!
//! Add new top-level flags here; keep this file focused on parsing only —
//! no business logic.

use clap::Parser;

/// Developer @tool for agent systems — multi-tier search, reading, and LLM fact-checking.
///
/// Examples:
///   fetchsys "who is Ada Lovelace"
///   fetchsys --json "latest Rust release"
///   fetchsys --providers searxng,brave "quantum computing news"
#[derive(Parser, Debug, Clone)]
#[command(
    name = "fetchsys",
    author,
    version,
    about = "Developer @tool: multi-tier search + LLM fact-check, designed for use inside agent systems",
    long_about = None,
    propagate_version = true,
)]
pub struct Opts {
    /// The search query (one or more words joined into a single query string).
    #[arg(required = true, num_args = 1.., value_name = "QUERY")]
    pub query: Vec<String>,

    /// Output strict JSON schema (agent/pipeline mode) instead of streamed Markdown.
    #[arg(long, short = 'j', env = "FETCHSYS_JSON")]
    pub json: bool,

    /// Maximum number of top URLs to read and fact-check (default: 5).
    #[arg(long, default_value = "5", env = "FETCHSYS_TOP_N")]
    pub top_n: usize,

    /// Comma-separated ordered list of search provider names to attempt.
    /// Valid values: searxng, duckduckgo, brave, bing, google, serper
    /// Defaults to the order defined in config.
    #[arg(long, value_delimiter = ',', env = "FETCHSYS_PROVIDERS")]
    pub providers: Option<Vec<String>>,

    /// SearXNG base URL (e.g. http://localhost:8888).
    #[arg(long, env = "SEARXNG_URL")]
    pub searxng_url: Option<String>,

    /// Brave Search API key.
    #[arg(long, env = "BRAVE_API_KEY")]
    pub brave_api_key: Option<String>,

    /// Serper.dev API key (Tier 3 paid fallback).
    #[arg(long, env = "SERPER_API_KEY")]
    pub serper_api_key: Option<String>,

    /// OpenAI-compatible LLM base URL (default: https://api.openai.com/v1).
    #[arg(long, env = "LLM_BASE_URL")]
    pub llm_base_url: Option<String>,

    /// LLM API key.
    #[arg(long, env = "LLM_API_KEY")]
    pub llm_api_key: Option<String>,

    /// LLM model name (default: gpt-4o-mini or local model path).
    #[arg(long, env = "LLM_MODEL")]
    pub llm_model: Option<String>,

    /// Skip LLM synthesis; return raw aggregated snippets only.
    #[arg(long)]
    pub no_llm: bool,

    /// Minimum quality score (0.0–1.0) before falling back to next search tier.
    #[arg(long, default_value = "0.3", env = "FETCHSYS_MIN_QUALITY")]
    pub min_quality: f64,

    /// Enable debug / verbose structured logging.
    #[arg(long, short = 'v')]
    pub verbose: bool,

    /// Output structured logs as JSON (useful when run inside a pipeline).
    #[arg(long)]
    pub log_json: bool,

    /// Path to TOML config file (default: config/default.toml if present).
    #[arg(long, env = "FETCHSYS_CONFIG")]
    pub config: Option<String>,
}

impl Opts {
    /// Convenience: join the query tokens into a single trimmed string.
    pub fn query_string(&self) -> String {
        self.query.join(" ").trim().to_string()
    }
}
