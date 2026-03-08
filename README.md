# fetchsys

> A developer `@tool` for agent systems — multi-tier web search, page reading, and LLM fact-checking in a single CLI call.

[![CI](https://github.com/samay2504/fetchsys/actions/workflows/ci.yml/badge.svg)](https://github.com/samay2504/fetchsys/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

---

## Overview

`fetchsys` is a single-binary CLI tool designed to be called as a **`@tool` inside other AI agent pipelines** (LangChain, AutoGen, CrewAI, custom orchestrators, etc.).  It takes a query string and returns a rich, structured JSON response containing search results, page content, and LLM-fact-checked claims — all in one shot.

```sh
fetchsys "who is Ada Lovelace"
fetchsys --json "latest Rust release"   # strict JSON for agents
```

### What it does

1. **Searches** the web via a 6-tier waterfall (SearXNG → DuckDuckGo → Brave → Bing → Google → Serper)
2. **Reads** the top results into LLM-friendly markdown via a 5-tier waterfall (Jina → Readability+htmd → anytomd → Firecrawl → raw HTTP)
3. **Fact-checks** claims across sources, flagging low-confidence or contradicted statements
4. **Outputs** either streamed Markdown (human mode) or a strict JSON schema (agent/pipeline mode)

Cold-start latency target: **< 150 ms** (excluding network I/O).

---

## Using as an `@tool` in agent systems

`fetchsys` is designed to be shell-executed by an orchestrating agent. The `--json` flag emits a machine-readable response that your agent code can parse directly.

### Shell invocation

```sh
fetchsys --json "your query here"
```

### LangChain / Python

```python
import subprocess, json

def fetchsys_tool(query: str) -> dict:
    result = subprocess.run(
        ["fetchsys", "--json", query],
        capture_output=True, text=True, timeout=60
    )
    return json.loads(result.stdout)

from langchain.tools import Tool

search_tool = Tool(
    name="fetchsys",
    description=(
        "Search the web and fact-check claims about a topic. "
        "Input: a plain-English question or search query. "
        "Output: JSON with answer, sources, fact_checks, and confidence score."
    ),
    func=lambda q: json.dumps(fetchsys_tool(q)),
)
```

### AutoGen

```python
import subprocess, json

def fetchsys(query: str) -> str:
    """Search the web and return fact-checked results as JSON."""
    return subprocess.run(
        ["fetchsys", "--json", query],
        capture_output=True, text=True, timeout=60
    ).stdout
```

### CrewAI

```python
from crewai_tools import tool

@tool("fetchsys")
def fetchsys_search(query: str) -> str:
    """
    Search the web, read top pages, and return LLM-fact-checked results.
    Use this tool whenever you need current, cited information.
    """
    import subprocess
    return subprocess.run(
        ["fetchsys", "--json", query],
        capture_output=True, text=True, timeout=60
    ).stdout
```

### Useful jq one-liners

```sh
# Get just the answer text
fetchsys --json "who built Rust" | jq -r '.answer.text'

# Get fact-check statuses
fetchsys --json "climate change causes" | jq '.fact_checks[] | {claim, status}'

# Get confidence score
fetchsys --json "quantum entanglement" | jq '.answer.confidence'
```

---

## Quick start

### Prerequisites

- Rust 1.71+ (stable) via [rustup](https://rustup.rs)
- (Optional) A running [SearXNG](https://searxng.org/) instance for Tier-1 search

```sh
# Clone and build
git clone https://github.com/your-org/fetchsys
cd fetchsys
cargo build --release

# Copy and populate secrets
cp .env.example .env
$EDITOR .env
```

> **Windows note:** When using the `x86_64-pc-windows-gnu` toolchain (MSYS2 default), the binary is placed at
> `target\x86_64-pc-windows-gnu\release\fetchsys.exe` rather than `target\release\fetchsys.exe`.

### Run

```sh
# Human-friendly streamed Markdown output
./target/release/fetchsys "who is Ada Lovelace"  # Linux/macOS
.\target\x86_64-pc-windows-gnu\release\fetchsys.exe "who is Ada Lovelace"  # Windows (GNU toolchain)

# Agent/pipeline JSON output
./target/release/fetchsys --json "latest Rust release"

# Pipe query via stdin (JSON mode)
echo "quantum computing" | xargs ./target/release/fetchsys --json

# Use only specific providers
./target/release/fetchsys --providers searxng,brave "climate change consensus"

# Disable LLM synthesis (return raw aggregated snippets)
./target/release/fetchsys --no-llm "Rust async runtime"

# Verbose structured logging
./target/release/fetchsys -v "OpenAI GPT models"

# JSON structured logs (useful inside pipelines)
./target/release/fetchsys --log-json --json "search query"
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `SEARXNG_URL` | No | `http://localhost:8888` | SearXNG instance base URL (Tier 1 search; connection is capped at 3 s to fail fast when Docker is offline) |
| `BRAVE_API_KEY` | No | — | [Brave Search API](https://api-dashboard.search.brave.com/) key (Tier 3 search, 2000 free/month) |
| `SERPER_API_KEY` | No | — | [Serper.dev](https://serper.dev/) API key (Tier 6 search, paid) |
| `GOOGLE_API_KEY` | No | — | Google AI Studio key — used for **both** Gemini (Tier 1 LLM) and the Google search scraper (Tier 5 search) |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model name |
| `GROQ_API_KEY` | No | — | [Groq](https://console.groq.com/) API key (Tier 2 LLM, free) |
| `OPENROUTER_API_KEY` | No | — | [OpenRouter](https://openrouter.ai/) API key (Tier 3 LLM) |
| `OPENROUTER_MODEL` | No | `openai/gpt-5.2-codex` | OpenRouter model name |
| `OLLAMA_URL` | No | `http://localhost:11434` | Ollama base URL (Tier 4 LLM, local) |
| `OLLAMA_MODEL` | No | `dolphin-llama3` | Ollama model name |
| `LLM_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI-compatible endpoint (Tier 5 LLM) |
| `LLM_API_KEY` | No | — | API key for the LLM endpoint |
| `LLM_MODEL` | No | `gpt-4o-mini` | Model name |
| `JINA_API_KEY` | No | — | [Jina Reader](https://jina.ai/reader/) API key (Tier 1 reader, 1M tokens/month free; circuit breaker skips on repeated failures) |
| `FIRECRAWL_API_KEY` | No | — | [Firecrawl](https://firecrawl.dev/) API key (Tier 4 reader; self-hostable OSS alternative to Jina) |
| `FIRECRAWL_BASE_URL` | No | `https://api.firecrawl.dev` | Firecrawl endpoint (override to point at a self-hosted instance) |
| `FETCHSYS_TOP_N` | No | `5` | Number of URLs to read and fact-check |
| `FETCHSYS_MIN_QUALITY` | No | `0.3` | Min quality score before falling back |
| `RUST_LOG` | No | — | Standard `tracing` log level filter |

See [`.env.example`](.env.example) for the full list.

---

## Provider tiers

### Search

Providers are tried in order; the first to return results above `min_quality_score` wins.

| Tier | Provider | Requires | Notes |
|---|---|---|---|
| 1 | **SearXNG** (self-hosted) | Docker + JSON enabled (see below) | Privacy-first; no API key; **3 s hard timeout** — fails instantly when Docker is not running |
| 2 | **DuckDuckGo** | Nothing | HTML scrape via POST form |
| 3 | **Brave Search API** | `BRAVE_API_KEY` | 2,000 free queries/month |
| 4 | **Bing** | Nothing | HTML scrape; base64 URL decode |
| 5 | **Google** | Nothing | HTML scrape; no API key required |
| 6 | **Serper.dev** | `SERPER_API_KEY` | Paid; Google-backed JSON API |

### SearXNG — Docker setup (one-time)

Public SearXNG instances disable their JSON API by default to prevent bot abuse.
Self-hosting guarantees a stable, rate-limit-free endpoint.

**1. Start the container:**
```sh
docker run -d -p 8888:8080 --name searxng searxng/searxng
```

**2. Enable the JSON format inside the container:**
```sh
docker exec searxng python3 -c "
content = open('/etc/searxng/settings.yml').read()
content = content.replace('  formats:\n    - html', '  formats:\n    - html\n    - json')
open('/etc/searxng/settings.yml', 'w').write(content)
print('Done')
"
```

**3. Restart the container:**
```sh
docker restart searxng
```

**4. Verify the JSON API:**
```sh
curl "http://localhost:8888/search?q=test&format=json"
# Should return: {"query": "test", "results": [...], ...}
```

**5. Set in `.env`:**
```dotenv
SEARXNG_URL=http://localhost:8888
```

> **Note:** The `settings.yml` edit lives inside the container's filesystem.
> If you `docker rm searxng` and recreate it, repeat step 2.
> To persist permanently, mount a local config:
> ```sh
> # Copy out the patched file first
> docker cp searxng:/etc/searxng/settings.yml ./searxng-settings.yml
> # Then on recreate, mount it back in
> docker run -d -p 8888:8080 --name searxng \
>   -v "$(pwd)/searxng-settings.yml:/etc/searxng/settings.yml" \
>   searxng/searxng
> ```

**To stop/start the container later:**
```sh
docker stop searxng
docker start searxng
```

### Reader / Scraper

Adapters are tried in order; the first to return non-empty content wins.

| Tier | Adapter | Notes |
|---|---|---|
| 1 | **Jina Reader** (`r.jina.ai`) | Fast, free (1M tokens/month); URL → Markdown; **circuit breaker** skips when repeated failures detected; omitted if `JINA_API_KEY` not set |
| 2 | **Readability + htmd** (local) | Rust-native article extraction (`readability` crate) + HTML→Markdown (`htmd` crate); no external service |
| 3 | **anytomd** (local) | `anytomd` crate; broader format support; no network |
| 4 | **Firecrawl** (`firecrawl.dev`) | Managed cloud reader; self-hostable OSS; **circuit breaker** active; omitted if `FIRECRAWL_API_KEY` not set |
| 5 | **Raw HTTP** | `reqwest` + `scraper` HTML text extraction; plain-text last resort |

**URL normalisation** applied before entering the waterfall:
- `reddit.com` / `www.reddit.com` URLs are rewritten to `old.reddit.com` (static HTML, avoids 403 bot-protection on the React SPA)

### LLM

Providers are tried in order via circuit-breaker waterfall. Each provider has an independent circuit breaker; consecutive failures open its circuit for a configurable cooldown window before retrying.

| Tier | Backend | Requires | Notes |
|---|---|---|---|
| 1 | **Google Gemini** | `GOOGLE_API_KEY` | Free tier; `gemini-2.5-flash`; **HTTP 429 immediately opens the circuit** (no gradual threshold) — falls back to Groq on first rate-limit |
| 2 | **Groq** | `GROQ_API_KEY` | Fast free inference; tries `llama-3.3-70b-versatile` → `llama-3.1-8b-instant` → `mixtral-8x7b-32768` in order |
| 3 | **OpenRouter** | `OPENROUTER_API_KEY` | Multi-model gateway; `openai/gpt-5.2-codex` |
| 4 | **Ollama** (local) | `OLLAMA_URL` | No key; `llama3.2` default |
| 5 | **OpenAI-compat** | `LLM_API_KEY` | Any OpenAI-compatible endpoint (Azure, LM Studio, etc.) |
| — | **Fallback echo** | — | Always-last safety net; returns a structured message if all real providers fail |

---

## Agent JSON schema

When `--json` is passed, output conforms to:

```json
{
  "query": "string",
  "answer": {
    "text": "string",
    "confidence": 0.0
  },
  "sources": [
    {
      "url": "https://...",
      "title": "string",
      "snippet": "string",
      "rank": 1,
      "reader_raw": "string"
    }
  ],
  "fact_checks": [
    {
      "claim": "string",
      "status": "confirmed|contradicted|inconclusive",
      "evidence_urls": ["https://..."]
    }
  ],
  "metadata": {
    "timestamp": "ISO-8601",
    "providers_used": ["searxng", "r.jina.ai"],
    "latency_ms": 123
  }
}
```

Schema validation is enforced in [`tests/schema_tests.rs`](tests/schema_tests.rs).

---

## Configuration

Settings are read (in ascending priority) from:

1. `config/default.toml` — committed defaults
2. `config/local.toml` — local overrides (gitignored)
3. Environment variables (`FETCHSYS__*` with double-underscore nesting)
4. CLI flags

Example `config/local.toml`:

```toml
[search]
searxng_url = "http://my-searxng.internal:8080"
min_quality_score = 0.5

[llm]
base_url = "http://localhost:11434/v1"
model = "llama3"
api_key = ""
```

---

## Architecture

```
CLI (clap)
  └─► Config (config crate + env)
        └─► SearchTool (6-tier waterfall)
              │  Tier 1: SearXNG      (self-hosted Docker, JSON API, 3 s hard timeout)
              │  Tier 2: DuckDuckGo   (HTML scrape, POST form)
              │  Tier 3: Brave        (API key)
              │  Tier 4: Bing         (HTML scrape, base64 URL decode)
              │  Tier 5: Google       (HTML scrape, no key)
              │  Tier 6: Serper       (API key, paid)
              └─► URLNormaliser (reddit.com → old.reddit.com, …)
                    └─► Reader (5-tier waterfall)
                          │  Tier 1: Jina Reader    (r.jina.ai, URL → Markdown, circuit breaker)
                          │  Tier 2: Readability+htmd (local article extraction + Markdown)
                          │  Tier 3: anytomd          (local HTML → Markdown)
                          │  Tier 4: Firecrawl        (cloud reader, circuit breaker)
                          │  Tier 5: Raw HTTP         (reqwest + scraper, plain-text)
                          └─► LLM (5-tier circuit-breaker waterfall)
                                │  Tier 1: Google Gemini  (429 → immediate circuit trip)
                                │  Tier 2: Groq           (multi-model fallback)
                                │  Tier 3: OpenRouter
                                │  Tier 4: Ollama (local)
                                │  Tier 5: OpenAI-compat
                                │  Tier 6: Fallback echo (always succeeds)
                                └─► FactChecker (claim extraction with JSON sanitiser +
                                      proximity-windowed negation + precision×coverage scoring)
                                          └─► Output (Markdown stream | JSON)
```

### Module map

| Path | Responsibility |
|---|---|
| `src/cli.rs` | `clap` argument definitions |
| `src/config.rs` | Config loading and merge |
| `src/schema.rs` | Agent JSON contract + validation |
| `src/search/mod.rs` | 6-tier search waterfall; SearXNG 3 s fail-fast timeout |
| `src/reader/mod.rs` | 5-tier reader waterfall; `normalize_scrape_url` (Reddit → old.reddit.com); Jina + Firecrawl circuit breakers |
| `src/extract/mod.rs` | Page classification (`classify_page`); refined login detection (strong/weak signals + visible text threshold); SSR-aware JS-heavy detection |
| `src/llm/mod.rs` | LLM provider trait + cloud/local impls; circuit breaker with immediate-open on HTTP 429 (`record_rate_limit`) |
| `src/factcheck/mod.rs` | Claim extraction; `extract_json_array` with `sanitize_json_inner_quotes` for LLM nested-quote output; cross-referencing + scoring |
| `src/login/mod.rs` | Static login form detection and credential submission |
| `src/fetch/mod.rs` | Tiered fetch (static + optional headless); CAPTCHA detection + UA-rotation retry loop |
| `src/main.rs` | Binary entry point (orchestration) |

---

## Resilience & production hardening

The following hardening changes are implemented across the codebase:

### Search
- **SearXNG fail-fast** (`src/search/mod.rs`): SearXNG's per-request timeout is capped at `min(configured_timeout, 3 s)`. When the local Docker container is not running the tier fails in under 3 seconds instead of hanging for the full 10 s × 3 retries.

### Reader
- **Reddit URL rewrite** (`src/reader/mod.rs`): All `reddit.com` / `www.reddit.com` URLs are silently rewritten to `old.reddit.com` before entering the adapter waterfall. The old-reddit HTML interface returns static pages (HTTP 200) where the React SPA returns HTTP 403 to scrapers.
- **Circuit breakers** (`src/reader/mod.rs`): Jina Reader and Firecrawl each maintain an independent `AtomicBool` circuit breaker. After a configurable number of consecutive failures the adapter is skipped for the remainder of the process lifetime.

### Page classification
- **Refined login detection** (`src/extract/mod.rs`): `LOGIN_RE` was split into two patterns:
  - `LOGIN_TEXT_RE` — explicit textual markers (`"login required"`, `"sign in to continue"`, etc.) always trigger `LoginRequired`.
  - `LOGIN_FIELD_RE` — password input field only triggers `LoginRequired` when visible text is < 1 500 bytes, preventing content-rich pages with a nav-bar login widget from being misclassified as login walls.
- **SSR-aware JS-heavy detection** (`src/extract/mod.rs`): Framework markers (React, Vue, Next.js, Nuxt, …) previously always returned `JsHeavy`. Now the classifier also measures visible text length: SSR pages that ship framework markers but have > 500 bytes of pre-rendered text are treated as `Static` and avoid pointless headless-browser fallback attempts.

### LLM
- **Gemini 429 immediate circuit trip** (`src/llm/mod.rs`): `CircuitBreaker` gained a `record_rate_limit()` method that sets the circuit to open *immediately* (bypassing the gradual failure counter). The `FallbackChainProvider` calls it whenever a provider error message contains `"429"` or `"rate_limit"`, so a Gemini quota exhaustion falls through to Groq on the very next call.

### Fact-checking
- **LLM JSON claim sanitiser** (`src/factcheck/mod.rs`): LLMs occasionally produce JSON arrays with unescaped inner quotes — e.g. `["The paper \"Attention Is All You Need\" introduced…"]` rendered without escapes. `extract_json_array` now calls `sanitize_json_inner_quotes` as a fallback when `serde_json` fails to parse the raw output. The sanitiser walks the byte stream, distinguishes element-boundary quotes from inner quotes by looking ahead for `,` or `]`, and escapes inner occurrences in-place.

---

## Adding a new search provider

1. Implement `SearchProvider` in `src/search/mod.rs`:

   ```rust
   pub struct MyProvider { /* ... */ }

   #[async_trait]
   impl SearchProvider for MyProvider {
       fn name(&self) -> &'static str { "myprovider" }
       async fn search(&self, query: &str, max: usize) -> anyhow::Result<Vec<SearchResult>> {
           // HTTP call here
       }
   }
   ```

2. Register it in `ProviderRegistry::build` with the appropriate config guard.
3. Add the name to `providers` in `config/default.toml`.
4. Document the required env var in `.env.example` and `README.md`.

---

## Development

```sh
# Run all tests
cargo test

# Run only unit tests
cargo test --lib

# Run integration tests
cargo test --test '*'

# Format
cargo fmt

# Lint
cargo clippy --all-targets -- -D warnings

# Security audit
cargo audit

# Release build
cargo build --release
```

---

## Roadmap

| Milestone | Summary |
|---|---|
| M1 Core | SearXNG + Jina reader + streamed Markdown output |
| M2 Fallbacks | Brave + Serper adapters, JSON agent output |
| M3 LLM | `candle`/`llm` crate local inference, cloud provider |
| M4 Hardening | CI matrix, static releases, full test coverage |
| M5 Optional | Ratatui TUI, Prometheus metrics, enterprise telemetry |

---

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
