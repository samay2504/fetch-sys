# Fetch-SyS

> A developer `@tool` for agent systems ; multi-tier web search, page reading, and LLM fact-checking in a single CLI call.

[![CI](https://github.com/samay2504/fetchsys/actions/workflows/ci.yml/badge.svg)](https://github.com/samay2504/fetchsys/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

---

## Overview

`Fetch-SyS` is a single-binary CLI tool designed to be called as a **`@tool` inside other AI agent pipelines** (LangChain, AutoGen, CrewAI, custom orchestrators, etc.).  It takes a query string and returns a rich, structured JSON response containing search results, page content, and LLM-fact-checked claims — all in one shot.

```sh
fetchsys "who is Ada Lovelace"
fetchsys --json "latest Rust release"   # strict JSON for agents
```

### What it does

1. **Searches** the web via a tiered waterfall (SearXNG → Brave → Serper)
2. **Reads** the top results into LLM-friendly markdown (Jina Reader → raw HTTP fallback)
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
git clone https://github.com/samay2504/fetchsys
cd fetchsys
cargo build --release

# Copy and populate secrets
cp .env.example .env
$EDITOR .env
```

### Run

```sh
# Human-friendly streamed Markdown output
./target/release/fetchsys "who is Ada Lovelace"

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
| `SEARXNG_URL` | No | `http://localhost:8888` | SearXNG instance base URL (Tier 1 search) |
| `BRAVE_API_KEY` | No | — | [Brave Search API](https://api-dashboard.search.brave.com/) key (Tier 3 search, 2000 free/month) |
| `SERPER_API_KEY` | No | — | [Serper.dev](https://serper.dev/) API key (Tier 5 search, paid) |
| `GOOGLE_API_KEY` | No | — | Google AI Studio key for Gemini (Tier 1 LLM) |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model name |
| `GROQ_API_KEY` | No | — | [Groq](https://console.groq.com/) API key (Tier 2 LLM, free) |
| `OPENROUTER_API_KEY` | No | — | [OpenRouter](https://openrouter.ai/) API key (Tier 3 LLM) |
| `OPENROUTER_MODEL` | No | `openai/gpt-5.2-codex` | OpenRouter model name |
| `OLLAMA_URL` | No | `http://localhost:11434` | Ollama base URL (Tier 4 LLM, local) |
| `OLLAMA_MODEL` | No | `dolphin-llama3` | Ollama model name |
| `LLM_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI-compatible endpoint (Tier 5 LLM) |
| `LLM_API_KEY` | No | — | API key for the LLM endpoint |
| `LLM_MODEL` | No | `gpt-4o-mini` | Model name |
| `JINA_API_KEY` | No | — | [Jina Reader](https://jina.ai/reader/) API key (Tier 1 reader, 1M tokens/month free) |
| `FETCHSYS_TOP_N` | No | `5` | Number of URLs to read and fact-check |
| `FETCHSYS_MIN_QUALITY` | No | `0.3` | Min quality score before falling back |
| `RUST_LOG` | No | — | Standard `tracing` log level filter |

See [`.env.example`](.env.example) for the full list.

---

## Provider tiers

### Search

Providers are tried in order; the first to return results wins.

| Tier | Provider | Requires | Notes |
|---|---|---|---|
| 1 | **SearXNG** (self-hosted) | Docker + JSON enabled (see below) | Privacy-first; no API key |
| 2 | **DuckDuckGo** | Nothing | HTML scrape via POST form |
| 3 | **Brave Search API** | `BRAVE_API_KEY` | 2000 free queries/month |
| 4 | **Bing** | Nothing | HTML scrape; base64 URL decode |
| 5 | **Serper.dev** | `SERPER_API_KEY` | Paid; Google-backed results |

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

Adapters are tried in order; the first to return content wins.

| Tier | Adapter | Notes |
|---|---|---|
| 1 | **Jina Reader** (`r.jina.ai`) | Fast, free (1M tokens/month); URL → Markdown |
| 2 | **anytomd** (local) | `anytomd` crate; HTML → Markdown, no network |
| 3 | **Raw HTTP** | Plain-text scraper last resort |

### LLM

Providers are tried in order via circuit-breaker waterfall.

| Tier | Backend | Requires | Notes |
|---|---|---|---|
| 1 | **Google Gemini** | `GOOGLE_API_KEY` | Free tier; `gemini-2.5-flash` |
| 2 | **Groq** | `GROQ_API_KEY` | Fast free inference; `llama-3.3-70b-versatile` |
| 3 | **OpenRouter** | `OPENROUTER_API_KEY` | Multi-model gateway; `openai/gpt-5.2-codex` |
| 4 | **Ollama** (local) | `OLLAMA_URL` | No key; `dolphin-llama3` default |
| 5 | **OpenAI-compat** | `LLM_API_KEY` | Any OpenAI-compatible endpoint |

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
        └─► SearchTool (5-tier waterfall)
              │  Tier 1: SearXNG      (self-hosted Docker, JSON API)
              │  Tier 2: DuckDuckGo   (HTML scrape, POST form)
              │  Tier 3: Brave        (API key)
              │  Tier 4: Bing         (HTML scrape, base64 URL decode)
              │  Tier 5: Serper       (API key, paid)
              └─► Reader (3-tier waterfall)
                    │  Tier 1: Jina Reader  (r.jina.ai, URL → Markdown)
                    │  Tier 2: anytomd      (local HTML → Markdown)
                    │  Tier 3: Raw HTTP     (plain-text scraper)
                    └─► LLM (5-tier circuit-breaker waterfall)
                          │  Tier 1: Google Gemini
                          │  Tier 2: Groq
                          │  Tier 3: OpenRouter
                          │  Tier 4: Ollama (local)
                          │  Tier 5: OpenAI-compat
                          └─► FactChecker (proximity-windowed negation + precision×coverage scoring)
                                └─► Output (Markdown stream | JSON)
```

### Module map

| Path | Responsibility |
|---|---|
| `src/cli.rs` | `clap` argument definitions |
| `src/config.rs` | Config loading and merge |
| `src/schema.rs` | Agent JSON contract + validation |
| `src/search/mod.rs` | Search providers + waterfall logic |
| `src/reader/mod.rs` | Reader adapters + URL→text pipeline |
| `src/llm/mod.rs` | LLM provider trait + cloud/local impls |
| `src/factcheck/mod.rs` | Claim extraction, cross-referencing, scoring |
| `src/main.rs` | Binary entry point (orchestration) |

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




