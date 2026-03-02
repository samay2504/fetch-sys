//! `fetchsys` binary entry point.
//!
//! Flow:
//!   CLI opts → Config → SearchTool (multi-tier) → Reader → LLM → FactCheck → Output

use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use fetchsys::cli;
use fetchsys::config;
use fetchsys::factcheck;
use fetchsys::llm;
use fetchsys::reader;
use fetchsys::schema;
use fetchsys::search;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let opts = cli::Opts::parse();

    // --------------- Initialise tracing ---------------
    init_tracing(&opts);

    let start = Instant::now();

    // --------------- Load & apply config ---------------
    let mut cfg = config::Config::load(opts.config.as_deref())
        .context("Failed to load configuration")?;
    cfg.apply_cli_overrides(&opts);

    let query = opts.query_string();
    if query.is_empty() {
        anyhow::bail!("Query must not be empty");
    }
    info!(%query, "Starting fetchsys");

    // --------------- Search ---------------
    let search_results = match search::multi_tier_search(&query, &cfg.search).await {
        Ok(r) => r,
        Err(e) => {
            error!(error = %e, "Search failed");
            return Err(e);
        }
    };

    info!(count = search_results.len(), "Search complete");

    let providers_used: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        search_results
            .iter()
            .map(|r| r.provider.clone())
            .filter(|p| seen.insert(p.clone()))
            .collect()
    };

    // Top N URLs to read
    let top = search_results
        .iter()
        .take(cfg.top_n)
        .cloned()
        .collect::<Vec<_>>();

    // --------------- Reader ---------------
    let docs = reader::read_top(&top, &cfg.reader).await;
    info!(count = docs.len(), "Reader complete");

    if docs.is_empty() {
        warn!("All reader adapters failed; output will be based purely on search snippets");
    }

    // --------------- LLM provider ---------------
    let llm_provider = llm::build_provider(&cfg.llm);
    let llm_enabled = cfg.llm.enabled;

    // --------------- Fact-check ---------------
    let fact_result =
        factcheck::check(&docs, &query, llm_provider.as_ref(), llm_enabled, &cfg.factcheck)
            .await;

    let latency_ms = start.elapsed().as_millis() as u64;

    // --------------- Output ---------------
    if opts.json {
        // Build AgentResponse
        let mut response = schema::AgentResponse::new(query.clone(), providers_used, latency_ms);

        response.answer = schema::Answer {
            text: fact_result.answer.clone(),
            confidence: fact_result.confidence,
        };

        response.sources = docs
            .iter()
            .zip(top.iter())
            .map(|(doc, sr)| schema::Source {
                url: sr.url.clone(),
                title: doc.title.clone(),
                snippet: sr.snippet.clone(),
                rank: sr.rank,
                reader_raw: doc.snippet(500),
            })
            .collect();

        response.fact_checks = fact_result
            .fact_checks
            .iter()
            .map(|fc| fc.to_fact_check(&cfg.factcheck))
            .collect();

        println!(
            "{}",
            serde_json::to_string_pretty(&response)
                .context("Failed to serialise agent response")?
        );
    } else {
        // Human-friendly streaming markdown
        println!("# Results for: {query}\n");
        println!("{}", fact_result.answer);

        if !top.is_empty() {
            println!("\n---\n## Sources\n");
            for (sr, doc) in top.iter().zip(docs.iter()) {
                println!(
                    "{}. **{}** — <{}>  \n   *{}*\n",
                    sr.rank,
                    if doc.title.is_empty() { sr.url.as_str() } else { doc.title.as_str() },
                    sr.url,
                    &sr.snippet,
                );
            }
        }

        if !fact_result.fact_checks.is_empty() {
            println!("\n---\n## Fact-checks\n");
            for fc in &fact_result.fact_checks {
                let checked = fc.to_fact_check(&cfg.factcheck);
                println!("- **{}** — `{}`", checked.claim, checked.status);
            }
        }

        let reader_providers: Vec<_> = {
            let mut seen = std::collections::HashSet::new();
            docs.iter()
                .map(|d| d.adapter.as_str())
                .filter(|a| seen.insert(*a))
                .collect()
        };

        println!(
            "\n---\n*Confidence: {:.0}% | Latency: {}ms | Readers: {}*",
            fact_result.confidence * 100.0,
            latency_ms,
            if reader_providers.is_empty() {
                "none".to_owned()
            } else {
                reader_providers.join(", ")
            },
        );
    }

    Ok(())
}

fn init_tracing(opts: &cli::Opts) {
    let level = if opts.verbose { "debug" } else { "info" };
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("fetchsys={level},warn")));

    if opts.log_json {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().json())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().with_target(false).compact())
            .init();
    }
}
