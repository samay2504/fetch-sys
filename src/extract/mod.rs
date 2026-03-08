//! Intelligent content extraction and page classification.
//!
//! ## Page classification
//! [`classify_page`] analyses HTML heuristics to decide which fetch strategy is
//! appropriate — static HTML, JS rendering, or skipping.
//!
//! ## Content extraction
//! [`extract_content`] implements a readability-style DOM density-scoring
//! algorithm to pull the main article body, stripping navigation, ads, and
//! boilerplate, then emits clean Markdown-like plain text.

use once_cell::sync::Lazy;
use regex::Regex;
use scraper::{Html, Selector};
use tracing::debug;

// ---------------------------------------------------------------------------
// Page classification
// ---------------------------------------------------------------------------

/// Heuristic classification of a page's rendering requirements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageType {
    /// Plain HTML — no JS required for content.
    Static,
    /// Heavy JS SPA/framework — needs browser rendering for content.
    JsHeavy,
    /// Login wall detected before meaningful content.
    LoginRequired,
    /// CAPTCHA or bot-challenge page.
    CaptchaProtected,
    /// Content is loaded via XHR/fetch — may need JS or API call.
    ApiBacked,
    /// Could not determine.
    Unknown,
}

// Lazy-compiled selectors & regexes used by classifier ----------------------

static JS_FRAMEWORK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)(react|angular|vue|next\.js|nuxt|svelte|ember|backbone|__NEXT_DATA__|window\.__INITIAL_STATE__|_\_nuxt)"#).unwrap()
});

static SPA_MARKER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)(id="(root|app|__next|__nuxt)"|data-reactroot|ng-app|x-data\s*=)"#).unwrap()
});

/// Strong login-wall signals — explicit textual markers.
static LOGIN_TEXT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)(login required|sign in to continue|please log in|you must be logged in|authentication required)"#).unwrap()
});

/// Weak login signal — page contains a password field (may be nav-bar widget).
static LOGIN_FIELD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)(type="password"|<input[^>]+name="password")"#).unwrap()
});

static XHR_PLACEHOLDER_RE: Lazy<Regex> = Lazy::new(|| {
    // Page has almost no inline text but loads data via XHR
    Regex::new(r#"(?i)(XMLHttpRequest|fetch\(|axios\.|graphql|application/json)"#).unwrap()
});

/// Approximate visible-text byte count (text nodes only, ignoring tags).
fn visible_text_len(html: &str) -> usize {
    html.split('>')
        .filter(|s| !s.starts_with('<') && !s.trim().is_empty())
        .map(|s| s.trim().len())
        .sum()
}

/// Classify a page based on its raw HTML.
pub fn classify_page(html: &str) -> PageType {
    // Check CAPTCHA first (delegate to captcha module heuristics inline to avoid dep cycles)
    if html.contains("cf-challenge")
        || html.contains("g-recaptcha")
        || html.contains("h-captcha")
        || html.contains("hcaptcha.com")
    {
        return PageType::CaptchaProtected;
    }

    // Compute visible text length once — used by multiple heuristics below.
    let visible_text = visible_text_len(html);

    // Login detection:
    //   • Explicit textual markers ("login required", …) → always LoginRequired.
    //   • Password field alone → only LoginRequired when the page is content-sparse
    //     (avoids false positives on blogs with a nav-bar login widget).
    if LOGIN_TEXT_RE.is_match(html) {
        return PageType::LoginRequired;
    }
    if LOGIN_FIELD_RE.is_match(html) && visible_text < 1500 {
        return PageType::LoginRequired;
    }

    // Count <script> tags + look for SPA markers
    let script_count = html.matches("<script").count();
    let has_framework = JS_FRAMEWORK_RE.is_match(html);
    let has_spa_root = SPA_MARKER_RE.is_match(html);

    // Heuristic: JS framework / high script density + SPA root → JsHeavy.
    // BUT: SSR frameworks (Next.js, Nuxt) embed framework markers yet render
    // content server-side.  Only classify JsHeavy when visible text is sparse;
    // pages with substantial SSR content are already usable as Static.
    if (script_count >= 8 && has_spa_root) || has_framework {
        if visible_text < 500 {
            return PageType::JsHeavy;
        }
        debug!(
            visible = visible_text,
            "JS framework detected but page has substantial SSR content — treating as Static"
        );
    }

    // Content-light page that loads via XHR
    if visible_text < 200 && XHR_PLACEHOLDER_RE.is_match(html) {
        return PageType::ApiBacked;
    }

    PageType::Static
}

// ---------------------------------------------------------------------------
// Content extraction
// ---------------------------------------------------------------------------

/// Boilerplate tag names to strip entirely.
#[allow(dead_code)]
const STRIP_TAGS: &[&str] = &[
    "script", "style", "nav", "header", "footer", "aside", "form",
    "noscript", "iframe", "figure", "button", "select", "input", "textarea",
    "menu", "dialog", "banner", "ad", "advertisement",
];

/// CSS selectors for candidate content containers (ranked by quality).
static CONTENT_SELECTORS: Lazy<Vec<(&'static str, f64)>> = Lazy::new(|| {
    vec![
        // Semantic HTML5 landmarks — high quality
        ("article", 1.0),
        ("main", 0.95),
        // Common content class patterns
        ("[class*='article']", 0.9),
        ("[class*='content']", 0.85),
        ("[class*='post']", 0.85),
        ("[class*='entry']", 0.80),
        ("[class*='body']", 0.75),
        ("[id*='article']", 0.9),
        ("[id*='content']", 0.85),
        ("[id*='main']", 0.8),
        ("[id*='post']", 0.8),
        // Generic container fallback
        ("div", 0.5),
        ("section", 0.6),
    ]
});

/// Extract clean, readable Markdown-like text from raw HTML.
///
/// Algorithm:
///  1. Parse DOM with `scraper`.
///  2. Score candidate content containers by text density and link density.
///  3. Pick the highest-scoring container.
///  4. Walk it, emitting headings as `##`/`###` and paragraphs as plain text.
pub fn extract_content(html: &str, _url: &str) -> String {
    let doc = Html::parse_document(html);

    // Try semantic selectors in priority order first
    if let Some(text) = try_semantic_extraction(&doc) {
        if text.split_whitespace().count() > 30 {
            debug!("Semantic extraction succeeded");
            return text;
        }
    }

    // Fallback: density-scored div extraction
    let text = density_extract(&doc);
    if text.split_whitespace().count() > 20 {
        text
    } else {
        // Last resort: just take all text
        all_text(&doc)
    }
}

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

/// Try known high-quality content selectors in order.
fn try_semantic_extraction(doc: &Html) -> Option<String> {
    for (sel_str, _weight) in CONTENT_SELECTORS.iter().take(9) {
        let sel = match Selector::parse(sel_str) {
            Ok(s) => s,
            Err(_) => continue,
        };
        for el in doc.select(&sel) {
            let text = element_text(el.html().as_str());
            // Accept if contains reasonable prose (not a nav full of short links)
            let word_count = text.split_whitespace().count();
            let link_density = link_density(&el);
            if word_count > 50 && link_density < 0.5 {
                return Some(text);
            }
        }
    }
    None
}

/// Compute the fraction of words inside <a> tags (high link density = nav/footer).
fn link_density(el: &scraper::ElementRef<'_>) -> f64 {
    let total: usize = el
        .text()
        .map(|t| t.split_whitespace().count())
        .sum();
    if total == 0 {
        return 1.0;
    }
    let link_sel = Selector::parse("a").unwrap();
    let linked: usize = el
        .select(&link_sel)
        .flat_map(|a| a.text())
        .map(|t| t.split_whitespace().count())
        .sum();
    linked as f64 / total as f64
}

/// Score all divs by text density, return text from the best one.
fn density_extract(doc: &Html) -> String {
    let div_sel = Selector::parse("div, section, article").unwrap();
    let mut best_score = 0.0f64;
    let mut best_text = String::new();

    for el in doc.select(&div_sel) {
        let html = el.html();
        let text = element_text(&html);
        let words = text.split_whitespace().count();
        if words < 30 {
            continue;
        }
        let link_d = link_density(&el);
        // Score: word count penalised by link density
        let score = words as f64 * (1.0 - link_d);
        if score > best_score {
            best_score = score;
            best_text = text;
        }
    }
    best_text
}

/// Extract all visible text from the document as a fallback.
fn all_text(doc: &Html) -> String {
    // Strip script/style tags' text, collect rest
    let body_sel = Selector::parse("body").unwrap();
    let body = doc.select(&body_sel).next();
    match body {
        Some(b) => element_text(&b.html()),
        None => doc
            .root_element()
            .text()
            .collect::<Vec<_>>()
            .join(" ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" "),
    }
}

/// Convert raw HTML element (as string) to clean Markdown-ish text.
/// Turns h1-h3 into `## heading`, p into prose paragraphs.
fn element_text(html: &str) -> String {
    let doc = Html::parse_fragment(html);
    let mut out = String::with_capacity(html.len() / 3);

    // We walk text nodes, but insert heading markers when in heading context.
    // Simple approach: use selectors to find headings + paragraphs in order.

    let heading_sel = Selector::parse("h1,h2,h3,h4").unwrap();
    let para_sel = Selector::parse("p,li,blockquote,td,th,dt,dd").unwrap();

    // Collect heading texts
    let mut heading_texts: Vec<String> = doc
        .select(&heading_sel)
        .map(|el| el.text().collect::<String>().trim().to_owned())
        .filter(|t| !t.is_empty())
        .collect();

    // Collect paragraph texts
    let para_texts: Vec<String> = doc
        .select(&para_sel)
        .map(|el| el.text().collect::<String>())
        .map(|t| {
            t.split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|t| !t.is_empty() && t.len() > 20)
        .collect();

    // Emit headings first
    for h in heading_texts.drain(..) {
        out.push_str("## ");
        out.push_str(&h);
        out.push('\n');
    }

    if !out.is_empty() && !para_texts.is_empty() {
        out.push('\n');
    }

    // Emit paragraphs
    for p in para_texts {
        // Skip boilerplate (pure nav links, one-word spans, etc.)
        if should_skip_paragraph(&p) {
            continue;
        }
        out.push_str(&p);
        out.push_str("\n\n");
    }

    out.trim().to_owned()
}

/// Return true for paragraphs that are obviously boilerplate.
fn should_skip_paragraph(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 4 {
        return true;
    }
    // Mostly uppercase (nav elements)
    let upper_count = words.iter().filter(|w| w.chars().all(|c| c.is_uppercase())).count();
    if upper_count > words.len() / 2 {
        return true;
    }
    false
}
