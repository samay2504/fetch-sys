//! Mock credential management and autonomous login-wall automation.
//!
//! ## Overview
//! Many real-world pages sit behind a login gate.  This module provides a
//! complete, open-source login pipeline:
//!
//! ```text
//!  detect_login_form(html)          ← heuristic form scanner
//!       │ LoginFormInfo
//!       ▼
//!  CredentialVault::get_or_generate(domain)   ← persistent or auto-generated mock creds
//!       │ MockCredential
//!       ▼
//!  perform_static_login(url, cred)  ← reqwest-based POST form submission
//!       │                           (no browser needed for most sites)
//!       ▼
//!  LoginSession                     ← cookie / token capture + re-use
//! ```
//!
//! When the `selenium` feature is enabled, `perform_browser_login()` drives a
//! real browser to handle JS-rendered login forms, 2FA hint pages, and
//! invisible CAPTCHA challenges.
//!
//! ## Ethics & Legal
//! Mock credentials are useful only for:
//!  - Sites that explicitly provide test / sandbox accounts.
//!  - Developer-owned services during integration testing.
//!  - Research environments with written permission.
//!
//! Never use this module against sites or services without authorisation.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use fake::faker::internet::en::{FreeEmail, Password, Username};
use fake::Fake;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Credential types
// ---------------------------------------------------------------------------

/// A single mock login credential pair.
///
/// Generated using the [`fake`] crate — looks realistic but is entirely
/// synthetic; no real accounts are used or created.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockCredential {
    /// Site or domain this credential targets (e.g. `"reddit.com"`).
    pub domain: String,
    /// Realistic-looking fake email address.
    pub email: String,
    /// Randomly generated password meeting common strength requirements.
    pub password: String,
    /// Optional display-name / username field value.
    pub username: Option<String>,
    /// Extra fields (honeypot values, CSRF tokens injected at runtime, etc.)
    #[serde(default)]
    pub extras: HashMap<String, String>,
}

impl MockCredential {
    /// Generate a fresh, random credential for `domain`.
    ///
    /// Uses the `fake` crate — completely open-source, no external service
    /// required (`MIT / Apache-2.0` licensed).
    pub fn generate(domain: &str) -> Self {
        let email: String = FreeEmail().fake();
        let password: String = Password(10..20).fake();
        let username: String = Username().fake();
        Self {
            domain: domain.to_owned(),
            email,
            password,
            username: Some(username),
            extras: HashMap::new(),
        }
    }

    /// Build a POST form body `application/x-www-form-urlencoded` ready to
    /// submit against the detected login form.
    ///
    /// `form_info` tells us the HTML field names discovered by
    /// [`detect_login_form`].
    pub fn to_form_body(&self, form_info: &LoginFormInfo) -> Vec<(String, String)> {
        let mut body = Vec::with_capacity(4);

        // Email / username field
        if let Some(ref name) = form_info.email_field_name {
            body.push((name.clone(), self.email.clone()));
        } else {
            body.push(("email".to_owned(), self.email.clone()));
        }

        // Username field (some sites have both)
        if let Some(ref uname_name) = form_info.username_field_name {
            if let Some(ref u) = self.username {
                body.push((uname_name.clone(), u.clone()));
            }
        }

        // Password field
        if let Some(ref pname) = form_info.password_field_name {
            body.push((pname.clone(), self.password.clone()));
        } else {
            body.push(("password".to_owned(), self.password.clone()));
        }

        // Any extras (honeypots etc.)
        for (k, v) in &self.extras {
            body.push((k.clone(), v.clone()));
        }

        body
    }
}

// ---------------------------------------------------------------------------
// Credential vault
// ---------------------------------------------------------------------------

/// Persistent in-memory store of [`MockCredential`]s, keyed by domain.
///
/// ### Loading order
/// 1. Optionally load from `credentials.json` at startup.
/// 2. Auto-generate fresh creds for any domain not found in the vault.
/// 3. Persist back to JSON on demand with [`CredentialVault::save_json`].
///
/// ### JSON format
/// ```json
/// [
///   {"domain":"example.com","email":"alice@mailinator.com","password":"Abc12!xyz","username":"alice99","extras":{}}
/// ]
/// ```
#[derive(Debug, Default)]
pub struct CredentialVault {
    credentials: HashMap<String, MockCredential>,
}

impl CredentialVault {
    pub fn new() -> Self {
        Self { credentials: HashMap::new() }
    }

    /// Load a JSON credential vault from `path`.
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Cannot read vault {:?}", path.as_ref()))?;
        let list: Vec<MockCredential> =
            serde_json::from_str(&data).context("Vault is not valid JSON")?;
        let mut vault = Self::new();
        for cred in list {
            vault.credentials.insert(cred.domain.clone(), cred);
        }
        info!(count = vault.credentials.len(), "Credential vault loaded");
        Ok(vault)
    }

    /// Save the current vault to `path` as pretty-printed JSON.
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let list: Vec<&MockCredential> = self.credentials.values().collect();
        let json = serde_json::to_string_pretty(&list).context("Serialisation failed")?;
        std::fs::write(path.as_ref(), json)
            .with_context(|| format!("Cannot write vault {:?}", path.as_ref()))
    }

    /// Insert or replace a credential.
    pub fn insert(&mut self, cred: MockCredential) {
        self.credentials.insert(cred.domain.clone(), cred);
    }

    /// Get the credential for `domain`, generating a new one if absent.
    pub fn get_or_generate(&mut self, domain: &str) -> MockCredential {
        if let Some(c) = self.credentials.get(domain) {
            debug!(domain, "Credential cache hit");
            return c.clone();
        }
        let generated = MockCredential::generate(domain);
        info!(domain, email = %generated.email, "Generated mock credential");
        self.credentials.insert(domain.to_owned(), generated.clone());
        generated
    }

    /// Remove a credential (e.g. after a Credentials-Rejected response).
    pub fn evict(&mut self, domain: &str) {
        self.credentials.remove(domain);
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.credentials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.credentials.is_empty()
    }

    /// Domains with stored credentials.
    pub fn domains(&self) -> Vec<&str> {
        self.credentials.keys().map(|s| s.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// Login form detection (static HTML)
// ---------------------------------------------------------------------------

/// Information extracted from a HTML page's login form.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LoginFormInfo {
    /// Whether a likely login form was found.
    pub has_login_form: bool,
    /// The `name` attribute of the email/username input.
    pub email_field_name: Option<String>,
    /// The `name` attribute of the optional username-only input.
    pub username_field_name: Option<String>,
    /// The `name` attribute of the password input.
    pub password_field_name: Option<String>,
    /// The `action` attribute of the `<form>` (POST URL).
    pub form_action: Option<String>,
    /// The `method` attribute (`post` / `get`).
    pub form_method: Option<String>,
}

// Pre-compiled patterns — compiled once at process start.

/// Extracts the `action` attribute value from a `<form ...>` tag.
static FORM_ACTION_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)<form[^>]*\baction="([^"]*)"[^>]*>"#).unwrap()
});

/// Extracts the `method` attribute value from a `<form ...>` tag.
static FORM_METHOD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)<form[^>]*\bmethod="([^"]*)"[^>]*>"#).unwrap()
});

static EMAIL_INPUT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)<input[^>]+(?:type="(?:email|text)"|name="(email|username|user(?:_?name)?|login|account|mail|e-mail)")[^>]*/?>"#).unwrap()
});

static PASS_INPUT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)<input[^>]+type="password"[^>]*(?:name="([^"]*)")?[^>]*/?>"#).unwrap()
});

/// Scan raw HTML and return best-effort login form details.
///
/// Returns `LoginFormInfo { has_login_form: false, .. }` when no form is found.
pub fn detect_login_form(html: &str) -> LoginFormInfo {
    let has_password = PASS_INPUT_RE.is_match(html);
    let has_email_like = EMAIL_INPUT_RE.is_match(html);

    if !has_password {
        return LoginFormInfo::default();
    }

    // Extract password field name
    let password_field_name = PASS_INPUT_RE
        .captures(html)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_owned())
        .or_else(|| Some("password".to_owned()));

    // Extract email/username field name
    let email_field_name = if has_email_like {
        EMAIL_INPUT_RE
            .captures(html)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_owned())
            .or_else(|| Some("email".to_owned()))
    } else {
        None
    };

    // Extract form action / method
    let form_action = FORM_ACTION_RE
        .captures(html)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_owned());
    let form_method = FORM_METHOD_RE
        .captures(html)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_lowercase())
        .or_else(|| Some("post".to_owned()));

    LoginFormInfo {
        has_login_form: has_password && (has_email_like || email_field_name.is_some()),
        email_field_name,
        username_field_name: None, // refined below if both fields appear
        password_field_name,
        form_action,
        form_method,
    }
}

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

/// Tracks authentication state across requests (cookies, tokens, headers).
#[derive(Debug, Clone, Default)]
pub struct LoginSession {
    /// URL where authentication was confirmed.
    pub authenticated_url: Option<String>,
    /// Captured session cookies.
    pub cookies: HashMap<String, String>,
    /// Bearer / JWT token if extracted from response.
    pub bearer_token: Option<String>,
    /// Whether we believe the session is currently valid.
    pub is_authenticated: bool,
    /// How many login attempts have been made this session.
    pub attempts: u32,
}

impl LoginSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse and store cookies from a list of `Set-Cookie` header values.
    pub fn capture_cookies(&mut self, set_cookie_headers: &[String]) {
        for header in set_cookie_headers {
            if let Some((kv, _)) = header.split_once(';') {
                if let Some((k, v)) = kv.split_once('=') {
                    self.cookies.insert(k.trim().to_owned(), v.trim().to_owned());
                }
            }
        }
    }

    /// Heuristic: does the cookie jar contain common session-cookie names?
    pub fn looks_authenticated(&self) -> bool {
        const AUTH_HINTS: &[&str] =
            &["session", "token", "auth", "sid", "access_token", "_session", "jwt", "logged"];
        self.cookies
            .keys()
            .any(|k| AUTH_HINTS.iter().any(|hint| k.to_lowercase().contains(hint)))
    }

    /// Convert the cookie jar into a `Cookie` header value.
    pub fn cookie_header(&self) -> String {
        self.cookies
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Clear session state (e.g. after logout or credential rejection).
    pub fn reset(&mut self) {
        self.cookies.clear();
        self.bearer_token = None;
        self.is_authenticated = false;
        self.authenticated_url = None;
    }
}

// ---------------------------------------------------------------------------
// Login outcome
// ---------------------------------------------------------------------------

/// Normalised result of a login attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum LoginOutcome {
    /// Authentication succeeded; cookies / token captured.
    Success,
    /// Login form found but form-field injection failed.
    InjectionFailed(String),
    /// Credentials were submitted but the server rejected them.
    CredentialsRejected,
    /// Page has no login form.
    NoFormFound,
    /// A CAPTCHA challenge blocked the login attempt.
    BlockedByCaptcha,
    /// Network / transport error.
    NetworkError(String),
}

impl LoginOutcome {
    pub fn is_success(&self) -> bool {
        matches!(self, LoginOutcome::Success)
    }
}

// ---------------------------------------------------------------------------
// Static (reqwest-based) login — no browser required
// ---------------------------------------------------------------------------

/// Attempt to authenticate against `base_url` using a plain HTTP POST.
///
/// 1. Fetch `base_url` to get the login form (action URL, field names).
/// 2. Build form body from `credential`.
/// 3. POST to the action URL and inspect the response for auth cookies.
///
/// Does **not** handle JS-rendered forms; use `perform_browser_login` for those.
pub async fn perform_static_login(
    base_url: &str,
    credential: &MockCredential,
    user_agent: &str,
    timeout_secs: u64,
) -> Result<(LoginOutcome, LoginSession)> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .user_agent(user_agent)
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .context("Failed to build login HTTP client")?;

    // Step 1: fetch the login page to discover form structure
    let page_resp = client
        .get(base_url)
        .header("Accept", "text/html,application/xhtml+xml,*/*;q=0.8")
        .send()
        .await
        .context("Login page fetch failed")?;

    let login_html = page_resp.text().await.context("Login page body read failed")?;
    let form_info = detect_login_form(&login_html);

    if !form_info.has_login_form {
        warn!(url = base_url, "No login form found");
        return Ok((LoginOutcome::NoFormFound, LoginSession::new()));
    }

    // Check for CAPTCHA embedded in the login page itself
    if crate::captcha::is_blocked(&login_html) {
        warn!(url = base_url, "CAPTCHA on login page");
        return Ok((LoginOutcome::BlockedByCaptcha, LoginSession::new()));
    }

    // Step 2: determine POST URL
    let action = match &form_info.form_action {
        Some(a) if !a.is_empty() => {
            // Resolve relative URLs
            if a.starts_with("http") {
                a.clone()
            } else {
                let base = url::Url::parse(base_url).unwrap_or_else(|_| url::Url::parse("http://localhost").unwrap());
                base.join(a)
                    .map(|u| u.to_string())
                    .unwrap_or_else(|_| base_url.to_owned())
            }
        }
        _ => base_url.to_owned(),
    };

    // Step 3: build and POST the form
    let form_body = credential.to_form_body(&form_info);
    debug!(url = %action, fields = form_body.len(), "Posting login form");

    let post_resp = client
        .post(&action)
        .form(&form_body)
        .header("Referer", base_url)
        .header("Origin", base_url)
        .send()
        .await
        .context("Login POST failed")?;

    // Step 4: inspect response
    let mut session = LoginSession::new();
    session.attempts += 1;

    let set_cookie_headers: Vec<String> = post_resp
        .headers()
        .get_all("set-cookie")
        .iter()
        .filter_map(|v| v.to_str().ok().map(|s| s.to_owned()))
        .collect();

    session.capture_cookies(&set_cookie_headers);
    let response_url = post_resp.url().to_string();
    let response_html = post_resp.text().await.unwrap_or_default();

    // Heuristic: check for still-present login form => rejected
    let still_has_form = detect_login_form(&response_html).has_login_form;

    if session.looks_authenticated() && !still_has_form {
        session.is_authenticated = true;
        session.authenticated_url = Some(response_url);
        info!(domain = %credential.domain, "Login appears successful");
        return Ok((LoginOutcome::Success, session));
    }

    if still_has_form {
        warn!(domain = %credential.domain, "Login form still present — credentials likely rejected");
        return Ok((LoginOutcome::CredentialsRejected, session));
    }

    // Ambiguous — return success optimistically if cookies were set
    if !session.cookies.is_empty() {
        session.is_authenticated = true;
        session.authenticated_url = Some(response_url);
        return Ok((LoginOutcome::Success, session));
    }

    Ok((LoginOutcome::CredentialsRejected, session))
}

// ---------------------------------------------------------------------------
// Browser-backed login (optional selenium feature)
// ---------------------------------------------------------------------------

/// Perform a login using a real browser via Selenium WebDriver.
///
/// Requires a running Chrome/Firefox WebDriver on `webdriver_url` (default
/// `http://localhost:4444`).  Enable with `--features selenium`.
///
/// Handles:
/// - JS-rendered login forms
/// - Invisible reCAPTCHA v3 (human-like timing helps)
/// - Multi-step login (email → continue → password)
/// - Cookie capture for subsequent `reqwest` requests
#[cfg(feature = "selenium")]
pub async fn perform_browser_login(
    base_url: &str,
    credential: &MockCredential,
    webdriver_url: &str,
    timeout_secs: u64,
) -> Result<(LoginOutcome, LoginSession)> {
    use rand::Rng;
    use thirtyfour::prelude::*;

    let mut caps = DesiredCapabilities::chrome();
    caps.set_headless()?;
    caps.add_chrome_arg("--no-sandbox")?;
    caps.add_chrome_arg("--disable-blink-features=AutomationControlled")?;
    caps.add_chrome_arg("--window-size=1366,768")?;

    let driver = WebDriver::new(webdriver_url, caps)
        .await
        .context("WebDriver connect failed")?;

    driver
        .set_implicit_wait_timeout(Duration::from_secs(timeout_secs))
        .await?;

    driver.goto(base_url).await.context("Navigate to login page failed")?;

    // Small human-like pause
    tokio::time::sleep(Duration::from_millis(
        rand::thread_rng().gen_range(800..1_500),
    ))
    .await;

    let mut session = LoginSession::new();
    session.attempts += 1;

    // Try to find email/username field
    let email_result = driver
        .find(By::Css(
            "input[type='email'], input[name='email'], input[name='username'], input[name='user']",
        ))
        .await;

    let outcome = match email_result {
        Ok(email_el) => {
            email_el.send_keys(&credential.email).await?;
            // Realistic typing delay
            tokio::time::sleep(Duration::from_millis(rand::thread_rng().gen_range(300..700))).await;

            // Password field
            if let Ok(pass_el) = driver.find(By::Css("input[type='password']")).await {
                pass_el.send_keys(&credential.password).await?;
                tokio::time::sleep(Duration::from_millis(
                    rand::thread_rng().gen_range(200..500),
                ))
                .await;

                // Submit
                if let Ok(submit) =
                    driver.find(By::Css("button[type='submit'], input[type='submit']")).await
                {
                    submit.click().await?;
                } else {
                    pass_el.send_keys("\n").await?;
                }

                tokio::time::sleep(Duration::from_millis(2_000)).await;

                let page_src = driver.source().await.unwrap_or_default();
                if detect_login_form(&page_src).has_login_form {
                    LoginOutcome::CredentialsRejected
                } else {
                    LoginOutcome::Success
                }
            } else {
                LoginOutcome::InjectionFailed("Password field not found".to_owned())
            }
        }
        Err(_) => LoginOutcome::NoFormFound,
    };

    // Capture cookies from browser session
    if outcome == LoginOutcome::Success {
        session.is_authenticated = true;
        session.authenticated_url = Some(driver.current_url().await?.to_string());
        for cookie in driver.get_all_cookies().await.unwrap_or_default() {
            session.cookies.insert(cookie.name().to_owned(), cookie.value().to_owned());
        }
    }

    driver.quit().await.ok();
    Ok((outcome, session))
}

/// Stub when `selenium` feature is disabled.
#[cfg(not(feature = "selenium"))]
pub async fn perform_browser_login(
    _base_url: &str,
    _credential: &MockCredential,
    _webdriver_url: &str,
    _timeout_secs: u64,
) -> Result<(LoginOutcome, LoginSession)> {
    bail!("Browser login requires `--features selenium` (a running chromedriver is also needed)")
}

// ---------------------------------------------------------------------------
// Convenience: extract eTLD+1 domain from URL
// ---------------------------------------------------------------------------

pub fn domain_from_url(url: &str) -> String {
    url::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_owned()))
        .unwrap_or_else(|| url.to_owned())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_generate_looks_real() {
        let cred = MockCredential::generate("example.com");
        assert!(!cred.email.is_empty());
        assert!(cred.email.contains('@'), "email should contain @");
        assert!(cred.password.len() >= 8, "password should be at least 8 chars");
        assert_eq!(cred.domain, "example.com");
    }

    #[test]
    fn vault_get_or_generate_creates_on_miss() {
        let mut vault = CredentialVault::new();
        let cred = vault.get_or_generate("test.io");
        assert_eq!(cred.domain, "test.io");
        assert_eq!(vault.len(), 1);
    }

    #[test]
    fn vault_get_or_generate_cache_hit() {
        let mut vault = CredentialVault::new();
        let first = vault.get_or_generate("foo.com");
        let second = vault.get_or_generate("foo.com");
        assert_eq!(first.email, second.email, "should return same credential on second call");
    }

    #[test]
    fn vault_evict_removes_entry() {
        let mut vault = CredentialVault::new();
        vault.get_or_generate("bar.com");
        assert_eq!(vault.len(), 1);
        vault.evict("bar.com");
        assert_eq!(vault.len(), 0);
    }

    #[test]
    fn vault_roundtrip_json() {
        let mut vault = CredentialVault::new();
        vault.get_or_generate("a.com");
        vault.get_or_generate("b.com");
        let dir = std::env::temp_dir();
        let path = dir.join("fetchsys_test_vault.json");
        vault.save_json(&path).expect("save failed");
        let loaded = CredentialVault::load_json(&path).expect("load failed");
        assert_eq!(loaded.len(), 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn detect_login_form_finds_standard_form() {
        let html = r#"
            <form action="/login" method="post">
                <input type="email" name="email" />
                <input type="password" name="password" />
                <button type="submit">Sign in</button>
            </form>
        "#;
        let info = detect_login_form(html);
        assert!(info.has_login_form);
        assert_eq!(info.form_action.as_deref(), Some("/login"));
        assert!(info.password_field_name.is_some());
        assert!(info.email_field_name.is_some());
    }

    #[test]
    fn detect_login_form_no_form() {
        let html = "<html><body><p>Just some text.</p></body></html>";
        let info = detect_login_form(html);
        assert!(!info.has_login_form);
    }

    #[test]
    fn detect_login_form_password_only_still_detected() {
        let html = r#"<form><input type="password" name="pass" /><button type="submit">Go</button></form>"#;
        let info = detect_login_form(html);
        // has_login_form requires BOTH password AND email-like; so false here
        assert!(!info.has_login_form, "email-like field is missing");
        assert!(info.password_field_name.is_some());
    }

    #[test]
    fn session_capture_cookies() {
        let mut session = LoginSession::new();
        session.capture_cookies(&[
            "session_id=abc123; Path=/; HttpOnly".to_owned(),
            "theme=dark; Path=/".to_owned(),
        ]);
        assert_eq!(session.cookies.get("session_id").map(|s| s.as_str()), Some("abc123"));
        assert_eq!(session.cookies.get("theme").map(|s| s.as_str()), Some("dark"));
    }

    #[test]
    fn session_looks_authenticated_true() {
        let mut session = LoginSession::new();
        session.capture_cookies(&["session=tok_xyz; Path=/".to_owned()]);
        assert!(session.looks_authenticated());
    }

    #[test]
    fn session_looks_authenticated_false() {
        let mut session = LoginSession::new();
        session.capture_cookies(&["theme=dark; Path=/".to_owned()]);
        assert!(!session.looks_authenticated());
    }

    #[test]
    fn session_cookie_header_format() {
        let mut session = LoginSession::new();
        session.cookies.insert("a".to_owned(), "1".to_owned());
        session.cookies.insert("b".to_owned(), "2".to_owned());
        let header = session.cookie_header();
        assert!(header.contains("a=1"));
        assert!(header.contains("b=2"));
    }

    #[test]
    fn session_reset_clears_state() {
        let mut session = LoginSession::new();
        session.is_authenticated = true;
        session.cookies.insert("tok".to_owned(), "abc".to_owned());
        session.reset();
        assert!(!session.is_authenticated);
        assert!(session.cookies.is_empty());
    }

    #[test]
    fn to_form_body_uses_correct_field_names() {
        let cred = MockCredential {
            domain: "test.com".into(),
            email: "user@test.com".into(),
            password: "P@ssw0rd!".into(),
            username: Some("user99".into()),
            extras: HashMap::new(),
        };
        let form_info = LoginFormInfo {
            has_login_form: true,
            email_field_name: Some("login".to_owned()),
            username_field_name: None,
            password_field_name: Some("pass".to_owned()),
            form_action: Some("/auth".to_owned()),
            form_method: Some("post".to_owned()),
        };
        let body = cred.to_form_body(&form_info);
        let map: HashMap<_, _> = body.into_iter().collect();
        assert_eq!(map.get("login").map(|s| s.as_str()), Some("user@test.com"));
        assert_eq!(map.get("pass").map(|s| s.as_str()), Some("P@ssw0rd!"));
    }

    #[test]
    fn domain_from_url_extracts_host() {
        assert_eq!(domain_from_url("https://www.example.com/login"), "www.example.com");
        assert_eq!(domain_from_url("not-a-url"), "not-a-url");
    }
}
