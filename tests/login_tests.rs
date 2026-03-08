//! Integration tests for credential vault, form detection, and session tracking.

use std::collections::HashMap;
use fetchsys::login::{
    detect_login_form, domain_from_url, CredentialVault, LoginFormInfo, LoginOutcome,
    LoginSession, MockCredential,
};

// ── MockCredential ────────────────────────────────────────────────────────────

#[test]
fn generate_produces_non_empty_email_and_password() {
    let cred = MockCredential::generate("example.com");
    assert!(!cred.email.is_empty(), "email should be non-empty");
    assert!(cred.email.contains('@'), "email should contain @");
    assert!(!cred.password.is_empty(), "password should be non-empty");
    assert!(cred.password.len() >= 8, "password should be at least 8 chars");
    assert_eq!(cred.domain, "example.com");
}

#[test]
fn generate_produces_different_credentials_each_time() {
    let c1 = MockCredential::generate("a.com");
    let c2 = MockCredential::generate("a.com");
    // There is an astronomically low chance of collision, so this should always pass
    assert_ne!(c1.email, c2.email, "Two generated credentials should differ");
}

#[test]
fn to_form_body_uses_detected_field_names() {
    let form_info = LoginFormInfo {
        has_login_form: true,
        email_field_name: Some("user_email".to_owned()),
        username_field_name: None,
        password_field_name: Some("user_pass".to_owned()),
        form_action: Some("/session".to_owned()),
        form_method: Some("post".to_owned()),
    };
    let cred = MockCredential {
        domain: "test.io".into(),
        email: "alice@fake.com".into(),
        password: "Abc123!XYZ".into(),
        username: None,
        extras: HashMap::new(),
    };
    let body: HashMap<String, String> = cred.to_form_body(&form_info).into_iter().collect();
    assert_eq!(body.get("user_email").map(|s| s.as_str()), Some("alice@fake.com"));
    assert_eq!(body.get("user_pass").map(|s| s.as_str()), Some("Abc123!XYZ"));
}

#[test]
fn to_form_body_falls_back_to_default_names() {
    let form_info = LoginFormInfo {
        has_login_form: true,
        email_field_name: None,
        username_field_name: None,
        password_field_name: None,
        form_action: None,
        form_method: None,
    };
    let cred = MockCredential::generate("x.com");
    let body: Vec<(String, String)> = cred.to_form_body(&form_info);
    // Should fall back to "email" and "password"
    let keys: Vec<&str> = body.iter().map(|(k, _)| k.as_str()).collect();
    assert!(keys.contains(&"email") || keys.contains(&"password"),
        "Should fall back to default field names");
}

#[test]
fn extras_included_in_form_body() {
    let mut extras = HashMap::new();
    extras.insert("_token".to_owned(), "csrf_abc123".to_owned());
    let cred = MockCredential {
        domain: "site.com".into(),
        email: "x@y.com".into(),
        password: "pass".into(),
        username: None,
        extras,
    };
    let form_info = LoginFormInfo::default();
    let body: HashMap<String, String> = cred.to_form_body(&form_info).into_iter().collect();
    assert_eq!(body.get("_token").map(|s| s.as_str()), Some("csrf_abc123"));
}

// ── CredentialVault ───────────────────────────────────────────────────────────

#[test]
fn vault_starts_empty() {
    let vault = CredentialVault::new();
    assert_eq!(vault.len(), 0);
    assert!(vault.is_empty());
}

#[test]
fn vault_generates_on_miss() {
    let mut vault = CredentialVault::new();
    let cred = vault.get_or_generate("new-site.com");
    assert_eq!(cred.domain, "new-site.com");
    assert_eq!(vault.len(), 1);
}

#[test]
fn vault_returns_same_credential_on_repeated_calls() {
    let mut vault = CredentialVault::new();
    let first = vault.get_or_generate("repeat.io");
    let second = vault.get_or_generate("repeat.io");
    assert_eq!(first.email, second.email, "Should return cached credential");
}

#[test]
fn vault_manages_multiple_domains_independently() {
    let mut vault = CredentialVault::new();
    let c1 = vault.get_or_generate("site1.com");
    let c2 = vault.get_or_generate("site2.com");
    // Domains should not share credentials
    assert_eq!(c1.domain, "site1.com");
    assert_eq!(c2.domain, "site2.com");
    assert_eq!(vault.len(), 2);
}

#[test]
fn vault_eviction_removes_entry() {
    let mut vault = CredentialVault::new();
    vault.get_or_generate("evict.me");
    assert_eq!(vault.len(), 1);
    vault.evict("evict.me");
    assert_eq!(vault.len(), 0);
    assert!(vault.is_empty());
}

#[test]
fn vault_insert_replaces_existing() {
    let mut vault = CredentialVault::new();
    vault.get_or_generate("test.org");
    let custom = MockCredential {
        domain: "test.org".into(),
        email: "custom@explicit.com".into(),
        password: "ExplicitP4ss!".into(),
        username: Some("admin".into()),
        extras: HashMap::new(),
    };
    vault.insert(custom.clone());
    let retrieved = vault.get_or_generate("test.org");
    assert_eq!(retrieved.email, "custom@explicit.com");
}

#[test]
fn vault_json_roundtrip() {
    let mut vault = CredentialVault::new();
    vault.get_or_generate("alpha.io");
    vault.get_or_generate("beta.io");
    vault.get_or_generate("gamma.io");

    let tmp = std::env::temp_dir().join("fetchsys_integration_vault_test.json");
    vault.save_json(&tmp).expect("save_json should succeed");

    let loaded = CredentialVault::load_json(&tmp).expect("load_json should succeed");
    assert_eq!(loaded.len(), 3, "Loaded vault should have 3 entries");

    // Verify all domains are present
    let mut domains = loaded.domains();
    domains.sort();
    assert!(domains.contains(&"alpha.io"));
    assert!(domains.contains(&"beta.io"));
    assert!(domains.contains(&"gamma.io"));

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn vault_load_json_invalid_path_returns_error() {
    let result = CredentialVault::load_json("/nonexistent/path/credentials.json");
    assert!(result.is_err(), "Loading from invalid path should fail");
}

// ── Form detection ────────────────────────────────────────────────────────────

#[test]
fn detects_standard_email_password_form() {
    let html = r#"
    <form action="/auth/login" method="post">
        <input type="email" name="email" placeholder="Email" />
        <input type="password" name="password" placeholder="Password" />
        <button type="submit">Sign In</button>
    </form>"#;
    let info = detect_login_form(html);
    assert!(info.has_login_form, "Should detect login form");
    assert_eq!(info.form_action.as_deref(), Some("/auth/login"));
    assert!(info.password_field_name.is_some());
    assert!(info.email_field_name.is_some());
}

#[test]
fn detects_username_password_form() {
    let html = r#"<form>
        <input type="text" name="username" />
        <input type="password" name="pass" />
        <input type="submit" value="Login" />
    </form>"#;
    let info = detect_login_form(html);
    assert!(info.has_login_form);
    // Username field name should be captured
    assert!(info.email_field_name.is_some());
}

#[test]
fn no_login_form_on_article_page() {
    let html = r#"<html><body>
        <article><p>Read all about Rust here.</p></article>
    </body></html>"#;
    let info = detect_login_form(html);
    assert!(!info.has_login_form);
}

#[test]
fn password_only_form_not_considered_login() {
    // No email or username — just password (e.g. change-password form)
    let html = r#"<form>
        <input type="password" name="new_pass" />
        <input type="password" name="confirm_pass" />
        <button type="submit">Update</button>
    </form>"#;
    let info = detect_login_form(html);
    // Should NOT flag password-only as login (no email-like field)
    assert!(!info.has_login_form, "Password-only form should not be a login form");
}

#[test]
fn login_form_default_method_is_post() {
    let html = r#"<form action="/login">
        <input type="email" name="email" />
        <input type="password" name="password" />
        <button type="submit">Go</button>
    </form>"#;
    let info = detect_login_form(html);
    // When method is absent, we default to "post"
    assert!(info.form_method.as_deref().map_or(true, |m| m == "post"),
        "Default method should be post");
}

// ── LoginSession ──────────────────────────────────────────────────────────────

#[test]
fn session_captures_multiple_cookies() {
    let mut session = LoginSession::new();
    session.capture_cookies(&[
        "session_id=abc123; Path=/; HttpOnly; Secure".to_owned(),
        "csrf_token=xyz789; Path=/".to_owned(),
        "user_id=42; Path=/; HttpOnly".to_owned(),
    ]);
    assert_eq!(session.cookies.get("session_id").map(|s| s.as_str()), Some("abc123"));
    assert_eq!(session.cookies.get("csrf_token").map(|s| s.as_str()), Some("xyz789"));
    assert_eq!(session.cookies.get("user_id").map(|s| s.as_str()), Some("42"));
}

#[test]
fn session_cookies_with_no_value_part_ignored() {
    let mut session = LoginSession::new();
    session.capture_cookies(&["InvalidCookie; Path=/".to_owned()]);
    // No key=value pair so should not crash and produce no entries
    // (may vary by implementation, just ensure no panic)
    let _ = session.cookies.len();
}

#[test]
fn session_looks_authenticated_with_session_cookie() {
    let mut session = LoginSession::new();
    session.capture_cookies(&["session=tok_abc; Path=/".to_owned()]);
    assert!(session.looks_authenticated(), "session cookie should signal auth");
}

#[test]
fn session_looks_authenticated_with_jwt() {
    let mut session = LoginSession::new();
    session.capture_cookies(&["jwt=eyJhbGc.eyJzdWIi.sig; Path=/".to_owned()]);
    assert!(session.looks_authenticated());
}

#[test]
fn session_not_authenticated_with_only_theme_cookie() {
    let mut session = LoginSession::new();
    session.capture_cookies(&["theme=dark; Path=/".to_owned()]);
    assert!(!session.looks_authenticated());
}

#[test]
fn session_cookie_header_format_correct() {
    let mut session = LoginSession::new();
    // Use a single known cookie for deterministic test
    session.cookies.insert("x".to_owned(), "1".to_owned());
    let header = session.cookie_header();
    assert_eq!(header, "x=1");
}

#[test]
fn session_reset_clears_everything() {
    let mut session = LoginSession::new();
    session.is_authenticated = true;
    session.bearer_token = Some("tok".to_owned());
    session.cookies.insert("s".to_owned(), "v".to_owned());
    session.authenticated_url = Some("https://example.com/home".to_owned());
    session.reset();
    assert!(!session.is_authenticated);
    assert!(session.bearer_token.is_none());
    assert!(session.cookies.is_empty());
    assert!(session.authenticated_url.is_none());
}

// ── LoginOutcome ──────────────────────────────────────────────────────────────

#[test]
fn login_outcome_success_is_success() {
    assert!(LoginOutcome::Success.is_success());
}

#[test]
fn login_outcome_rejection_is_not_success() {
    assert!(!LoginOutcome::CredentialsRejected.is_success());
    assert!(!LoginOutcome::NoFormFound.is_success());
    assert!(!LoginOutcome::BlockedByCaptcha.is_success());
    assert!(!LoginOutcome::InjectionFailed("x".into()).is_success());
    assert!(!LoginOutcome::NetworkError("y".into()).is_success());
}

// ── domain_from_url ───────────────────────────────────────────────────────────

#[test]
fn domain_from_url_extracts_host() {
    assert_eq!(domain_from_url("https://www.example.com/login?next=/home"), "www.example.com");
    assert_eq!(domain_from_url("http://sub.domain.io:8080/path"), "sub.domain.io");
}

#[test]
fn domain_from_url_handles_bare_domain() {
    // Not a valid URL → returns input
    let result = domain_from_url("not-a-url");
    // Should not panic
    assert!(!result.is_empty());
}

#[test]
fn domain_from_url_https_scheme() {
    assert_eq!(domain_from_url("https://api.github.com/repos"), "api.github.com");
}
