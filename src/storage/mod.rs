//! In-memory TTL page cache backed by [`dashmap::DashMap`].
//!
//! Avoids re-fetching the same URL within a configurable window.
//! Thread-safe: every method takes `&self` (interior mutability via DashMap).
//!
//! # Features
//! - TTL-based expiration with automatic eviction on insert when at capacity
//! - Bounded capacity (default 512) to prevent unbounded memory growth
//! - `Arc<CachedPage>` storage to avoid cloning large markdown on cache hit
//!
//! # Usage
//! ```rust,ignore
//! let cache = PageCache::new(300);                  // 5-minute TTL
//! cache.insert("https://example.com".into(), page);
//! if let Some(p) = cache.get("https://example.com") { ... }
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tracing::debug;

/// Default max entries before LRU-style eviction of expired entries.
const DEFAULT_MAX_CAPACITY: usize = 512;

// ---------------------------------------------------------------------------
// Cached page entry (independent of the crawler module to avoid circular deps)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CachedPage {
    pub url: String,
    pub title: String,
    pub markdown: String,
    pub adapter: String,
}

// ---------------------------------------------------------------------------
// PageCache
// ---------------------------------------------------------------------------

/// Thread-safe, TTL-bounded in-memory cache for crawled pages.
pub struct PageCache {
    store: Arc<DashMap<String, (Arc<CachedPage>, Instant)>>,
    ttl: Duration,
    max_capacity: usize,
}

impl PageCache {
    /// Create a new cache with the given time-to-live in seconds.
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            store: Arc::new(DashMap::new()),
            ttl: Duration::from_secs(ttl_secs),
            max_capacity: DEFAULT_MAX_CAPACITY,
        }
    }

    /// Create a new cache with a custom max capacity.
    pub fn with_max_capacity(ttl_secs: u64, max_capacity: usize) -> Self {
        Self {
            store: Arc::new(DashMap::new()),
            ttl: Duration::from_secs(ttl_secs),
            max_capacity,
        }
    }

    /// Look up a URL. Returns `None` if absent or expired.
    pub fn get(&self, url: &str) -> Option<CachedPage> {
        let entry = self.store.get(url)?;
        let (page, inserted_at) = entry.value();
        if inserted_at.elapsed() <= self.ttl {
            debug!(url, "Cache hit");
            // Deref Arc — clone the inner page (needed by caller)
            Some(CachedPage {
                url: page.url.clone(),
                title: page.title.clone(),
                markdown: page.markdown.clone(),
                adapter: page.adapter.clone(),
            })
        } else {
            debug!(url, "Cache expired");
            drop(entry);
            self.store.remove(url);
            None
        }
    }

    /// Insert or overwrite a cache entry.
    /// If at capacity, evicts expired entries first; if still full, evicts the oldest entry.
    pub fn insert(&self, url: String, page: CachedPage) {
        // Evict expired entries if we're at or over capacity
        if self.store.len() >= self.max_capacity {
            self.evict_expired();
        }
        // If still at capacity after eviction, remove the oldest entry
        if self.store.len() >= self.max_capacity {
            if let Some(oldest_key) = self.oldest_key() {
                self.store.remove(&oldest_key);
                debug!(evicted = %oldest_key, "Evicted oldest cache entry to make room");
            }
        }
        self.store.insert(url, (Arc::new(page), Instant::now()));
    }

    /// Find the key with the oldest insertion time.
    fn oldest_key(&self) -> Option<String> {
        self.store
            .iter()
            .min_by_key(|entry| entry.value().1)
            .map(|entry| entry.key().clone())
    }

    /// Number of entries currently in the cache (including possibly-expired ones).
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Evict all entries that have exceeded their TTL.
    pub fn evict_expired(&self) {
        let ttl = self.ttl;
        self.store.retain(|_, (_, inserted_at)| inserted_at.elapsed() <= ttl);
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_page(url: &str) -> CachedPage {
        CachedPage {
            url: url.to_owned(),
            title: "Test".into(),
            markdown: "# Test content".into(),
            adapter: "test".into(),
        }
    }

    #[test]
    fn insert_and_get() {
        let cache = PageCache::new(60);
        cache.insert("https://a.com".into(), make_page("https://a.com"));
        assert!(cache.get("https://a.com").is_some());
        assert!(cache.get("https://b.com").is_none());
    }

    #[test]
    fn max_capacity_evicts_oldest() {
        let cache = PageCache::with_max_capacity(600, 2);
        cache.insert("https://a.com".into(), make_page("https://a.com"));
        cache.insert("https://b.com".into(), make_page("https://b.com"));
        // At capacity — inserting c should evict the oldest (a)
        cache.insert("https://c.com".into(), make_page("https://c.com"));
        assert_eq!(cache.len(), 2);
        assert!(cache.get("https://b.com").is_some());
        assert!(cache.get("https://c.com").is_some());
    }

    #[test]
    fn evict_expired_removes_old_entries() {
        let cache = PageCache::new(0); // 0-second TTL = immediately expired
        cache.insert("https://a.com".into(), make_page("https://a.com"));
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.evict_expired();
        assert!(cache.is_empty());
    }
}
