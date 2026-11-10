//! AI integration — daimon/hoosh client for pravash.

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DaimonConfig {
    pub endpoint: String,
    pub api_key: Option<String>,
}

impl std::fmt::Debug for DaimonConfig {
    /// Redacts `api_key` so it never reaches logs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaimonConfig")
            .field("endpoint", &self.endpoint)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}
impl Default for DaimonConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8090".into(),
            api_key: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HooshConfig {
    pub endpoint: String,
}
impl Default for HooshConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8088".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_defaults() {
        assert_eq!(DaimonConfig::default().endpoint, "http://localhost:8090");
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let cfg = DaimonConfig {
            api_key: Some("secret-token".into()),
            ..DaimonConfig::default()
        };
        let dbg = format!("{cfg:?}");
        assert!(!dbg.contains("secret-token"));
        assert!(dbg.contains("<redacted>"));
    }
}
