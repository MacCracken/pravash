//! AI integration — daimon/hoosh client for pravash.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaimonConfig {
    pub endpoint: String,
    pub api_key: Option<String>,
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
}
