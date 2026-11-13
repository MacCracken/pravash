//! `ai.tsv` — `rust-old/src/ai.rs` (M7 part b, seed 0x70_c0_0b): the daimon and hoosh service configs.
//!
//! Tokens: a `DaimonConfig` is `s:<endpoint>` then `s:<api_key>`, or `n:` for no key; a `HooshConfig` is
//! `s:<endpoint>`. Every row is class A:
//! - `DaimonConfig::default` and `HooshConfig::default` give the config;
//! - `::from_json` takes `s:<document>` and gives the config serde_json reads, or `E:Deserialize`;
//! - `::to_json` takes a document that reads and gives `s:<serde_json::to_string>` of what it read;
//! - `::debug` takes a document that reads and gives `s:<{:?}>` and `s:<{:#?}>` of what it read: Rust's derived
//!   `Debug` for `HooshConfig` and the hand-written one for `DaimonConfig`, which writes `Some("<redacted>")` for a
//!   key.
//!
//! The documents: serde_json's text of configs whose strings mix printable ASCII, the control characters, Latin-1,
//! combining marks, the non-printable classes (format, separators, private use, unassigned) and astral text, seeded
//! and hand-picked; and hand-written ones for each escape serde_json reads and each it refuses (a raw control
//! character, a lone surrogate either way, an unknown escape), the sequence forms, and fields missing, null,
//! duplicated, extra and wrong-typed.

use pravash::ai::{DaimonConfig, HooshConfig};

use crate::grids::Rng;
use crate::tsv::{Class, Module, Tok, s};

fn daimon_toks(c: &DaimonConfig) -> Vec<Tok> {
    vec![
        s(c.endpoint.clone()),
        c.api_key.clone().map_or(Tok::None, s),
    ]
}

fn hoosh_toks(c: &HooshConfig) -> Vec<Tok> {
    vec![s(c.endpoint.clone())]
}

/// One char of a seeded string: printable ASCII most often, then the classes `Debug` treats apart.
fn draw_char(rng: &mut Rng) -> char {
    const PICKS: [u32; 16] = [
        0x200B, 0x200C, 0x200D, 0x2028, 0x2029, 0xFEFF, 0x00A0, 0x00AD, 0x3000, 0xE000, 0xFFFD,
        0xFFFF, 0x1F600, 0xE0041, 0x10FFFF, 0x0301,
    ];
    loop {
        let u = match rng.int(0, 11) {
            0..=4 => rng.int(0x20, 0x7E),
            5 => {
                if rng.int(0, 7) == 0 {
                    0x7F
                } else {
                    rng.int(0, 0x1F)
                }
            }
            6 => rng.int(0x80, 0xFF),
            7 => rng.int(0x300, 0x36F),
            8 => rng.int(0x100, 0xFFFF),
            9 => rng.int(0x1_0000, 0x10_FFFF),
            _ => i64::from(PICKS[usize::try_from(rng.int(0, 15)).unwrap_or(0)]),
        };
        if let Some(c) = u32::try_from(u).ok().and_then(char::from_u32) {
            return c;
        }
    }
}

fn draw_string(rng: &mut Rng) -> String {
    let n = rng.int(0, 12);
    (0..n).map(|_| draw_char(rng)).collect()
}

/// The strings every config row is built from, besides the seeded ones.
fn picked() -> Vec<String> {
    [
        "http://localhost:8090",
        "https://daimon.example:443/v1/agents?x=1&y=\"2\"",
        "",
        "a\"b\\c/d",
        "\t\r\n\u{0}\u{1}\u{1f}\u{7f}",
        "\u{8}\u{c}\u{b}",
        "é ñ ü 中文 日本語",
        "e\u{301} \u{300}start",
        "\u{200b}\u{200c}\u{200d}\u{2060}\u{feff}",
        "\u{a0}\u{2000}\u{3000} \u{2028}\u{2029}",
        "\u{e000}\u{f8ff}\u{f0000}\u{10fffd}",
        "\u{378}\u{fffe}\u{ffff}\u{10ffff}",
        "😀👍🏽🇺🇳",
        "\u{e0041}\u{e007f}\u{fe0f}",
        "\u{85}\u{9f}\u{ad}",
        "'single' `back`",
    ]
    .iter()
    .map(|x| (*x).to_string())
    .collect()
}

fn daimon_of(endpoint: &str, key: Option<&str>) -> DaimonConfig {
    let mut c = DaimonConfig::default();
    c.endpoint = endpoint.to_string();
    c.api_key = key.map(str::to_string);
    c
}

fn hoosh_of(endpoint: &str) -> HooshConfig {
    let mut c = HooshConfig::default();
    c.endpoint = endpoint.to_string();
    c
}

/// Hand-written daimon documents: the escapes serde_json reads and those it refuses, the forms and the fields.
const DAIMON_DOCS: &[&str] = &[
    r#"{"endpoint":"x"}"#,
    r#"{"endpoint":"x","api_key":null}"#,
    r#"{"endpoint":"x","api_key":""}"#,
    r#"{"api_key":"k","endpoint":"x"}"#,
    r#"{"endpoint":"\/\b\f\n\r\t\"\\\u00e9\u0000\u001F\u007f"}"#,
    r#"{"endpoint":"\ud83d\ude00\uD83D\uDE00\u20ac"}"#,
    r#"{"endpoint":"\ud800"}"#,
    r#"{"endpoint":"\udc00"}"#,
    r#"{"endpoint":"\ud800A"}"#,
    r#"{"endpoint":"\ud800\n"}"#,
    r#"{"endpoint":"\ud800x"}"#,
    r#"{"endpoint":"\x41"}"#,
    r#"{"endpoint":"\u12"}"#,
    "{\"endpoint\":\"a\u{1}b\"}",
    "{\"endpoint\":\"a\tb\"}",
    "{\"endpoint\":\"a\u{7f}b\"}",
    r#"{"endpoint":1}"#,
    r#"{"endpoint":null}"#,
    r#"{"endpoint":"x","api_key":5}"#,
    r#"{"endpoint":"x","api_key":["k"]}"#,
    r#"{"api_key":"k"}"#,
    r#"{"endpoint":"x","endpoint":"y"}"#,
    r#"{"endpoint":"x","extra":{"a":[1,"A"]}}"#,
    r#"["x"]"#,
    r#"["x",null]"#,
    r#"["x","k"]"#,
    r#"["x","k",1]"#,
    r#"[]"#,
    r#"null"#,
    r#""x""#,
    r#"{}"#,
    r#" { "endpoint" : "x" , "api_key" : "k" } "#,
];

/// Hand-written hoosh documents.
const HOOSH_DOCS: &[&str] = &[
    r#"{"endpoint":"x"}"#,
    r#"{"endpoint":"x","api_key":"ignored"}"#,
    r#"{"endpoint":"\u00e9\ud83d\ude00"}"#,
    r#"{"endpoint":"\udfff"}"#,
    "{\"endpoint\":\"a\u{1f}b\"}",
    r#"{"endpoint":true}"#,
    r#"{}"#,
    r#"["x"]"#,
    r#"["x",1]"#,
    r#"[]"#,
    r#"{"endpoint":"x","endpoint":"x"}"#,
];

fn daimon_rows(m: &mut Module, docs: &[String]) {
    for doc in docs {
        let read = serde_json::from_str::<DaimonConfig>(doc);
        let out = match &read {
            Ok(c) => daimon_toks(c),
            Err(_) => vec![Tok::Err("Deserialize".to_string())],
        };
        m.row("DaimonConfig::from_json", Class::A, &[s(doc.clone())], &out);
        if let Ok(c) = read {
            let text = serde_json::to_string(&c).unwrap_or_default();
            m.row(
                "DaimonConfig::to_json",
                Class::A,
                &[s(doc.clone())],
                &[s(text)],
            );
            m.row(
                "DaimonConfig::debug",
                Class::A,
                &[s(doc.clone())],
                &[s(format!("{c:?}")), s(format!("{c:#?}"))],
            );
        }
    }
}

fn hoosh_rows(m: &mut Module, docs: &[String]) {
    for doc in docs {
        let read = serde_json::from_str::<HooshConfig>(doc);
        let out = match &read {
            Ok(c) => hoosh_toks(c),
            Err(_) => vec![Tok::Err("Deserialize".to_string())],
        };
        m.row("HooshConfig::from_json", Class::A, &[s(doc.clone())], &out);
        if let Ok(c) = read {
            let text = serde_json::to_string(&c).unwrap_or_default();
            m.row(
                "HooshConfig::to_json",
                Class::A,
                &[s(doc.clone())],
                &[s(text)],
            );
            m.row(
                "HooshConfig::debug",
                Class::A,
                &[s(doc.clone())],
                &[s(format!("{c:?}")), s(format!("{c:#?}"))],
            );
        }
    }
}

pub fn emit() -> Module {
    let mut m = Module::new("ai");
    let mut rng = Rng::new(0x70_c0_0b);
    m.row(
        "DaimonConfig::default",
        Class::A,
        &[],
        &daimon_toks(&DaimonConfig::default()),
    );
    m.row(
        "HooshConfig::default",
        Class::A,
        &[],
        &hoosh_toks(&HooshConfig::default()),
    );
    let mut strings = picked();
    for _ in 0..40 {
        strings.push(draw_string(&mut rng));
    }
    let mut ddocs: Vec<String> = Vec::new();
    for (k, e) in strings.iter().enumerate() {
        let key = &strings[(k * 7 + 3) % strings.len()];
        ddocs.push(serde_json::to_string(&daimon_of(e, None)).unwrap_or_default());
        ddocs.push(serde_json::to_string(&daimon_of(e, Some(key))).unwrap_or_default());
    }
    ddocs.extend(DAIMON_DOCS.iter().map(|d| (*d).to_string()));
    daimon_rows(&mut m, &ddocs);
    let mut hdocs: Vec<String> = strings
        .iter()
        .map(|e| serde_json::to_string(&hoosh_of(e)).unwrap_or_default())
        .collect();
    hdocs.extend(HOOSH_DOCS.iter().map(|d| (*d).to_string()));
    hoosh_rows(&mut m, &hdocs);
    m
}
