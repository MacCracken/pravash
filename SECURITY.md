# Security Policy

## Supported Versions

| Version | Language | Status |
|---|---|---|
| 2.0.x | Cyrius | Supported: the current line, the port of 1.3.0 (`docs/development/pravash-port-plan.md`) |
| 1.3.x | Rust | Supported. The final Rust line, frozen in `rust-old/` as the port's oracle. Security fixes ship as 1.3.x |
| < 1.3 | Rust | Not supported |

## Reporting a Vulnerability

If you discover a security vulnerability in pravash, please report it
responsibly through **GitHub Security Advisories**:

1. Go to the [Security tab](../../security/advisories) of this repository.
2. Click **"Report a vulnerability"**.
3. Fill in the details and submit.

**Do not open a public issue for security vulnerabilities.**

## Response Timeline

| Action | Target |
|---|---|
| Acknowledgement | Within **48 hours** |
| Initial assessment | Within **5 business days** |
| Fix for critical severity | Within **14 days** |
| Fix for high severity | Within **30 days** |
| Fix for moderate/low severity | Next scheduled release |

## Scope

This policy covers the `pravash` Rust crate and its published API, and the Cyrius library as it is ported.
The Rust crate's hidden `oracle` feature is port tooling, not public API. Vulnerabilities in dependencies
(hisab, the Cyrius stdlib, or the Rust crates) should be reported to the respective upstream projects (and
flagged here if they affect pravash users).

## Disclosure

We follow coordinated disclosure. Once a fix is released, we will publish a
security advisory crediting the reporter (unless anonymity is requested).
