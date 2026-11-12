#!/usr/bin/env bash
set -euo pipefail

# Bump version in VERSION file and Cargo.toml
NEW_VERSION="${1:?Usage: $0 <new-version>}"

echo "$NEW_VERSION" > VERSION

# Update Cargo.toml version
sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" Cargo.toml

# Update Cargo.lock
cargo check --quiet 2>/dev/null || true

echo "Bumped to $NEW_VERSION"
