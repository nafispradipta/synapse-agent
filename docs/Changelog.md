# Changelog Policy

## Canonical history

The root [`CHANGELOG.md`](../CHANGELOG.md) is the only changelog history for this project. It follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Do not copy release notes into this directory. Two histories drift, and readers cannot tell which one is current.

## When to add an entry

Add an entry under `[Unreleased]` in the root file for any user-visible change:

- Added: new capability, tool, option, or document set.
- Changed: behaviour or defaults that existing users will notice.
- Fixed: a defect with user-visible impact.
- Removed: a capability that no longer exists.
- Security: a vulnerability fix or a hardened boundary.

Internal refactors with no user-visible effect may be omitted from the changelog, but must still be described in the commit message.

## How to write an entry

- Start with a bold summary, then the concrete effect. For example: `**Vault Path Containment:** writeback is confined to the configured vault directory.`
- Name the observable behaviour, not the internal helper.
- Reference the test or document that pins the behaviour when it helps a future reader.
- Keep the newest release at the top. `[Unreleased]` always sits above released versions.

## Release steps

1. Move `[Unreleased]` entries under a new version heading with the release date.
2. Bump `pyproject.toml`.
3. Confirm `synapse/mcp.py` reports the same version through package metadata.
4. Run the full test suite.
5. Commit the release. Do not push without the owner's decision.

## Documentation changes

Documentation-only changes that alter what readers rely on, such as a new required document or a corrected setup command, get an entry under `[Unreleased]` with the `docs:` commit type.
