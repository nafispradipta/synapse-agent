# Documentation Index

This directory is the single source of truth for how `synapse-cortex` is designed, operated, secured, and tracked. `AGENTS.md` at the repository root carries the rules an AI coding agent must follow.

## Documents

| Document | Audience | Update when |
| --- | --- | --- |
| [Architecture.md](Architecture.md) | Developers, maintainers | Modules, data flow, schema, retrieval weights, or optional engines change |
| [Product.md](Product.md) | Maintainers, product owners | Scope, capabilities, non-goals, or user-facing limits change |
| [Operations.md](Operations.md) | Operators, integrators | Install steps, environment variables, MCP setup, scripts, or troubleshooting change |
| [Harness_Setup.md](Harness_Setup.md) | Operators, integrators | Harness detection, CLI setup, or per-client MCP configuration changes |
| [Security_Privacy.md](Security_Privacy.md) | Maintainers, security reviewers | Trust boundaries, data handling, credentials, or a security fix changes |
| [Status.md](Status.md) | Everyone | Every completed task, blocker, or change of focus |
| [Roadmap.md](Roadmap.md) | Maintainers, users | A candidate is accepted, rejected, or completed |
| [Changelog.md](Changelog.md) | Everyone | Policy changes only; history lives in the root `CHANGELOG.md` |
| [Dokumentasi_user.md](Dokumentasi_user.md) | Non-technical readers | The user-visible story changes |
| [archive/README.md](archive/README.md) | Maintainers | An archive rule changes |

## Rules

- `AGENTS.md` owns agent behaviour. Technical documents own facts about the system.
- Root `CHANGELOG.md` is the only changelog history. Do not copy release notes into this directory.
- Every command, path, version, and environment variable must be verifiable from the repository. If it cannot be verified, it does not belong here.
- Roadmap items are candidates, not promises. Never add dates or delivery commitments without the owner's explicit decision.
- Superseded documents move to [archive/](archive/) instead of being deleted, and the replacement is named at the top of the archived copy.
- `docs/superpowers/` contains git-ignored working plans and specs. It is not part of the published documentation set.
- Relative links must resolve. Run the documentation check described in [Status.md](Status.md) before committing documentation changes.
