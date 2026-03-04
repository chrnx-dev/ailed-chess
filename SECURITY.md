# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 1.x (latest) | Yes |
| < 1.0 | No |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Use [GitHub's private vulnerability reporting](https://github.com/chrnx-dev/ailed-chess/security/advisories/new) to submit a report confidentially. This ensures the issue can be assessed and patched before public disclosure.

### What to include

- A clear description of the vulnerability
- Steps to reproduce or a minimal proof of concept
- The component affected (e.g., UCI engine, dependency, data pipeline)
- Potential impact

### Response timeline

| Stage | Target |
|---|---|
| Acknowledgement | Within 7 days |
| Assessment | Within 14 days |
| Fix / advisory | Dependent on severity |

## Scope

This is a chess engine middleware library. Security-relevant areas include:

- **Dependency vulnerabilities** — `python-chess`, `pydantic`, `torch`
- **Input validation** — malformed UCI commands or PGN input that could cause unexpected behavior
- **Logic correctness in critical paths** — bugs that silently corrupt psyche state or move selection in ways that could affect downstream systems

Out of scope: psyche tuning disagreements, ELO-related behavior differences, personality configuration choices.
