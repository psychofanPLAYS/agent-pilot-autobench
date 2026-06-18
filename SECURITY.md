# Security Policy

## Supported Version

Security fixes target the latest commit on `main`. The project is currently alpha
software and has not published a stable compatibility guarantee.

## Reporting a Vulnerability

Please use GitHub's private security-advisory flow:

https://github.com/psychofanPLAYS/agent-pilot-autobench/security/advisories/new

Do not include credentials, private model prompts, local receipts, or exploit details in
a public issue. Include the affected command, version/commit, impact, and a minimal safe
reproduction in the private report.

## Security Boundaries

- Benchmark-controlled HTTP servers bind to `127.0.0.1` by default.
- Core subprocess calls use argument lists rather than a shell.
- Extra llama-server arguments cannot override the managed model, host, or port.
- Environment values whose names contain key, token, secret, or password markers are
  redacted from benchmark-suite receipts.
- Models, databases, virtual environments, and receipts remain local and ignored.

This tool executes user-selected local binaries and optional benchmark-plan commands.
Review third-party binaries, model files, plans, and plugins before running them. Do not
run untrusted plans merely because they are valid JSON.
