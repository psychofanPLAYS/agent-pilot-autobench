# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses
semantic versioning for published versions.

## [Unreleased]

### Added

- Independent llama-server flag ablations with SimpleBench accuracy, TTFT, throughput,
  slowdown, warning, command, and receipt evidence.
- MTP draft-profile experiments for models explicitly identified as MTP-capable.
- Package-contained, MIT-licensed SimpleBench public data with upstream provenance.
- Architecture/code map, contributor guide, security policy, and release-quality CI job.

### Changed

- Autoresearch inputs now reject invalid datasets, impossible numeric values, and extra
  arguments that could override benchmark-managed model or network bindings.
- One SimpleBench attempt now shares a bounded deadline across its question batch.
- Package metadata and CLI help now use a consistent public product description.
- The codebase is checked by MyPy in addition to Ruff, pytest, compilation, and builds.

### Fixed

- Revised final answers are scored using the latest explicit answer marker.
- Accuracy now always outranks throughput; speed is only a bounded tie-breaker.
- `--dry-run` cannot fall through to a real benchmark without `--flag-ladder`.
- Budget-limited ladders are marked partial, suppress a champion, and bound each
  server attempt to the remaining run budget. Partial evidence is excluded from
  project-wide champion promotion.
- Thread-sweep profiles inherit the documented q8 KV settings.
- Exact server launch arguments are stored as non-executable JSON instead of a `.cmd` file.
- The optional benchmark dependency lock now uses patched `aiohttp 3.14.1`.
- Built wheels include the default SimpleBench dataset and system prompt.
- Duplicate helper definitions and new runner typing regressions were removed.
