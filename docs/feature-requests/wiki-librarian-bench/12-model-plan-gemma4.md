# 12 - Model plan: Gemma 4 26B-A4B IT

Status: current Google-family comparator, 2026-07-03

This is the current Gemma-family comparison lane for wiki-librarian work.

## Identity

| Field | Value |
|-------|-------|
| Canonical HF slug | `google/gemma-4-26B-A4B-it` |
| Architecture | `gemma4` |
| Recommended local lane | text-only GGUF or QAT GGUF |
| Role in pilotBENCHY | Google-family challenger against Qwen3.6 |

## Serving Contract

Gemma 4 uses a new chat-template format and native system role support. For
llama.cpp, use the model's Jinja template path or the runtime's Gemma template
support, then prove it in preflight before scoring.

Minimum launch guidance:

```powershell
--jinja `
--temp 1.0 `
--top-p 0.95 `
--top-k 64
```

For thinking-mode tests, enable Gemma 4's template-level thinking control in the
system prompt or template kwargs supported by the specific runtime build, then
verify the answer channel still emits `Final Answer:`.

## Evaluation Use

Use this plan only as the current Google-family challenger. The default
librarian-agent decision lane remains Qwen3.6-first until real local receipts say
otherwise.
