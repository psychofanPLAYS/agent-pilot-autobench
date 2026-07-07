# 11 - Flag Doctor Doctrine

Status: implementation target, 2026-07-03

pilotBENCHY should discover good llama.cpp settings the way an expert operator
does:

1. Inspect the local command or profile.
2. Check the installed `llama-server --help`.
3. Verify model path, template path, template repo version, and live `/props`.
4. Start from the known-good baseline.
5. Change one behavior at a time.
6. Run tiny probes before expensive benchmark packs.
7. Write receipts for every probe.
8. Treat invalid setup as `preflight_fail`, not as a bad model score.

For Qwen/Froggeric v21.3, the default thinking block is:

```powershell
--jinja `
--chat-template-file 'G:\AI\models\Qwen-Fixed-Chat-Templates\chat_template.jinja' `
--chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' `
--reasoning on `
--reasoning-format deepseek
```

Froggeric tags such as `<|think_on|>` and `<|think_off|>` belong in the message
text, usually the system message. They are not llama.cpp command-line flags.

For any Gemma 4 challenger run, the first checks are template correctness,
thinking/answer-channel discipline, and final-answer parseability.

The cockpit should show the operator these facts before a run:

- installed llama.cpp build and supported flags
- disk template version
- live template version from `/props`
- whether live template matches disk
- Qwen thinking evidence source, either `reasoning_content` or visible content tags
- Gemma BOS result
- final answer parse result

## Current Model Evaluation Shape

The evidence lane should stay split instead of forcing one blended winner:

- Reader lane: long-context QA, summarization, multilingual comprehension.
- Librarian-agent lane: retrieval, browsing, tool use, structured collection,
  citation fidelity, and recovery from bad tool output.

Current selection should be Qwen3.6-first for the librarian-agent lane. If a
Google-family challenger is needed, use Gemma 4.
