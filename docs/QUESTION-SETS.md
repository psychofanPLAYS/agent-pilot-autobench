# Writing pilotBENCHY question sets

A question set is **one self-contained file** — system prompt, questions, and correct
answers all together — so it's easy for both humans and models to read, inspect, and
extend. YAML is the canonical authoring format (JSON also loads). Drop a `.yaml` file
in `src/gguf_limit_bench/data/packs/` and the engine picks it up by its `id`.

## Schema

```yaml
id: my-set                      # unique id; the file is loaded by this id
title: My Set                   # human label
tier: easy                      # optional: easy | medium | hard | unknown
answer_type: exact              # exact | multiple_choice
system_prompt: |                # UP TOP, multiline, editable prose
  You are careful and precise. Think step by step.
  You MUST finish with a line: Final Answer: X
questions:
  - id: q1                      # unique within the set
    prompt: |                   # the question (multiline ok)
      How many R's are in STRAWBERRY?
    answer: "3"                 # the correct answer
    accept: [three]             # optional: other answers that also count as correct
    tags: [counting]            # optional
  # multiple_choice questions add a `choices` list OR embed options in the prompt:
  - id: q2
    prompt: |
      Which is a mammal?
    answer: B                   # a single letter A–F
    choices: ["Trout", "Whale", "Sparrow", "Frog"]
```

## Rules the loader enforces (you get a clear error if you break one)
- `id`, `answer_type`, `system_prompt`, and non-empty `questions` are required.
- Each question needs `id`, `prompt`, and `answer`.
- `multiple_choice` answers are a single letter `A`–`F`. `choices` is **optional** —
  omit it if the options are written into the prompt (as SimpleBench does).
- `exact` answers are matched case-insensitively with light normalization; add
  alternates via `accept:`.

## Legacy formats (still load, but don't use for new sets)
- `pack_id` instead of `id`, `question_id` instead of question `id`.
- `eval_data:` instead of `questions:`.
- `system_prompt_ref: some_file.txt` (external system prompt). **Prefer inline
  `system_prompt`** — the whole point is one self-contained file.

## Where sets live
Bundled sets: `src/gguf_limit_bench/data/packs/*.yaml|*.json`. `simple-bench` is the
migrated flagship set (`packs/simple-bench.yaml`). Add your own `.yaml` beside it and
run it by `id` from the cockpit or CLI.
