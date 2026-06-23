# Question packs

This folder is the single home for benchmark **question packs**. Drop a new
`*.json` file here and pilotBENCHY will pick it up automatically — no code
changes needed. Each pack is **self-contained**: it carries its own system
prompt, its questions, and the correct answers, so the exact same prompt is
loaded with the questions every time.

## File format

```jsonc
{
  "pack_id": "my-pack",                 // unique id; must match the filename stem
  "title": "Human readable title",
  "tier": "easy",                       // easy | medium | hard (free-form label)
  "answer_type": "multiple_choice",     // "multiple_choice" or "exact"
  "system_prompt": "You are an expert reasoner. Think step by step and take as much reasoning as you need; do not rush. You MUST finish with a line in exactly this format: Final Answer: X",
  "questions": [
    {
      "question_id": "my-0001",
      "prompt": "Which planet is closest to the Sun?",
      "answer": "A",                    // MC: a single letter A-F
      "choices": ["Mercury", "Venus", "Earth", "Mars"],
      "tags": ["astronomy"]
    },
    {
      "question_id": "my-0002",
      "prompt": "How many sides does a hexagon have?",
      "answer": "6",                    // exact: the canonical answer string
      "accept": ["six"],                // optional alternative spellings accepted
      "tags": ["geometry"]
    }
  ]
}
```

### Fields

- **`system_prompt`** — loaded as the system message for every question in the
  pack. Keep the "Final Answer: X" instruction so answers can be extracted even
  from long reasoning. Models are **not** capped on output length while
  answering (they think until they stop), so write prompts that let them reason.
  - You may instead use `"system_prompt_ref": "some_file.txt"` to point at a text
    file in this folder, but inlining keeps the pack self-contained.
- **`answer_type`**
  - `multiple_choice` — each question needs `choices` and an `answer` that is a
    single letter `A`-`F` indexing into `choices`.
  - `exact` — `answer` is the canonical string; add `accept` for alternative
    spellings (e.g. number words). Scoring normalises number words and accepts
    phrase containment.
- **`accept`** — optional list of additional answers counted as correct.
- **`tags`** — optional labels for your own filtering/reporting.

## Notes

- `simple-bench` is loaded from `../simple_bench_public.json` (+
  `../system_prompt.txt`) for licensing reasons; the other packs live here.
- The authoritative answer keys for the bundled `easy-mc` pack and licensing
  details are in [`LICENSES.md`](LICENSES.md).
- After adding a pack, confirm it loads: `apb packs` lists everything found.
