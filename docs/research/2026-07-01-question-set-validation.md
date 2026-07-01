# Question-Set Validation — 2026-07-01

Validation of the AGENT-PULLED benchmark question sets: verifying that each stated
`answer` is actually correct. `simple-bench` (human-pulled, trusted) was intentionally
skipped. No question files were modified — this is a read-only audit.

Method: multiple-choice factual items (`easy-mc`) were cross-checked against source-dataset
gold labels / general knowledge and web search; `easy-gotcha` items were verified by careful
first-principles reasoning (counting, arithmetic, riddle logic); librarian packs were
spot-checked (3–4 items each) by loading the full rendered prompt — including the randomized
choice ordering — and checking the letter answer against the source facts.

## Summary table

| Pack | Total | Verified | Suspect count |
|------|-------|----------|---------------|
| easy-mc | 26 | 26 | 0 |
| easy-gotcha | 24 | 24 | 1 |
| librarian-write-entry | 16 | 4 (spot) | 0 |
| librarian-triage | 16 | 5 (spot) | 0 |
| librarian-dedupe | 12 | 4 (spot) | 0 |
| librarian-gate | 11 | 4 (spot) | 0 |
| librarian-rerank | 14 | 2 (spot) | 0 |
| librarian-compress | 16 | 3 (spot) | 0 |
| librarian-contradiction | 14 | 3 (spot) | 0 |

Net: 1 confirmed suspect (`easy-gotcha` / `bookshelf-worm`). Everything else spot-checked clean.

## Suspects (detail)

### 1. easy-gotcha / `bookshelf-worm` — CONFIRMED WRONG (answer does not match the prompt wording)

Prompt: *"A bookworm starts eating from the **front cover of volume 1** and eats straight
through to the **back cover of volume 2**. If each book is 2 inches thick plus 0.25-inch
covers on each side, and the books are on a shelf in order, how far does the worm travel?"*

Stated answer: **`0.5`**

Why it's wrong: `0.5` is the answer to the *classic* form of this puzzle, which asks for the
distance from the **first page of Vol 1** to the **last page of Vol 2**. With books shelved in
order (V1 left, V2 right), page 1 of V1 sits on its *right* (inner) side and the last page of
V2 sits on its *left* (inner) side, so the worm crosses only the two inner covers:
`0.25 + 0.25 = 0.5`. That is the intended gotcha.

But this item was reworded to say **front cover of V1 → back cover of V2**. Those are the two
*outer* covers (V1's front is the far left, V2's back is the far right). Traversing from one
outer face to the other means eating through *both entire books*:

    V1 front cover (0.25) + V1 pages (2) + V1 back cover (0.25)
  + V2 front cover (0.25) + V2 pages (2) + V2 back cover (0.25)  =  5.0 inches

So under the literal prompt the correct answer is **`5.0`**, not `0.5`. The rewrite inverted the
endpoints and silently broke the puzzle: the "clever" 0.5 answer no longer follows from the text.

Recommended fix (human review): either (a) restore the classic wording — "from the **first
page** of volume 1 to the **last page** of volume 2" — keeping answer `0.5`; or (b) keep the
current front-cover/back-cover wording and change the answer to `5` / `5.0`. Option (a)
preserves the intended gotcha; option (b) makes it a trivial through-both-books sum.

## Notes / non-issues (checked, not flagged)

- **easy-gotcha counting & arithmetic** all verified: `straw-r`=3, `straw-total`=10,
  `mississippi-s`=4, `mississippi-i`=4, `banana-count`=6, `pencil-letters`=6,
  `word-few` ("the alphabet")=11, `half-of-twelve`=6, `multiply-by-zero`=0,
  `divide-30`=29, `months-28`=12 (all months have ≥28 days), `age-riddle`=12
  (3T+12 = 2(T+12) ⇒ T=12), `friday-born`=2024, `two-coins`=quarter+nickel,
  `sister-riddle`=his son, `doctor-riddle`=his mother, `rooster-egg`=roosters don't lay eggs,
  `word-incorrectly`=incorrectly, `yesterday-tomorrow`=in a dictionary,
  `zero-point-nine` (1 > 0.999)=1, decimal comparisons (`9.9>9.11`, `8.9>8.11`) and
  `float-sort` (9.1, 9.11, 9.9) all correct.

- **easy-mc / `csqa-0003`** (fox from city into forest → "natural habitat", C) initially
  looked debatable ("hen house" is a distractor) but matches the official CommonsenseQA gold
  label. Not flagged. All other ARC-Easy / OpenBookQA / CommonsenseQA answers match the
  expected source labels (e.g. fever→bacterial population, lichen algae→food, calcium
  carbonate→neutralizes acid, freezing→fixed position, Earth's core→iron, GPS→atlas,
  fountain-pen ink→blotter, aromas→kitchen).

- **Librarian packs**: the answer letters correspond to *shuffled* choice orderings that differ
  per question, so verification required the full rendered prompt. Every spot-checked item was
  internally consistent, e.g.:
  - `librarian-dedupe-s0-0` pytest≡pytest → A (Duplicate) ✓;
    `-s0-5` bearer-auth vs rate-limit → C (Related) ✓; `-s0-8` React vs Vite → A (Related) ✓.
  - `librarian-gate-s0-2` (query optimization ← reports.py index fact) → A (Inject) ✓;
    `-s0-3` (query optimization ← muted Slack channel) → A (Skip) ✓.
  - `librarian-compress-s0-0` picks the summary with exactly the 3 real facts, excluding the
    "camelCase" fabrication → B ✓; `-s0-7` includes all three user facts, no fabricated
    timezone → A ✓.
  - `librarian-contradiction-s0-0` snake_case→camelCase = Contradicts (C) ✓;
    `-s0-12` master→main rename = Contradicts (A) ✓; `-s0-5` us-east-1 latency = Confirms (C) ✓.
  - `librarian-rerank-s0-0` / `-s0-5`: no snippet answers the query → correctly point at the
    "None of these" option ✓.
  - `librarian-triage` fact-counting (`-s0-13`=2, `-s0-14`=3, `-s0-15`=5) correctly counts
    *distinct durable* facts and ignores filler/duplicate sentences ✓.
