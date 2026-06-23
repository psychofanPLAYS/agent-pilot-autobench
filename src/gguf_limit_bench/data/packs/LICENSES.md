# Data Pack Licenses and Attribution

## easy-gotcha.json

Questions in this pack are common-knowledge facts and well-known logical riddles that
have circulated widely as benchmarks for LLM evaluation. No dataset license applies.
Answers have been independently verified. Provenance is noted in the pack's `source`
field: "curated, web-sourced well-known LLM gotchas".

---

## easy-mc.json

This pack contains a frozen sample drawn from three permissively-licensed datasets.

### ARC-Easy (allenai/ai2_arc)

- **Dataset:** allenai/ai2_arc, config `ARC-Easy`, split `train`, rows 0–24
- **Source URL:** https://huggingface.co/datasets/allenai/ai2_arc
- **Paper:** Clark et al., 2018. "Think you have solved question answering? Try ARC,
  the AI2 Reasoning Challenge." https://arxiv.org/abs/1803.05457
- **License:** CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
- **Creator:** Allen Institute for AI (AI2)
- **Notes:** The ARC dataset is licensed under Creative Commons Attribution-ShareAlike
  4.0 International. Redistribution requires attribution and the same license.

### OpenBookQA (allenai/openbookqa)

- **Dataset:** allenai/openbookqa, config `main`, split `train`, rows 0–24
- **Source URL:** https://huggingface.co/datasets/allenai/openbookqa
- **GitHub:** https://github.com/allenai/OpenBookQA
- **Paper:** Mihaylov et al., 2018. "Can a Suit of Armor Conduct Electricity? A New
  Dataset for Open Book Question Answering." https://arxiv.org/abs/1809.02789
- **License:** Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
- **Creator:** Allen Institute for AI (AI2)
- **Notes:** HuggingFace Hub lists the license as "unknown" due to a metadata gap,
  but the official GitHub repository (allenai/OpenBookQA) is explicitly Apache-2.0
  licensed, confirmed via the repository license file.

### CommonsenseQA (tau/commonsense_qa)

- **Dataset:** tau/commonsense_qa, config `default`, split `train`, rows 0–14
- **Source URL:** https://huggingface.co/datasets/tau/commonsense_qa
- **Paper:** Talmor et al., 2019. "CommonsenseQA: A Question Answering Challenge
  Targeting Commonsense Knowledge." https://arxiv.org/abs/1811.00937
- **License:** MIT (https://opensource.org/licenses/MIT)
- **Creator:** Tel Aviv University / tau group
- **Notes:** The dataset is licensed under the MIT License, confirmed on the
  HuggingFace Hub dataset card.
