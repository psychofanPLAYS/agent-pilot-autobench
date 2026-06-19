---
license: apache-2.0
base_model:
  - Qwen/Qwen3.6-27B
---

# Qwen3.6 27B MTP Pi Tune

Recommended non-thinking profile:

```bash
llama-server -hf bytkim/Qwen3.6-27B-MTP-pi-tune-GGUF:Q4_K_M \
  --spec-type draft-mtp --spec-draft-n-max 3 -np 1 \
  --jinja -ngl 99 -fa -c 131072 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
  --presence-penalty 1.5
```
