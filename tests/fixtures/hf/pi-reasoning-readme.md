---
license: apache-2.0
base_model:
  - Qwen/Qwen3.6-27B
---

# Qwen3.6 27B MTP Pi Reasoning

Recommended llama.cpp profile:

```bash
llama-server -hf bytkim/Qwen3.6-27B-MTP-pi-reasoning-GGUF:Q4_K_M \
  --spec-type draft-mtp --spec-draft-n-max 3 -np 1 \
  --jinja -ngl 99 -fa -c 131072 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --temp 1.0 --top-p 0.95 --top-k 0 --min-p 0
```
