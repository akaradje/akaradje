# Router LoRA Fine-Tuning Pipeline

## Goal

Replace the prompt-based router (which uses a full LLM call per query)
with a **fine-tuned small model** that classifies query complexity in <50ms.

## Architecture

```
Query → Fine-tuned classifier (1-8B) → difficulty_score (0.0 - 1.0)
                                           │
                        ┌──────────────────┴──────────────────┐
                        │ 0.0-0.2: TRIVIAL (thinking OFF)      │
                        │ 0.2-0.6: STANDARD (effort medium)    │
                        │ 0.6-0.8: COMPLEX (effort high)       │
                        │ 0.8-1.0: VERY_COMPLEX (effort max)   │
                        └──────────────────────────────────────┘
```

## Training Strategy

1. **Data Generation** — Use DeepSeek V4 Pro to label 10k+ queries with
   difficulty scores (teacher labeling)
2. **Base Model** — Qwen3-1.7B or DeepSeek-V4-Flash distill (small, fast)
3. **Method** — QLoRA (4-bit base + BF16 adapters) via Unsloth/TRL
4. **Output** — Single float (regression) or 4-class softmax
5. **Export** — ONNX for CPU inference (<50ms) or vLLM for GPU

## Files

| File | Purpose |
|------|---------|
| `generate_data.py` | Create labeled training data using V4 Pro as teacher |
| `train_lora.py` | Fine-tune with QLoRA via TRL/Unsloth |
| `export_onnx.py` | Export trained model to ONNX for fast inference |
| `config.yaml` | Axolotl-compatible training config |
| `evaluate.py` | Test accuracy on held-out set |

## Hardware Requirements

- **Training**: 1x RTX 4090 (24GB) or A100 (40GB) — QLoRA on 1-8B model
- **Inference**: CPU only (ONNX) or any GPU (vLLM)
- **Data generation**: DeepSeek API access (~$5-10 for 10k labels)

## Quick Start

```bash
# 1. Generate training data
python generate_data.py --n 10000 --output ../data/router_train.jsonl

# 2. Train
python train_lora.py --config config.yaml

# 3. Export
python export_onnx.py --checkpoint output/final --output router.onnx

# 4. Evaluate
python evaluate.py --model router.onnx --test ../data/router_test.jsonl
```
