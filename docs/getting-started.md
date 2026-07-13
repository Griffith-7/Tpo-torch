# TPO-Torch Documentation

## Overview

TPO-Torch is a PyTorch implementation of **Target Policy Optimization (TPO)**, an experimental RLHF training method based on [arXiv:2604.06159](https://arxiv.org/abs/2604.06159) (Kaddour, 2026).

## How TPO Works

Standard RL methods (PPO, GRPO) answer two questions simultaneously:
1. Which completions should gain probability mass?
2. How should the parameters move to realize that change?

TPO **decouples** these. Given scored completions:

```
target_prob = sigmoid(log_odds(P_ref) + advantage / beta)
loss        = -target_prob * log P_policy(token)
```

The gradient is `p^theta - q`, which **vanishes** once the policy matches the target — no clipping, no critic, no importance ratios.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from tpo_torch import TPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Dataset must have 'advantages' column (float: higher = better)
trainer = TPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=0.1,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
```

## CLI Usage

```bash
# Train with synthetic data
tpo train --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 100

# Train with a HuggingFace dataset
tpo train --model Qwen/Qwen2.5-0.5B-Instruct --dataset my-dataset --split train

# Run benchmarks
tpo bench

# Show info
tpo info
```

## API Reference

### `tpo_loss_from_logits(policy_logits, reference_logits, labels, advantages, beta, attention_mask)`

Compute TPO loss from raw model logits.

| Parameter | Type | Description |
|-----------|------|-------------|
| `policy_logits` | `Tensor` | Policy model output logits `[B, T, V]` |
| `reference_logits` | `Tensor` | Reference model output logits `[B, T, V]` |
| `labels` | `Tensor` | Token labels `[B, T]` |
| `advantages` | `Tensor` | Advantage scores `[B]` or `[B, T]` |
| `beta` | `float` | Temperature (default: `0.1`) |
| `attention_mask` | `Tensor` | Optional mask `[B, T]` |

### `TPOTrainer`

Subclass of HuggingFace `Trainer`. Adds `ref_model` and `beta` parameters.

### `TPODataCollator`

Custom data collator that preserves the `advantages` column through batching.

### `TPOModel`

Model wrapper with frozen reference policy mechanism.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- transformers >= 4.40
- datasets >= 2.14
- accelerate >= 0.27
- peft >= 0.10 (optional, for LoRA)

## Citation

```bibtex
@misc{kaddour2026targetpolicyoptimization,
    title={Target Policy Optimization},
    author={Jean Kaddour},
    year={2026},
    eprint={2604.06159},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
