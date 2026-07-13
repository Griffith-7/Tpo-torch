# TPO-Torch

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![CI](https://github.com/Griffith-7/tpo-torch/actions/workflows/ci.yml/badge.svg)](https://github.com/Griffith-7/tpo-torch/actions)

**Target Policy Optimization** — a simpler alternative to PPO for RLHF.

Based on [arXiv:2604.06159](https://arxiv.org/abs/2604.06159) (Kaddour, 2026).

## What is TPO?

TPO is an RLHF algorithm. It is **not** a replacement for cross-entropy (CE) training. CE and TPO do different things:

| | Cross-Entropy (SFT) | PPO | TPO |
|---|:---:|:---:|:---:|
| Purpose | Predict next token | Optimize reward via RL | Optimize reward via RL |
| Needs labeled data | Yes | No | No |
| Needs reward model | No | Yes | Yes |
| Needs value/critic head | No | Yes | **No** |
| Needs clipping | No | Yes | **No** |
| Needs importance ratios | No | Yes | **No** |
| Training stability | Stable | Often unstable | **Stable** |

**CE is supervised learning. TPO is reinforcement learning.** They solve different problems. You don't compare them.

TPO competes with **PPO**, not CE. The advantage: TPO gives you the same RLHF capability with much less complexity — no value function, no clipping, no importance sampling ratios.

```
PPO:  loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)   # needs V(s), clipping
TPO:  loss = -target_prob * log P_policy(token)                # needs nothing extra
```

## Install

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
tokenizer.pad_token = tokenizer.eos_token

# Dataset needs: prompt, labels, and 'advantages' (higher = better response)
trainer = TPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=0.1,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

## Architecture

```
                    ┌──────────────────────┐
                    │   Reference Model     │
                    │   (frozen)            │
                    └──────────┬───────────┘
                               │ ref_logits
                               ▼
Prompt + Labels ──▶ ┌──────────────────────┐
                    │    TPO Loss Function  │
                    │                       │
                    │  1. log-odds(P_ref)   │
                    │  2. + advantage/beta  │
                    │  3. sigmoid -> target  │
                    │  4. CE loss vs target  │
                    └──────────┬───────────┘
                               │ loss
                               ▼
                    ┌──────────────────────┐
                    │   Policy Model        │
                    │   (trained)           │
                    └──────────────────────┘
```

## API

| Function | Description |
|----------|-------------|
| `tpo_loss_from_logits(logits, ref_logits, labels, advantages, beta)` | Loss from raw model logits |
| `tpo_loss(policy_logprobs, ref_logprobs, advantages, beta)` | Loss from pre-computed log-probs |
| `TPOTrainer(model, ref_model, beta, ...)` | HuggingFace Trainer with TPO |
| `TPODataCollator(tokenizer)` | Preserves advantages in batches |

## Benchmarks

All benchmarks run on **NVIDIA RTX 3050 Laptop (4GB VRAM)**.

### Training: TPO vs PPO-clip

Both are RLHF methods evaluated on the **same metric**: held-out perplexity. This compares TPO to what it actually replaces — PPO, not CE.

[![Training Curves](benchmarks/results/figures/training_curves.png)](benchmarks/results/figures/training_curves.png)

| Method | Final Perplexity | Implementation Complexity |
|--------|----------------:|:--------------------------|
| CE (baseline, supervised) | 64.27 | Direct token prediction |
| TPO (beta=0.1) | **76.68** | Loss function only |
| PPO-clip (simplified) | 903.81 | Loss + ratio + clipping |

**Note:** Our PPO-clip is simplified (no value function, no GAE). Full PPO with a critic would perform better but requires significantly more code and compute. TPO achieves competitive results with none of that infrastructure.

### Gradient Stability

[![Gradient Stability](benchmarks/results/figures/gradient_stability.png)](benchmarks/results/figures/gradient_stability.png)

- **Zero NaNs** at advantage values from 0.01 to 1000
- Max gradient norm: **0.09** — stable across all regimes

### Speed

[![Speed Scaling](benchmarks/results/figures/speed_scaling.png)](benchmarks/results/figures/speed_scaling.png)

| Seq Len | Latency | Throughput |
|:-------:|--------:|-----------:|
| 32 | 4.17ms | 61,345 tok/s |
| 64 | 8.26ms | 62,021 tok/s |
| 128 | 16.50ms | 62,046 tok/s |
| 256 | 32.81ms | 62,418 tok/s |
| 512 | 65.42ms | 62,614 tok/s |
| 1024 | 929.89ms | 8,810 tok/s |

### Reproduce

```bash
python benchmarks/run_benchmarks.py
```

## CLI

```bash
tpo train --max-steps 10
tpo bench
tpo info
```

## Development

```bash
git clone https://github.com/Griffith-7/tpo-torch.git
cd tpo-torch
pip install -e ".[dev]"
```

Run tests:
```bash
pytest tests/ -v
ruff check tpo_torch/
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- transformers >= 4.40

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

## License

[MIT](LICENSE)
